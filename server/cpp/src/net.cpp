#include "net.h"

#include <spdlog/spdlog.h>
#include <winsock2.h>
#include <ws2tcpip.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <optional>
#include <set>
#include <thread>
#include <vector>


#pragma comment(lib, "ws2_32.lib")

using json = nlohmann::json;

// ---------------------------------------------------------------- SHA-1 + base64 (handshake only)
namespace {
std::string sha1(const std::string& msg) {
    uint32_t h[5] = {0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0};
    std::string m = msg;
    m.push_back((char)0x80);
    while (m.size() % 64 != 56) m.push_back(0);
    const uint64_t bits = (uint64_t)msg.size() * 8;
    for (int i = 7; i >= 0; --i) m.push_back((char)(bits >> (8 * i)));
    auto rol = [](uint32_t v, int s) { return (v << s) | (v >> (32 - s)); };
    for (size_t off = 0; off < m.size(); off += 64) {
        uint32_t w[80];
        for (int i = 0; i < 16; ++i)
            w[i] = ((uint32_t)(uint8_t)m[off + 4 * i] << 24) | ((uint32_t)(uint8_t)m[off + 4 * i + 1] << 16) |
                   ((uint32_t)(uint8_t)m[off + 4 * i + 2] << 8) | (uint32_t)(uint8_t)m[off + 4 * i + 3];
        for (int i = 16; i < 80; ++i) w[i] = rol(w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16], 1);
        uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4];
        for (int i = 0; i < 80; ++i) {
            uint32_t f, k;
            if (i < 20) f = (b & c) | (~b & d), k = 0x5A827999;
            else if (i < 40) f = b ^ c ^ d, k = 0x6ED9EBA1;
            else if (i < 60) f = (b & c) | (b & d) | (c & d), k = 0x8F1BBCDC;
            else f = b ^ c ^ d, k = 0xCA62C1D6;
            const uint32_t t = rol(a, 5) + f + e + k + w[i];
            e = d, d = c, c = rol(b, 30), b = a, a = t;
        }
        h[0] += a, h[1] += b, h[2] += c, h[3] += d, h[4] += e;
    }
    std::string out;
    for (uint32_t v : h)
        for (int i = 3; i >= 0; --i) out.push_back((char)(v >> (8 * i)));
    return out;
}
}  // namespace

std::string sha1_base64(const std::string& in) {
    static const char* T = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    const std::string d = sha1(in);
    std::string out;
    for (size_t i = 0; i < d.size(); i += 3) {
        const uint32_t v = ((uint8_t)d[i] << 16) | (i + 1 < d.size() ? (uint8_t)d[i + 1] << 8 : 0) | (i + 2 < d.size() ? (uint8_t)d[i + 2] : 0);
        out.push_back(T[(v >> 18) & 63]), out.push_back(T[(v >> 12) & 63]);
        out.push_back(i + 1 < d.size() ? T[(v >> 6) & 63] : '=');
        out.push_back(i + 2 < d.size() ? T[v & 63] : '=');
    }
    return out;
}

// ---------------------------------------------------------------- message validation (mirrors fallback_server.parse)
ParsedMessage parse_message(const unsigned char* data, size_t stored, size_t total, size_t max_bytes) {
    constexpr size_t kHeader = 20;
    ParsedMessage r;
    uint32_t magic = 0;
    uint16_t ver = 0;
    if (stored >= kHeader) {
        uint64_t ts_us;
        std::memcpy(&magic, data, 4), std::memcpy(&ver, data + 4, 2), std::memcpy(&r.flags, data + 6, 2);
        r.flags &= 0xFFFF;
        std::memcpy(&r.frame_id, data + 8, 4), std::memcpy(&ts_us, data + 12, 8);
        r.ts_ms = ts_us / 1000.0;
        r.has_frame_id = magic == 0x534C4E47;
    }
    auto err = [&](const char* code, std::string detail) {
        r.error_code = code, r.error_detail = std::move(detail);
        return r;
    };
    if (total > max_bytes) return err("too_large", std::to_string(total) + " bytes > " + std::to_string(max_bytes));
    if (total < kHeader) return err("too_short", std::to_string(total) + " bytes < 20-byte header");
    if (magic != 0x534C4E47) return err("bad_magic", "magic mismatch");
    if (ver != 1) return err("bad_version", "version " + std::to_string(ver));
    r.payload_offset = kHeader;
    if (total == kHeader && !(r.flags & 1)) return err("empty_payload", "no JPEG bytes");
    return r;
}

// ---------------------------------------------------------------- server
namespace {
using Clock = std::chrono::steady_clock;

constexpr size_t kHardCap = 16u << 20;   // messages up to this size are kept (and validated); bigger ones are drained -> too_large
constexpr size_t kMaxDrain = 64u << 20;  // beyond this a message is refused outright: close 1009, never read it
constexpr size_t kMaxHeader = 16384;
constexpr size_t kMaxHeaders = 100;

struct Frame {
    unsigned id;
    std::vector<unsigned char> payload;
    bool eos, want_keypoints;
    double ts_ms;
};

// RAII socket with a small read buffer. Reads honour a deadline (handshake) or block (after the upgrade).
class Socket {
public:
    explicit Socket(SOCKET s) : s_(s) {}
    ~Socket() { close(); }
    Socket(const Socket&) = delete;
    Socket& operator=(const Socket&) = delete;

    // deadline == time_point{} means no deadline. False = closed, error or deadline passed.
    bool fill(Clock::time_point deadline = {}) {
        if (deadline != Clock::time_point{}) {
            const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - Clock::now()).count();
            if (left <= 0) return false;
            DWORD ms = (DWORD)left;
            setsockopt(s_, SOL_SOCKET, SO_RCVTIMEO, (const char*)&ms, sizeof ms);
        } else {
            DWORD ms = 0;
            setsockopt(s_, SOL_SOCKET, SO_RCVTIMEO, (const char*)&ms, sizeof ms);
        }
        char tmp[65536];
        const int n = recv(s_, tmp, sizeof tmp, 0);
        if (n <= 0) return false;
        buf_.append(tmp, (size_t)n);
        return true;
    }
    bool read_exact(void* dst, size_t n) {
        while (buf_.size() - pos_ < n)
            if (!fill()) return false;
        std::memcpy(dst, buf_.data() + pos_, n);
        pos_ += n;
        compact();
        return true;
    }
    // Up to `n` bytes appended to out when `keep` (at least 1 unless the connection ends); got = bytes consumed.
    bool read_some(std::vector<unsigned char>& out, size_t n, bool keep, size_t& got) {
        if (buf_.size() == pos_ && !fill()) return false;
        const size_t k = std::min(n, buf_.size() - pos_);
        if (keep) out.insert(out.end(), buf_.begin() + (std::ptrdiff_t)pos_, buf_.begin() + (std::ptrdiff_t)(pos_ + k));
        got = k;
        pos_ += k;
        compact();
        return true;
    }
    enum class Head { ok, closed, too_large, timeout };
    // Reads up to and including "\r\n\r\n" before the deadline.
    Head read_head(std::string& head, Clock::time_point deadline) {
        for (;;) {
            const size_t e = buf_.find("\r\n\r\n", pos_);
            if (e != std::string::npos) {
                if (e + 4 - pos_ > kMaxHeader) return Head::too_large;
                head = buf_.substr(pos_, e + 4 - pos_);
                pos_ = e + 4;
                compact();
                return Head::ok;
            }
            if (buf_.size() - pos_ > kMaxHeader) return Head::too_large;
            if (Clock::now() >= deadline) return Head::timeout;
            if (!fill(deadline)) return Clock::now() >= deadline ? Head::timeout : Head::closed;
        }
    }
    bool send_all(const void* p, size_t n) {
        const char* c = static_cast<const char*>(p);
        while (n > 0) {
            const int k = send(s_, c, (int)std::min<size_t>(n, 1 << 20), 0);
            if (k <= 0) return false;
            c += k, n -= (size_t)k;
        }
        return true;
    }
    void close() {
        if (s_ == INVALID_SOCKET) return;
        shutdown(s_, SD_BOTH);
        closesocket(s_);
        s_ = INVALID_SOCKET;
    }

private:
    void compact() {
        if (pos_ == buf_.size()) buf_.clear(), pos_ = 0;
    }
    SOCKET s_;
    std::string buf_;
    size_t pos_ = 0;
};

std::string lower(std::string s) {
    for (auto& c : s) c = (char)std::tolower((unsigned char)c);
    return s;
}

bool constant_time_equal(const std::string& a, const std::string& b) {
    unsigned char diff = a.size() != b.size();
    for (size_t i = 0; i < std::min(a.size(), b.size()); ++i) diff |= (unsigned char)(a[i] ^ b[i]);
    return diff == 0;
}

bool http_reply(Socket& s, int code, const char* reason, const std::string& body, const char* extra = "") {
    std::string r = "HTTP/1.1 " + std::to_string(code) + " " + reason + "\r\nContent-Type: application/json\r\nContent-Length: " +
                    std::to_string(body.size()) + "\r\nConnection: close\r\n" + extra + "\r\n" + body;
    return s.send_all(r.data(), r.size());
}

// One WebSocket connection after the handshake.
class Connection {
public:
    Connection(Socket& s, IProcessor& proc, size_t max_bytes) : s_(s), proc_(proc), max_bytes_(max_bytes), state_(proc.make_state()) {}

    void run() {
        std::thread worker([this] { work(); });
        read_loop();
        {
            std::lock_guard<std::mutex> lk(m_);
            closed_ = true;
        }
        cv_.notify_all();
        worker.join();
    }

private:
    bool send_frame(unsigned char opcode, const std::string& payload) {
        std::string f;
        f.push_back((char)(0x80 | opcode));
        if (payload.size() < 126) f.push_back((char)payload.size());
        else if (payload.size() < 65536) f.push_back(126), f.push_back((char)(payload.size() >> 8)), f.push_back((char)payload.size());
        else {
            f.push_back(127);
            for (int i = 7; i >= 0; --i) f.push_back((char)((uint64_t)payload.size() >> (8 * i)));
        }
        f += payload;
        std::lock_guard<std::mutex> lk(send_m_);
        return s_.send_all(f.data(), f.size());
    }
    void send_close(unsigned code, const std::string& echo_tail = "") {
        std::string p;
        p.push_back((char)(code >> 8)), p.push_back((char)code);
        send_frame(0x8, p + echo_tail);
    }
    void send_json(const json& j) { send_frame(0x1, j.dump()); }
    void send_error(const std::string& code, const std::string& detail, std::optional<unsigned> fid) {
        json j = {{"type", "error"}, {"code", code}, {"detail", detail}};
        if (fid) j["frame_id"] = *fid;
        send_json(j);
    }

    // RFC 6455 framing, strict: anything the protocol does not use gets a close frame (1002 protocol error / 1009 too big) and ends the connection.
    void read_loop() {
        std::vector<unsigned char> msg;
        size_t total = 0;
        unsigned char msg_op = 0;
        bool in_msg = false;
        for (;;) {
            unsigned char h[2];
            if (!s_.read_exact(h, 2)) return;
            const bool fin = h[0] & 0x80;
            const unsigned char op = h[0] & 0x0F;
            const bool masked = h[1] & 0x80;
            uint64_t len = h[1] & 0x7F;
            if ((h[0] & 0x70) || !masked) return send_close(1002);  // reserved bits set, or a client frame that is not masked
            if (!(op == 0 || op == 1 || op == 2 || op == 8 || op == 9 || op == 10)) return send_close(1002);  // unknown opcode
            if (len == 126) {
                unsigned char e[2];
                if (!s_.read_exact(e, 2)) return;
                len = (uint64_t)(e[0] << 8) | e[1];
            } else if (len == 127) {
                unsigned char e[8];
                if (!s_.read_exact(e, 8)) return;
                len = 0;
                for (int i = 0; i < 8; ++i) len = (len << 8) | e[i];
                if (len >> 63) return send_close(1002);  // the top bit must be 0
            }
            unsigned char mask[4];
            if (!s_.read_exact(mask, 4)) return;
            if (op >= 8) {  // control frame: not fragmented, payload <= 125 bytes
                if (!fin || len > 125) return send_close(1002);
                std::string p(len, 0);
                if (len && !s_.read_exact(p.data(), len)) return;
                for (size_t i = 0; i < len; ++i) p[i] = (char)(p[i] ^ mask[i & 3]);
                if (op == 0x8) {  // close: echo the status code (if any) and end
                    if (len == 1) return send_close(1002);
                    send_frame(0x8, p.substr(0, 2));
                    return;
                }
                if (op == 0x9) send_frame(0xA, p);
                continue;
            }
            if (op == 0 && !in_msg) return send_close(1002);   // continuation without a start frame
            if (op != 0 && in_msg) return send_close(1002);    // new message while one is unfinished
            if (op != 0) msg.clear(), total = 0, msg_op = op, in_msg = true;
            if (len > kMaxDrain || total + len > kMaxDrain) return send_close(1009);  // never read it
            // payload in chunks; beyond kHardCap only the length is counted (so a big message still gets a too_large reply)
            const size_t frame_start = msg.size();
            for (uint64_t left = len; left > 0;) {
                const bool keep = msg.size() < kHardCap;
                size_t got = 0;
                if (!s_.read_some(msg, (size_t)std::min<uint64_t>(left, 1 << 16), keep, got)) return;
                left -= got;
                total += got;
            }
            for (size_t i = frame_start; i < msg.size(); ++i) msg[i] ^= mask[(i - frame_start) & 3];  // unmask the stored prefix
            if (!fin) continue;
            handle_message(msg_op, msg, total);
            msg.clear(), total = 0, in_msg = false;
        }
    }

    void handle_message(unsigned char op, const std::vector<unsigned char>& msg, size_t total) {
        if (op != 0x2) return send_error("bad_type", "binary messages only", std::nullopt);
        const ParsedMessage p = parse_message(msg.data(), msg.size(), total, max_bytes_);
        if (!p.error_code.empty()) return send_error(p.error_code, p.error_detail, p.has_frame_id ? std::optional<unsigned>(p.frame_id) : std::nullopt);
        const bool eos = p.flags & 1, want_kp = p.flags & 2;
        std::lock_guard<std::mutex> lk(m_);
        if (eos && total == p.payload_offset) {  // empty end_of_stream: no result, fresh decision state
            frame_.reset();
            reset_ = true;
        } else {
            if (frame_) dropped_++;  // latest-only: the waiting frame is replaced, never queued
            frame_ = Frame{p.frame_id, std::vector<unsigned char>(msg.begin() + (std::ptrdiff_t)p.payload_offset, msg.end()), eos, want_kp, p.ts_ms};
        }
        cv_.notify_one();
    }

    void work() {
        for (;;) {
            std::optional<Frame> f;
            {
                std::unique_lock<std::mutex> lk(m_);
                cv_.wait(lk, [&] { return closed_ || frame_ || reset_; });
                if (closed_) return;
                if (reset_) state_.reset(), reset_ = false;
                f = std::move(frame_);
                frame_.reset();
            }
            if (!f) continue;
            try {
                const Result r = proc_.process(state_, f->payload.data(), f->payload.size(), f->ts_ms, f->want_keypoints);
                json j = {{"type", "result"}, {"frame_id", f->id}, {"label", r.label}, {"confidence", r.confidence},
                          {"state", r.state}, {"committed_label", r.committed ? json(*r.committed) : json(nullptr)},
                          {"server_ms", {{"decode", round2(r.ms.decode)}, {"keypoints", round2(r.ms.keypoints)},
                                         {"temporal", round2(r.ms.temporal)}, {"total", round2(r.ms.total)}}},
                          {"dropped_frames", dropped_.load()}, {"profile", r.profile}};
                if (!r.keypoints.empty()) {  // debug view: [[index, x, y, score], ...] in frame pixels
                    json pts = json::array();
                    for (size_t i = 0; i + 3 < r.keypoints.size(); i += 4)
                        pts.push_back({(int)r.keypoints[i], round2(r.keypoints[i + 1]), round2(r.keypoints[i + 2]), round2(r.keypoints[i + 3])});
                    j["keypoints"] = {{"w", r.kp_w}, {"h", r.kp_h}, {"thr", r.kp_thr}, {"pts", pts}};
                }
                send_json(j);
            } catch (const DecodeError& e) {
                send_error("decode_failed", e.what(), f->id);
            } catch (const std::exception& e) {  // never let one bad frame kill the connection
                spdlog::error("processing failed: {}", e.what());
                send_error("internal", "processing failed", f->id);
            }
            if (f->eos) state_.reset();
        }
    }
    static double round2(double v) { return std::round(v * 100) / 100; }

    Socket& s_;
    IProcessor& proc_;
    size_t max_bytes_;
    StreamState state_;
    std::mutex m_, send_m_;
    std::condition_variable cv_;
    std::optional<Frame> frame_;
    bool closed_ = false, reset_ = false;
    std::atomic<int> dropped_{0};
};

// "GET /path HTTP/1.1" -> method, path; false if it is not exactly three tokens or not HTTP/1.x.
bool parse_request_line(const std::string& line, std::string& method, std::string& path) {
    const size_t a = line.find(' ');
    if (a == std::string::npos || a == 0) return false;
    const size_t b = line.find(' ', a + 1);
    if (b == std::string::npos || b == a + 1) return false;
    method = line.substr(0, a), path = line.substr(a + 1, b - a - 1);
    const std::string ver = line.substr(b + 1);
    return ver == "HTTP/1.1" || ver == "HTTP/1.0";
}
}  // namespace

struct NetServer::Impl {
    const Config& cfg;
    IProcessor& proc;
    std::string expected;  // "Bearer <token>"
    SOCKET listen_sock = INVALID_SOCKET;
    std::atomic<bool> stopping{false};
    std::mutex m;
    std::set<SOCKET> live;
    std::atomic<int> active{0};

    void handle(SOCKET raw) {
        Socket s(raw);
        struct Guard {  // runs BEFORE ~Socket, so stop() never shuts down a socket handle that was already closed and reused
            Impl& i;
            SOCKET r;
            ~Guard() { std::lock_guard<std::mutex> lk(i.m); i.live.erase(r); i.active--; }
        } guard{*this, raw};
        try {
            std::string head;
            const auto deadline = Clock::now() + std::chrono::milliseconds(cfg.handshake_timeout_ms);  // whole request, not per byte (slowloris)
            switch (s.read_head(head, deadline)) {
                case Socket::Head::ok: break;
                case Socket::Head::too_large: http_reply(s, 431, "Request Header Fields Too Large", R"({"detail":"headers too large"})"); return;
                case Socket::Head::timeout: http_reply(s, 408, "Request Timeout", R"({"detail":"request timeout"})"); return;
                case Socket::Head::closed: return;
            }
            const size_t eol = head.find("\r\n");
            std::string method, path;
            if (!parse_request_line(head.substr(0, eol), method, path)) {
                http_reply(s, 400, "Bad Request", R"({"detail":"bad request line"})");
                return;
            }
            std::vector<std::pair<std::string, std::string>> headers;
            for (size_t p = eol; p != std::string::npos && p + 2 < head.size() && headers.size() < kMaxHeaders;) {
                const size_t e = head.find("\r\n", p + 2);
                const std::string l = head.substr(p + 2, e == std::string::npos ? std::string::npos : e - p - 2);
                p = e;
                const size_t c = l.find(':');
                if (c == std::string::npos || c == 0) continue;  // malformed header line: ignored
                const size_t b = l.find_first_not_of(" \t", c + 1);
                headers.emplace_back(lower(l.substr(0, c)), b == std::string::npos ? "" : l.substr(b));
            }
            auto header = [&](const char* k) -> std::string {
                for (auto& h : headers) if (h.first == k) return h.second;
                return "";
            };
            if (method != "GET") {
                http_reply(s, 405, "Method Not Allowed", R"({"detail":"GET only"})");
            } else if (path == "/health") {
                http_reply(s, 200, "OK", json({{"ok", true}, {"server", "cpp"}, {"version", kServerVersion}}).dump());
            } else if (path == "/ws") {
                if (!constant_time_equal(header("authorization"), expected)) {
                    spdlog::warn("rejected /ws upgrade: bad or missing token (HTTP 401)");
                    http_reply(s, 401, "Unauthorized", R"({"detail":"unauthorized"})");
                } else if (lower(header("upgrade")) != "websocket" || header("sec-websocket-key").empty()) {
                    http_reply(s, 400, "Bad Request", R"({"detail":"WebSocket upgrade required"})");
                } else if (header("sec-websocket-version") != "13") {
                    http_reply(s, 426, "Upgrade Required", R"({"detail":"WebSocket version 13 required"})", "Sec-WebSocket-Version: 13\r\n");
                } else {
                    const std::string acc = sha1_base64(header("sec-websocket-key") + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11");
                    const std::string r = "HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Accept: " + acc + "\r\n\r\n";
                    if (!s.send_all(r.data(), r.size())) return;
                    spdlog::info("client connected");
                    Connection(s, proc, cfg.max_message_bytes).run();
                    spdlog::info("client disconnected");
                }
            } else {
                http_reply(s, 404, "Not Found", R"({"detail":"not found"})");
            }
        } catch (const std::exception& e) {
            spdlog::error("connection error: {}", e.what());
        }
    }
};

NetServer::NetServer(const Config& cfg, IProcessor& proc, std::string token) : impl_(new Impl{cfg, proc, "Bearer " + std::move(token)}) {
    WSADATA w;
    if (WSAStartup(MAKEWORD(2, 2), &w) != 0) throw std::runtime_error("WSAStartup failed");
}

NetServer::~NetServer() {
    stop();
    delete impl_;
    WSACleanup();
}

int NetServer::active_connections() const { return impl_->active.load(); }

void NetServer::run() {
    Impl& i = *impl_;
    i.listen_sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (i.listen_sock == INVALID_SOCKET) throw std::runtime_error("socket() failed");
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_port = htons((u_short)i.cfg.port);
    if (inet_pton(AF_INET, i.cfg.host.c_str(), &a.sin_addr) != 1) throw std::runtime_error("bad host '" + i.cfg.host + "'");
    if (bind(i.listen_sock, (sockaddr*)&a, sizeof a) != 0)
        throw std::runtime_error("cannot bind " + i.cfg.host + ":" + std::to_string(i.cfg.port) + " (is another server, e.g. the Python fallback, still running?)");
    if (listen(i.listen_sock, 64) != 0) throw std::runtime_error("listen() failed");
    spdlog::info("signlang_server v{} listening on {}:{} (profile {})", kServerVersion, i.cfg.host, i.cfg.port, i.cfg.profile.name);
    while (!i.stopping) {
        SOCKET c = accept(i.listen_sock, nullptr, nullptr);
        if (c == INVALID_SOCKET) break;
        const int one = 1;
        setsockopt(c, IPPROTO_TCP, TCP_NODELAY, (const char*)&one, sizeof one);
        if (i.active >= i.cfg.max_connections) {  // refuse instead of spawning unbounded threads
            Socket s(c);
            http_reply(s, 503, "Service Unavailable", R"({"detail":"too many connections"})");
            continue;
        }
        {
            std::lock_guard<std::mutex> lk(i.m);
            i.live.insert(c);
            i.active++;
        }
        std::thread([&i, c] { i.handle(c); }).detach();
    }
    stop();
    for (int n = 0; n < 50 && i.active > 0; ++n) Sleep(100);  // let connection threads finish before the pipeline goes away
}

void NetServer::stop() {
    Impl& i = *impl_;
    if (i.stopping.exchange(true)) return;
    if (i.listen_sock != INVALID_SOCKET) closesocket(i.listen_sock);
    std::lock_guard<std::mutex> lk(i.m);
    for (SOCKET c : i.live) shutdown(c, SD_BOTH);
}
