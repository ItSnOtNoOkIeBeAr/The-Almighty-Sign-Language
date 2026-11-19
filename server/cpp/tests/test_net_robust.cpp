// Robustness of the hand-written HTTP/WebSocket layer, against a real NetServer on localhost with a fake (GPU-free) processor:
// malformed HTTP, oversized headers, slowloris, bad WebSocket frames, fragmentation, ping/pong/close, rapid connect/disconnect,
// connection cap. After every abuse the server must still answer /health. Also run under AddressSanitizer (server\scripts\sanitize.ps1).
#include <gtest/gtest.h>
#include <winsock2.h>
#include <ws2tcpip.h>

#include <nlohmann/json.hpp>

#include <atomic>
#include <chrono>
#include <cstring>
#include <random>
#include <thread>

#include "net.h"

using namespace std::chrono_literals;
using Clock = std::chrono::steady_clock;

namespace {
constexpr const char* kToken = "test-token";

struct FakeProcessor : IProcessor {
    Result process(StreamState& st, const unsigned char* d, size_t n, double ts_ms, bool) override {
        if (n >= 3 && std::memcmp(d, "BAD", 3) == 0) throw DecodeError("not a decodable JPEG");
        st.push(ts_ms, feat::FrameRow{});
        Result r;
        r.label = "none", r.state = "idle", r.confidence = 0.5;
        return r;
    }
    StreamState make_state() const override { return StreamState({"none", "hello"}, DecisionCfg{}, 100.0); }
};

int next_port() {
    static std::atomic<int> p{19100};
    return p++;
}

struct TestServer {
    int port;  // first: the initialisers below use it
    Config cfg;
    FakeProcessor proc;
    NetServer net;
    std::thread th;

    static Config make_cfg(int port, int timeout_ms, int max_conn) {
        Config c;
        c.host = "127.0.0.1", c.port = port, c.max_message_bytes = 2 * 1024 * 1024;
        c.handshake_timeout_ms = timeout_ms, c.max_connections = max_conn;
        c.profile.name = "test";
        return c;
    }
    explicit TestServer(int timeout_ms = 300, int max_conn = 64) : port(next_port()), cfg(make_cfg(port, timeout_ms, max_conn)), net(cfg, proc, kToken) {
        th = std::thread([this] { net.run(); });
        for (int i = 0; i < 100; ++i) {  // wait until it listens
            SOCKET s = dial_raw();
            if (s != INVALID_SOCKET) {
                closesocket(s);
                return;
            }
            std::this_thread::sleep_for(20ms);
        }
        ADD_FAILURE() << "server did not start";
    }
    ~TestServer() {
        net.stop();
        if (th.joinable()) th.join();
    }
    SOCKET dial_raw() const {
        SOCKET s = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
        sockaddr_in a{};
        a.sin_family = AF_INET, a.sin_port = htons((u_short)port);
        inet_pton(AF_INET, "127.0.0.1", &a.sin_addr);
        if (connect(s, (sockaddr*)&a, sizeof a) != 0) {
            closesocket(s);
            return INVALID_SOCKET;
        }
        return s;
    }
};

void set_timeout(SOCKET s, int ms) {
    DWORD t = (DWORD)ms;
    setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, (const char*)&t, sizeof t);
}
bool send_raw(SOCKET s, const std::string& d) {
    size_t off = 0;
    while (off < d.size()) {
        const int k = send(s, d.data() + off, (int)std::min<size_t>(d.size() - off, 1 << 20), 0);
        if (k <= 0) return false;
        off += (size_t)k;
    }
    return true;
}
// Everything the server sends until it closes (or `ms` pass).
std::string recv_all(SOCKET s, int ms) {
    std::string out;
    const auto end = Clock::now() + std::chrono::milliseconds(ms);
    char buf[4096];
    while (Clock::now() < end) {
        set_timeout(s, (int)std::chrono::duration_cast<std::chrono::milliseconds>(end - Clock::now()).count() + 1);
        const int n = recv(s, buf, sizeof buf, 0);
        if (n <= 0) break;
        out.append(buf, (size_t)n);
    }
    return out;
}
bool peer_closed(SOCKET s, int ms) {  // true once recv sees EOF/reset within ms (data that arrives first is read and ignored)
    const auto end = Clock::now() + std::chrono::milliseconds(ms);
    char buf[4096];
    while (Clock::now() < end) {
        set_timeout(s, (int)std::chrono::duration_cast<std::chrono::milliseconds>(end - Clock::now()).count() + 1);
        const int n = recv(s, buf, sizeof buf, 0);
        if (n == 0) return true;
        if (n < 0) return WSAGetLastError() != WSAETIMEDOUT;
    }
    return false;
}

std::string http_get_raw(const TestServer& srv, const std::string& request, int wait_ms = 2000) {
    SOCKET s = srv.dial_raw();
    if (s == INVALID_SOCKET) return "<no connection>";
    send_raw(s, request);
    std::string r = recv_all(s, wait_ms);
    closesocket(s);
    return r;
}
std::string get(const TestServer& srv, const std::string& path, const std::string& extra = "") {
    return http_get_raw(srv, "GET " + path + " HTTP/1.1\r\nHost: x\r\n" + extra + "\r\n");
}
bool healthy(const TestServer& srv) { return get(srv, "/health").rfind("HTTP/1.1 200", 0) == 0; }

std::string upgrade_request(const std::string& token = kToken, const std::string& version = "13") {
    return "GET /ws HTTP/1.1\r\nHost: x\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n"
           "Sec-WebSocket-Version: " + version + "\r\n" + (token.empty() ? "" : "Authorization: Bearer " + token + "\r\n") + "\r\n";
}
SOCKET ws_connect(const TestServer& srv) {
    SOCKET s = srv.dial_raw();
    EXPECT_NE(s, INVALID_SOCKET);
    send_raw(s, upgrade_request());
    std::string head;
    char c;
    set_timeout(s, 2000);
    while (head.find("\r\n\r\n") == std::string::npos && recv(s, &c, 1, 0) == 1) head.push_back(c);
    EXPECT_EQ(head.rfind("HTTP/1.1 101", 0), 0u) << head;
    EXPECT_NE(head.find("s3pPLMBiTxaQ9kYGzzhZRbK+xOo="), std::string::npos);
    return s;
}

// Client frame, masked unless told otherwise. `len_override` writes a different declared length than the payload (for bad lengths).
std::string frame(unsigned char op, const std::string& payload, bool fin = true, bool masked = true, int rsv = 0) {
    const unsigned char mask[4] = {1, 2, 3, 4};
    std::string f;
    f.push_back((char)((fin ? 0x80 : 0) | (rsv << 4) | op));
    const unsigned m = masked ? 0x80 : 0;
    if (payload.size() < 126) f.push_back((char)(m | payload.size()));
    else if (payload.size() < 65536) f.push_back((char)(m | 126)), f.push_back((char)(payload.size() >> 8)), f.push_back((char)payload.size());
    else {
        f.push_back((char)(m | 127));
        for (int i = 7; i >= 0; --i) f.push_back((char)((uint64_t)payload.size() >> (8 * i)));
    }
    if (masked) {
        f.append((const char*)mask, 4);
        for (size_t i = 0; i < payload.size(); ++i) f.push_back((char)(payload[i] ^ mask[i & 3]));
    } else {
        f += payload;
    }
    return f;
}
// Header of a frame that declares `len` payload bytes (7/16/64-bit form chosen by the caller via `form`), masked, no payload.
std::string header_only(unsigned char op, uint64_t len, int form) {
    std::string f;
    f.push_back((char)(0x80 | op));
    if (form == 7) f.push_back((char)(0x80 | len));
    else if (form == 16) f.push_back((char)(0x80 | 126)), f.push_back((char)(len >> 8)), f.push_back((char)len);
    else {
        f.push_back((char)(0x80 | 127));
        for (int i = 7; i >= 0; --i) f.push_back((char)(len >> (8 * i)));
    }
    f.append("\1\2\3\4", 4);
    return f;
}

struct WsFrame {
    unsigned op = 0;
    std::string payload;
};
bool read_frame(SOCKET s, WsFrame& f, int ms = 2000) {
    auto exact = [&](void* dst, size_t n) {
        char* p = (char*)dst;
        while (n) {
            set_timeout(s, ms);
            const int k = recv(s, p, (int)n, 0);
            if (k <= 0) return false;
            p += k, n -= (size_t)k;
        }
        return true;
    };
    unsigned char h[2];
    if (!exact(h, 2)) return false;
    f.op = h[0] & 0x0F;
    uint64_t len = h[1] & 0x7F;
    if (len == 126) { unsigned char e[2]; if (!exact(e, 2)) return false; len = (uint64_t)(e[0] << 8) | e[1]; }
    else if (len == 127) { unsigned char e[8]; if (!exact(e, 8)) return false; len = 0; for (int i = 0; i < 8; ++i) len = (len << 8) | e[i]; }
    f.payload.assign((size_t)len, 0);
    return len == 0 || exact(f.payload.data(), (size_t)len);
}
unsigned close_code(const WsFrame& f) { return f.payload.size() >= 2 ? ((unsigned char)f.payload[0] << 8) | (unsigned char)f.payload[1] : 0; }

std::string protocol_msg(unsigned fid, const std::string& payload, uint16_t flags = 0) {
    std::string m(20, 0);
    const uint32_t magic = 0x534C4E47;
    const uint16_t ver = 1;
    const uint64_t ts = 1000ull * fid;
    std::memcpy(&m[0], &magic, 4), std::memcpy(&m[4], &ver, 2), std::memcpy(&m[6], &flags, 2), std::memcpy(&m[8], &fid, 4), std::memcpy(&m[12], &ts, 8);
    return m + payload;
}
bool wait_idle(const TestServer& srv, int ms = 3000) {
    const auto end = Clock::now() + std::chrono::milliseconds(ms);
    while (Clock::now() < end) {
        if (srv.net.active_connections() == 0) return true;
        std::this_thread::sleep_for(20ms);
    }
    return false;
}
}  // namespace

TEST(NetRobust, MalformedHttpIsRejectedAndServerSurvives) {
    TestServer srv;
    std::mt19937 rng(5);
    std::string noise(5000, 0);
    for (auto& c : noise) c = (char)rng();
    const std::vector<std::string> cases = {
        std::string("\x00\x01\x02\xff garbage\r\n\r\n", 17), "GET\r\n\r\n", "GET /ws\r\n\r\n", "GET  HTTP/1.1\r\n\r\n", "POST /health HTTP/1.1\r\n\r\n",
        "GET /health HTTP/2.0\r\n\r\n", "\r\n\r\n", "GET /ws HTTP/1.1\r\nbadheaderline\r\n: novalue\r\n\r\n", "GET /nothing HTTP/1.1\r\n\r\n",
        noise + "\r\n\r\n", "GET /ws HTTP/1.1 extra tokens here\r\n\r\n", "GET / HTTP/1.1\r\nHost: x\r\n\r\n"};
    for (size_t i = 0; i < cases.size(); ++i) {
        const std::string r = http_get_raw(srv, cases[i]);
        EXPECT_TRUE(r.empty() || r.rfind("HTTP/1.1 4", 0) == 0) << "case " << i << " got: " << r.substr(0, 60);
        EXPECT_EQ(r.find("101"), std::string::npos) << "case " << i;
    }
    EXPECT_TRUE(healthy(srv));
}

TEST(NetRobust, HandshakeRejections) {
    TestServer srv;
    EXPECT_EQ(http_get_raw(srv, upgrade_request("")).rfind("HTTP/1.1 401", 0), 0u);
    EXPECT_EQ(http_get_raw(srv, upgrade_request("wrong")).rfind("HTTP/1.1 401", 0), 0u);
    EXPECT_EQ(http_get_raw(srv, upgrade_request(kToken, "8")).rfind("HTTP/1.1 426", 0), 0u);
    EXPECT_EQ(http_get_raw(srv, "GET /ws HTTP/1.1\r\nAuthorization: Bearer test-token\r\n\r\n").rfind("HTTP/1.1 400", 0), 0u);  // no upgrade headers
    EXPECT_TRUE(healthy(srv));
}

TEST(NetRobust, OversizedHeaders) {
    TestServer srv;
    EXPECT_EQ(get(srv, "/health", "X-Big: " + std::string(20000, 'a') + "\r\n").rfind("HTTP/1.1 431", 0), 0u);
    const std::string no_end = "GET /health HTTP/1.1\r\nX: " + std::string(200000, 'b');  // never terminated
    const std::string r = http_get_raw(srv, no_end);
    EXPECT_TRUE(r.empty() || r.rfind("HTTP/1.1 431", 0) == 0) << r.substr(0, 40);
    std::string many;
    for (int i = 0; i < 150; ++i) many += "X-" + std::to_string(i) + ": v\r\n";  // more headers than the cap: extras ignored, request still handled
    EXPECT_EQ(get(srv, "/health", many).rfind("HTTP/1.1 200", 0), 0u);
    EXPECT_TRUE(healthy(srv));
}

TEST(NetRobust, SlowlorisPartialRequestTimesOut) {
    TestServer srv(300);
    SOCKET s = srv.dial_raw();
    ASSERT_NE(s, INVALID_SOCKET);
    send_raw(s, "GET /ws HTT");  // then silence
    const auto t0 = Clock::now();
    const std::string r = recv_all(s, 3000);
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - t0).count();
    closesocket(s);
    EXPECT_LT(ms, 1500) << "connection was held open";
    EXPECT_TRUE(r.empty() || r.rfind("HTTP/1.1 408", 0) == 0) << r.substr(0, 40);
    EXPECT_TRUE(wait_idle(srv));
}

TEST(NetRobust, SlowlorisTrickleIsCutOffByTheTotalDeadline) {
    TestServer srv(300);
    SOCKET s = srv.dial_raw();
    ASSERT_NE(s, INVALID_SOCKET);
    const std::string req = "GET /health HTTP/1.1\r\nX-Slow: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\r\n";  // one byte per 100 ms never finishes in time
    const auto t0 = Clock::now();
    bool cut = false;
    u_long nb = 1;
    ioctlsocket(s, FIONBIO, &nb);
    for (size_t i = 0; i < req.size() && !cut; ++i) {
        if (send(s, &req[i], 1, 0) <= 0) cut = true;
        std::this_thread::sleep_for(100ms);
        char b[256];
        const int n = recv(s, b, sizeof b, 0);
        if (n == 0 || (n < 0 && WSAGetLastError() != WSAEWOULDBLOCK)) cut = true;
        if (n > 0) cut = true;  // the 408 reply
    }
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - t0).count();
    closesocket(s);
    EXPECT_TRUE(cut) << "server kept a trickling connection open for " << ms << " ms";
    EXPECT_LT(ms, 1500);
    EXPECT_TRUE(wait_idle(srv));
}

TEST(NetRobust, BadWebSocketFramesGetACloseFrameAndTheServerSurvives) {
    TestServer srv;
    struct Case { const char* name; std::string bytes; unsigned want; };
    const std::string text_fin0 = frame(1, "abc", false);
    const std::vector<Case> cases = {
        {"opcode 3", frame(3, "x"), 1002}, {"opcode 11", frame(11, "x"), 1002}, {"unmasked text", frame(1, "hi", true, false), 1002},
        {"rsv1 set", frame(2, "x", true, true, 4), 1002}, {"continuation without start", frame(0, "x"), 1002},
        {"fragmented ping", frame(9, "x", false), 1002}, {"ping with 126-byte payload", frame(9, std::string(126, 'p')), 1002},
        {"64-bit length with top bit", header_only(2, 0x8000000000000001ull, 64), 1002},
        {"declared 100 MB", header_only(2, 100u << 20, 64), 1009}, {"declared 2^62", header_only(2, 1ull << 62, 64), 1009},
        {"declared 17 GB in 7+64 bit form", header_only(2, 17ull << 30, 64), 1009},
        {"new message while one is unfinished", text_fin0 + frame(2, "y"), 1002}, {"close with 1-byte payload", frame(8, "x"), 1002}};
    for (const auto& c : cases) {
        SOCKET s = ws_connect(srv);
        send_raw(s, c.bytes);
        WsFrame f;
        ASSERT_TRUE(read_frame(s, f)) << c.name << ": no close frame";
        EXPECT_EQ(f.op, 8u) << c.name;
        EXPECT_EQ(close_code(f), c.want) << c.name;
        EXPECT_TRUE(peer_closed(s, 2000)) << c.name << ": TCP connection stayed open";
        closesocket(s);
        EXPECT_TRUE(healthy(srv)) << "after " << c.name;
    }
    EXPECT_TRUE(wait_idle(srv));
}

TEST(NetRobust, FragmentedMessageWithInterleavedPingWorks) {
    TestServer srv;
    SOCKET s = ws_connect(srv);
    const std::string m = protocol_msg(77, std::string(300, 'j'));
    const std::string a = m.substr(0, 10), b = m.substr(10, 100), c = m.substr(110);
    send_raw(s, frame(2, a, false) + frame(0, b, false) + frame(9, "mid") + frame(0, c, true));
    WsFrame f;
    ASSERT_TRUE(read_frame(s, f));
    EXPECT_EQ(f.op, 10u);  // pong first (it arrived before the message was complete)
    EXPECT_EQ(f.payload, "mid");
    ASSERT_TRUE(read_frame(s, f));
    ASSERT_EQ(f.op, 1u);
    const auto j = nlohmann::json::parse(f.payload);
    EXPECT_EQ(j["type"], "result");
    EXPECT_EQ(j["frame_id"], 77);
    closesocket(s);
}

TEST(NetRobust, PingPongAndCloseHandshake) {
    TestServer srv;
    SOCKET s = ws_connect(srv);
    send_raw(s, frame(9, "abc"));
    WsFrame f;
    ASSERT_TRUE(read_frame(s, f));
    EXPECT_EQ(f.op, 10u);
    EXPECT_EQ(f.payload, "abc");
    send_raw(s, frame(8, std::string("\x03\xE8", 2)));  // close 1000
    ASSERT_TRUE(read_frame(s, f));
    EXPECT_EQ(f.op, 8u);
    EXPECT_EQ(close_code(f), 1000u);
    EXPECT_TRUE(peer_closed(s, 2000));
    closesocket(s);
    EXPECT_TRUE(wait_idle(srv));
}

TEST(NetRobust, TextAndBadMessagesGetErrorRepliesAndTheConnectionLives) {
    TestServer srv;
    SOCKET s = ws_connect(srv);
    auto reply = [&]() { WsFrame f; EXPECT_TRUE(read_frame(s, f)); return nlohmann::json::parse(f.payload); };
    send_raw(s, frame(1, "hello"));
    EXPECT_EQ(reply()["code"], "bad_type");
    send_raw(s, frame(2, "SLNG"));
    EXPECT_EQ(reply()["code"], "too_short");
    send_raw(s, frame(2, std::string(40, 'z')));
    EXPECT_EQ(reply()["code"], "bad_magic");
    send_raw(s, frame(2, protocol_msg(5, "BAD jpeg")));
    const auto e = reply();
    EXPECT_EQ(e["code"], "decode_failed");
    EXPECT_EQ(e["frame_id"], 5);
    send_raw(s, frame(2, protocol_msg(6, "fine")));
    EXPECT_EQ(reply()["type"], "result");
    closesocket(s);
}

TEST(NetRobust, HugeAnnouncedMessageThenDisconnectDoesNotLeakOrCrash) {
    TestServer srv;
    SOCKET s = ws_connect(srv);
    send_raw(s, header_only(2, 5u << 20, 64) + std::string(1 << 20, 'q'));  // announces 5 MB, delivers 1 MB, vanishes
    closesocket(s);
    EXPECT_TRUE(wait_idle(srv));
    s = ws_connect(srv);
    send_raw(s, frame(2, protocol_msg(1, std::string(5 << 20, 'x'))));  // a real 5 MB message: too_large, connection survives
    WsFrame f;
    ASSERT_TRUE(read_frame(s, f, 5000));
    EXPECT_EQ(nlohmann::json::parse(f.payload)["code"], "too_large");
    send_raw(s, frame(2, protocol_msg(2, "ok")));
    ASSERT_TRUE(read_frame(s, f));
    EXPECT_EQ(nlohmann::json::parse(f.payload)["type"], "result");
    closesocket(s);
    EXPECT_TRUE(healthy(srv));
}

TEST(NetRobust, RapidConnectDisconnect) {
    TestServer srv;
    for (int i = 0; i < 300; ++i) {
        SOCKET s = srv.dial_raw();
        ASSERT_NE(s, INVALID_SOCKET) << i;
        switch (i % 5) {
            case 0: break;                                                                  // connect, close
            case 1: send_raw(s, "GET /ws HTTP/1.1\r\nHost"); break;                         // partial request
            case 2: send_raw(s, upgrade_request()); break;                                  // handshake sent, never read
            case 3: { send_raw(s, upgrade_request()); recv_all(s, 50); send_raw(s, frame(2, protocol_msg(i, "x")).substr(0, 9)); break; }  // half a frame
            case 4: { send_raw(s, upgrade_request()); recv_all(s, 50); send_raw(s, frame(2, protocol_msg(i, "x"))); break; }              // frame, then vanish
        }
        closesocket(s);
    }
    EXPECT_TRUE(wait_idle(srv, 5000)) << srv.net.active_connections() << " connections still counted";
    EXPECT_TRUE(healthy(srv));
}

TEST(NetRobust, ConnectionCapAnswers503) {
    TestServer srv(5000, 4);
    std::vector<SOCKET> idle;
    for (int i = 0; i < 4; ++i) idle.push_back(srv.dial_raw());
    const auto end = Clock::now() + 2s;
    while (srv.net.active_connections() < 4 && Clock::now() < end) std::this_thread::sleep_for(10ms);
    ASSERT_EQ(srv.net.active_connections(), 4);
    EXPECT_EQ(http_get_raw(srv, "GET /health HTTP/1.1\r\n\r\n").rfind("HTTP/1.1 503", 0), 0u);
    for (SOCKET s : idle) closesocket(s);
    EXPECT_TRUE(wait_idle(srv));
    EXPECT_TRUE(healthy(srv));
}
