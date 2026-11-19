// signlang_server: C++/CUDA sign-language recognition server.
//   signlang_server [--root DIR] [--profile primary|fallback] [--port N] [--mode full|safe]  serve (needs SIGNLANG_TOKEN); safe = fallback profile
//                  [--inject-delay-ms N]  failure injection: slows every frame, to exercise the tier-1 degrade
//   signlang_server --extract IN_DIR OUT.json [--blob OUT.f32]           keypoints of every *.jpg in IN_DIR (parity checks)
//   signlang_server --bench N JPEG_FILE                                  per-stage latency percentiles on one frame
#include <spdlog/spdlog.h>
#include <windows.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>

#include "net.h"
#include "pipeline.h"

namespace fs = std::filesystem;

namespace {
NetServer* g_server = nullptr;
BOOL WINAPI on_ctrl(DWORD) {
    if (g_server) g_server->stop();
    return TRUE;
}

std::vector<unsigned char> read_file(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) throw std::runtime_error("cannot read " + p.string());
    return {std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
}

int run_extract(Pipeline& pipe, const std::string& dir, const std::string& out_json, const std::string& blob_path) {
    std::vector<fs::path> files;
    for (auto& e : fs::directory_iterator(dir))
        if (e.path().extension() == ".jpg") files.push_back(e.path());
    std::sort(files.begin(), files.end());
    nlohmann::json out = nlohmann::json::array();
    for (size_t i = 0; i < files.size(); ++i) {
        const auto jpg = read_file(files[i]);
        std::vector<float> blob;
        const Keypoints k = pipe.extract(jpg.data(), jpg.size(), &blob);
        if (i == 0 && !blob_path.empty()) std::ofstream(blob_path, std::ios::binary).write((const char*)blob.data(), blob.size() * sizeof(float));
        out.push_back({{"name", files[i].filename().string()}, {"w", k.width}, {"h", k.height}, {"kpt", k.v}});
    }
    std::ofstream(out_json) << out.dump();
    spdlog::info("extracted {} frames -> {}", files.size(), out_json);
    return 0;
}

int run_bench(Pipeline& pipe, int n, const std::string& jpeg_file) {
    const auto jpg = read_file(jpeg_file);
    StreamState st = pipe.make_state();
    std::vector<double> dec, kp, tmp, tot;
    for (int i = 0; i < n + 20; ++i) {
        const Result r = pipe.process(st, jpg.data(), jpg.size(), i * 100.0);
        if (i < 20) continue;  // warm-up
        dec.push_back(r.ms.decode), kp.push_back(r.ms.keypoints), tmp.push_back(r.ms.temporal), tot.push_back(r.ms.total);
    }
    auto pct = [](std::vector<double> v, double q) { std::sort(v.begin(), v.end()); return v[std::min(v.size() - 1, (size_t)(q * v.size()))]; };
    for (auto& [name, v] : {std::pair<const char*, std::vector<double>&>{"decode", dec}, {"keypoints", kp}, {"temporal", tmp}, {"total", tot}})
        std::printf("%-10s p50 %6.2f  p95 %6.2f  p99 %6.2f ms\n", name, pct(v, 0.5), pct(v, 0.95), pct(v, 0.99));
    return 0;
}
}  // namespace

int main(int argc, char** argv) {
    std::string root_hint, profile, extract_dir, extract_out, blob_path, bench_file;
    int bench_n = 0, port = 0, inject_delay = 0;
    std::string mode = "full";
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&] { if (i + 1 >= argc) { std::fprintf(stderr, "missing value after %s\n", a.c_str()); std::exit(2); } return std::string(argv[++i]); };
        if (a == "--root") root_hint = next();
        else if (a == "--profile") profile = next();
        else if (a == "--port") port = std::stoi(next());
        else if (a == "--mode") mode = next();
        else if (a == "--inject-delay-ms") inject_delay = std::stoi(next());
        else if (a == "--extract") extract_dir = next(), extract_out = next();
        else if (a == "--blob") blob_path = next();
        else if (a == "--bench") bench_n = std::stoi(next()), bench_file = next();
        else { std::fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }
    try {
        spdlog::set_pattern("%Y-%m-%d %H:%M:%S %l %v");
        const std::string root = find_root(root_hint.empty() ? (fs::path(argv[0]).parent_path()).string() : root_hint);
        if (mode != "full" && mode != "safe") { std::fprintf(stderr, "--mode must be full or safe\n"); return 2; }
        if (mode == "safe") profile = "fallback";  // safe mode: the small keypoint model (RTMW-l-m 256x192) from the start
        Config cfg = load_config(root, profile);
        if (port > 0) cfg.port = port;  // e.g. a second instance for testing next to the live one
        const bool serve = extract_dir.empty() && bench_n == 0;
        const char* token = std::getenv("SIGNLANG_TOKEN");
        if (serve && (!token || !*token)) {
            std::fprintf(stderr, "SIGNLANG_TOKEN is not set. Set it with setx, then open a NEW terminal.\n");
            return 2;
        }
        Pipeline pipe(cfg);
        if (inject_delay > 0) pipe.set_inject_delay_ms(inject_delay);  // failure injection: every frame is this much slower
        if (!extract_dir.empty()) return run_extract(pipe, extract_dir, extract_out, blob_path);
        if (bench_n > 0) return run_bench(pipe, bench_n, bench_file);
        NetServer server(cfg, pipe, token);
        g_server = &server;
        SetConsoleCtrlHandler(on_ctrl, TRUE);
        server.run();
        spdlog::info("shut down cleanly");
        g_server = nullptr;
        return 0;
    } catch (const std::exception& e) {
        spdlog::critical("fatal: {}", e.what());
        return 1;
    }
}
