#include "config.h"

#include <nlohmann/json.hpp>

#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {
json read_json(const fs::path& p) {
    std::ifstream f(p);
    if (!f) throw std::runtime_error("cannot open " + p.string());
    return json::parse(f);
}

void apply_decision(DecisionCfg& d, const json& j) {
    d.gate_frames = j.value("gate_frames", d.gate_frames);
    d.gate_min = j.value("gate_min", d.gate_min);
    d.ema_alpha = j.value("ema_alpha", d.ema_alpha);
    d.commit_threshold = j.value("commit_threshold", d.commit_threshold);
    d.commit_consecutive_M = j.value("commit_consecutive_M", d.commit_consecutive_M);
    d.cooldown_ms = j.value("cooldown_ms", d.cooldown_ms);
    d.join_ms = j.value("join_ms", d.join_ms);
    d.release_frames = j.value("release_frames", d.release_frames);
    d.release_none_frames = j.value("release_none_frames", d.release_none_frames);
    d.repeat_rest_frames = j.value("repeat_rest_frames", d.repeat_rest_frames);
    d.repeat_rest_p = j.value("repeat_rest_p", d.repeat_rest_p);
    d.repeat_cooldown_ms = j.value("repeat_cooldown_ms", d.repeat_cooldown_ms);
    for (auto it = j.begin(); it != j.end(); ++it)  // thr_<word>: per-word commit threshold
        if (it.key().rfind("thr_", 0) == 0 && it.value().is_number()) d.word_threshold[it.key().substr(4)] = it.value().get<double>();
}

ProfileCfg parse_profile(const std::string& root, const json& j, const std::string& name) {
    const json& m = j["models"][name];
    ProfileCfg p;
    p.name = name;
    p.keypoint_engine = (fs::path(root) / m["keypoint_engine"].get<std::string>()).string();
    p.temporal_engine = (fs::path(root) / m["temporal_engine"].get<std::string>()).string();
    p.in_w = m["keypoint_input_wh"][0], p.in_h = m["keypoint_input_wh"][1];
    p.keypoint_thr = m["keypoint_thr"], p.hand_present_thr = m["hand_present_thr"];
    apply_decision(p.decision, j["decision"]);
    if (m.contains("decision")) apply_decision(p.decision, m["decision"]);  // per-profile overrides, as in the Python server
    return p;
}
}  // namespace

std::string find_root(const std::string& hint) {
    for (fs::path start : {fs::path(hint), fs::current_path()}) {
        if (start.empty()) continue;
        for (fs::path p = fs::absolute(start); !p.empty(); p = p.parent_path()) {
            if (fs::exists(p / "protocol" / "labels.json")) return p.string();
            if (p == p.parent_path()) break;
        }
    }
    throw std::runtime_error("repo root (a folder containing protocol/labels.json) not found; pass --root");
}

Config load_config(const std::string& root, const std::string& profile_name) {
    const json j = read_json(fs::path(root) / "server" / "config" / "server.json");
    Config c;
    c.root = root;
    c.host = j["net"]["host"];
    c.port = j["net"]["port"];
    c.max_message_bytes = j["net"]["max_message_bytes"];
    c.handshake_timeout_ms = j["net"].value("handshake_timeout_ms", c.handshake_timeout_ms);
    c.max_connections = j["net"].value("max_connections", c.max_connections);
    c.step_ms = j["window"]["step_ms"];
    c.vram_min_free_gb = j["vram"]["min_free_gb"];
    c.vram_max_use_gb = j["vram"]["max_use_gb"];
    if (j.contains("max_frame")) c.max_frame_w = j["max_frame"][0], c.max_frame_h = j["max_frame"][1];
    c.labels = read_json(fs::path(root) / "protocol" / "labels.json")["labels"].get<std::vector<std::string>>();

    const std::string name = profile_name.empty() ? j["models"]["default_profile"].get<std::string>() : profile_name;
    if (!j["models"].contains(name) || !j["models"][name].is_object()) throw std::runtime_error("unknown profile '" + name + "'");
    c.profile = parse_profile(root, j, name);
    if (name == "primary" && j["models"].contains("fallback")) c.alt = parse_profile(root, j, "fallback");  // degrade target
    if (j.contains("degrade")) {
        const json& g = j["degrade"];
        c.degrade.enabled = g.value("enabled", c.degrade.enabled);
        c.degrade.latency_ms = g.value("latency_ms", c.degrade.latency_ms);
        c.degrade.frames = g.value("frames", c.degrade.frames);
        c.degrade.min_free_vram_gb = g.value("min_free_vram_gb", c.degrade.min_free_vram_gb);
    }
    return c;
}
