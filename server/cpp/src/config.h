// server/config/server.json + protocol/labels.json, resolved against the repo root.
#pragma once
#include <optional>
#include <string>
#include <vector>

#include "decision.h"

struct ProfileCfg {
    std::string name, keypoint_engine, temporal_engine;
    int in_w = 0, in_h = 0;
    float keypoint_thr = 0, hand_present_thr = 0;
    DecisionCfg decision;
};

// Tier-1 degrade (BUILD_SPEC section 7): drop to the small fallback keypoint model when ...
struct DegradeCfg {
    bool enabled = true;
    double latency_ms = 80;         // ... server time per frame stays above this for `frames` frames in a row,
    int frames = 30;
    double min_free_vram_gb = 1.0;  // ... or free VRAM falls below this (checked every 100 frames)
};

struct Config {
    std::string root, host;
    int port = 0;
    size_t max_message_bytes = 0;
    int handshake_timeout_ms = 5000;  // the whole HTTP request must arrive within this (slowloris)
    int max_connections = 64;
    int max_frame_w = 3840, max_frame_h = 2160;
    double step_ms = 100;
    double vram_min_free_gb = 4, vram_max_use_gb = 10;
    std::vector<std::string> labels;
    ProfileCfg profile;                // the profile the server starts on
    std::optional<ProfileCfg> alt;     // degrade target: the fallback profile, when the server starts on the primary one
    DegradeCfg degrade;
};

// Finds the repo root (folder holding protocol/labels.json) from `hint` upwards, then from the working directory.
std::string find_root(const std::string& hint);
Config load_config(const std::string& root, const std::string& profile_name);  // "" = default_profile
