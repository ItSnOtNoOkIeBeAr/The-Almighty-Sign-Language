// What the network layer needs from the recognition pipeline, without any CUDA types: the real GPU pipeline implements it,
// and the network tests use a fake one (so the hand-written HTTP/WebSocket code can run under AddressSanitizer without a GPU).
#pragma once
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "features.h"

struct DecodeError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

struct StageMs {
    double decode = 0, keypoints = 0, temporal = 0, total = 0;
};

struct Result {
    std::string label, state, profile;  // profile: which keypoint model produced this (primary | fallback)
    double confidence = 0;
    std::optional<std::string> committed;
    StageMs ms;
    // Only filled when the client asked for debug keypoints: flat [index, x, y, score] per drawn point, in frame pixels.
    std::vector<float> keypoints;
    int kp_w = 0, kp_h = 0;
    float kp_thr = 0;  // the profile's keypoint threshold: points scoring below it are unreliable (the client hides them)
};

class IProcessor {
public:
    virtual ~IProcessor() = default;
    // One frame of one stream. Throws DecodeError for an undecodable frame. Must be thread-safe.
    virtual Result process(StreamState& st, const unsigned char* jpeg, size_t n, double ts_ms, bool want_keypoints) = 0;
    virtual StreamState make_state() const = 0;
};
