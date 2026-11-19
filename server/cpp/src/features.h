// Keypoints -> temporal-model features. Port of training/signfeat.py (the reference): 57 points + 5 flags per frame,
// 30-frame window normalised by its own median mid-shoulder and shoulder width, features = xy + velocity + flags = 233.
#pragma once
#include <array>
#include <deque>
#include <vector>

#include "decision.h"

namespace feat {
constexpr int kWindow = 30, kPts = 57, kFlags = 5, kDim = kPts * 2 * 2 + kFlags;  // 233
constexpr int kKeypoints = 133;

struct FrameRow {
    std::array<std::array<float, 2>, kPts> pts{};
    std::array<bool, kFlags> flags{};  // loose L, loose R, shoulders ok, strict L, strict R
};

// kpt: 133 x (x, y) in frame pixels; score: 133. thr = loose per-model threshold, present_thr = strict one.
FrameRow frame_row(const float* kpt, const float* score, float thr, float present_thr, int width, int height);

// rows: kWindow frames, oldest first -> out: kWindow x kDim floats.
void window_features(const FrameRow* const* rows, float* out);
}  // namespace feat

// Per connection: recent keypoint rows (kept by client timestamp), the decision state machine and the drop counter.
class StreamState {
public:
    StreamState(std::vector<std::string> labels, DecisionCfg cfg, double step_ms)
        : decision(labels, cfg), labels_(std::move(labels)), step_ms_(step_ms) {}
    // Switch to another keypoint profile (different thresholds and decision settings): everything restarts.
    void set_decision(DecisionCfg cfg) {
        decision = Decision(labels_, cfg);
        reset();
    }
    void reset();
    void push(double ts_ms, const feat::FrameRow& row);
    // kWindow frames at step_ms spacing ending now; each slot holds the newest frame at or before its time.
    void window(const feat::FrameRow* out[feat::kWindow]) const;
    double t_last() const { return t_last_; }

    Decision decision;
    int profile_id = 0;  // which pipeline stage produced the rows in the window

private:
    std::vector<std::string> labels_;
    double step_ms_;
    std::deque<double> times_;
    std::deque<feat::FrameRow> rows_;
    double t_last_ = -1.0;
};
