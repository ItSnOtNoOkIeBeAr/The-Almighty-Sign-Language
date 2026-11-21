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
    std::array<unsigned char, 2> hand_pts{};  // diagnostics only: hand keypoints above the loose threshold (of 21 each), left and right
};

// kpt: 133 x (x, y) in frame pixels; score: 133. thr = loose per-model threshold, present_thr = strict one.
FrameRow frame_row(const float* kpt, const float* score, float thr, float present_thr, int width, int height);

// Hand movement between two consecutive ticks in shoulder widths: {wrist L, tips L, wrist R, tips R, motion}. Port of signfeat.hand_motion().
std::array<float, 5> hand_motion(const FrameRow& prev, const FrameRow& cur);

// rows: kWindow frames, oldest first -> out: kWindow x kDim floats.
void window_features(const FrameRow* const* rows, float* out);
}  // namespace feat

// Per connection: the 10 fps recognition clock, the window of tick frames, the decision state machine and the drop counter.
// Port of server/fallback/ticks.py. The client may send 10-30 fps and keypoints are computed for every frame, but the temporal model and the
// decision see ONE frame per 100 ms tick (the frame whose client timestamp is closest to the tick), so a 30-tick window still covers 3 s and
// the decision settings keep their meaning. Per frame: push(), then pop_tick() until it returns false.
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
    // One frame with its client timestamp (every frame, whatever the rate).
    void push(double ts_ms, const feat::FrameRow& row);
    // Next 100 ms tick that fell due with the frame just pushed (false = none left): appends the frame closest to the tick to the tick window.
    // run_model = false for ticks of a gap in the stream: they only fill the window (the model runs for the last kMaxRun due ticks).
    bool pop_tick(double* t_ms, bool* run_model);
    // kWindow ticks ending with the newest; before the stream began the first tick is repeated.
    void window(const feat::FrameRow* out[feat::kWindow]) const;
    // Decision output of a tick: stored; committed words are queued until a reply can carry them (one per reply).
    void record(const DecisionOut& d);
    // What the reply to the current frame says: the newest decision output and the oldest unreported committed word.
    DecisionOut report();
    double t_last() const { return t_last_; }

    Decision decision;
    int profile_id = 0;  // which pipeline stage produced the rows in the window

private:
    static constexpr int kMaxRun = 3;
    static constexpr double kMaxGapMs = 5000.0;
    std::vector<std::string> labels_;
    double step_ms_;
    std::deque<feat::FrameRow> ticks_;
    std::deque<std::string> events_;
    DecisionOut last_{"none", 0.0, "idle", std::nullopt};
    feat::FrameRow cur_, prev_;
    double cur_ts_ = 0, prev_ts_ = 0, next_tick_ = 0, interval_ = -1.0, t_last_ = -1.0;
    bool have_cur_ = false, have_prev_ = false, have_next_ = false;
    int due_ = 0, skip_ = 0, j_ = 0;
};
