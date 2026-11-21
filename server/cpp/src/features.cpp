#include "features.h"

#include <algorithm>
#include <cmath>

namespace feat {
namespace {
constexpr int kL0 = 15, kR0 = 36;  // first hand point inside the 57
constexpr float kDefaultCx = 320.0f, kDefaultCy = 300.0f, kDefaultScale = 170.0f;  // only when no frame has reliable shoulders

// Indices into the 133 RTMW keypoints: shoulders/elbows/wrists, nose, 4 eyebrow + 4 mouth points, left hand, right hand.
std::array<int, kPts> make_sel() {
    std::array<int, kPts> s{};
    int n = 0;
    for (int i = 5; i <= 10; ++i) s[n++] = i;
    s[n++] = 0;
    for (int i : {19, 24, 21, 22, 48, 54, 51, 57}) s[n++] = 23 + i;
    for (int i = 91; i < 133; ++i) s[n++] = i;
    return s;
}
const std::array<int, kPts> kSel = make_sel();

bool hand_ok(const float* kpt, const float* score, int first, float thr, bool inside_needed, int w, int h) {
    int n = 0;
    for (int i = first; i < first + 21; ++i) n += score[i] >= thr;
    if (n < 15) return false;
    if (!inside_needed) return true;
    int inside = 0;
    for (int i = first; i < first + 21; ++i) {
        const float x = kpt[2 * i], y = kpt[2 * i + 1];
        inside += (x >= 0 && x < w && y >= 0 && y < h - 20);
    }
    return static_cast<double>(inside) / 21.0 >= 0.9;
}

// torch.nanmedian semantics: NaNs ignored, an even count returns the LOWER middle value, no values -> NaN.
float nan_median(std::vector<float>& v) {
    if (v.empty()) return NAN;
    std::sort(v.begin(), v.end());
    return v[(v.size() - 1) / 2];
}
}  // namespace

FrameRow frame_row(const float* kpt, const float* score, float thr, float present_thr, int width, int height) {
    FrameRow r;
    for (int i = 0; i < kPts; ++i) r.pts[i] = {kpt[2 * kSel[i]], kpt[2 * kSel[i] + 1]};
    r.flags = {hand_ok(kpt, score, 91, thr, false, width, height), hand_ok(kpt, score, 112, thr, false, width, height),
               score[5] >= thr && score[6] >= thr, hand_ok(kpt, score, 91, present_thr, true, width, height),
               hand_ok(kpt, score, 112, present_thr, true, width, height)};
    return r;
}

std::array<float, 5> hand_motion(const FrameRow& prev, const FrameRow& cur) {
    float w = 0;
    if (cur.flags[2]) w = std::hypot(cur.pts[0][0] - cur.pts[1][0], cur.pts[0][1] - cur.pts[1][1]);
    if (w < 20) w = kDefaultScale;
    std::array<float, 5> out{};
    float m = 0;
    for (int h = 0; h < 2; ++h) {
        if (!(prev.flags[h] && cur.flags[h])) continue;
        const int base = h == 0 ? kL0 : kR0;
        auto d = [&](int k) { return std::hypot(cur.pts[base + k][0] - prev.pts[base + k][0], cur.pts[base + k][1] - prev.pts[base + k][1]) / w; };
        float tips = 0;
        for (int k : {4, 8, 12, 16, 20}) tips += d(k);
        tips /= 5.0f;
        out[2 * h] = d(0), out[2 * h + 1] = tips;
        m = std::max(m, 0.5f * (out[2 * h] + out[2 * h + 1]));
    }
    out[4] = m;
    return out;
}

void window_features(const FrameRow* const* rows, float* out) {
    // window centre and scale: median over the frames whose shoulders are reliable
    std::vector<float> cx, cy, sw;
    for (int t = 0; t < kWindow; ++t) {
        if (!rows[t]->flags[2]) continue;
        const auto& a = rows[t]->pts[0];
        const auto& b = rows[t]->pts[1];
        cx.push_back(0.5f * (a[0] + b[0]));
        cy.push_back(0.5f * (a[1] + b[1]));
        sw.push_back(std::sqrt((a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1])));
    }
    float mx = nan_median(cx), my = nan_median(cy), s = nan_median(sw);
    if (std::isnan(mx)) mx = kDefaultCx;
    if (std::isnan(my)) my = kDefaultCy;
    if (std::isnan(s) || s < 20) s = kDefaultScale;

    static thread_local float norm[kWindow][kPts][2];
    for (int t = 0; t < kWindow; ++t) {
        for (int p = 0; p < kPts; ++p) {
            const bool present = p < kL0 ? true : (p < kR0 ? rows[t]->flags[0] : rows[t]->flags[1]);
            norm[t][p][0] = present ? (rows[t]->pts[p][0] - mx) / s : 0.0f;
            norm[t][p][1] = present ? (rows[t]->pts[p][1] - my) / s : 0.0f;
        }
    }
    for (int t = 0; t < kWindow; ++t) {
        float* o = out + t * kDim;
        for (int p = 0; p < kPts; ++p) {
            o[2 * p] = norm[t][p][0];
            o[2 * p + 1] = norm[t][p][1];
            // velocity only where the point exists in both frames
            bool both = true;
            if (t == 0) both = false;
            else if (p >= kR0) both = rows[t]->flags[1] && rows[t - 1]->flags[1];
            else if (p >= kL0) both = rows[t]->flags[0] && rows[t - 1]->flags[0];
            o[114 + 2 * p] = both ? norm[t][p][0] - norm[t - 1][p][0] : 0.0f;
            o[114 + 2 * p + 1] = both ? norm[t][p][1] - norm[t - 1][p][1] : 0.0f;
        }
        for (int f = 0; f < kFlags; ++f) o[228 + f] = rows[t]->flags[f] ? 1.0f : 0.0f;
    }
}
}  // namespace feat

void StreamState::reset() {
    ticks_.clear(), events_.clear();
    last_ = {"none", 0.0, "idle", std::nullopt};
    have_cur_ = have_prev_ = have_next_ = false;
    interval_ = -1.0, t_last_ = -1.0, due_ = skip_ = j_ = 0;
    decision.reset();
}

void StreamState::push(double ts_ms, const feat::FrameRow& row) {
    const double ts = t_last_ = std::max(ts_ms, t_last_ + 1.0);  // timestamps must increase even if a client sends duplicates
    if (have_cur_) prev_ = cur_, prev_ts_ = cur_ts_, have_prev_ = true;
    cur_ = row, cur_ts_ = ts, have_cur_ = true;
    if (have_prev_) {
        const double dt = std::min(std::max(ts - prev_ts_, 1.0), 300.0);
        interval_ = interval_ < 0 ? dt : 0.9 * interval_ + 0.1 * dt;  // running average of the frame interval
    }
    const double half = std::min(std::max(0.5 * (interval_ < 0 ? step_ms_ : interval_), 5.0), 50.0);  // a tick fires this early
    if (!have_next_ || ts - next_tick_ > kMaxGapMs) next_tick_ = ts, have_next_ = true;  // first frame, or a long pause: restart the grid
    due_ = skip_ = j_ = 0;
    if (ts >= next_tick_ - half) {
        due_ = static_cast<int>(std::floor((ts - (next_tick_ - half)) / step_ms_)) + 1;
        skip_ = std::max(0, due_ - kMaxRun);
    }
}

bool StreamState::pop_tick(double* t_ms, bool* run_model) {
    if (j_ >= due_) return false;
    const double t = next_tick_;
    const bool use_prev = have_prev_ && std::fabs(prev_ts_ - t) < std::fabs(cur_ts_ - t);  // the closer frame; ties take the newer
    ticks_.push_back(use_prev ? prev_ : cur_);
    if (ticks_.size() > static_cast<size_t>(feat::kWindow)) ticks_.pop_front();
    *t_ms = t, *run_model = j_ >= skip_;
    next_tick_ += step_ms_;
    ++j_;
    return true;
}

void StreamState::window(const feat::FrameRow* out[feat::kWindow]) const {
    const int n = static_cast<int>(ticks_.size());
    for (int k = 0; k < feat::kWindow; ++k) {
        const int i = std::max(0, k - (feat::kWindow - n));
        out[k] = n ? &ticks_[(size_t)i] : &cur_;
    }
}

void StreamState::record(const DecisionOut& d) {
    if (d.committed) events_.push_back(*d.committed);
    last_ = d;
    if (last_.state == "committed") last_.state = "signing";
    last_.committed.reset();
}

DecisionOut StreamState::report() {
    DecisionOut d = last_;
    if (!events_.empty()) {
        d.committed = events_.front();
        d.state = "committed";
        events_.pop_front();
    }
    return d;
}
