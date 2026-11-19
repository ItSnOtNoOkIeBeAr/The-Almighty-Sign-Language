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
    times_.clear(), rows_.clear();
    t_last_ = -1.0;
    decision.reset();
}

void StreamState::push(double ts_ms, const feat::FrameRow& row) {
    t_last_ = std::max(ts_ms, t_last_ + 1.0);  // timestamps must increase even if a client sends duplicates
    times_.push_back(t_last_);
    rows_.push_back(row);
    while (times_.size() > 120) times_.pop_front(), rows_.pop_front();
}

void StreamState::window(const feat::FrameRow* out[feat::kWindow]) const {
    for (int k = 0; k < feat::kWindow; ++k) {
        const double target = t_last_ - (feat::kWindow - 1 - k) * step_ms_;
        const auto it = std::upper_bound(times_.begin(), times_.end(), target);  // bisect_right
        const std::ptrdiff_t i = std::max<std::ptrdiff_t>(0, (it - times_.begin()) - 1);
        out[k] = &rows_[(size_t)i];
    }
}
