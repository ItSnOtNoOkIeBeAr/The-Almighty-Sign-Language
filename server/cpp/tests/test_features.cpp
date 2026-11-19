// features.cpp against the Python reference (training/signfeat.py) on golden data, plus the window selection by timestamp.
#include <gtest/gtest.h>

#include <cmath>

#include "features.h"
#include "golden.h"

TEST(Features, WindowFeaturesMatchPython) {
    int ci = 0;
    for (const auto& c : golden()["features"]) {
        std::vector<feat::FrameRow> rows;
        for (int t = 0; t < feat::kWindow; ++t) {
            const auto kpt = c["kpt"][t].get<std::vector<float>>();
            const auto score = c["score"][t].get<std::vector<float>>();
            rows.push_back(feat::frame_row(kpt.data(), score.data(), 1.9f, 3.44f, 640, 480));
            for (int f = 0; f < feat::kFlags; ++f) ASSERT_EQ((int)rows.back().flags[f], c["flags"][t][f].get<int>()) << "case " << ci << " frame " << t << " flag " << f;
        }
        std::vector<const feat::FrameRow*> ptrs;
        for (auto& r : rows) ptrs.push_back(&r);
        std::vector<float> out(feat::kWindow * feat::kDim);
        feat::window_features(ptrs.data(), out.data());
        const auto want = c["features"].get<std::vector<float>>();
        ASSERT_EQ(want.size(), out.size());
        double worst = 0;
        for (size_t i = 0; i < out.size(); ++i) worst = std::max(worst, (double)std::fabs(out[i] - want[i]));
        EXPECT_LT(worst, 2e-5) << "case " << ci;  // float32 vs float32 maths; golden is rounded to 6 decimals
        ++ci;
    }
}

TEST(Features, WindowPicksNewestFrameAtOrBeforeEachSlot) {
    StreamState st({"none", "hello"}, DecisionCfg{}, 100.0);
    feat::FrameRow r;
    for (int i = 0; i < 5; ++i) {  // frames at 0, 100, 200, 300, 400 ms; marker in the first point
        r.pts[0][0] = (float)i;
        st.push(i * 100.0, r);
    }
    const feat::FrameRow* w[feat::kWindow];
    st.window(w);
    // slot k holds the frame at t_last - (29-k)*100 ms: frames 0..4 for the last five slots, the first frame before the stream began
    for (int k = 0; k < 25; ++k) EXPECT_EQ(w[k]->pts[0][0], 0.0f) << k;
    for (int k = 25; k < 30; ++k) EXPECT_EQ(w[k]->pts[0][0], (float)(k - 25)) << k;
}

TEST(Features, DroppedFramesDoNotStretchTime) {
    StreamState st({"none", "hello"}, DecisionCfg{}, 100.0);
    feat::FrameRow r;
    for (double t : {0.0, 100.0, 500.0}) {  // a gap: 200..400 ms never arrived
        r.pts[0][0] = (float)t;
        st.push(t, r);
    }
    const feat::FrameRow* w[feat::kWindow];
    st.window(w);
    EXPECT_EQ(w[29]->pts[0][0], 500.0f);
    EXPECT_EQ(w[28]->pts[0][0], 100.0f);  // slots at 400, 300, 200 ms repeat the newest earlier frame
    EXPECT_EQ(w[25]->pts[0][0], 100.0f);
    EXPECT_EQ(w[24]->pts[0][0], 0.0f);    // slot at 0 ms: the first frame (as are all earlier slots)
    EXPECT_EQ(w[23]->pts[0][0], 0.0f);
}

TEST(Features, DuplicateTimestampsStillIncrease) {
    StreamState st({"none", "hello"}, DecisionCfg{}, 100.0);
    feat::FrameRow r;
    st.push(1000.0, r);
    st.push(1000.0, r);
    EXPECT_EQ(st.t_last(), 1001.0);
}
