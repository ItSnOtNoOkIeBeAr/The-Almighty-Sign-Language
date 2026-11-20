// features.cpp against the Python reference (training/signfeat.py) on golden data, plus the window selection by timestamp.
#include <gtest/gtest.h>

#include <cstdlib>
#include <vector>

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

namespace {
// Pushes frames with the given client timestamps (marker = frame index in the first point) and returns every tick as {tick ms, chosen frame, run model}.
struct TickRec { double t; int frame; bool run; };
std::vector<TickRec> run_ticks(StreamState& st, const std::vector<double>& ts) {
    std::vector<TickRec> out;
    feat::FrameRow r;
    for (size_t i = 0; i < ts.size(); ++i) {
        r.pts[0][0] = (float)i;
        st.push(ts[i], r);
        double t = 0;
        bool run = false;
        while (st.pop_tick(&t, &run)) {
            const feat::FrameRow* w[feat::kWindow];
            st.window(w);
            out.push_back({t, (int)w[feat::kWindow - 1]->pts[0][0], run});
        }
    }
    return out;
}
StreamState make_stream() { return StreamState({"none", "hello"}, DecisionCfg{}, 100.0); }
std::vector<double> stream_at(double fps, double seconds, double jitter, unsigned seed) {
    std::srand(seed);
    std::vector<double> ts;
    for (int i = 0; i < (int)(seconds * fps); ++i)
        ts.push_back(1000.0 + i * 1000.0 / fps + (i ? ((std::rand() / (double)RAND_MAX) * 2 - 1) * jitter : 0.0));
    return ts;
}
}  // namespace

TEST(Ticks, ExactlyTenTicksPerSecondAtEveryRate) {
    for (double fps : {10.0, 12.0, 15.0, 20.0, 25.0, 30.0}) {
        auto st = make_stream();
        const auto ticks = run_ticks(st, stream_at(fps, 10.0, 0.0, 1));
        ASSERT_GE(ticks.size(), 95u) << fps;
        for (size_t i = 1; i < ticks.size(); ++i) EXPECT_NEAR(ticks[i].t - ticks[i - 1].t, 100.0, 1e-9) << fps;
    }
}

TEST(Ticks, TenFpsInputPassesThroughFrameForFrame) {
    auto st = make_stream();
    const auto ticks = run_ticks(st, stream_at(10.0, 5.0, 8.0, 3));
    for (size_t i = 0; i < ticks.size(); ++i) EXPECT_EQ(ticks[i].frame, (int)i);
}

TEST(Ticks, EachTickUsesTheFrameClosestToIt) {
    for (double fps : {15.0, 20.0, 30.0}) {
        const auto ts = stream_at(fps, 6.0, 3.0, 5);
        auto st = make_stream();
        const auto ticks = run_ticks(st, ts);
        for (size_t k = 0; k + 1 < ticks.size(); ++k) {
            double best = 1e9;
            for (double t : ts) best = std::min(best, std::fabs(t - ticks[k].t));
            EXPECT_LE(std::fabs(ts[(size_t)ticks[k].frame] - ticks[k].t), best + 1000.0 / fps / 2 + 1e-9) << fps << " tick " << k;
        }
    }
}

TEST(Ticks, AGapFillsTheMissedTicksAndOnlyRunsTheLastOnes) {
    auto st = make_stream();
    const auto ticks = run_ticks(st, {0.0, 100.0, 200.0, 1000.0});  // 0.8 s without frames
    ASSERT_EQ(ticks.size(), 11u);
    EXPECT_EQ(ticks[3].t, 300.0);
    for (size_t i = 3; i < 11; ++i) EXPECT_EQ(ticks[i].run, i >= 8) << i;  // 8 gap ticks, only the last 3 run the model
    for (size_t i = 3; i < 6; ++i) EXPECT_EQ(ticks[i].frame, 2) << i;      // held: the nearest frame
    for (size_t i = 8; i < 11; ++i) EXPECT_EQ(ticks[i].frame, 3) << i;
}

TEST(Ticks, ALongPauseRestartsTheGrid) {
    auto st = make_stream();
    const auto ticks = run_ticks(st, {0.0, 100.0, 20000.0, 20100.0});
    ASSERT_EQ(ticks.size(), 4u);
    EXPECT_EQ(ticks[2].t, 20000.0);
    EXPECT_EQ(ticks[3].t, 20100.0);
}

TEST(Ticks, WindowRepeatsTheFirstTickBeforeTheStreamBegan) {
    auto st = make_stream();
    feat::FrameRow r;
    for (int i = 0; i < 5; ++i) {
        r.pts[0][0] = (float)i;
        st.push(i * 100.0, r);
        double t;
        bool run;
        while (st.pop_tick(&t, &run)) {}
    }
    const feat::FrameRow* w[feat::kWindow];
    st.window(w);
    for (int k = 0; k < 25; ++k) EXPECT_EQ(w[k]->pts[0][0], 0.0f) << k;
    for (int k = 25; k < 30; ++k) EXPECT_EQ(w[k]->pts[0][0], (float)(k - 25)) << k;
}

TEST(Ticks, DuplicateTimestampsStillIncrease) {
    auto st = make_stream();
    feat::FrameRow r;
    st.push(1000.0, r);
    st.push(1000.0, r);
    EXPECT_EQ(st.t_last(), 1001.0);
}

TEST(Ticks, CommittedWordsAreReportedOncePerReply) {
    auto st = make_stream();
    st.record({"hello", 0.9, "committed", std::string("hello")});
    st.record({"hello", 0.9, "committed", std::string("what")});  // two events in one frame (after a gap): one per reply
    DecisionOut a = st.report(), b = st.report(), c = st.report();
    EXPECT_EQ(a.committed.value_or(""), "hello");
    EXPECT_EQ(a.state, "committed");
    EXPECT_EQ(b.committed.value_or(""), "what");
    EXPECT_FALSE(c.committed.has_value());
    EXPECT_EQ(c.state, "signing");  // frames between ticks repeat the newest output, never "committed"
}

TEST(Ticks, MatchesPythonOnGoldenStreams) {
    int n_streams = 0;
    for (const auto& g : golden()["ticks"]) {
        auto st = make_stream();
        const auto ts = g["ts"].get<std::vector<double>>();
        const auto ticks = run_ticks(st, ts);
        const auto& exp = g["expected"];
        ASSERT_EQ(ticks.size(), exp.size()) << "stream " << n_streams;
        for (size_t i = 0; i < ticks.size(); ++i) {
            ASSERT_NEAR(ticks[i].t, exp[i][0].get<double>(), 1e-6) << "stream " << n_streams << " tick " << i;  // golden holds 3 decimals
            ASSERT_EQ(ticks[i].frame, exp[i][1].get<int>()) << "stream " << n_streams << " tick " << i;
            ASSERT_EQ(ticks[i].run, exp[i][2].get<bool>()) << "stream " << n_streams << " tick " << i;
        }
        ++n_streams;
    }
    EXPECT_EQ(n_streams, 5);
}
