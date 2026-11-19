// The 20 cases of server/fallback/test_decision.py, ported one to one: same synthetic sequences, same expectations,
// same settings (server/config/server.json). Both implementations must give the same commits.
#include <gtest/gtest.h>

#include <vector>

#include "config.h"
#include "decision.h"

namespace {
const Config& cfg() {
    static const Config c = load_config(find_root(""), "primary");
    return c;
}
const DecisionCfg& CFG() { return cfg().profile.decision; }
constexpr double DT = 100;  // ms per frame (10 fps)
int warm() { return CFG().gate_min - 1; }  // the idle gate needs the hands in gate_min frames

std::vector<double> p(const std::string& label, double conf = 0.95) {
    const auto& L = cfg().labels;
    const double rest = (1 - conf) / (L.size() - 1);
    std::vector<double> v;
    for (auto& n : L) v.push_back(n == label ? conf : rest);
    return v;
}

struct Seq {
    Decision d;
    double t = 0;
    std::vector<std::pair<double, std::string>> events;
    std::vector<DecisionOut> frames;
    explicit Seq(DecisionCfg c) : d(cfg().labels, c) {}
    Seq() : Seq(CFG()) {}
    Seq& feed(const std::string& label, int n, bool hands = true, double conf = 0.95) {
        for (int i = 0; i < n; ++i) {
            frames.push_back(d.step(p(label, conf), hands, t));
            if (frames.back().committed) events.emplace_back(t, *frames.back().committed);
            t += DT;
        }
        return *this;
    }
    std::vector<std::string> committed() const {
        std::vector<std::string> v;
        for (auto& e : events) v.push_back(e.second);
        return v;
    }
};
using V = std::vector<std::string>;
}  // namespace

TEST(Decision, IdleGateBlocksEverything) {
    Seq r;
    r.feed("hello", 40, false);
    EXPECT_TRUE(r.committed().empty());
    for (auto& f : r.frames) EXPECT_TRUE(f.state == "idle" && f.label == "none");
}

TEST(Decision, CommitsAfterExactlyMFrames) {
    Seq r;
    r.feed("hello", warm() + CFG().commit_consecutive_M - 1);
    EXPECT_TRUE(r.committed().empty());
    r.feed("hello", 1);
    EXPECT_EQ(r.committed(), V{"hello"});
    EXPECT_EQ(r.frames.back().state, "committed");
}

TEST(Decision, BelowThresholdNeverCommits) {
    Seq r;
    r.feed("hello", 40, true, CFG().commit_threshold - 0.1);
    EXPECT_TRUE(r.committed().empty());
    for (size_t i = warm(); i < r.frames.size(); ++i) EXPECT_EQ(r.frames[i].state, "signing");
}

TEST(Decision, SingleFrameSpikeIsSmoothedAway) {
    Seq r;
    r.feed("none", 10).feed("what", 1).feed("none", 10);
    EXPECT_TRUE(r.committed().empty());
}

TEST(Decision, InterruptedRunRestartsTheCount) {
    Seq r;
    r.feed("why", 3).feed("none", 3).feed("why", 3);
    EXPECT_TRUE(r.committed().empty());
}

TEST(Decision, HoldingOneSignFiresOnce) {
    Seq r;
    r.feed("hello", 60);
    EXPECT_EQ(r.committed(), V{"hello"});
}

TEST(Decision, SameSignAgainAfterHandsDownAndCooldown) {
    Seq r;
    r.feed("hello", 10).feed("none", 8, false).feed("hello", 10);
    EXPECT_EQ(r.committed(), (V{"hello", "hello"}));
}

TEST(Decision, CooldownDelaysAQuickSecondCommit) {
    // gate and release shortened so only the cooldown can hold the second commit back
    DecisionCfg c = CFG();
    c.gate_frames = 1, c.gate_min = 1, c.release_frames = 1, c.release_none_frames = 1, c.cooldown_ms = 800;
    Seq r(c);
    r.feed("hello", 4).feed("none", 1).feed("hello", 12);
    EXPECT_EQ(r.committed(), (V{"hello", "hello"}));
    EXPECT_GE(r.events[1].first - r.events[0].first, 800);
}

TEST(Decision, MidSignFlickerToNoneDoesNotRefire) {
    Seq r;
    r.feed("hello", 8).feed("none", 5).feed("hello", 12);
    EXPECT_EQ(r.committed(), V{"hello"});
}

TEST(Decision, LongNoneWithHandsUpReleasesTheWord) {
    Seq r;
    r.feed("hello", 8).feed("none", CFG().release_none_frames + 4).feed("hello", 12);
    EXPECT_EQ(r.committed(), (V{"hello", "hello"}));
}

TEST(Decision, HowThenYouJoins) {
    Seq r;
    r.feed("how", 6).feed("you", 8).feed("none", 20, false);
    EXPECT_EQ(r.committed(), V{"how_are_you"});
}

TEST(Decision, WhereThenYouJoins) {
    Seq r;
    r.feed("where", 6).feed("you", 8).feed("none", 20, false);
    EXPECT_EQ(r.committed(), V{"where_are_you"});
}

TEST(Decision, YouThenWhereJoins) {
    Seq r;
    r.feed("you", 6).feed("where", 8).feed("none", 20, false);
    EXPECT_EQ(r.committed(), V{"where_are_you"});
}

TEST(Decision, HowAloneIsEmittedAfterTheJoinWindow) {
    Seq r;
    r.feed("how", 6);
    EXPECT_TRUE(r.committed().empty());  // held while waiting for YOU
    const double t_sign_end = r.t - DT;  // last frame in which `how` was still the prediction
    r.feed("none", 30, false);
    EXPECT_EQ(r.committed(), V{"how"});
    EXPECT_GT(r.events[0].first - t_sign_end, CFG().join_ms - DT);
}

TEST(Decision, LongSignDoesNotEatTheJoinWindow) {
    // HOW is held for 3 s (commits early), then a 1 s pause, then YOU: still one sentence
    Seq r;
    r.feed("how", 30).feed("none", 10, false).feed("you", 8).feed("none", 20, false);
    EXPECT_EQ(r.committed(), V{"how_are_you"});
}

TEST(Decision, YouAfterTheJoinWindowIsSeparate) {
    Seq r;
    r.feed("how", 6).feed("none", 20, false).feed("you", 8).feed("none", 20, false);
    EXPECT_EQ(r.committed(), (V{"how", "you"}));
}

TEST(Decision, UnrelatedWordFlushesTheHeldWord) {
    Seq r;
    r.feed("how", 6).feed("hello", 8).feed("none", 20, false);
    EXPECT_EQ(r.committed(), (V{"how", "hello"}));  // one event per frame: the second goes out one frame later
    EXPECT_EQ(r.events[1].first - r.events[0].first, DT);
}

TEST(Decision, HelloWhatWhyAreNotDelayed) {
    for (const char* w : {"hello", "what", "why"}) {
        Seq r;
        r.feed(w, 6);
        EXPECT_EQ(r.committed(), V{w});
    }
}

TEST(Decision, SentenceIsOneEventNotTwo) {
    Seq r;
    r.feed("how", 6).feed("you", 8);
    ASSERT_EQ(r.events.size(), 1u);
    EXPECT_EQ(r.events[0].second, "how_are_you");
}

TEST(Decision, ResetClearsState) {
    Seq r;
    r.feed("how", 6);
    r.d.reset();
    r.feed("none", 30, false);
    EXPECT_TRUE(r.committed().empty());
}
