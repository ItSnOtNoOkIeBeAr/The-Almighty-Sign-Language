// The 20 cases of server/fallback/test_decision.py, ported one to one: same synthetic sequences, same expectations,
// same settings (server/config/server.json). Both implementations must give the same commits.
#include <gtest/gtest.h>

#include <vector>

#include "config.h"
#include "decision.h"
#include "golden.h"

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
    Seq& feed(const std::string& label, int n, bool hands = true, double conf = 0.95, double motion = 0.0) {
        for (int i = 0; i < n; ++i) {
            frames.push_back(d.step(p(label, conf), hands, t, motion));
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
    // (and the same-word guard off: this test is about the generic cooldown, the guard has its own tests)
    DecisionCfg c = CFG();
    c.gate_frames = 1, c.gate_min = 1, c.release_frames = 1, c.release_none_frames = 1, c.cooldown_ms = 800, c.repeat_rest_frames = 0;
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


// ---- same-word guard: ported one to one from SameWordGuardTests in server/fallback/test_decision.py ----
namespace {
const char* const WORDS[] = {"hello", "what", "how", "why", "where", "you"};

// One long sign that the hand tracker loses twice (the idle gate closes while the model still says `w`), plus a 2-frame dip to another word.
Seq& one_sign_with_dip(Seq& r, const std::string& w, const std::string& other) {
    return r.feed(w, 8).feed(w, 6, false).feed(w, 8).feed(other, 2).feed(w, 8).feed(w, 5, false).feed(w, 10);
}
}  // namespace

TEST(SameWordGuard, IsOnInTheDeployedSettings) { EXPECT_GT(CFG().repeat_rest_frames, 0); }

TEST(SameWordGuard, OneLongSignWithAMidSignDipCommitsOnceForEveryWord) {
    for (const std::string w : WORDS) {
        Seq r;
        one_sign_with_dip(r, w, w == "what" ? "why" : "what").feed("none", 30, false);
        EXPECT_EQ(r.committed(), V{w}) << w;
    }
}

TEST(SameWordGuard, WithoutTheGuardTheSameSignCommitsMoreThanOnce) {
    DecisionCfg c = CFG();
    c.repeat_rest_frames = 0;
    Seq r(c);
    r.feed("hello", 8).feed("hello", 6, false).feed("hello", 10).feed("none", 30, false);
    EXPECT_GE(r.committed().size(), 2u);
}

TEST(SameWordGuard, TwoSeparateSignsWithTheHandsDownBetweenCommitTwiceForEveryWord) {
    for (const std::string w : WORDS) {
        Seq r;
        r.feed(w, 8).feed("none", 10, false).feed(w, 8).feed("none", 30, false);
        EXPECT_EQ(r.committed(), (V{w, w})) << w;
    }
}

TEST(SameWordGuard, TwoSignsSeparatedOnlyByTheModelFallingBackToNoneCommitTwice) {
    Seq r;
    r.feed("hello", 8).feed("none", CFG().release_none_frames + 2).feed("hello", 8).feed("none", 30, false);
    EXPECT_EQ(r.committed(), (V{"hello", "hello"}));
}

TEST(SameWordGuard, AShorterWobbleThanTheRestTimeIsStillOneSign) {
    Seq r;
    r.feed("hello", 8).feed("none", CFG().repeat_rest_frames - 1).feed("hello", 8).feed("none", 30, false);
    EXPECT_EQ(r.committed(), V{"hello"});
}

TEST(SameWordGuard, TimeFallbackTheWordMayCommitAgainAfterTheCooldownWithoutBeingSeen) {
    DecisionCfg c = CFG();
    c.repeat_rest_p = 0.0;  // `rest` can never happen
    const int gap = static_cast<int>(CFG().repeat_cooldown_ms / DT) + 3;
    Seq r(c);
    r.feed("hello", 8).feed("what", 2).feed("none", gap, true, 0.35).feed("hello", 8).feed("none", 30, false);
    EXPECT_EQ(r.committed(), (V{"hello", "hello"}));
}

TEST(SameWordGuard, HowThenYouIsExactlyOneSentenceEvenWithFlicker) {
    Seq r;
    one_sign_with_dip(r, "how", "what");
    one_sign_with_dip(r, "you", "what").feed("none", 40, false);
    EXPECT_EQ(r.committed(), V{"how_are_you"});
}

TEST(SameWordGuard, WhereThenYouIsExactlyOneSentenceEvenWithFlicker) {
    Seq r;
    one_sign_with_dip(r, "where", "what");
    one_sign_with_dip(r, "you", "what").feed("none", 40, false);
    EXPECT_EQ(r.committed(), V{"where_are_you"});
}

TEST(SameWordGuard, YouThenWhereIsExactlyOneSentenceEvenWithFlicker) {
    Seq r;
    one_sign_with_dip(r, "you", "what");
    one_sign_with_dip(r, "where", "what").feed("none", 40, false);
    EXPECT_EQ(r.committed(), V{"where_are_you"});
}

TEST(SameWordGuard, YouAloneCommitsOnceAfterTheJoinWindow) {
    Seq r;
    one_sign_with_dip(r, "you", "what");
    EXPECT_TRUE(r.committed().empty());  // held while waiting for HOW / WHERE
    r.feed("none", 30, false);
    EXPECT_EQ(r.committed(), V{"you"});
}

TEST(SameWordGuard, AHeldWordIsNeverEmittedAloneAndAgainInsideItsSentence) {
    const char* pairs[3][2] = {{"how", "you"}, {"where", "you"}, {"you", "where"}};
    for (auto& pr : pairs) {
        Seq r;
        r.feed(pr[0], 8).feed(pr[0], 6, false).feed(pr[0], 6);
        r.feed(pr[1], 8).feed(pr[1], 6, false).feed(pr[1], 6).feed("none", 40, false);
        ASSERT_EQ(r.committed().size(), 1u) << pr[0] << "+" << pr[1];
        EXPECT_NE(r.committed()[0].find('_'), std::string::npos);
    }
}

TEST(SameWordGuard, ADifferentWordIsNotBlockedByTheGuard) {
    Seq r;
    r.feed("hello", 8).feed("what", 8).feed("why", 8).feed("none", 30, false);
    EXPECT_EQ(r.committed(), (V{"hello", "what", "why"}));
}

TEST(SameWordGuard, AThirdWordBetweenTwoEqualOnesLetsTheFirstWordCommitAgain) {
    Seq r;
    r.feed("hello", 8).feed("what", 8).feed("hello", 8).feed("none", 30, false);
    EXPECT_EQ(r.committed(), (V{"hello", "what", "hello"}));
}

TEST(SameWordGuard, ResetClearsTheGuard) {
    Seq r;
    r.feed("hello", 8);
    r.d.reset();
    r.feed("hello", 8);
    EXPECT_EQ(r.committed(), (V{"hello", "hello"}));
}

TEST(PerWordThreshold, DeployedSettingsRaiseWhy) {
    ASSERT_TRUE(CFG().word_threshold.count("why"));
    EXPECT_GT(CFG().word_threshold.at("why"), CFG().commit_threshold);
}

TEST(PerWordThreshold, ABelowItsOwnThresholdWordDoesNotCommitButTheOthersDo) {
    const double conf = (CFG().commit_threshold + CFG().word_threshold.at("why")) / 2;  // above the common threshold, below why's
    Seq a;
    a.feed("why", 12, true, conf).feed("none", 30, false);
    EXPECT_TRUE(a.committed().empty());
    for (const char* w : {"hello", "what"}) {
        Seq b;
        b.feed(w, 12, true, conf).feed("none", 30, false);
        EXPECT_EQ(b.committed(), V{w}) << w;
    }
}

TEST(PerWordThreshold, AWordAboveItsOwnThresholdCommits) {
    Seq r;
    r.feed("why", 12, true, CFG().word_threshold.at("why") + 0.05).feed("none", 30, false);
    EXPECT_EQ(r.committed(), V{"why"});
}

TEST(PerWordThreshold, WithoutOneEveryWordUsesTheCommonThreshold) {
    DecisionCfg c = CFG();
    c.word_threshold.clear();
    for (const char* w : {"hello", "what", "why"}) {
        Seq r(c);
        r.feed(w, 12, true, c.commit_threshold + 0.05).feed("none", 30, false);
        EXPECT_EQ(r.committed(), V{w}) << w;
    }
}

TEST(PerWordThreshold, ARaisedThresholdOnAJoiningWordOnlyAffectsThatWord) {
    DecisionCfg c = CFG();
    c.word_threshold["you"] = 0.9;
    Seq a(c), b(c);
    a.feed("how", 8, true, 0.8).feed("none", 40, false);
    b.feed("you", 8, true, 0.8).feed("none", 40, false);
    EXPECT_EQ(a.committed(), V{"how"});
    EXPECT_TRUE(b.committed().empty());
}

// The same long synthetic streams (words, wobble, hands flickering out of view, rests) through decision.py and decision.cpp:
// every frame must give the same label, state and committed event.
TEST(DecisionGolden, MatchesPythonFrameByFrame) {
    int streams = 0, commits = 0;
    for (const auto& st : golden()["decision"]) {
        Decision d(cfg().labels, CFG());
        const auto& pr = st["probs"];
        const auto& hands = st["hands"];
        const auto& exp = st["expected"];
        const size_t n_lab = cfg().labels.size();
        ASSERT_EQ(pr.size(), hands.size() * n_lab);
        for (size_t i = 0; i < hands.size(); ++i) {
            std::vector<double> probs(n_lab);
            for (size_t k = 0; k < n_lab; ++k) probs[k] = pr[i * n_lab + k].get<double>();
            const DecisionOut o = d.step(probs, hands[i].get<int>() != 0, i * 100.0);
            ASSERT_EQ(o.label, exp[i][0].get<std::string>()) << "stream " << streams << " frame " << i;
            ASSERT_EQ(o.state, exp[i][1].get<std::string>()) << "stream " << streams << " frame " << i;
            ASSERT_EQ(o.committed.value_or(""), exp[i][2].get<std::string>()) << "stream " << streams << " frame " << i;
            commits += !exp[i][2].get<std::string>().empty();
        }
        ++streams;
    }
    EXPECT_EQ(streams, 3);
    EXPECT_GT(commits, 20);  // the streams really contain words and sentences
}

namespace {
DecisionCfg gate_cfg() {  // 10 ticks x 0.2 = 2.0 for a moving hand, 0 for a held one
    DecisionCfg c = CFG();
    c.motion_min = 1.0, c.motion_window = 10;
    return c;
}
}  // namespace

TEST(MotionGate, OffByDefaultAStillHandCommits) {
    DecisionCfg c = CFG();
    c.motion_min = 0;
    for (const char* w : {"hello", "what", "why"}) {
        Seq r(c);
        r.feed(w, 10, true, 0.95, 0.0).feed("none", 30, false);
        EXPECT_EQ(r.committed(), V{w}) << w;
    }
}

TEST(MotionGate, AHeldStillHandNeverCommitsForEveryWord) {
    for (const auto& w : cfg().labels) {
        if (w == "none") continue;
        Seq r(gate_cfg());
        r.feed(w, 60, true, 0.95, 0.0).feed("none", 40, false);
        EXPECT_TRUE(r.committed().empty()) << w;
    }
}

TEST(MotionGate, AMovingHandCommitsForEveryWord) {
    for (const auto& w : cfg().labels) {
        if (w == "none") continue;
        Seq r(gate_cfg());
        r.feed(w, 12, true, 0.95, 0.2).feed("none", 40, false);
        EXPECT_EQ(r.committed(), V{w}) << w;
    }
}

TEST(MotionGate, AWordHeldStillCommitsAsSoonAsTheHandMoves) {
    Seq r(gate_cfg());
    r.feed("hello", 10, true, 0.95, 0.0).feed("hello", 10, true, 0.95, 0.2).feed("none", 40, false);
    EXPECT_EQ(r.committed(), V{"hello"});
}

TEST(MotionGate, MotionOlderThanTheWindowDoesNotCount) {
    DecisionCfg c = gate_cfg();
    c.motion_window = 2;
    Seq r(c);
    r.feed("hello", 1, true, 0.95, 1.0).feed("hello", 12, true, 0.95, 0.0).feed("none", 40, false);
    EXPECT_TRUE(r.committed().empty());
}

TEST(MotionGate, AStillHandCannotCompleteASentence) {
    const char* pairs[3][2] = {{"how", "you"}, {"where", "you"}, {"you", "where"}};
    for (auto& pr : pairs) {
        DecisionCfg c = gate_cfg();  // a short window and a short rest between the signs, so the first sign's motion has left the window
        c.motion_min = 0.5, c.motion_window = 4;
        Seq r(c);
        r.feed(pr[0], 8, true, 0.95, 0.2).feed("none", 4, true, 0.95, 0.0).feed(pr[1], 12, true, 0.95, 0.0).feed("none", 40, false);
        EXPECT_EQ(r.committed(), V{pr[0]}) << pr[0] << "+" << pr[1];
    }
}

TEST(MotionGate, TwoMovingSignsStillMakeTheSentences) {
    const char* cases[3][3] = {{"how", "you", "how_are_you"}, {"where", "you", "where_are_you"}, {"you", "where", "where_are_you"}};
    for (auto& c : cases) {
        Seq r(gate_cfg());
        r.feed(c[0], 8, true, 0.95, 0.2).feed(c[1], 8, true, 0.95, 0.2).feed("none", 40, false);
        EXPECT_EQ(r.committed(), V{c[2]}) << c[2];
    }
}

TEST(MotionGate, RollingMotionScoresForTheDiagnosticLog) {
    Seq r(gate_cfg());
    r.feed("none", 30, true, 0.95, 0.1);
    EXPECT_NEAR(r.d.mot_1s(), 0.1, 1e-9);
    EXPECT_NEAR(r.d.mot_3s(), 0.1, 1e-9);
    r.feed("none", 10, true, 0.95, 0.0);
    EXPECT_NEAR(r.d.mot_1s(), 0.0, 1e-9);
    EXPECT_NEAR(r.d.mot_3s(), 0.1 * 20 / 30, 1e-9);
}

// Streams with hand motion and the motion gate on: labels, states, events and the rolling motion scores must equal Python's.
TEST(DecisionGolden, MotionGateMatchesPythonFrameByFrame) {
    const auto& g = golden()["decision_motion"];
    DecisionCfg c = CFG();
    c.motion_min = g["cfg"]["motion_min"].get<double>();
    c.motion_window = g["cfg"]["motion_window"].get<int>();
    int streams = 0, commits = 0;
    for (const auto& st : g["streams"]) {
        Decision d(cfg().labels, c);
        const auto& pr = st["probs"];
        const auto& hands = st["hands"];
        const auto& mot = st["motion"];
        const auto& exp = st["expected"];
        const size_t n_lab = cfg().labels.size();
        for (size_t i = 0; i < hands.size(); ++i) {
            std::vector<double> probs(n_lab);
            for (size_t k = 0; k < n_lab; ++k) probs[k] = pr[i * n_lab + k].get<double>();
            const DecisionOut o = d.step(probs, hands[i].get<int>() != 0, i * 100.0, mot[i].get<double>());
            ASSERT_EQ(o.label, exp[i][0].get<std::string>()) << "stream " << streams << " frame " << i;
            ASSERT_EQ(o.state, exp[i][1].get<std::string>()) << "stream " << streams << " frame " << i;
            ASSERT_EQ(o.committed.value_or(""), exp[i][2].get<std::string>()) << "stream " << streams << " frame " << i;
            ASSERT_NEAR(d.mot_1s(), exp[i][3].get<double>(), 1e-5) << "stream " << streams << " frame " << i;
            ASSERT_NEAR(d.mot_3s(), exp[i][4].get<double>(), 1e-5) << "stream " << streams << " frame " << i;
            commits += !exp[i][2].get<std::string>().empty();
        }
        ++streams;
    }
    EXPECT_EQ(streams, 2);
    EXPECT_GT(commits, 10);
}
