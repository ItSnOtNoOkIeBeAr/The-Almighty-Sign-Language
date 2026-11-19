// Decision logic (BUILD_SPEC section 8). Line-by-line port of server/fallback/decision.py, which stays the reference:
// idle gate -> EMA -> candidate -> commit after M agreeing frames -> fired/release -> word joining.
#pragma once
#include <deque>
#include <optional>
#include <string>
#include <vector>

struct DecisionCfg {
    int gate_frames = 5, gate_min = 2;
    double ema_alpha = 0.7, commit_threshold = 0.5;
    int commit_consecutive_M = 4;
    double cooldown_ms = 800, join_ms = 1500;
    int release_frames = 3, release_none_frames = 8;
};

struct DecisionOut {
    std::string label;
    double confidence;
    std::string state;                       // idle | signing | committed
    std::optional<std::string> committed;
};

class Decision {
public:
    Decision(std::vector<std::string> labels, DecisionCfg cfg);
    void reset();
    DecisionOut step(const std::vector<double>& probs, bool hands_present, double t_ms);

private:
    struct Pending { std::string word; double t_ms; };
    std::vector<std::string> commit(const std::string& word, double t_ms);
    DecisionOut out(int k, double conf, const std::vector<std::string>& emitted, bool gate_open);
    int index_of(const std::string& label) const;

    std::vector<std::string> labels_;
    DecisionCfg cfg_;
    int none_ = 0;
    std::deque<bool> hands_;
    std::vector<double> ema_;                // empty = no EMA yet
    int cand_ = -1, consec_ = 0, fired_ = -1, away_ = 0;
    double last_commit_ = -1e18;
    std::optional<Pending> pending_;         // word waiting for a join partner
    std::deque<std::string> outbox_;         // the protocol allows one event per frame
};
