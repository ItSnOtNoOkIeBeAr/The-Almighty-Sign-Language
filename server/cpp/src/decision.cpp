#include "decision.h"

#include <algorithm>
#include <cmath>
#include <utility>

namespace {
// HOW then YOU, WHERE then YOU, YOU then WHERE
const char* join_of(const std::string& a, const std::string& b) {
    if (a == "how" && b == "you") return "how_are_you";
    if ((a == "where" && b == "you") || (a == "you" && b == "where")) return "where_are_you";
    return nullptr;
}
bool starts_join(const std::string& w) { return w == "how" || w == "where" || w == "you"; }
}  // namespace

Decision::Decision(std::vector<std::string> labels, DecisionCfg cfg) : labels_(std::move(labels)), cfg_(cfg) {
    none_ = index_of("none");
    reset();
}

int Decision::index_of(const std::string& label) const {
    return static_cast<int>(std::find(labels_.begin(), labels_.end(), label) - labels_.begin());
}

void Decision::reset() {
    hands_.clear();
    ema_.clear();
    cand_ = -1, consec_ = 0, fired_ = -1, away_ = 0;
    last_commit_ = -1e18;
    pending_.reset();
    outbox_.clear();
}

DecisionOut Decision::step(const std::vector<double>& probs, bool hands_present, double t_ms) {
    std::vector<std::string> emitted;
    if (pending_ && t_ms - pending_->t_ms > cfg_.join_ms) {
        emitted.push_back(pending_->word);
        pending_.reset();
    }
    hands_.push_back(hands_present);
    if (static_cast<int>(hands_.size()) > cfg_.gate_frames) hands_.pop_front();
    if (std::count(hands_.begin(), hands_.end(), true) < cfg_.gate_min) {  // idle gate closed
        ema_.clear(), cand_ = -1, consec_ = 0, fired_ = -1, away_ = 0;
        return out(none_, probs[none_], emitted, false);
    }
    const double a = cfg_.ema_alpha;
    if (ema_.empty()) {
        ema_ = probs;
    } else {
        for (size_t i = 0; i < ema_.size(); ++i) ema_[i] = a * probs[i] + (1 - a) * ema_[i];
    }
    const int k = static_cast<int>(std::max_element(ema_.begin(), ema_.end()) - ema_.begin());  // first max, like Python max()
    const double conf = ema_[k];
    if (pending_ && labels_[k] == pending_->word) pending_->t_ms = t_ms;  // the held word's sign is still going on
    if (fired_ >= 0) {
        away_ = (k == fired_) ? 0 : away_ + 1;
        if (away_ >= (k == none_ ? cfg_.release_none_frames : cfg_.release_frames)) fired_ = -1;
    }
    if (k != none_ && conf >= cfg_.commit_threshold) {
        consec_ = (k == cand_) ? consec_ + 1 : 1;
        cand_ = k;
    } else {
        cand_ = -1, consec_ = 0;
    }
    if (cand_ >= 0 && consec_ >= cfg_.commit_consecutive_M && cand_ != fired_ && t_ms - last_commit_ >= cfg_.cooldown_ms) {
        for (auto& e : commit(labels_[cand_], t_ms)) emitted.push_back(std::move(e));
        fired_ = cand_, away_ = 0, last_commit_ = t_ms, consec_ = 0;
    }
    return out(k, conf, emitted, true);
}

std::vector<std::string> Decision::commit(const std::string& word, double t_ms) {
    std::vector<std::string> res;
    if (pending_ && t_ms - pending_->t_ms <= cfg_.join_ms) {
        if (const char* j = join_of(pending_->word, word)) {
            pending_.reset();
            return {j};
        }
    }
    if (pending_) {  // not a pair: the held word goes out now
        res.push_back(pending_->word);
        pending_.reset();
    }
    if (starts_join(word)) pending_ = Pending{word, t_ms};
    else res.push_back(word);
    return res;
}

DecisionOut Decision::out(int k, double conf, const std::vector<std::string>& emitted, bool gate_open) {
    for (auto& e : emitted) outbox_.push_back(e);  // two events in one frame only when a held word is flushed by an unrelated new word
    std::optional<std::string> committed;
    if (!outbox_.empty()) {
        committed = outbox_.front();
        outbox_.pop_front();
    }
    const char* state = committed ? "committed" : ((gate_open || pending_) ? "signing" : "idle");
    return {labels_[k], std::round(conf * 1e4) / 1e4, state, committed};
}
