"""Decision logic (BUILD_SPEC section 8): turns per-frame class probabilities into committed words and joined sentences.

Pure Python, no numpy/torch, driven by explicit timestamps, so it is unit-tested on synthetic sequences and is the
reference for the C++ port.  Per frame, in order:
  1. idle gate (rule based): the hands must be seen in >= gate_min of the last gate_frames frames, else nothing can happen;
  2. EMA smoothing of the probabilities (reset whenever the gate closes);
  3. candidate = best non-`none` class with smoothed confidence >= commit_threshold;
  4. commit after M consecutive frames with the same candidate, if it is not still "fired" and the cooldown has passed.
     A committed label stays fired until another word has been predicted for release_frames frames, `none` for
     release_none_frames frames (a sign flickers to `none` mid-way), or the gate closes, so one sign never fires twice;
  5. word joining: HOW then YOU -> how_are_you, WHERE then YOU or YOU then WHERE -> where_are_you, within join_ms.
     `how`, `where`, `you` are held up to join_ms waiting for their partner, then emitted alone. The join timer restarts
     every frame the held word is still the prediction, so it measures the pause after the first SIGN ended (a long sign
     commits early and must not eat the window).
"""
from collections import deque

JOINS = {("how", "you"): "how_are_you", ("where", "you"): "where_are_you", ("you", "where"): "where_are_you"}
JOIN_STARTS = {a for a, _ in JOINS}


class Decision:
    def __init__(self, labels, cfg):
        self.labels, self.cfg = list(labels), cfg
        self.none = self.labels.index("none")
        self.reset()

    def reset(self):
        self.hands = deque(maxlen=self.cfg["gate_frames"])
        self.ema = None
        self.cand, self.consec = None, 0
        self.fired, self.away = None, 0
        self.last_commit = -1e18
        self.pending = None  # (word, commit time ms) waiting for a join partner
        self.outbox = deque()  # events not yet reported: the protocol allows one event per frame

    def step(self, probs, hands_present, t_ms):
        """probs: one probability per label. Returns (label, confidence, state, committed_label)."""
        c = self.cfg
        emitted = []
        if self.pending and t_ms - self.pending[1] > c["join_ms"]:
            emitted.append(self.pending[0])
            self.pending = None
        self.hands.append(bool(hands_present))
        if sum(self.hands) < c["gate_min"]:  # idle gate closed
            self.ema, self.cand, self.consec, self.fired, self.away = None, None, 0, None, 0
            return self._out(self.none, probs[self.none], emitted, gate_open=False)
        a = c["ema_alpha"]
        self.ema = list(probs) if self.ema is None else [a * p + (1 - a) * e for p, e in zip(probs, self.ema)]
        k = max(range(len(self.ema)), key=self.ema.__getitem__)
        conf = self.ema[k]
        if self.pending and self.labels[k] == self.pending[0]:
            self.pending = (self.pending[0], t_ms)  # the held word's sign is still going on
        if self.fired is not None:
            self.away = 0 if k == self.fired else self.away + 1
            if self.away >= (c["release_none_frames"] if k == self.none else c["release_frames"]):
                self.fired = None
        if k != self.none and conf >= c["commit_threshold"]:
            self.consec = self.consec + 1 if k == self.cand else 1
            self.cand = k
        else:
            self.cand, self.consec = None, 0
        if (self.cand is not None and self.consec >= c["commit_consecutive_M"] and self.cand != self.fired
                and t_ms - self.last_commit >= c["cooldown_ms"]):
            emitted += self._commit(self.labels[self.cand], t_ms)
            self.fired, self.away, self.last_commit, self.consec = self.cand, 0, t_ms, 0
        return self._out(k, conf, emitted, gate_open=True)

    def _commit(self, word, t_ms):
        """Word joining. Returns the strings to emit now (0, 1 or 2)."""
        p, out = self.pending, []
        if p and t_ms - p[1] <= self.cfg["join_ms"] and (p[0], word) in JOINS:
            self.pending = None
            return [JOINS[(p[0], word)]]
        if p:  # not a pair: the held word goes out now
            out.append(p[0])
            self.pending = None
        if word in JOIN_STARTS:
            self.pending = (word, t_ms)
        else:
            out.append(word)
        return out

    def _out(self, k, conf, emitted, gate_open):
        self.outbox.extend(emitted)  # two events in one frame only when a held word is flushed by an unrelated new word
        committed = self.outbox.popleft() if self.outbox else None
        state = "committed" if committed else ("signing" if gate_open or self.pending else "idle")
        return self.labels[k], round(float(conf), 4), state, committed
