"""Decision logic (BUILD_SPEC section 8): turns per-frame class probabilities into committed words and joined sentences.

Pure Python, no numpy/torch, driven by explicit timestamps, so it is unit-tested on synthetic sequences and is the
reference for the C++ port.  Per frame, in order:
  1. idle gate (rule based): the hands must be seen in >= gate_min of the last gate_frames frames, else nothing can happen;
  2. EMA smoothing of the probabilities (reset whenever the gate closes);
  3. candidate = best non-`none` class with smoothed confidence >= commit_threshold;
  3b. the threshold can be set per word (`thr_<word>`, e.g. thr_why 0.7); a word without one uses commit_threshold;
  3c. motion gate (motion_min > 0): a word may commit only if the hands moved during the sign: the summed hand motion (shoulder widths per tick,
      see signfeat.hand_motion) over the last motion_window ticks must reach motion_min. A hand held still in view (typing, resting) never commits, nor
      does it complete a sentence: every word of a sentence passes the gate by itself. motion is optional per step; without it the gate sees zeros.
  4. commit after M consecutive frames with the same candidate, if it is not still "fired" and the cooldown has passed.
     A committed label stays fired until another word has been predicted for release_frames frames, `none` for
     release_none_frames frames (a sign flickers to `none` mid-way), or the gate closes, so one sign never fires twice;
  5. word joining: HOW then YOU -> how_are_you, WHERE then YOU or YOU then WHERE -> where_are_you, within join_ms.
     `how`, `where`, `you` are held up to join_ms waiting for their partner, then emitted alone. The join timer restarts
     every frame the held word is still the prediction, so it measures the pause after the first SIGN ended (a long sign
     commits early and must not eat the window).
  6. same-word guard (repeat_rest_frames > 0): after a word commits, the SAME word cannot commit again until the model has dropped that word
     (raw probability < repeat_rest_p) for repeat_rest_frames frames in a row, i.e. the sign ended, or has not seen it (probability below
     the commit threshold) for repeat_cooldown_ms. One long sign whose hand flickers out of view (closing the idle gate) or whose probability wobbles commits once; two
     signs with the model back at rest in between commit twice. A DIFFERENT word is never blocked, and committing it replaces the guard
     (HOW then YOU still joins). The guard also covers held words (how/where/you), so a word is never emitted alone and again in its sentence.
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
        self.guard = None  # (class index, ms the model last predicted it) of the last committed word, while it may not commit again
        self.rest_run = 0  # consecutive frames in which the guarded word's raw probability is low (the sign has ended)
        self.outbox = deque()  # events not yet reported: the protocol allows one event per frame
        self.mot = deque(maxlen=30)  # hand motion per tick, newest last (3 s)

    @property
    def mot_1s(self):
        """Mean hand motion over the last 10 ticks (1 s); diagnostics."""
        return sum(list(self.mot)[-10:]) / 10.0

    @property
    def mot_3s(self):
        """Mean hand motion over the last 30 ticks (3 s); diagnostics."""
        return sum(self.mot) / 30.0

    def step(self, probs, hands_present, t_ms, motion=0.0):
        """probs: one probability per label; motion: hand motion since the previous tick (signfeat.hand_motion). Returns (label, confidence, state, committed_label)."""
        c = self.cfg
        self.mot.append(float(motion))
        emitted = []
        if self.pending and t_ms - self.pending[1] > c["join_ms"]:
            emitted.append(self.pending[0])
            self.pending = None
        self.hands.append(bool(hands_present))
        n_rest = c.get("repeat_rest_frames", 0)
        if n_rest and self.guard is not None:  # same-word guard bookkeeping: runs even while the idle gate is closed (the gate flickers inside a sign)
            p = probs[self.guard[0]]
            self.rest_run = self.rest_run + 1 if p < c.get("repeat_rest_p", 0.2) else 0
            if self.rest_run >= n_rest or t_ms - self.guard[1] >= c.get("repeat_cooldown_ms", 2000):
                self.guard = None  # the sign ended (the model dropped the word) or the word has not been seen for a while
            elif p >= c["commit_threshold"]:
                self.guard = (self.guard[0], t_ms)  # the model still sees that sign: the clock restarts
        if sum(self.hands) < c["gate_min"]:  # idle gate closed
            self.ema, self.cand, self.consec, self.fired, self.away = None, None, 0, None, 0
            return self._out(self.none, probs[self.none], emitted, gate_open=False)
        a = c["ema_alpha"]
        self.ema = list(probs) if self.ema is None else [a * p + (1 - a) * e for p, e in zip(probs, self.ema)]
        k = max(range(len(self.ema)), key=self.ema.__getitem__)
        conf = self.ema[k]
        word = self.labels[k]
        if self.pending and self.labels[k] == self.pending[0]:
            self.pending = (self.pending[0], t_ms)  # the held word's sign is still going on
        if self.fired is not None:
            self.away = 0 if k == self.fired else self.away + 1
            if self.away >= (c["release_none_frames"] if k == self.none else c["release_frames"]):
                self.fired = None
        if k != self.none and conf >= c.get("thr_" + word, c["commit_threshold"]):  # per-word threshold (thr_<word>), else the common one
            self.consec = self.consec + 1 if k == self.cand else 1
            self.cand = k
        else:
            self.cand, self.consec = None, 0
        repeat = self.guard is not None and self.cand == self.guard[0]  # the same word again, without the sign having ended in between
        moved = c.get("motion_min", 0) <= 0 or sum(list(self.mot)[-c.get("motion_window", 10):]) >= c["motion_min"]  # the hands moved during the sign
        if (self.cand is not None and moved and self.consec >= c["commit_consecutive_M"] and self.cand != self.fired
                and t_ms - self.last_commit >= c["cooldown_ms"] and not repeat):
            emitted += self._commit(self.labels[self.cand], t_ms)
            self.fired, self.away, self.last_commit, self.consec = self.cand, 0, t_ms, 0
            if n_rest:
                self.guard, self.rest_run = (self.cand, t_ms), 0
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
