"""Unit tests for decision.py on synthetic probability sequences. Run: python -m unittest server/fallback/test_decision.py -v"""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from decision import Decision  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
LABELS = json.loads((ROOT / "protocol" / "labels.json").read_text())["labels"]
CFG = {**json.loads((ROOT / "server" / "config" / "server.json").read_text())["decision"]}
DT = 100  # ms per frame (10 fps)
WARM = CFG["gate_min"] - 1  # the idle gate needs the hands in gate_min frames, so the first usable frame is the (WARM+1)th


def p(label, conf=0.95):
    """Probability vector with `conf` on `label` and the rest spread evenly."""
    rest = (1 - conf) / (len(LABELS) - 1)
    return [conf if n == label else rest for n in LABELS]


class Run:
    """Feeds frames to a Decision and records (time_ms, committed) events."""

    def __init__(self, cfg=None):
        self.d, self.t, self.events, self.frames = Decision(LABELS, cfg or CFG), 0, [], []

    def feed(self, label, n, hands=True, conf=0.95):
        for _ in range(n):
            out = self.d.step(p(label, conf), hands, self.t)
            self.frames.append(out)
            if out[3]:
                self.events.append((self.t, out[3]))
            self.t += DT
        return self

    def committed(self):
        return [w for _, w in self.events]


class DecisionTests(unittest.TestCase):
    def test_idle_gate_blocks_everything(self):
        r = Run().feed("hello", 40, hands=False)
        self.assertEqual(r.committed(), [])
        self.assertTrue(all(f[2] == "idle" and f[0] == "none" for f in r.frames))

    def test_commits_after_exactly_M_frames(self):
        r = Run().feed("hello", WARM + CFG["commit_consecutive_M"] - 1)
        self.assertEqual(r.committed(), [])
        r.feed("hello", 1)
        self.assertEqual(r.committed(), ["hello"])
        self.assertEqual(r.frames[-1][2], "committed")

    def test_below_threshold_never_commits(self):
        r = Run().feed("hello", 40, conf=CFG["commit_threshold"] - 0.1)
        self.assertEqual(r.committed(), [])
        self.assertTrue(all(f[2] == "signing" for f in r.frames[WARM:]))

    def test_single_frame_spike_is_smoothed_away(self):
        r = Run().feed("none", 10).feed("what", 1).feed("none", 10)
        self.assertEqual(r.committed(), [])

    def test_interrupted_run_restarts_the_count(self):
        r = Run().feed("why", 3).feed("none", 3).feed("why", 3)
        self.assertEqual(r.committed(), [])

    def test_holding_one_sign_fires_once(self):
        r = Run().feed("hello", 60)
        self.assertEqual(r.committed(), ["hello"])

    def test_same_sign_again_after_hands_down_and_cooldown(self):
        r = Run().feed("hello", 10).feed("none", 8, hands=False).feed("hello", 10)
        self.assertEqual(r.committed(), ["hello", "hello"])

    def test_cooldown_delays_a_quick_second_commit(self):
        # gate and release shortened so only the cooldown can hold the second commit back
        cfg = {**CFG, "gate_frames": 1, "gate_min": 1, "release_frames": 1, "release_none_frames": 1, "cooldown_ms": 800}
        r = Run(cfg).feed("hello", 4).feed("none", 1).feed("hello", 12)
        self.assertEqual(r.committed(), ["hello", "hello"])
        self.assertGreaterEqual(r.events[1][0] - r.events[0][0], 800)

    def test_mid_sign_flicker_to_none_does_not_refire(self):
        # hands stay up; the prediction dips to `none` for 5 frames (< release_none_frames) and comes back: still one commit
        r = Run().feed("hello", 8).feed("none", 5).feed("hello", 12)
        self.assertEqual(r.committed(), ["hello"])

    def test_long_none_with_hands_up_releases_the_word(self):
        r = Run().feed("hello", 8).feed("none", CFG["release_none_frames"] + 4).feed("hello", 12)
        self.assertEqual(r.committed(), ["hello", "hello"])

    def test_how_then_you_joins(self):
        r = Run().feed("how", 6).feed("you", 8).feed("none", 20, hands=False)
        self.assertEqual(r.committed(), ["how_are_you"])

    def test_where_then_you_joins(self):
        r = Run().feed("where", 6).feed("you", 8).feed("none", 20, hands=False)
        self.assertEqual(r.committed(), ["where_are_you"])

    def test_you_then_where_joins(self):
        r = Run().feed("you", 6).feed("where", 8).feed("none", 20, hands=False)
        self.assertEqual(r.committed(), ["where_are_you"])

    def test_how_alone_is_emitted_after_the_join_window(self):
        r = Run().feed("how", 6)
        self.assertEqual(r.committed(), [])  # held while waiting for YOU
        t_sign_end = r.t - DT  # last frame in which `how` was still the prediction
        r.feed("none", 30, hands=False)
        self.assertEqual(r.committed(), ["how"])
        self.assertGreater(r.events[0][0] - t_sign_end, CFG["join_ms"] - DT)

    def test_long_sign_does_not_eat_the_join_window(self):
        # HOW is held for 3 s (commits early), then a 1 s pause, then YOU: still one sentence, because the timer runs from the sign's end
        r = Run().feed("how", 30).feed("none", 10, hands=False).feed("you", 8).feed("none", 20, hands=False)
        self.assertEqual(r.committed(), ["how_are_you"])

    def test_you_after_the_join_window_is_separate(self):
        r = Run().feed("how", 6).feed("none", 20, hands=False).feed("you", 8).feed("none", 20, hands=False)
        self.assertEqual(r.committed(), ["how", "you"])

    def test_unrelated_word_flushes_the_held_word(self):
        r = Run().feed("how", 6).feed("hello", 8).feed("none", 20, hands=False)
        self.assertEqual(r.committed(), ["how", "hello"])  # one event per frame: the second goes out one frame later
        self.assertEqual(r.events[1][0] - r.events[0][0], DT)

    def test_hello_what_why_are_not_delayed(self):
        for w in ("hello", "what", "why"):
            r = Run().feed(w, 6)
            self.assertEqual(r.committed(), [w])

    def test_sentence_is_one_event_not_two(self):
        r = Run().feed("how", 6).feed("you", 8)
        self.assertEqual(len(r.events), 1)
        self.assertEqual(r.events[0][1], "how_are_you")

    def test_reset_clears_state(self):
        r = Run().feed("how", 6)
        r.d.reset()
        r.feed("none", 30, hands=False)
        self.assertEqual(r.committed(), [])


if __name__ == "__main__":
    unittest.main()
