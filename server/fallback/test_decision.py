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

    def feed(self, label, n, hands=True, conf=0.95, motion=0.0):
        for _ in range(n):
            out = self.d.step(p(label, conf), hands, self.t, motion)
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
        # (and the same-word guard off: this test is about the generic cooldown, the guard has its own tests)
        cfg = {**CFG, "gate_frames": 1, "gate_min": 1, "release_frames": 1, "release_none_frames": 1, "cooldown_ms": 800, "repeat_rest_frames": 0}
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


WORDS = LABELS[1:]


def one_sign_with_dip(r, w, other):
    """One long sign that the hand tracker loses twice: the hands flicker out of view (idle gate closes) while the model still says `w`,
    then the model briefly says another word (2 frames) and comes back. The first long hold of the sign is 3 s."""
    return r.feed(w, 8).feed(w, 6, hands=False).feed(w, 8).feed(other, 2).feed(w, 8).feed(w, 5, hands=False).feed(w, 10)


class SameWordGuardTests(unittest.TestCase):
    """The same word must not commit twice for one sign (repeat_rest_frames / repeat_rest_p / repeat_cooldown_ms), for every word."""

    def test_guard_is_on_in_the_deployed_settings(self):
        self.assertGreater(CFG["repeat_rest_frames"], 0)

    def test_one_long_sign_with_a_mid_sign_dip_commits_once_for_every_word(self):
        for w in WORDS:
            with self.subTest(word=w):
                other = "why" if w == "what" else "what"
                r = one_sign_with_dip(Run(), w, other).feed("none", 30, hands=False)
                self.assertEqual(r.committed(), [w])

    def test_without_the_guard_the_same_sign_commits_more_than_once(self):
        # documents the defect the guard fixes: hello, hands flicker out of view (closing the idle gate) while the model keeps saying hello
        cfg = {**CFG, "repeat_rest_frames": 0}
        r = Run(cfg).feed("hello", 8).feed("hello", 6, hands=False).feed("hello", 10).feed("none", 30, hands=False)
        self.assertGreaterEqual(len(r.committed()), 2)

    def test_two_separate_signs_with_the_hands_down_between_commit_twice_for_every_word(self):
        for w in WORDS:
            with self.subTest(word=w):
                r = Run().feed(w, 8).feed("none", 10, hands=False).feed(w, 8).feed("none", 30, hands=False)
                self.assertEqual(r.committed(), [w, w])

    def test_two_signs_separated_only_by_the_model_falling_back_to_none_commit_twice(self):
        # hands stay in view the whole time; only the probability of the word drops (long enough to release the "fired" state too)
        r = Run().feed("hello", 8).feed("none", CFG["release_none_frames"] + 2, hands=True).feed("hello", 8).feed("none", 30, hands=False)
        self.assertEqual(r.committed(), ["hello", "hello"])

    def test_a_shorter_wobble_than_the_rest_time_is_still_one_sign(self):
        r = Run().feed("hello", 8).feed("none", CFG["repeat_rest_frames"] - 1).feed("hello", 8).feed("none", 30, hands=False)
        self.assertEqual(r.committed(), ["hello"])

    def test_time_fallback_the_word_may_commit_again_after_the_cooldown_without_being_predicted(self):
        # no probability falls below repeat_rest_p (so no "rest"), but the word is not the prediction for longer than repeat_cooldown_ms
        cfg = {**CFG, "repeat_rest_p": 0.0}  # `rest` can never happen
        gap = int(CFG["repeat_cooldown_ms"] / DT) + 3
        r = Run(cfg).feed("hello", 8).feed("what", 2).feed("none", gap, conf=0.35).feed("hello", 8).feed("none", 30, hands=False)
        self.assertEqual(r.committed(), ["hello", "hello"])

    def test_how_then_you_is_exactly_one_sentence_even_with_flicker(self):
        r = one_sign_with_dip(Run(), "how", "what")
        one_sign_with_dip(r, "you", "what").feed("none", 40, hands=False)
        self.assertEqual(r.committed(), ["how_are_you"])

    def test_where_then_you_is_exactly_one_sentence_even_with_flicker(self):
        r = one_sign_with_dip(Run(), "where", "what")
        one_sign_with_dip(r, "you", "what").feed("none", 40, hands=False)
        self.assertEqual(r.committed(), ["where_are_you"])

    def test_you_then_where_is_exactly_one_sentence_even_with_flicker(self):
        r = one_sign_with_dip(Run(), "you", "what")
        one_sign_with_dip(r, "where", "what").feed("none", 40, hands=False)
        self.assertEqual(r.committed(), ["where_are_you"])

    def test_you_alone_commits_once_after_the_join_window(self):
        r = one_sign_with_dip(Run(), "you", "what")
        self.assertEqual(r.committed(), [])  # held while waiting for HOW / WHERE
        r.feed("none", 30, hands=False)
        self.assertEqual(r.committed(), ["you"])

    def test_a_held_word_is_never_emitted_alone_and_again_inside_its_sentence(self):
        for first, second in (("how", "you"), ("where", "you"), ("you", "where")):
            with self.subTest(first=first, second=second):
                r = Run().feed(first, 8).feed(first, 6, hands=False).feed(first, 6)  # first word, hand flicker, still the same sign
                r.feed(second, 8).feed(second, 6, hands=False).feed(second, 6).feed("none", 40, hands=False)
                self.assertEqual(len(r.committed()), 1)
                self.assertIn("_", r.committed()[0])

    def test_a_different_word_is_not_blocked_by_the_guard(self):
        r = Run().feed("hello", 8).feed("what", 8).feed("why", 8).feed("none", 30, hands=False)
        self.assertEqual(r.committed(), ["hello", "what", "why"])

    def test_a_third_word_between_two_equal_ones_lets_the_first_word_commit_again(self):
        # HELLO, WHAT, HELLO with the hands staying up: the guard only covers the most recent word
        r = Run().feed("hello", 8).feed("what", 8).feed("hello", 8).feed("none", 30, hands=False)
        self.assertEqual(r.committed(), ["hello", "what", "hello"])

    def test_reset_clears_the_guard(self):
        r = Run().feed("hello", 8)
        r.d.reset()
        r.feed("hello", 8)
        self.assertEqual(r.committed(), ["hello", "hello"])


class PerWordThresholdTests(unittest.TestCase):
    """thr_<word> sets the commit threshold of one word; the others keep commit_threshold."""

    def test_deployed_settings_raise_the_why_threshold(self):
        self.assertGreater(CFG["thr_why"], CFG["commit_threshold"])

    def test_a_word_below_its_own_threshold_does_not_commit_but_the_others_do(self):
        conf = (CFG["commit_threshold"] + CFG["thr_why"]) / 2  # above the common threshold, below why's
        self.assertEqual(Run().feed("why", 12, conf=conf).feed("none", 30, hands=False).committed(), [])
        for w in ("hello", "what"):
            self.assertEqual(Run().feed(w, 12, conf=conf).feed("none", 30, hands=False).committed(), [w], w)

    def test_a_word_above_its_own_threshold_commits(self):
        r = Run().feed("why", 12, conf=CFG["thr_why"] + 0.05).feed("none", 30, hands=False)
        self.assertEqual(r.committed(), ["why"])

    def test_without_a_per_word_threshold_every_word_uses_the_common_one(self):
        cfg = {k: v for k, v in CFG.items() if not k.startswith("thr_")}
        conf = CFG["commit_threshold"] + 0.05
        for w in ("hello", "what", "why"):
            self.assertEqual(Run(cfg).feed(w, 12, conf=conf).feed("none", 30, hands=False).committed(), [w], w)

    def test_a_raised_threshold_on_a_joining_word_only_affects_that_word(self):
        cfg = {**CFG, "thr_you": 0.9}
        conf = 0.8
        self.assertEqual(Run(cfg).feed("how", 8, conf=conf).feed("none", 40, hands=False).committed(), ["how"])
        self.assertEqual(Run(cfg).feed("you", 8, conf=conf).feed("none", 40, hands=False).committed(), [])


GATE = {**CFG, "motion_min": 1.0, "motion_window": 10}  # 10 ticks x 0.2 = 2.0 for a moving hand, 0 for a held one
SHORT = {**CFG, "motion_min": 0.5, "motion_window": 4}
MOVING, STILL = 0.2, 0.0


class MotionGateTests(unittest.TestCase):
    """A word may commit only if the hands moved: the summed motion over the last motion_window ticks must reach motion_min."""

    def test_with_motion_min_zero_a_still_hand_commits(self):
        cfg = {**CFG, "motion_min": 0}
        for w in ("hello", "what", "why"):
            self.assertEqual(Run(cfg).feed(w, 10, motion=STILL).feed("none", 30, hands=False).committed(), [w], w)

    def test_a_held_still_hand_never_commits_for_every_word(self):
        for w in WORDS:
            with self.subTest(word=w):
                self.assertEqual(Run(GATE).feed(w, 60, motion=STILL).feed("none", 40, hands=False).committed(), [])

    def test_a_moving_hand_commits_for_every_word(self):
        for w in WORDS:
            with self.subTest(word=w):
                self.assertEqual(Run(GATE).feed(w, 12, motion=MOVING).feed("none", 40, hands=False).committed(), [w])

    def test_a_word_held_still_commits_as_soon_as_the_hand_moves(self):
        r = Run(GATE).feed("hello", 10, motion=STILL).feed("hello", 10, motion=MOVING).feed("none", 40, hands=False)
        self.assertEqual(r.committed(), ["hello"])

    def test_motion_older_than_the_window_does_not_count(self):
        cfg = {**GATE, "motion_window": 2}
        r = Run(cfg).feed("hello", 1, motion=1.0).feed("hello", 12, motion=STILL).feed("none", 40, hands=False)
        self.assertEqual(r.committed(), [])

    def test_a_still_hand_cannot_complete_a_sentence(self):
        # the first word moves, the partner is held still: the first word goes out alone, never the sentence
        for first, partner in (("how", "you"), ("where", "you"), ("you", "where")):
            with self.subTest(first=first, partner=partner):
                # a short window and a short rest between the signs, so the first sign's motion has left the window when the partner is judged
                r = Run(SHORT).feed(first, 8, motion=MOVING).feed("none", 4, motion=STILL).feed(partner, 12, motion=STILL).feed("none", 40, hands=False)
                self.assertEqual(r.committed(), [first])

    def test_two_moving_signs_still_make_the_sentences(self):
        for a, b, s in (("how", "you", "how_are_you"), ("where", "you", "where_are_you"), ("you", "where", "where_are_you")):
            with self.subTest(sentence=s):
                r = Run(GATE).feed(a, 8, motion=MOVING).feed(b, 8, motion=MOVING).feed("none", 40, hands=False)
                self.assertEqual(r.committed(), [s])

    def test_a_held_still_you_alone_never_appears(self):
        self.assertEqual(Run(GATE).feed("you", 60, motion=STILL).feed("none", 60, hands=False).committed(), [])

    def test_rolling_motion_scores_for_the_diagnostic_log(self):
        r = Run(GATE).feed("none", 30, motion=0.1)
        self.assertAlmostEqual(r.d.mot_1s, 0.1)
        self.assertAlmostEqual(r.d.mot_3s, 0.1)
        r.feed("none", 10, motion=0.0)
        self.assertAlmostEqual(r.d.mot_1s, 0.0)
        self.assertAlmostEqual(r.d.mot_3s, 0.1 * 20 / 30)


if __name__ == "__main__":
    unittest.main()
