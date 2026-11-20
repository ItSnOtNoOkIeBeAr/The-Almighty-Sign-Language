"""Unit tests for ticks.py (the 10 fps recognition clock). Run: python -m unittest server/fallback/test_ticks.py -v"""
import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ticks import TickGrid  # noqa: E402


def run(timestamps, grid=None):
    """Push frames (item = frame index) and collect every tick as (tick_ms, frame index, run_model)."""
    g, ticks = grid or TickGrid(), []
    for i, ts in enumerate(timestamps):
        ticks += g.push(ts, i)
    return ticks


def stream(fps, seconds=10.0, jitter=0.0, seed=1, t0=1000.0):
    rng = random.Random(seed)
    n = int(seconds * fps)
    return [t0 + i * 1000.0 / fps + rng.uniform(-jitter, jitter) * (i > 0) for i in range(n)]


class TickGridTests(unittest.TestCase):
    def test_exactly_ten_ticks_per_second_at_every_rate(self):
        for fps in (10, 12, 15, 20, 25, 30):
            with self.subTest(fps=fps):
                ts = stream(fps, 10.0)
                ticks = run(ts)
                span = ticks[-1][0] - ticks[0][0]
                self.assertEqual(round(span / 100.0), len(ticks) - 1)  # one tick per 100 ms, none missing, none doubled
                self.assertTrue(all(abs((b[0] - a[0]) - 100.0) < 1e-9 for a, b in zip(ticks, ticks[1:])))
                self.assertGreaterEqual(len(ticks), 95)  # ~10 s of stream

    def test_each_tick_uses_the_frame_closest_to_it(self):
        for fps in (15, 20, 30):
            with self.subTest(fps=fps):
                ts = stream(fps, 6.0, jitter=3.0)
                for t, idx, _ in run(ts)[:-1]:
                    best = min(range(len(ts)), key=lambda i: abs(ts[i] - t))
                    # causal choice: the frame found at tick time is at most half a frame interval off the optimum
                    self.assertLessEqual(abs(ts[idx] - t), abs(ts[best] - t) + 1000.0 / fps / 2 + 1e-9)

    def test_ten_fps_input_is_passed_through_frame_for_frame(self):
        ts = stream(10, 5.0, jitter=8.0)
        self.assertEqual([i for _, i, _ in run(ts)], list(range(len(ts))))

    def test_ticks_are_never_delayed_by_a_whole_frame(self):
        # the tick fires on the frame that is close to it, not on the next one
        for fps in (10, 20, 30):
            ts = stream(fps, 5.0, jitter=2.0)
            g, lag = TickGrid(), []
            for i, t in enumerate(ts):
                for tick, _, _ in g.push(t, i):
                    lag.append(t - tick)
            self.assertLess(max(lag), 1000.0 / fps / 2 + 3.0)

    def test_a_gap_fills_the_missed_ticks_and_only_runs_the_last_ones(self):
        ts = [0.0, 100.0, 200.0, 1000.0]  # 0.8 s without frames
        ticks = run(ts)
        self.assertEqual([t for t, _, _ in ticks][:3], [0.0, 100.0, 200.0])
        gap = ticks[3:]
        self.assertEqual(len(gap), 8)  # ticks 300 ... 1000
        self.assertEqual([r for _, _, r in gap], [False] * 5 + [True] * 3)  # max_run = 3
        self.assertTrue(all(i in (2, 3) for _, i, _ in gap))  # held: the nearest of the two frames
        self.assertEqual([i for _, i, _ in gap][:3], [2, 2, 2])  # near the old frame -> old frame, near the new one -> new one
        self.assertEqual([i for _, i, _ in gap][-3:], [3, 3, 3])

    def test_a_long_pause_restarts_the_grid(self):
        ticks = run([0.0, 100.0, 20000.0, 20100.0])
        self.assertEqual([t for t, _, _ in ticks], [0.0, 100.0, 20000.0, 20100.0])

    def test_jittery_30_fps_with_dropped_frames_still_gives_10_ticks_per_second(self):
        ts = [t for i, t in enumerate(stream(30, 12.0, jitter=4.0)) if i % 7 not in (3, 4)]  # the server dropped ~2 of 7 frames
        ticks = run(ts)
        self.assertGreaterEqual(len(ticks), 115)
        self.assertTrue(all(abs((b[0] - a[0]) - 100.0) < 1e-9 for a, b in zip(ticks, ticks[1:])))

    def test_first_frame_is_tick_zero(self):
        ticks = run([5000.0])
        self.assertEqual(ticks, [(5000.0, 0, True)])

    def test_reset_starts_a_new_grid(self):
        g = TickGrid()
        run([0.0, 100.0, 200.0], g)
        g.reset()
        self.assertEqual(g.push(7777.0, "x"), [(7777.0, "x", True)])


if __name__ == "__main__":
    unittest.main()
