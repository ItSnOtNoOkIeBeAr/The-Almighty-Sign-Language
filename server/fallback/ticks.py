"""10 fps recognition clock, driven by client timestamps (pure Python, unit-tested; the C++ server ports it line by line).

The client may send 10-30 fps. Keypoints are computed for EVERY frame (smooth skeleton on the laptop), but the temporal model and the
decision logic must see exactly one frame per 100 ms "tick", so a 30-frame window still covers 3 s and the decision settings (M,
cooldowns, repeat_rest_frames, gate) keep their meaning. For each tick the frame whose client timestamp is CLOSEST to the tick is used.

Tick n is at T0 + n * step_ms (T0 = timestamp of the first frame). It fires as soon as a frame arrives that is at most half a frame
interval before the tick (so at 10 fps a tick is never delayed by a frame, at 30 fps it fires within ~17 ms); the frame used is the
closer of that frame and the previous one. After a gap several ticks fall due at once: each takes the nearest frame (hold), only the last
`max_run` of them are run through the model (the earlier ones only fill the window), and a pause longer than `max_gap_ms` restarts the grid.
"""
import math


class TickGrid:
    def __init__(self, step_ms=100.0, max_run=3, max_gap_ms=5000.0):
        self.step, self.max_run, self.max_gap = step_ms, max_run, max_gap_ms
        self.reset()

    def reset(self):
        self.prev = None          # (client ts ms, item) of the previous frame
        self.next_tick = None     # ms of the next tick
        self.interval = None      # running average of the frame interval, ms (seeded by the first interval seen)

    def push(self, ts, item):
        """One frame (client timestamp ms, any payload) -> list of (tick_ms, item_closest_to_tick, run_model) that fell due."""
        out = []
        if self.prev is not None:
            dt = min(max(ts - self.prev[0], 1.0), 300.0)
            self.interval = dt if self.interval is None else 0.9 * self.interval + 0.1 * dt
        half = min(max(0.5 * (self.step if self.interval is None else self.interval), 5.0), 50.0)
        if self.next_tick is None or ts - self.next_tick > self.max_gap:
            self.next_tick = ts  # first frame, or a long pause: the grid restarts at this frame
        if ts >= self.next_tick - half:
            n_due = int(math.floor((ts - (self.next_tick - half)) / self.step)) + 1
            skip = max(0, n_due - self.max_run)
            for j in range(n_due):
                t = self.next_tick
                pick = item
                if self.prev is not None and abs(self.prev[0] - t) < abs(ts - t):
                    pick = self.prev[1]
                out.append((t, pick, j >= skip))
                self.next_tick += self.step
        self.prev = (ts, item)
        return out
