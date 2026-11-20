"""Replay a live diagnostic log (logs/diag_*.csv) through the decision logic, with or without settings overrides.

  training\\.venv\\Scripts\\python server\\scripts\\replay_diag.py logs\\diag_<time>.csv [--set repeat_rest_frames=3 --set repeat_cooldown_ms=2000] [--list]

The log holds, per frame, the raw class probabilities and the hand-detection flags the server used, so the decision is replayed exactly
(a new stream starts when the client timestamp goes backwards or jumps by more than 5 s). Prints the committed events (time, word) of the
LOGGED run, of the replay with the deployed settings (a check that the replay matches), and of the replay with --set overrides; then
the repeats: the same word committed twice within 6 s. Numbers only, no images.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "server" / "fallback"))
from decision import Decision  # noqa: E402


def streams(rows):
    cur, last = [], None
    for r in rows:
        t = int(r["ts_ms"])
        if last is not None and (t < last or t - last > 5000):
            yield cur
            cur = []
        cur.append(r)
        last = t
    if cur:
        yield cur


def replay(rows, labels, cfg):
    events = []
    for s, stream in enumerate(streams(rows)):
        dec = Decision(labels, cfg)
        t0 = int(stream[0]["ts_ms"])
        for r in stream:
            probs = [float(r["p_" + n]) for n in labels]
            hands = r["hand_L_strict"] == "1" or r["hand_R_strict"] == "1"
            out = dec.step(probs, hands, float(r["ts_ms"]))
            if out[3]:
                events.append((s, (int(r["ts_ms"]) - t0) / 1000.0, out[3]))
    return events


def repeats(events, window=6.0):
    return [(a, b) for i, a in enumerate(events) for b in events[i + 1:] if a[0] == b[0] and a[2] == b[2] and b[1] - a[1] <= window]


def dip_between(rows, a, b):
    """Lowest raw p(word) between two commits of the same word (stream-relative seconds), and the longest run of frames below 0.2: one sign vs two."""
    st = list(streams(rows))[a[0]]
    t0 = int(st[0]["ts_ms"])
    seg = [float(r["p_" + a[2]]) for r in st if a[1] <= (int(r["ts_ms"]) - t0) / 1000.0 <= b[1]]
    run = best = 0
    for p in seg:
        run = run + 1 if p < 0.2 else 0
        best = max(best, run)
    return min(seg), best


def show(title, events, rows=None, labels=None):
    print(f"\n{title}: {len(events)} commits")
    for s, t, w in events:
        print(f"  stream {s} t={t:7.1f}s  {w}")
    rep = repeats(events)
    print(f"  same word twice within 6 s: {len(rep)}")
    for a, b in rep:
        if rows is not None:
            lo, run = dip_between(rows, a, b)
            print(f"    {a[2]:<6} at {a[1]:6.1f}s and {b[1]:6.1f}s: p(word) in between min {lo:.2f}, longest run below 0.2: {run} frames -> "
                  + ("ONE sign (no dip)" if run < 3 else "two signs (clear dip)"))
        else:
            print(f"    {a[2]} at {a[1]:.1f}s and {b[1]:.1f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--set", action="append", default=[], help="decision override, e.g. repeat_rest_frames=3 (repeatable)")
    ap.add_argument("--list", action="store_true", help="print every commit, not only the counts")
    ap.add_argument("--profile", default="primary", choices=["primary", "fallback"])
    a = ap.parse_args()
    srv = json.loads((ROOT / "server" / "config" / "server.json").read_text())
    labels = json.loads((ROOT / "protocol" / "labels.json").read_text())["labels"]
    cfg = {**srv["decision"], **srv["models"][a.profile].get("decision", {})}
    over = {k: (float(v) if "." in v else int(v)) for k, v in (s.split("=") for s in a.set)}
    rows = list(csv.DictReader(open(a.log, encoding="utf-8")))
    print(f"{a.log}: {len(rows)} frames, {sum(1 for _ in streams(rows))} stream(s)")
    logged = []
    for s, st in enumerate(streams(rows)):
        t0 = int(st[0]["ts_ms"])
        logged += [(s, (int(r["ts_ms"]) - t0) / 1000.0, r["committed"]) for r in st if r["committed"]]
    deployed = replay(rows, labels, {**cfg, "repeat_rest_frames": 0})
    for title, ev in (("LOGGED by the server", logged), ("REPLAY with the deployed settings (must equal the log)", deployed)):
        show(title, ev, rows) if a.list else print(f"\n{title}: {len(ev)} commits; same word twice within 6 s: {len(repeats(ev))}")
    print("\nreplay equals the log:", [e[2] for e in logged] == [e[2] for e in deployed])
    if over:
        show("REPLAY with overrides " + json.dumps(over), replay(rows, labels, {**cfg, **over}), rows)


if __name__ == "__main__":
    main()
