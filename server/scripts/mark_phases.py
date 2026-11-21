r"""Turn a diagnostic log plus the signer's own markers into phases (no guessing) and summarise each phase.

  client\.venv\Scripts\python server\scripts\mark_phases.py logs\diag_X.csv [--markers logs\markers.csv] [--at no_sign=21:04:10 --at gap=21:07:10 --at words=21:07:30 --at end=21:13:00]
Markers come from client\mark.py (epoch ms). --at gives local clock times instead (same machine as the client; same day as the log), and overrides a marker.
Writes logs\diag_X.phases.json = {"no_sign": [start_ms, end_ms], "gap": [...], "words": [...]} (client timestamps) for training\guard_experiments.py and prints,
per phase: ticks, share of ticks with a hand in view, mean hand motion, and the committed words. A calm gap that is not calm is reported.
"""
import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--markers", default=str(Path(__file__).resolve().parents[2] / "logs" / "markers.csv"))
    ap.add_argument("--at", action="append", default=[], help="name=HH:MM:SS local clock time")
    a = ap.parse_args()
    rows = list(csv.DictReader(open(a.log)))
    t_first, t_last = int(rows[0]["ts_ms"]), int(rows[-1]["ts_ms"])
    marks = {}
    if Path(a.markers).exists():
        for line in open(a.markers):
            n, _, t = line.strip().partition(",")
            if t and t_first - 60_000 <= int(t) <= t_last + 600_000:
                marks[n] = int(t)  # the latest marker of each name that belongs to this log
    for item in a.at:
        n, _, hms = item.partition("=")
        day = time.localtime(t_first / 1000)
        marks[n] = int(time.mktime(time.strptime(f"{day.tm_year}-{day.tm_mon}-{day.tm_mday} {hms}", "%Y-%m-%d %H:%M:%S")) * 1000)
    missing = [n for n in ("no_sign", "gap", "words") if n not in marks]
    if missing:
        raise SystemExit(f"missing markers: {missing} (client/mark.py <name> while recording, or --at name=HH:MM:SS)")
    end = marks.get("end", t_last + 1)
    ph = {"no_sign": [marks["no_sign"], marks["gap"]], "gap": [marks["gap"], marks["words"]], "words": [marks["words"], end]}
    Path(a.log).with_suffix(".phases.json").write_text(json.dumps(ph))
    for name, (lo, hi) in ph.items():
        rs = [r for r in rows if lo <= int(r["ts_ms"]) < hi]
        if not rs:
            print(f"{name:<8} no ticks in {(hi - lo) / 1000:.0f} s")
            continue
        up = sum(r["hand_L_strict"] == "1" or r["hand_R_strict"] == "1" for r in rs) / len(rs)
        mot = [float(r["mot"]) for r in rs] if "mot" in rs[0] else [0.0]
        com = Counter(r["committed"] for r in rs if r["committed"])
        print(f"{name:<8} {(hi - lo) / 1000:6.0f} s  {len(rs):>5} ticks  hands in view {100 * up:3.0f}%  mean motion/tick {sum(mot) / len(mot):.3f}  commits {sum(com.values()):>3} {dict(com)}")
    gap = [r for r in rows if ph["gap"][0] <= int(r["ts_ms"]) < ph["gap"][1]]
    if gap and (any(r["committed"] for r in gap) or sum(r["hand_L_strict"] == "1" or r["hand_R_strict"] == "1" for r in gap) > 0.2 * len(gap)):
        print("WARNING: the calm gap is not calm (commits or hands in view): check the markers")
    print("wrote", Path(a.log).with_suffix(".phases.json"))


if __name__ == "__main__":
    main()
