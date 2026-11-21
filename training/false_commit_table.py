"""List every commit of the "not signing" phase of a live diagnostic log with the features the guards could use (numbers only).

  training\.venv\Scripts\python training\false_commit_table.py logs\diag_20261121_044956.csv --no-sign-ticks 1800
Per commit: tick, word, peak raw probability of the word in its streak, frames above the threshold in that streak, raw top1-top2 margin at the commit,
hands at the commit (L/R strict), share of the streak frames with both hands, strict-hand points (mean/std over the streak: a motion proxy),
and the rest (frames without a strict hand) in the 30 frames before the streak began.
"""
import argparse
import statistics as st
import sys

import guard_experiments as G


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--no-sign-ticks", type=int, default=1800)
    ap.add_argument("--set", action="append", default=[])
    a = ap.parse_args()
    import csv
    rows = []
    for r in csv.DictReader(open(a.log)):
        if rows and int(r["ts_ms"]) - int(rows[-1]["ts_ms"]) > 3000:
            break
        rows.append(r)
    probs, left, right, ts, split = G.load_live(a.log, a.no_sign_ticks)
    cfg = G.base_cfg()
    for s in a.set:
        k, v = s.split("=")
        cfg[k] = float(v) if "." in v else int(v)
    ev = G.run_stream(probs, left, right, cfg, t0=float(ts[0]))
    labels = G.D.LABELS
    print(f"{'tick':>5} {'word':<14}{'peak':>6}{'frames>=thr':>12}{'margin':>8} {'L R':>4}{'both%':>7}{'pts mean':>9}{'pts std':>8}{'rest before':>12}")
    for i, w in ev:
        if i >= split:
            continue
        word = w.split("_")[-1] if "_" in w else w  # a sentence is reported with its last word
        k = labels.index(word)
        if probs[i][k] < cfg["commit_threshold"]:  # a held word (how/where/you) is emitted up to join_ms after its sign: look back for the sign
            back = [j for j in range(max(0, i - 30), i + 1) if probs[j][k] >= cfg["commit_threshold"]]
            if back:
                i = back[-1]
        lo = i
        while lo > 0 and probs[lo - 1][k] >= cfg["commit_threshold"]:
            lo -= 1
        hi = i
        while hi + 1 < len(probs) and probs[hi + 1][k] >= cfg["commit_threshold"]:
            hi += 1
        peak = max(probs[j][k] for j in range(lo, hi + 1))
        srt = sorted(probs[i], reverse=True)
        pts = [int(rows[j]["pts_L"]) + int(rows[j]["pts_R"]) for j in range(lo, i + 1)]
        both = sum(left[j] and right[j] for j in range(lo, i + 1)) / (i - lo + 1)
        rest = sum(not (left[j] or right[j]) for j in range(max(0, lo - 30), lo))
        print(f"{i:>5} {w:<14}{peak:>6.2f}{hi - lo + 1:>12}{srt[0] - srt[1]:>8.2f} {int(left[i])} {int(right[i])}{100 * both:>7.0f}{st.mean(pts):>9.1f}{(st.pstdev(pts) if len(pts) > 1 else 0):>8.1f}{rest:>12}")


if __name__ == "__main__":
    main()
