"""Summarise a live diagnostic log (logs/diag_*.csv, written by the server's --diag-log; numbers only, no images).

  client\\.venv\\Scripts\\python server\\scripts\\diag_report.py logs\\diag_20261120_101500.csv [--word you]

Answers: did the target word EVER get probability, and if not, were the hands even detected?
Frames are grouped by "hands up" = strict hand presence (the same rule that opens the server's idle gate).
Standard library only, so it also runs on the laptop.
"""
import argparse
import csv
from collections import Counter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--word", default="you", help="class to look at (default: you)")
    a = ap.parse_args()
    rows = list(csv.DictReader(open(a.log, encoding="utf-8")))
    if not rows:
        raise SystemExit("empty log")
    classes = [c[2:] for c in rows[0] if c.startswith("p_")]
    f = lambda r, k: float(r[k])  # noqa: E731
    n = len(rows)
    up = [r for r in rows if r["hand_L_strict"] == "1" or r["hand_R_strict"] == "1"]
    print(f"{a.log}: {n} frames ({n / 10:.0f} s at 10 fps), profile {rows[0]['profile']}")
    print(f"hands up (strict presence): {len(up)} frames ({100 * len(up) / n:.0f}%)")
    print(f"hand points above the keypoint threshold ({rows[0]['thr']}), hands-up frames: "
          f"left median {sorted(int(r['pts_L']) for r in up)[len(up) // 2] if up else 0}/21, "
          f"right median {sorted(int(r['pts_R']) for r in up)[len(up) // 2] if up else 0}/21 (a hand needs 15)")
    print("\nclass     max p   frames p>0.10  p>0.30  p>0.50   top-1 frames (hands up)")
    top1 = Counter(r["label"] for r in up)
    for c in classes:
        p = [f(r, "p_" + c) for r in up]
        print(f"{c:<9} {max(p, default=0):5.3f}   {sum(x > .1 for x in p):8d} {sum(x > .3 for x in p):7d} {sum(x > .5 for x in p):7d}   {top1.get(c, 0):6d}")
    committed = [r["committed"] for r in rows if r["committed"]]
    print(f"\ncommitted words: {committed if committed else 'none'}")
    # bursts of hands-up frames = one try each
    bursts, cur = [], []
    for r in rows:
        if r["hand_L_strict"] == "1" or r["hand_R_strict"] == "1":
            cur.append(r)
        elif cur:
            bursts.append(cur)
            cur = []
    if cur:
        bursts.append(cur)
    bursts = [b for b in bursts if len(b) >= 3]
    print(f"\nhands-up bursts (>= 3 frames): {len(bursts)}  | per burst: frames, peak p_{a.word}, peak class other than none, hand points L/R (median)")
    for i, b in enumerate(bursts, 1):
        peak = max(b, key=lambda r: f(r, "p_" + a.word))
        other = max(((c, max(f(r, "p_" + c) for r in b)) for c in classes if c != "none"), key=lambda t: t[1])
        pl, pr = sorted(int(r["pts_L"]) for r in b)[len(b) // 2], sorted(int(r["pts_R"]) for r in b)[len(b) // 2]
        print(f"  {i:2d}: {len(b):3d} frames  peak p_{a.word} {f(peak, 'p_' + a.word):.3f}  best non-none {other[0]} {other[1]:.3f}  hand pts {pl}/{pr}  committed {[r['committed'] for r in b if r['committed']]}")


if __name__ == "__main__":
    main()
