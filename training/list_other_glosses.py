"""Step 1 of the YOU investigation: which OTHER (= `none`) training glosses look like YOU, and how much of `none` do they supply?

  python training/list_other_glosses.py --model rtmw_xl_384x288 [--ckpt <best.pt>]

Prints (1) every OTHER gloss in the training split with clip count, (2) a pointing score per gloss, measured on the cached hand keypoints:
the share of active frames in which one hand has the index finger long and the other three fingers curled (2D distance to the wrist
over distance to the knuckle; a finger pointing AT the camera is foreshortened, so this finds sideways / upward points best),
(3) the highest probability the checkpoint gives to YOU on each OTHER gloss, and (4) the share of one epoch's windows (train.py sampling).
"""
import argparse
from collections import defaultdict

import numpy as np
import torch

import replay_eval as RE
import sign_data as D
import signfeat as S
import train as T
from sign_model import SignNet

POINT_FRAC = 0.6  # a gloss counts as "pointing" at or above this share of its active frames


def point_frac(d, c):
    s0 = d["start"][c]
    a0, a1 = d["seg"][c]
    best = 0.0
    for h0, f in ((S.L0, 0), (S.R0, 1)):
        P = d["pts"][s0 + a0:s0 + a1 + 1, h0:h0 + 21]
        r = np.linalg.norm(P - P[:, 0:1], axis=-1)
        ratio = lambda tip, mcp: r[:, tip] / np.maximum(r[:, mcp], 1e-3)  # noqa: E731
        ok = d["flg"][s0 + a0:s0 + a1 + 1, f] & (ratio(8, 5) > 1.6) & (ratio(12, 9) < 1.4) & (ratio(16, 13) < 1.4) & (ratio(20, 17) < 1.4)
        best = max(best, float(ok.mean()) if len(ok) else 0.0)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="rtmw_xl_384x288")
    ap.add_argument("--ckpt", default=None, help="checkpoint for the p(you) column (default runs/<model>/best.pt)")
    a = ap.parse_args()
    d = D.load(a.model)
    net = SignNet(n_cls=len(D.LABELS)).cuda().eval()
    net.load_state_dict(torch.load(a.ckpt or T.RUNS / a.model / "best.pt"))
    you = D.LABELS.index("you")
    by = defaultdict(list)
    for c in range(len(d["len"])):
        by[d["gloss"][c]].append(c)
    rows = []
    with torch.no_grad():
        for g, cs in by.items():
            pf = float(np.mean([point_frac(d, c) for c in cs]))
            py = float(np.mean([RE.stream_probs(net, *RE.clip_rows(d, c))[:, you].max() for c in cs]))
            rows.append((g, D.LABELS[d["cls"][cs[0]]] if d["cls"][cs[0]] else "other", len(cs), sum(d["split"][c] == "train" for c in cs), pf, py))
    other = sorted([r for r in rows if r[1] == "other"], key=lambda r: -r[4])
    print(f"{len(other)} OTHER glosses, {sum(r[2] for r in other)} clips ({sum(r[3] for r in other)} in train). Pointing score >= {POINT_FRAC} marked *")
    print(f"{'gloss':<14}{'clips':>6}{'train':>6}{'point':>7}{'max p(you)':>12}")
    for g, _, n, ntr, pf, py in other:
        print(f"{g:<14}{n:>6}{ntr:>6}{pf:>7.2f}{py:>12.2f} {'*' if pf >= POINT_FRAC else ''}")
    yr = [r for r in rows if r[1] == "you"][0]
    print(f"\nYOU itself: {yr[2]} clips ({yr[3]} train), pointing score {yr[4]:.2f} (low because it points AT the camera), max p(you) {yr[5]:.2f}")
    flagged = [r for r in other if r[4] >= POINT_FRAC]
    print("\nIndex-point glosses in OTHER:", ", ".join(f"{r[0]}({r[3]})" for r in flagged))
    print("Deictic points among the OTHER glosses: HE" + (" (ME, I, SHE, IT, THEY, THERE, THAT, HERE, WE, THIS, YOUR are not in the 150-gloss sample at all)"))
    # one epoch of the current sampling: none windows = NONE_RATIO x largest word class, half of them from OTHER clips chosen uniformly
    tr_other = [c for c in range(len(d["len"])) if d["split"][c] == "train" and d["cls"][c] == 0]
    big = max(T.REPEATS * int(((d["split"] == "train") & (d["cls"] == k)).sum()) for k in range(1, len(D.LABELS)))
    from_other = T.NONE_RATIO * big // 2
    n_you = T.REPEATS * int(((d["split"] == "train") & (d["cls"] == you)).sum())
    for name, glosses in (("HE only", ["HE"]), ("all flagged", [r[0] for r in flagged])):
        k = sum(d["gloss"][c] in glosses for c in tr_other)
        print(f"{name}: {k} of {len(tr_other)} train OTHER clips = {from_other * k / len(tr_other):.0f} of {T.NONE_RATIO * big} `none` windows per epoch; YOU has {n_you} windows per epoch")


if __name__ == "__main__":
    main()
