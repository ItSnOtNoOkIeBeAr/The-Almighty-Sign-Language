"""Phase 7/8 end-to-end replay on cached keypoints: clip streams -> signfeat -> SignNet -> Decision (same code as the server).

  python training/replay_eval.py --model rtmw_xl_384x288 --tune      # grid on VAL, then report TEST with the chosen setting
Each clip becomes a 10 fps stream: 1 s of its first frame (resting), the clip, 2 s of its last frame (flushes held words).
A word clip is correct when the committed events are exactly [its word]; an `other`-gloss clip is a false trigger if anything commits.
Sentences are built from two clips of the same signer (HOW+YOU, WHERE+YOU, YOU+WHERE) and must give exactly [sentence].
"""
import argparse
import itertools
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "server" / "fallback"))
import sign_data as D  # noqa: E402
import signfeat as S  # noqa: E402
import train as T  # noqa: E402
from decision import Decision  # noqa: E402
from sign_model import SignNet  # noqa: E402

LEAD, TAIL, GAP = 10, 20, 3  # GAP: rest frames between two signs of a sentence (0.3 s)


def clip_rows(d, c):
    s, n = d["start"][c], d["len"][c]
    p, f = d["pts"][s:s + n], d["flg"][s:s + n]
    return np.concatenate([p[:1].repeat(LEAD, 0), p, p[-1:].repeat(TAIL, 0)]), np.concatenate([f[:1].repeat(LEAD, 0), f, f[-1:].repeat(TAIL, 0)])


@torch.no_grad()
def stream_probs(model, pts, flg, dev="cuda"):
    n = len(pts)
    idx = np.clip(np.arange(n)[:, None] - np.arange(S.WINDOW - 1, -1, -1)[None], 0, None)
    out = []
    for i in range(0, n, 512):
        w = torch.from_numpy(pts[idx[i:i + 512]]).to(dev), torch.from_numpy(flg[idx[i:i + 512]]).to(dev)
        out.append(model(S.window_features(*w)).float().softmax(-1).cpu())
    return torch.cat(out).numpy()


def run_decision(probs, flg, cfg):
    dec, events = Decision(D.LABELS, cfg), []
    for i in range(len(probs)):
        out = dec.step(probs[i].tolist(), bool(flg[i, 3] or flg[i, 4]), i * 100.0)
        if out[3]:
            events.append(out[3])
    return events


def build(model, d, split):
    """-> list of (kind, expected_events, probs, flg). Probabilities are computed once; the decision grid reuses them."""
    items = []
    sp = np.where(d["split"] == split)[0]
    cache = {}

    def get(c):
        if c not in cache:
            p, f = clip_rows(d, c)
            cache[c] = (stream_probs(model, p, f), f)
        return cache[c]

    rng = np.random.default_rng(0)
    for c in sp:
        if d["cls"][c] > 0:
            items.append(("word", [D.LABELS[d["cls"][c]]], *get(c)))
    other = [c for c in sp if d["cls"][c] == 0]
    for c in rng.choice(other, min(len(other), 400), replace=False):  # 400 random other-gloss clips as the false-trigger check
        items.append(("other", [], *get(c)))
    by = defaultdict(lambda: defaultdict(list))
    for c in sp:
        if d["cls"][c] > 0:
            by[d["signer"][c]][D.LABELS[d["cls"][c]]].append(c)

    def sign_rows(c):
        """The sign itself: active segment +- 2 frames, so back-to-back signs are separated only by GAP frames."""
        s0, n = d["start"][c], d["len"][c]
        lo, hi = max(0, d["seg"][c][0] - 2), min(n - 1, d["seg"][c][1] + 2)
        return d["pts"][s0 + lo:s0 + hi + 1], d["flg"][s0 + lo:s0 + hi + 1]

    for sg, w in by.items():
        for a, b, sent in (("how", "you", "how_are_you"), ("where", "you", "where_are_you"), ("you", "where", "where_are_you")):
            if w[a] and w[b]:
                (pa, fa), (pb, fb) = sign_rows(w[a][0]), sign_rows(w[b][0])
                pts = np.concatenate([pa[:1].repeat(LEAD, 0), pa, pa[-1:].repeat(GAP, 0), pb, pb[-1:].repeat(TAIL, 0)])
                flg = np.concatenate([fa[:1].repeat(LEAD, 0), fa, fa[-1:].repeat(GAP, 0), fb, fb[-1:].repeat(TAIL, 0)])
                items.append(("sentence", [sent], stream_probs(model, pts, flg), flg))
    return items


def score(items, cfg):
    res = defaultdict(Counter)
    detail = defaultdict(Counter)
    for kind, exp, probs, flg in items:
        ev = run_decision(probs, flg, cfg)
        key = exp[0] if exp else "other"
        if kind == "other":
            res[kind]["false_trigger" if ev else "clean"] += 1
        else:
            res[kind]["correct" if ev == exp else ("missed" if not ev else ("extra" if exp[0] in ev else "wrong"))] += 1
            detail[key if kind == "word" else "sentence:" + key]["correct" if ev == exp else "bad"] += 1
    return res, detail


def summary(res):
    w, o, s = res["word"], res["other"], res["sentence"]
    wn, on, sn = sum(w.values()), sum(o.values()), sum(s.values())
    return {"word_correct": w["correct"] / max(wn, 1), "other_false_trigger": o["false_trigger"] / max(on, 1), "sentence_correct": s["correct"] / max(sn, 1),
            "n": (wn, on, sn)}


def grid_val(val, cfg0):
    """Best decision setting on VAL: (score, cfg). score = word_correct - 2 x other_false_trigger + sentence_correct."""
    best, best_s, rows = cfg0, -9, []
    for thr, m, al in itertools.product((0.5, 0.6, 0.7, 0.8), (2, 3, 4), (0.5, 0.7)):
        cfg = {**cfg0, "commit_threshold": thr, "commit_consecutive_M": m, "ema_alpha": al}
        sm = summary(score(val, cfg)[0])
        sc = sm["word_correct"] - 2 * sm["other_false_trigger"] + sm["sentence_correct"]
        rows.append((thr, m, al, sm, sc))
        if sc > best_s:
            best, best_s = cfg, sc
    return best_s, best, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tune", action="store_true", help="grid the decision settings on VAL for every seed, keep the best seed (copied to best.pt)")
    a = ap.parse_args()
    d = D.load(a.model)
    cfg0 = json.loads((ROOT / "server" / "config" / "server.json").read_text())["decision"]
    run = T.RUNS / a.model
    model = SignNet(n_cls=len(D.LABELS)).cuda().eval()
    best = cfg0
    if a.tune:
        results = []
        for ck in sorted(run.glob("seed*.pt")):
            model.load_state_dict(torch.load(ck))
            sc, cfg, _ = grid_val(build(model, d, "val"), cfg0)
            results.append((sc, ck, cfg))
            print(f"  {ck.name}: best val replay score {sc:.3f}  thr {cfg['commit_threshold']} M {cfg['commit_consecutive_M']} alpha {cfg['ema_alpha']}", flush=True)
        sc, ck, best = max(results, key=lambda r: r[0])
        torch.save(torch.load(ck), run / "best.pt")
        print(f"chosen on val: {ck.name}, score {sc:.3f}, " + json.dumps({k: best[k] for k in ("commit_threshold", "commit_consecutive_M", "ema_alpha")}))
        json.dump({"seed": ck.name, "val_replay_score": sc, "decision": best}, open(run / "selected.json", "w"), indent=1)
    model.load_state_dict(torch.load(run / "best.pt"))
    test = build(model, d, "test")
    res, detail = score(test, best)
    sm = summary(res)
    print(f"\n=== TEST replay ({a.model}), decision {json.dumps({k: best[k] for k in ('commit_threshold', 'commit_consecutive_M', 'ema_alpha', 'cooldown_ms', 'join_ms')})} ===")
    print(f"word clips {sm['n'][0]}: exactly-correct {sm['word_correct']:.3f}  {dict(res['word'])}")
    for k in sorted(detail):
        if k != "other":
            print(f"   {k:>6}: {detail[k]['correct']}/{detail[k]['correct'] + detail[k]['bad']}")
    print(f"other-gloss clips {sm['n'][1]}: false-trigger rate {sm['other_false_trigger']:.3f}  {dict(res['other'])}")
    print(f"sentences {sm['n'][2]}: exactly-correct {sm['sentence_correct']:.3f}  {dict(res['sentence'])}")
    json.dump({"decision": best, "test": {"summary": sm, "word": dict(res["word"]), "other": dict(res["other"]), "sentence": dict(res["sentence"]),
                                           "per_word": {k: dict(v) for k, v in detail.items()}}}, open(run / "replay.json", "w"), indent=1)


if __name__ == "__main__":
    main()
