"""Compare training recipes on VAL (never TEST) with one fixed decision setting.

  python training/variant_report.py --model rtmw_xl_384x288 base he_only you_x3 ...      (tags = folders under runs/<model>/; "." = the deployed one)
For every seed checkpoint of a tag: window macro-F1 and per-class recall on the VAL signers, plus the end-to-end replay (words exactly right,
false triggers on other-gloss clips, sentences) with the decision settings from server.json. Shown as mean over seeds (and the worst seed).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "server" / "fallback"))
import replay_eval as RE  # noqa: E402
import sign_data as D  # noqa: E402
import train as T  # noqa: E402
from sign_model import SignNet  # noqa: E402


def evaluate(model_name, ck, d, bank, plan, items, cfg):
    net = SignNet(n_cls=len(D.LABELS)).cuda().eval()
    net.load_state_dict(torch.load(ck))
    pr = T.predict(net, bank, plan)
    y, yh = plan["label"], pr.argmax(1)
    rec = [float((yh[y == k] == k).mean()) if (y == k).any() else float("nan") for k in range(len(D.LABELS))]
    m = T.metrics(pr, y)
    items = RE.build(net, d, "val") if items is None else items
    res, per = RE.score(items, cfg)
    sm = RE.summary(res)
    word = {k.split(":")[-1]: v["correct"] / max(v["correct"] + v["bad"], 1) for k, v in per.items() if k != "other" and not k.startswith("sentence")}
    return {"f1": m["macro_f1"], "acc": m["acc"], "recall": rec, "word": sm["word_correct"], "ft": sm["other_false_trigger"], "sent": sm["sentence_correct"], "per_word": word}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="rtmw_xl_384x288")
    ap.add_argument("--split", default="val", choices=["val", "test"], help="test only for the final report of the chosen recipe")
    ap.add_argument("tags", nargs="+")
    a = ap.parse_args()
    d = D.load(a.model)
    bank = D.Bank(d, "cuda")
    plan = T.eval_set(d, a.split)
    cfg = json.loads((ROOT / "server" / "config" / "server.json").read_text())["decision"]
    rows = {}
    for tag in a.tags:
        folder = T.RUNS / a.model / ("" if tag == "." else tag)
        cks = sorted(folder.glob("seed*.pt"))
        rows[tag] = [evaluate(a.model, ck, d, bank, plan, None, cfg) for ck in cks]
        print(f"[{tag}] {len(cks)} seeds done", flush=True)
    names = D.LABELS
    print(f"\n{a.split.upper()} signers, decision settings fixed (thr {cfg['commit_threshold']}, M {cfg['commit_consecutive_M']}); mean over seeds (worst seed in brackets)")
    print(f"{'recipe':<16}{'F1':>7}{'acc':>7}{'words':>14}{'false trig':>14}{'sentences':>13}   window recall: " + " ".join(f"{n:>6}" for n in names))
    for tag, rs in rows.items():
        f = lambda k: np.array([r[k] for r in rs])  # noqa: E731
        rec = np.nanmean(np.array([r["recall"] for r in rs]), 0)
        print(f"{tag:<16}{f('f1').mean():7.3f}{f('acc').mean():7.3f}{f('word').mean():8.3f} ({f('word').min():.2f}){f('ft').mean():8.3f} ({f('ft').max():.3f}){f('sent').mean():8.3f} ({f('sent').min():.2f})   "
              + "                " + " ".join(f"{v:6.2f}" for v in rec))
    print("\nend-to-end per word (exactly right), mean over seeds:")
    print(f"{'recipe':<16}" + " ".join(f"{n:>7}" for n in names[1:]))
    for tag, rs in rows.items():
        print(f"{tag:<16}" + " ".join(f"{np.mean([r['per_word'].get(n, np.nan) for r in rs]):7.2f}" for n in names[1:]))
    json.dump(rows, open(T.RUNS / a.model / f"variants_{a.split}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
