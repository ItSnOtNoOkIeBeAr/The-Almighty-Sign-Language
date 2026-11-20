"""Double commits on the dataset replay: one word clip = one sign, so the same word committed more than once for it is a double.

  python training/double_commit_report.py --model rtmw_xl_384x288 --split val [--set repeat_cooldown_ms=2000 --set repeat_rest_frames=3] [--ckpt PATH]
Runs the Decision (server/fallback/decision.py, the code the servers port) on every word clip of the split with the DEPLOYED settings
(runs/<model>/selected.json) plus any --set overrides, and prints per word: clips, exactly right, double-commit rate, and the
other-gloss false triggers and sentence results. Doubles: the clip's word (or, for HOW/WHERE/YOU, the word or a sentence containing it) commits twice.
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "server" / "fallback"))
import replay_eval as RE  # noqa: E402
import sign_data as D  # noqa: E402
import train as T  # noqa: E402
from sign_model import SignNet  # noqa: E402


def parse_sets(items):
    out = {}
    for s in items:
        k, v = s.split("=")
        out[k] = float(v) if "." in v else int(v)
    return out


def double_commits(ev, word):
    """Same sign reported more than once: the word twice, or the word alone plus again inside a sentence."""
    parts = [word] if word not in ("how", "where", "you") else [word]
    n = sum(e == word for e in ev) + sum(word in e.split("_") and "_" in e for e in ev)
    return n >= 2


def report(items, cfg, label=""):
    per = defaultdict(Counter)
    res = defaultdict(Counter)
    for kind, exp, probs, flg in items:
        ev = RE.run_decision(probs, flg, cfg)
        if kind == "word":
            w = exp[0]
            per[w]["clips"] += 1
            per[w]["right"] += ev == exp
            per[w]["double"] += double_commits(ev, w)
        elif kind == "other":
            res["other"]["false_trigger" if ev else "clean"] += 1
        else:
            res["sentence"]["correct" if ev == exp else "bad"] += 1
            per["sentence:" + exp[0]]["double"] += len(ev) > 1 and exp[0] in ev
    return per, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="rtmw_xl_384x288")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--tail", type=int, default=35, help="frames of the last frame appended to every stream (replay_eval uses 20: too short for a HOW/WHERE/YOU held 1.5 s "
                    "after its sign, which then is still unreported when the stream ends and can hide a second, double commit)")
    ap.add_argument("--set", action="append", default=[], help="decision override, e.g. repeat_cooldown_ms=2000 (repeatable)")
    a = ap.parse_args()
    d = D.load(a.model)
    run = T.RUNS / a.model
    cfg = json.loads((run / "selected.json").read_text())["decision"]
    srv = json.loads((ROOT / "server" / "config" / "server.json").read_text())
    cfg = {**srv["decision"], **srv["models"]["primary" if "xl" in a.model else "fallback"].get("decision", {}), **{k: v for k, v in cfg.items()}}
    net = SignNet(n_cls=len(D.LABELS)).cuda().eval()
    net.load_state_dict(torch.load(a.ckpt or run / "best.pt"))
    RE.TAIL = a.tail
    items = RE.build(net, d, a.split)
    for title, over in (("BEFORE (settings as deployed)", {"repeat_cooldown_ms": 0, "repeat_rest_frames": 0}), ("AFTER (with overrides)", parse_sets(a.set))):
        if title.startswith("AFTER") and not a.set:
            continue
        c = {**cfg, **over}
        per, res = report(items, c)
        print(f"\n=== {a.model} {a.split.upper()} {title}: " + json.dumps({k: c[k] for k in c if k in ("commit_threshold", "commit_consecutive_M", "ema_alpha", "repeat_cooldown_ms", "repeat_rest_frames")}))
        print(f"{'word':<8}{'clips':>6}{'right':>7}{'double':>8}{'double %':>10}")
        tot = Counter()
        for w in D.LABELS[1:]:
            p = per[w]
            tot.update(p)
            print(f"{w:<8}{p['clips']:>6}{p['right']:>7}{p['double']:>8}{100 * p['double'] / max(p['clips'], 1):>9.1f}%")
        print(f"{'all':<8}{tot['clips']:>6}{tot['right']:>7}{tot['double']:>8}{100 * tot['double'] / max(tot['clips'], 1):>9.1f}%   words exactly right {tot['right'] / max(tot['clips'], 1):.3f}")
        o, s = res["other"], res["sentence"]
        print(f"other-gloss false triggers {o['false_trigger']}/{o['false_trigger'] + o['clean']} = {o['false_trigger'] / max(sum(o.values()), 1):.3f};"
              f" sentences exactly right {s['correct']}/{s['correct'] + s['bad']} = {s['correct'] / max(sum(s.values()), 1):.3f}")


if __name__ == "__main__":
    main()
