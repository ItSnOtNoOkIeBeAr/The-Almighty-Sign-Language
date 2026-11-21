"""Evaluate decision-logic settings (motion gate, per-word thresholds, ...) on the dataset replay (VAL, TEST) and on a live diagnostic log.

  training\\.venv\\Scripts\\python training\\guard_experiments.py [--live logs\\diag_X.csv --phases logs\\diag_X.phases.json] [--configs cfgs.json] [--only a,b]
  training\\.venv\\Scripts\\python training\\guard_experiments.py --motion-stats          # how much the hands move in word clips (VAL+TEST)

Per configuration (named overrides on the deployed decision settings) it prints, for VAL and TEST: words exactly right, doubles, other-gloss false
triggers, sentences exactly right; and for the live log: commits inside the "not signing" phase (all false) and commits per word in the "words"
phase. Settings are CHOSEN on VAL + the live log, TEST is only reported. Dataset streams get the same per-tick hand motion the servers compute
(signfeat.hand_motion between consecutive ticks); the live log carries it in its `mot` column (written by -Diag).
The phases file comes from server\\scripts\\mark_phases.py (the signer's own markers, not guesses). Numbers only.
"""
import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "server" / "fallback"))
import replay_eval as RE  # noqa: E402
import sign_data as D  # noqa: E402
import signfeat as S  # noqa: E402
import train as T  # noqa: E402
from decision import Decision  # noqa: E402
from sign_model import SignNet  # noqa: E402

MODEL = "rtmw_xl_384x288"
WORDS = D.LABELS[1:]
MOTION = {}  # id(probs array) -> per-tick hand motion of that stream (the arrays stay alive in the item lists)


def stream_motion(pts, flg):
    m = np.zeros(len(pts), np.float32)
    for i in range(1, len(pts)):
        m[i] = S.hand_motion(pts[i - 1], pts[i], flg[i - 1], flg[i])[4]
    return m


_orig_stream_probs = RE.stream_probs


def _stream_probs(model, pts, flg, dev="cuda"):
    probs = _orig_stream_probs(model, pts, flg, dev)
    MOTION[id(probs)] = stream_motion(pts, flg)
    return probs


RE.stream_probs = _stream_probs  # replay_eval.build() calls it for every stream: record the motion alongside


def base_cfg():
    run = T.RUNS / MODEL
    srv = json.loads((ROOT / "server" / "config" / "server.json").read_text())
    prof = "primary" if "xl" in MODEL else "fallback"
    return {**srv["decision"], **srv["models"][prof].get("decision", {}), **json.loads((run / "selected.json").read_text())["decision"]}


def run_stream(probs, left, right, cfg, motion, t0=0.0):
    dec, ev = Decision(D.LABELS, cfg), []
    for i in range(len(probs)):
        out = dec.step(list(map(float, probs[i])), bool(left[i] or right[i]), t0 + i * 100.0, float(motion[i]))
        if out[3]:
            ev.append((i, out[3]))
    return ev


def eval_items(items, cfg):
    word, other, sent, dbl = Counter(), Counter(), Counter(), 0
    for kind, exp, probs, flg in items:
        ev = [e for _, e in run_stream(probs, flg[:, 3], flg[:, 4], cfg, MOTION[id(probs)])]
        if kind == "word":
            word["n"] += 1
            word["right"] += ev == exp
            dbl += sum(e == exp[0] for e in ev) + sum(exp[0] in e.split("_") and "_" in e for e in ev) >= 2
        elif kind == "other":
            other["n"] += 1
            other["ft"] += bool(ev)
        else:
            sent["n"] += 1
            sent["right"] += ev == exp
    return word, other, sent, dbl


def load_live(path, phases_path):
    rows = list(csv.DictReader(open(path)))
    if "mot" not in rows[0]:
        sys.exit(f"{path} has no motion columns: record it with a server built after the motion change (start_server.ps1 -Diag)")
    ph = json.loads(Path(phases_path).read_text())
    streams, cur = [], []
    for r in rows:
        if cur and int(r["ts_ms"]) - int(cur[-1]["ts_ms"]) > 3000:
            streams.append(cur)
            cur = []
        cur.append(r)
    streams.append(cur)
    out = []
    for st in streams:
        out.append(([[float(r["p_" + n]) for n in D.LABELS] for r in st], [r["hand_L_strict"] == "1" for r in st], [r["hand_R_strict"] == "1" for r in st],
                    [float(r["mot"]) for r in st], [int(r["ts_ms"]) for r in st]))
    return out, ph


def eval_live(live, cfg):
    streams, ph = live
    false, after = [], Counter()
    for probs, left, right, mot, ts in streams:
        for i, e in run_stream(probs, left, right, cfg, mot, t0=float(ts[0])):
            t = ts[i]
            if ph["no_sign"][0] <= t < ph["no_sign"][1]:
                false.append(e)
            elif ph["words"][0] <= t < ph["words"][1]:
                after[e] += 1
    return false, after


CONFIGS = {"deployed": {}}


def main():
    global MODEL
    ap = argparse.ArgumentParser()
    ap.add_argument("--live")
    ap.add_argument("--phases")
    ap.add_argument("--only", default="")
    ap.add_argument("--configs", help="JSON file: {name: overrides} replacing the built-in list")
    ap.add_argument("--model", default=MODEL, help="rtmw_xl_384x288 (primary) or rtmw_lm_256x192 (fallback profile)")
    ap.add_argument("--motion-stats", action="store_true")
    a = ap.parse_args()
    MODEL = a.model
    d = D.load(MODEL)
    run = T.RUNS / MODEL
    net = SignNet(n_cls=len(D.LABELS)).cuda().eval()
    net.load_state_dict(torch.load(run / "best.pt"))
    RE.TAIL = 35
    splits = {s: RE.build(net, d, s) for s in ("val", "test")}
    if a.motion_stats:
        motion_stats(splits)
        return
    live = load_live(a.live, a.phases) if a.live else None
    base = base_cfg()
    cfgs = json.loads(Path(a.configs).read_text()) if a.configs else CONFIGS
    only = set(filter(None, a.only.split(",")))
    print(f"{'config':<26}| {'VAL right dbl FT sent':<26}| {'TEST right dbl FT sent':<26}| LIVE: false | words in the words phase (hello what how why where you | sentences)")
    for name, over in cfgs.items():
        if only and name not in only:
            continue
        cfg = {**base, **over}
        cols = []
        for s in ("val", "test"):
            w, o, se, dbl = eval_items(splits[s], cfg)
            cols.append(f"{w['right']:>3}/{w['n']} {dbl:>2} {o['ft']:>3}/{o['n']} {se['right']:>2}/{se['n']}")
        tail = ""
        if live:
            false, after = eval_live(live, cfg)
            tail = f"{len(false):>4}  " + " ".join(f"{after[w]:>2}" for w in WORDS) + f" | {after['how_are_you'] + after['where_are_you']}   false={dict(Counter(false))}"
        print(f"{name:<26}| {cols[0]:<25} | {cols[1]:<25} | {tail}")


def gate_value(m, end, window):
    return float(m[max(0, end - window + 1):end + 1].sum())


def motion_stats(splits):
    """The best the motion gate could see in a stream: the maximum over the stream of the summed motion of W consecutive ticks, per clip kind."""
    for W in (10, 20, 30):
        for kind in ("word", "other", "sentence"):
            vals = np.array([max(gate_value(MOTION[id(p)], i, W) for i in range(len(p))) for k, _, p, _ in splits["val"] + splits["test"] if k == kind])
            q = np.percentile(vals, [0, 5, 10, 25, 50, 75])
            print(f"W={W:>2} {kind:<9} n={len(vals):>3} max-sum percentiles 0/5/10/25/50/75: " + " ".join(f"{x:.2f}" for x in q))


if __name__ == "__main__":
    main()
