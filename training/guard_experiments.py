"""False-trigger guards: evaluate decision-logic settings on the dataset replay (VAL, TEST) and on a live diagnostic log.

  training\\.venv\\Scripts\\python training\\guard_experiments.py --live logs\\diag_20261121_044956.csv --no-sign-ticks 1800 [--only name1,name2]

Per configuration (named overrides on the deployed decision settings, see CONFIGS below) it prints, for VAL and TEST: words exactly right, doubles,
other-gloss false triggers, sentences exactly right; and for the live log: commits during the "not signing" phase (all false) and the commits
per word afterwards (the signer did each word 3 times, so these are a recall proxy). Settings are CHOSEN on VAL + live, TEST is only reported.
Numbers only. The live log is read from logs/ (gitignored).
"""
import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "server" / "fallback"))
import replay_eval as RE  # noqa: E402
import sign_data as D  # noqa: E402
import train as T  # noqa: E402
from decision import Decision  # noqa: E402
from sign_model import SignNet  # noqa: E402

MODEL = "rtmw_xl_384x288"
WORDS = D.LABELS[1:]


class Ensemble(torch.nn.Module):
    """Mean of the logits of several SignNets (one ONNX graph when exported)."""

    def __init__(self, nets):
        super().__init__()
        self.nets = torch.nn.ModuleList(nets)

    mode = "logit"

    def forward(self, x):
        out = torch.stack([n(x).float() for n in self.nets])
        return out.mean(0) if self.mode == "logit" else out.softmax(-1).mean(0).clamp_min(1e-9).log()


def load_net(paths):
    nets = []
    for p in paths:
        n = SignNet(n_cls=len(D.LABELS)).cuda().eval()
        n.load_state_dict(torch.load(p))
        nets.append(n)
    return nets[0] if len(nets) == 1 else Ensemble(nets).cuda().eval()


def base_cfg():
    run = T.RUNS / MODEL
    srv = json.loads((ROOT / "server" / "config" / "server.json").read_text())
    prof = "primary" if "xl" in MODEL else "fallback"
    return {**srv["decision"], **srv["models"][prof].get("decision", {}), **json.loads((run / "selected.json").read_text())["decision"]}


def run_stream(probs, left, right, cfg, t0=0.0):
    dec, ev = Decision(D.LABELS, cfg), []
    for i in range(len(probs)):
        out = dec.step(list(map(float, probs[i])), bool(left[i] or right[i]), t0 + i * 100.0)
        if out[3]:
            ev.append((i, out[3]))
    return ev


def eval_items(items, cfg):
    word, other, sent, dbl = Counter(), Counter(), Counter(), 0
    for kind, exp, probs, flg in items:
        ev = [e for _, e in run_stream(probs, flg[:, 3], flg[:, 4], cfg)]
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


def load_live(path, split_tick):
    rows = list(csv.DictReader(open(path)))
    seg0 = []
    for r in rows:  # the first stream of the log (the log restarts a stream when the client clock jumps)
        if seg0 and int(r["ts_ms"]) - int(seg0[-1]["ts_ms"]) > 3000:
            break
        seg0.append(r)
    probs = [[float(r["p_" + n]) for n in D.LABELS] for r in seg0]
    return probs, [r["hand_L_strict"] == "1" for r in seg0], [r["hand_R_strict"] == "1" for r in seg0], [int(r["ts_ms"]) for r in seg0], split_tick


def eval_live(live, cfg):
    probs, left, right, ts, split = live
    ev = run_stream(probs, left, right, cfg, t0=float(ts[0]))
    false = [e for i, e in ev if i < split]
    after = Counter(e for i, e in ev if i >= split)
    return false, after


# name -> overrides on the deployed settings. Chosen on VAL + live log; TEST is reported only.
CONFIGS = {
    "deployed": {},
    "no_guard": {"repeat_rest_frames": 0},
}


def main():
    global MODEL
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", required=True)
    ap.add_argument("--no-sign-ticks", type=int, default=1800)
    ap.add_argument("--only", default="")
    ap.add_argument("--ensemble", action="store_true", help="deployed checkpoint + the 4 retrained seeds (runs/<model>/base/seed*.pt), logits averaged")
    ap.add_argument("--ens-mode", default="logit", choices=["logit", "prob"], help="average the logits (one ONNX graph) or the probabilities")
    ap.add_argument("--only-seeds", default="", help="comma list of base seeds to use, e.g. 1,2 (default all)")
    ap.add_argument("--configs", help="JSON file: {name: overrides} replacing the built-in list")
    ap.add_argument("--model", default=MODEL, help="rtmw_xl_384x288 (primary) or rtmw_l_m_256x192 (fallback profile)")
    a = ap.parse_args()
    MODEL = a.model
    d = D.load(MODEL)
    run = T.RUNS / MODEL
    seeds = sorted((run / "base").glob("seed*.pt"))
    if a.only_seeds:
        seeds = [q for q in seeds if q.stem[4:] in a.only_seeds.split(",")]
    paths = [run / "best.pt"] + (seeds if a.ensemble else [])
    net = load_net(paths)
    if a.ensemble:
        net.mode = a.ens_mode
    print(f"{len(paths)} checkpoint(s)")
    RE.TAIL = 35
    splits = {s: RE.build(net, d, s) for s in ("val", "test")}
    live = load_live(a.live, a.no_sign_ticks)
    base = base_cfg()
    cfgs = json.loads(Path(a.configs).read_text()) if a.configs else CONFIGS
    only = set(filter(None, a.only.split(",")))
    print(f"{'config':<28}| {'VAL right dbl FT sent':<26}| {'TEST right dbl FT sent':<26}| LIVE false  words after (hello what how why where you | sentences)")
    for name, over in cfgs.items():
        if only and name not in only:
            continue
        cfg = {**base, **over}
        cols = []
        for s in ("val", "test"):
            w, o, se, dbl = eval_items(splits[s], cfg)
            cols.append(f"{w['right']:>3}/{w['n']} {dbl:>2} {o['ft']:>3}/{o['n']} {se['right']:>2}/{se['n']}")
        false, after = eval_live(live, cfg)
        per = " ".join(f"{after[w]:>2}" for w in WORDS)
        sents = after["how_are_you"] + after["where_are_you"]
        print(f"{name:<28}| {cols[0]:<25} | {cols[1]:<25} | {len(false):>4}        {per} | {sents}   false={dict(Counter(false))}")


if __name__ == "__main__":
    main()
