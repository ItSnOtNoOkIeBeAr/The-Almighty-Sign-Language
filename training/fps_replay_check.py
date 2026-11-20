"""Does the client send rate change the answers? Replays the TEST clips through a RUNNING server at 10 / 20 / 30 fps input.

  python training/fps_replay_check.py --url ws://127.0.0.1:8766/ws --model rtmw_xl_384x288 --fps 10 20 30 [--other 400]
Every clip is sent as a live stream (640x480 letterboxed JPEG q70, client timestamps exactly 1000/fps ms apart, 1 s of the first frame before,
3.5 s of the last after so held words are emitted); sentences are two signs of one signer with 0.3 s of rest between. Prints, per rate:
words exactly right, double commits, other-gloss false triggers, sentences exactly right, and the same numbers for the offline 10 fps replay
(cached keypoints + decision.py with the server's settings) as the reference. Needs SIGNLANG_TOKEN. Numbers only; nothing is saved.
"""
import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch
from websockets.sync.client import connect

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "server" / "fallback"))
sys.path.insert(0, str(ROOT / "client"))
import phase3_eval as E  # noqa: E402
import replay_eval as RE  # noqa: E402
import sign_data as D  # noqa: E402
import train as T  # noqa: E402
from common import encode_jpeg, pack_frame  # noqa: E402
from sign_model import SignNet  # noqa: E402

LEAD_S, TAIL_S, GAP_S = 1.0, 3.5, 0.3


def letterbox(f):
    h, w = f.shape[:2]
    s = min(E.LIVE_W / w, E.LIVE_H / h)
    nw, nh = int(round(w * s)), int(round(h * s))
    canvas = np.zeros((E.LIVE_H, E.LIVE_W, 3), np.uint8)
    canvas[(E.LIVE_H - nh) // 2:(E.LIVE_H - nh) // 2 + nh, (E.LIVE_W - nw) // 2:(E.LIVE_W - nw) // 2 + nw] = cv2.resize(f, (nw, nh), interpolation=cv2.INTER_AREA)
    return cv2.imdecode(np.frombuffer(encode_jpeg(canvas, E.LIVE_Q), np.uint8), cv2.IMREAD_COLOR)


def clip_frames(path, fps, t_range=None):
    """Dataset clip -> live frames at `fps` (picked by time from the source video; 10 fps = exactly phase3_eval.to_live_frames).
    t_range = (t0, t1) seconds keeps only that part (sentences use the sign itself)."""
    if abs(fps - E.LIVE_FPS) < 1e-9 and t_range is None:
        return E.to_live_frames(path)
    cap = cv2.VideoCapture(str(path))
    src = cap.get(cv2.CAP_PROP_FPS) or 30.0
    allf = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        allf.append(f)
    cap.release()
    n_out = int(len(allf) / src * fps + 1e-9)
    idx = [min(len(allf) - 1, int(round(j * src / fps))) for j in range(n_out)]
    if t_range is not None:
        idx = [i for j, i in enumerate(idx) if t_range[0] <= j / fps <= t_range[1]]
    return [letterbox(allf[i]) for i in idx]


def stream_server(url, token, frames, fps):
    events = []
    with connect(url, additional_headers={"Authorization": f"Bearer {token}"}, max_size=None) as ws:
        for i, f in enumerate(frames):
            ws.send(pack_frame(i, encode_jpeg(f, E.LIVE_Q), ts=i / fps))
            r = json.loads(ws.recv(timeout=60))
            if r.get("committed_label"):
                events.append(r["committed_label"])
    return events


def plan(d, rows, n_other):
    """The same clips as replay_eval.build(model, d, 'test'): every word clip, 400 random other clips, one sentence per signer pair."""
    sp = np.where(d["split"] == "test")[0]
    items = [("word", [D.LABELS[d["cls"][c]]], c, None) for c in sp if d["cls"][c] > 0]
    other = [c for c in sp if d["cls"][c] == 0]
    for c in np.random.default_rng(0).choice(other, min(len(other), 400), replace=False)[:n_other]:
        items.append(("other", [], int(c), None))
    by = defaultdict(lambda: defaultdict(list))
    for c in sp:
        if d["cls"][c] > 0:
            by[d["signer"][c]][D.LABELS[d["cls"][c]]].append(c)
    for sg, w in by.items():
        for a, b, sent in (("how", "you", "how_are_you"), ("where", "you", "where_are_you"), ("you", "where", "where_are_you")):
            if w[a] and w[b]:
                items.append(("sentence", [sent], w[a][0], w[b][0]))
    return items


def build_stream(d, rows, item, fps):
    kind, exp, c, c2 = item
    video = lambda cc: E.DATA / "videos" / rows[cc]["file"]  # noqa: E731
    n = lambda s: max(1, int(round(s * fps)))  # noqa: E731
    if kind != "sentence":
        fr = clip_frames(video(c), fps)
        return [fr[0]] * n(LEAD_S) + fr + [fr[-1]] * n(TAIL_S)

    def sign(cc):  # the sign itself: active segment +- 2 frames of the 10 fps cache
        a0, a1 = d["seg"][cc]
        return clip_frames(video(cc), fps, ((a0 - 2) / 10.0, (a1 + 2) / 10.0))
    fa, fb = sign(c), sign(c2)
    return [fa[0]] * n(LEAD_S) + fa + [fa[-1]] * n(GAP_S) + fb + [fb[-1]] * n(TAIL_S)


def score(items, events_list):
    res = Counter()
    per = defaultdict(Counter)
    for (kind, exp, c, c2), ev in zip(items, events_list):
        if kind == "word":
            per[exp[0]]["n"] += 1
            per[exp[0]]["right"] += ev == exp
            res["word_right"] += ev == exp
            res["word_n"] += 1
            res["double"] += sum(e == exp[0] or (exp[0] in e.split("_") and "_" in e) for e in ev) >= 2
        elif kind == "other":
            res["other_n"] += 1
            res["false_trigger"] += bool(ev)
        else:
            res["sent_n"] += 1
            res["sent_right"] += ev == exp
    return res, per


def fmt(res):
    return (f"words {res['word_right']}/{res['word_n']} ({100 * res['word_right'] / max(res['word_n'], 1):.1f}%)  doubles {res['double']}  "
            f"false triggers {res['false_trigger']}/{res['other_n']} ({100 * res['false_trigger'] / max(res['other_n'], 1):.2f}%)  "
            f"sentences {res['sent_right']}/{res['sent_n']} ({100 * res['sent_right'] / max(res['sent_n'], 1):.1f}%)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://127.0.0.1:8766/ws")
    ap.add_argument("--model", default="rtmw_xl_384x288")
    ap.add_argument("--fps", type=float, nargs="+", default=[10, 20, 30])
    ap.add_argument("--other", type=int, default=400, help="how many of the 400 other-gloss clips to send")
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    token = os.environ["SIGNLANG_TOKEN"]
    d = D.load(a.model)
    rows = list(csv.DictReader(open(E.DATA / "manifest.csv", encoding="utf-8")))
    items = plan(d, rows, a.other)
    print(f"{len(items)} streams: {sum(i[0] == 'word' for i in items)} words, {sum(i[0] == 'other' for i in items)} other-gloss clips, "
          f"{sum(i[0] == 'sentence' for i in items)} sentences", flush=True)

    # offline reference at 10 fps: cached keypoints + decision.py with the server's settings
    srv = json.loads((ROOT / "server" / "config" / "server.json").read_text())
    prof = "primary" if "xl" in a.model else "fallback"
    sel = json.loads((T.RUNS / a.model / "selected.json").read_text())["decision"]
    cfg = {**srv["decision"], **srv["models"][prof].get("decision", {}), **{k: v for k, v in sel.items() if not k.startswith("repeat_")}}
    net = SignNet(n_cls=len(D.LABELS)).cuda().eval()
    net.load_state_dict(torch.load(T.RUNS / a.model / "best.pt"))
    RE.TAIL = int(round(TAIL_S * 10))
    off = RE.build(net, d, "test")
    ref = Counter()
    for kind, exp, probs, flg in off:
        ev = RE.run_decision(probs, flg, cfg)
        if kind == "word":
            ref["word_n"] += 1
            ref["word_right"] += ev == exp
            ref["double"] += sum(e == exp[0] or (exp[0] in e.split("_") and "_" in e) for e in ev) >= 2
        elif kind == "other":
            ref["other_n"] += 1
            ref["false_trigger"] += bool(ev)
        else:
            ref["sent_n"] += 1
            ref["sent_right"] += ev == exp
    print("offline replay, 10 fps (cached keypoints):  " + fmt(ref), flush=True)

    def one(fps):
        def run(item):
            return stream_server(a.url, token, build_stream(d, rows, item, fps), fps)
        with ThreadPoolExecutor(a.workers) as ex:
            evs = list(ex.map(run, items))
        res, per = score(items, evs)
        print(f"server, {fps:g} fps input:" + " " * (14 - len(f"{fps:g}")) + fmt(res), flush=True)
        print("      per word: " + "  ".join(f"{w} {per[w]['right']}/{per[w]['n']}" for w in D.LABELS[1:]), flush=True)
        return evs

    all_ev = {fps: one(fps) for fps in a.fps}
    if len(a.fps) > 1:
        base = all_ev[a.fps[0]]
        for fps in a.fps[1:]:
            same = sum(x == y for x, y in zip(base, all_ev[fps]))
            print(f"identical committed events, {a.fps[0]:g} fps vs {fps:g} fps: {same}/{len(items)} streams")


if __name__ == "__main__":
    main()
