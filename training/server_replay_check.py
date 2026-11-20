"""Integration check: test-split clips -> frames (live format) -> the RUNNING fallback server over WebSocket -> committed events,
compared with the offline replay (cached keypoints + same model + same Decision) of the same clips.

  python training/server_replay_check.py --url ws://127.0.0.1:8765/ws --model rtmw_xl_384x288 [--per-word 4]
Frames are sent one at a time with client timestamps 100 ms apart (no dropping), 1 s of the first frame before and 2 s of the last after.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from websockets.sync.client import connect

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "server" / "fallback"))
import phase3_eval as E  # noqa: E402
import replay_eval as RE  # noqa: E402
import sign_data as D  # noqa: E402
import train as T  # noqa: E402
from sign_model import SignNet  # noqa: E402

sys.path.insert(0, str(ROOT / "client"))
from common import encode_jpeg, pack_frame  # noqa: E402


def stream_server(url, token, frames):
    """Returns (events, n_results). One fresh connection per clip = fresh server state."""
    events, n = [], 0
    with connect(url, additional_headers={"Authorization": f"Bearer {token}"}, max_size=None) as ws:
        for i, f in enumerate(frames):
            msg = bytearray(pack_frame(i, encode_jpeg(f, E.LIVE_Q)))
            msg[12:20] = int(i * 100_000).to_bytes(8, "little")  # client_ts_us: exactly 10 fps
            ws.send(bytes(msg))
            r = json.loads(ws.recv(timeout=30))
            n += 1
            if r.get("committed_label"):
                events.append(r["committed_label"])
    return events, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://127.0.0.1:8765/ws")
    ap.add_argument("--model", default="rtmw_xl_384x288")
    ap.add_argument("--per-word", type=int, default=4)
    a = ap.parse_args()
    token = os.environ["SIGNLANG_TOKEN"]
    sel = json.loads((T.RUNS / a.model / "selected.json").read_text())
    srv_cfg = json.loads((ROOT / "server" / "config" / "server.json").read_text())
    # the settings the server really uses: those selected on val plus the same-word guard settings from server.json
    cfg = {**sel["decision"], **{k: v for k, v in srv_cfg["decision"].items() if k.startswith("repeat_")}}
    print("replay decision settings:", {k: cfg.get(k) for k in ("commit_threshold", "commit_consecutive_M", "ema_alpha", "repeat_rest_frames", "repeat_rest_p", "repeat_cooldown_ms")})
    d = D.load(a.model)
    net = SignNet(n_cls=len(D.LABELS)).cuda().eval()
    net.load_state_dict(torch.load(T.RUNS / a.model / "best.pt"))
    rows = list(__import__("csv").DictReader(open(E.DATA / "manifest.csv", encoding="utf-8")))
    rng = np.random.default_rng(5)
    picks = []
    for cls in range(1, len(D.LABELS)):
        pool = np.where((d["split"] == "test") & (d["cls"] == cls))[0]
        picks += list(rng.choice(pool, min(a.per_word, len(pool)), replace=False))
    picks += list(rng.choice(np.where((d["split"] == "test") & (d["cls"] == 0))[0], a.per_word * 2, replace=False))
    agree = right_srv = right_off = 0
    for c in picks:
        frames = E.to_live_frames(E.DATA / "videos" / rows[c]["file"])
        stream = [frames[0]] * RE.LEAD + frames + [frames[-1]] * RE.TAIL
        want = [D.LABELS[d["cls"][c]]] if d["cls"][c] else []
        srv, n = stream_server(a.url, token, stream)
        pts, flg = RE.clip_rows(d, c)
        off = RE.run_decision(RE.stream_probs(net, pts, flg), flg, cfg)
        agree += srv == off
        right_srv += srv == want
        right_off += off == want
        print(f"  {rows[c]['gloss']:>12}  want {want!s:12} server {srv!s:14} offline {off!s:14} {'same' if srv == off else 'DIFFERENT'}")
    n = len(picks)
    print(f"\n{n} clips: server == offline replay on {agree}/{n}; exactly right: server {right_srv}/{n}, offline {right_off}/{n}")


if __name__ == "__main__":
    main()
