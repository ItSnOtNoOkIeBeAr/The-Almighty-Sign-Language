"""Fallback server (safety net): same protocol as the C++ server, onnxruntime-gpu instead of custom CUDA.

Pipeline: JPEG decode -> RTMW keypoints (full-frame box, training/rtmw_ref.py) for EVERY frame (smooth skeleton on the client);
then a 10 fps clock driven by the client timestamps (ticks.py) picks, for each 100 ms tick, the frame closest to it: 57-point features
(training/signfeat.py) -> 30-tick window -> SignNet (ONNX) -> decision logic (server/fallback/decision.py) run once per tick, so the
window still covers 3 s and the decision settings keep their meaning at any client rate (10-30 fps).
Models come from server/config/server.json (profile primary or fallback).
Run: python server/fallback/fallback_server.py [--profile fallback]   (needs SIGNLANG_TOKEN in the environment)
"""
import argparse
import asyncio
import hmac
import json
import logging
import os
import struct
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ort_cuda  # noqa: E402  (must be imported before onnxruntime; registers CUDA DLL folders)
import rtmw_ref as R  # noqa: E402
import signfeat as S  # noqa: E402
import torch  # noqa: E402  (CPU only here: signfeat's tensor maths, shared with training)
from decision import Decision  # noqa: E402
from ticks import TickGrid  # noqa: E402

import uvicorn  # noqa: E402
from fastapi import FastAPI, WebSocket  # noqa: E402

VERSION = "0.1.0"
MAGIC, PROTO_VERSION = 0x534C4E47, 1
HEADER = struct.Struct("<IHHIQ")  # magic, version, flags, frame_id, client_ts_us = 20 bytes
CFG = json.loads((ROOT / "server" / "config" / "server.json").read_text())
LABELS = json.loads((ROOT / "protocol" / "labels.json").read_text())["labels"]
MAX_BYTES = CFG["net"]["max_message_bytes"]
assert (CFG["window"]["frames"], CFG["window"]["feature_dim"]) == (S.WINDOW, S.F_DIM), "server.json window does not match signfeat.py"
log = logging.getLogger("fallback")


class DecodeError(Exception):
    pass


class DiagLog:
    """Live diagnostics (--diag-log): one CSV line per 10 fps decision tick, same columns as the C++ server. Numbers only, no images."""

    def __init__(self, path):
        self.f = open(path, "w", encoding="utf-8", newline="\n")
        self.f.write("ts_ms,profile," + ",".join("p_" + n for n in LABELS) + ",top1,top2,top3,label,conf,state,committed,"
                     "hand_L_loose,hand_R_loose,hand_L_strict,hand_R_strict,pts_L,pts_R,thr,"
                     "mot_L_wrist,mot_L_tips,mot_R_wrist,mot_R_tips,mot,mot_1s,mot_3s\n")
        log.info("diagnostic log (numbers only, no images): %s", path)

    def write(self, ts_ms, profile, probs, counts, flags, thr, mot, mot_1s, mot_3s, label, conf, state, committed):
        top = sorted(range(len(LABELS)), key=lambda i: -probs[i])[:3]
        n_l, n_r = counts  # hand keypoints above the loose threshold (of 21 each)
        self.f.write(f"{int(ts_ms)},{profile}," + ",".join(f"{p:.4f}" for p in probs) + "," + ",".join(f"{LABELS[i]}:{probs[i]:.3f}" for i in top)
                     + f",{label},{conf},{state},{committed or ''},{int(flags[0])},{int(flags[1])},{int(flags[3])},{int(flags[4])},{n_l},{n_r},{thr},"
                     + ",".join(f"{m:.4f}" for m in mot) + f",{mot_1s:.4f},{mot_3s:.4f}\n")
        self.f.flush()


class Engine:
    """Shared ORT sessions. Per-connection state lives in ConnState so clients never mix windows."""

    def __init__(self, profile, diag=None):
        self.profile, self.diag = profile, None  # the diagnostic log starts after the warm-up frames
        m = CFG["models"][profile]
        self.kp = ort_cuda.make_session(str(ROOT / m["keypoint"]))      # raises if CPU-only
        self.temporal = ort_cuda.make_session(str(ROOT / m["temporal"]), tf32=False)  # exact FP32: the logits match PyTorch to 1e-5
        self.decision_cfg = {**CFG["decision"], **m.get("decision", {})}  # per-profile overrides (selected on val per model)
        self.wh, self.thr, self.present_thr = tuple(m["keypoint_input_wh"]), m["keypoint_thr"], m["hand_present_thr"]
        log.info("profile %s, onnxruntime providers: %s | %s", profile, self.kp.get_providers(), self.temporal.get_providers())
        blank = np.zeros((480, 640, 3), np.uint8)
        for _ in range(2):  # warm-up
            self.process(ConnState(self.decision_cfg), cv2.imencode(".jpg", blank)[1].tobytes(), 0)
        self.diag = diag

    def process(self, st: "ConnState", jpeg: bytes, ts_ms: float, want_kp: bool = False) -> dict:
        t0 = time.perf_counter()
        img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise DecodeError("not a decodable JPEG")
        t1 = time.perf_counter()
        h, w = img.shape[:2]
        blob, centre, scale = R.pose_crop(img, [0, 0, w, h], self.wh, True)  # full frame, no detector (Phase 3 decision)
        sx, sy = self.kp.run(None, {"input": blob})
        kpt, score = R.simcc_decode(sx[0], sy[0], self.wh, centre, scale)
        pts, flags = S.frame_row(kpt, score, self.thr, self.present_thr, (w, h))
        counts = (int((score[91:112] >= self.thr).sum()), int((score[112:133] >= self.thr).sum()))
        t2 = time.perf_counter()
        # 10 fps recognition clock: every 100 ms tick takes the frame closest to it; the model and the decision run once per tick
        for t_tick, (tp, tf, tc), run_model in st.push(ts_ms, (pts, flags, counts)):
            st.window.append((tp, tf))
            if not run_model:  # a gap in the stream: the missed ticks only fill the window
                continue
            wp, wf = st.window_arrays()
            feats = S.window_features(torch.from_numpy(wp)[None], torch.from_numpy(wf)[None]).numpy()
            logits = self.temporal.run(None, {"x": feats})[0][0]
            probs = np.exp(logits - logits.max())
            probs /= probs.sum()
            mot = S.hand_motion(wp[-2], wp[-1], wf[-2], wf[-1])  # movement since the previous tick (0 at the start of a stream)
            out = st.decision.step(probs.tolist(), bool(tf[3] or tf[4]), t_tick, mot[4])  # strict presence drives the idle gate
            st.record(out)
            if self.diag:
                self.diag.write(t_tick, self.profile, probs.tolist(), tc, tf, self.thr, mot, st.decision.mot_1s, st.decision.mot_3s, *out)
        t3 = time.perf_counter()
        label, conf, state, committed = st.report()
        res = {
            "type": "result", "label": label, "confidence": conf, "state": state, "committed_label": committed, "profile": self.profile,
            "server_ms": {"decode": round((t1 - t0) * 1e3, 2), "keypoints": round((t2 - t1) * 1e3, 2),
                          "temporal": round((t3 - t2) * 1e3, 2), "total": round((t3 - t0) * 1e3, 2)},
        }
        if want_kp:  # debug view, same format as the C++ server: body 0-12 and both hands 91-132 as [index, x, y, score]
            res["keypoints"] = {"w": w, "h": h, "thr": float(self.thr),
                                "pts": [[i, round(float(kpt[i][0]), 2), round(float(kpt[i][1]), 2), round(float(score[i]), 2)] for i in list(range(13)) + list(range(91, 133))]}
        return res


class ConnState:
    """Per connection: the 10 fps recognition clock, the window of tick frames and the decision state machine."""

    def __init__(self, decision_cfg):
        self.dropped = 0
        self.decision = Decision(LABELS, decision_cfg)
        self.reset()

    def reset(self):
        self.t_last = -1.0
        self.grid = TickGrid(CFG["window"]["step_ms"])
        self.window = deque(maxlen=S.WINDOW)  # (pts, flags) of the last 30 ticks, oldest first
        self.events = deque()                 # committed words not reported yet: one per reply
        self.last = ("none", 0.0, "idle")     # newest decision output (label, confidence, state)
        self.decision.reset()

    def push(self, ts_ms, item):
        """Every frame, with its client timestamp -> the 100 ms ticks that fell due: [(tick_ms, item, run_model)]."""
        self.t_last = max(ts_ms, self.t_last + 1.0)  # timestamps must increase even if a client sends duplicates
        return self.grid.push(self.t_last, item)

    def window_arrays(self):
        """WINDOW ticks ending with the newest; before the stream began the first tick is repeated."""
        rows = list(self.window)
        rows = [rows[0]] * (S.WINDOW - len(rows)) + rows
        return np.stack([r[0] for r in rows]), np.stack([r[1] for r in rows])

    def record(self, out):
        """Store a decision output (label, confidence, state, committed); committed words are queued until a reply can carry them."""
        label, conf, state, committed = out
        if committed:
            self.events.append(committed)
        self.last = (label, conf, "signing" if state == "committed" else state)

    def report(self):
        """What the reply to the current frame says: the newest decision output, and the oldest unreported committed word."""
        label, conf, state = self.last
        if self.events:
            return label, conf, "committed", self.events.popleft()
        return label, conf, state, None


def parse(data: bytes):
    """Returns (error_code, detail, frame_id, flags, payload, client_ts_ms). frame_id is None if the header is unusable."""
    fid, ts = None, 0.0
    if len(data) >= HEADER.size:
        magic, ver, flags, fid, ts_us = HEADER.unpack_from(data)
        ts = ts_us / 1000.0
        if magic != MAGIC:
            fid = None
    if len(data) > MAX_BYTES:
        return "too_large", f"{len(data)} bytes > {MAX_BYTES}", fid, 0, b"", ts
    if len(data) < HEADER.size:
        return "too_short", f"{len(data)} bytes < {HEADER.size}-byte header", None, 0, b"", ts
    if magic != MAGIC:
        return "bad_magic", f"magic 0x{magic:08X}", None, 0, b"", ts
    if ver != PROTO_VERSION:
        return "bad_version", f"version {ver}", fid, 0, b"", ts
    payload = data[HEADER.size:]
    if not payload and not flags & 1:
        return "empty_payload", "no JPEG bytes", fid, flags, b"", ts
    return None, "", fid, flags, payload, ts


class AuthWS:
    """Rejects /ws upgrades with HTTP 401 *before* accepting (FastAPI's own close would give 403)."""

    def __init__(self, app, token):
        self.app, self.expected = app, f"Bearer {token}"

    async def __call__(self, scope, receive, send):
        if scope["type"] == "websocket" and scope["path"] == "/ws":
            got = dict(scope["headers"]).get(b"authorization", b"").decode("latin1")
            if not hmac.compare_digest(got.encode(), self.expected.encode()):
                log.warning("rejected /ws upgrade from %s: bad or missing token (HTTP 401)", (scope.get("client") or ["?"])[0])
                await receive()  # websocket.connect
                await send({"type": "websocket.http.response.start", "status": 401,
                            "headers": [(b"content-type", b"application/json")]})
                await send({"type": "websocket.http.response.body", "body": b'{"detail":"unauthorized"}'})
                return
        await self.app(scope, receive, send)


app = FastAPI()
engine: Engine | None = None


@app.get("/health")
def health():
    return {"ok": True, "server": "fallback", "version": VERSION}


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    st, slot, wake, send_lock = ConnState(engine.decision_cfg), None, asyncio.Event(), asyncio.Lock()
    loop = asyncio.get_running_loop()

    async def send(obj):
        async with send_lock:
            await ws.send_text(json.dumps(obj))

    async def worker():
        nonlocal slot
        while True:
            await wake.wait()
            wake.clear()
            item, slot = slot, None
            if item is None:
                continue
            fid, payload, eos, ts_ms, want_kp = item
            try:
                res = await loop.run_in_executor(None, engine.process, st, payload, ts_ms, want_kp)
                await send({**res, "frame_id": fid, "dropped_frames": st.dropped})
            except DecodeError as e:
                await send({"type": "error", "code": "decode_failed", "detail": str(e), "frame_id": fid})
            except Exception as e:  # never let one bad frame kill the connection
                log.exception("processing failed")
                await send({"type": "error", "code": "internal", "detail": type(e).__name__, "frame_id": fid})
            if eos:
                st.reset()

    task = asyncio.create_task(worker())
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            data = msg.get("bytes")
            if data is None:
                await send({"type": "error", "code": "bad_type", "detail": "binary messages only"})
                continue
            code, detail, fid, flags, payload, ts_ms = parse(data)
            if code:
                await send({"type": "error", "code": code, "detail": detail,
                            **({"frame_id": fid} if fid is not None else {})})
            elif flags & 1 and not payload:
                slot = None
                st.reset()  # empty end_of_stream: no result, fresh decision state
            else:
                if slot is not None:
                    st.dropped += 1  # latest-only: the waiting frame is replaced, never queued
                slot = (fid, payload, bool(flags & 1), ts_ms, bool(flags & 2))
                wake.set()
    finally:
        task.cancel()


def main():
    global engine
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default=CFG["models"]["default_profile"], choices=[k for k in CFG["models"] if k != "default_profile"])
    ap.add_argument("--port", type=int, default=None, help="listen on this port instead of server.json (a second instance for testing next to the live one)")
    ap.add_argument("--diag-log", default=None, help="write one CSV line per frame (probabilities, decision, hand detection); numbers only")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = ap.parse_args()
    profile = args.profile
    token = os.environ.get("SIGNLANG_TOKEN")
    if not token:
        sys.exit("SIGNLANG_TOKEN is not set. Set it with setx, then open a NEW terminal.")
    # uvicorn logs "returned without completing handshake" after every correct 401 denial (it never marks the
    # handshake done for denial responses, then skips its 500 because a response was already sent).
    # Verified in uvicorn/protocols/websockets/websockets_sansio_impl.py; drop only that exact message.
    logging.getLogger("uvicorn.error").addFilter(lambda rec: "without completing handshake" not in rec.getMessage())
    engine = Engine(profile, DiagLog(args.diag_log) if args.diag_log else None)
    net = {**CFG["net"], **({"port": args.port} if args.port else {})}
    log.info("fallback server v%s listening on %s:%d", VERSION, net["host"], net["port"])
    # ws_max_size is far above MAX_BYTES so oversized frames get a too_large reply instead of a closed socket.
    uvicorn.run(AuthWS(app, token), host=net["host"], port=net["port"], ws_max_size=16 * 1024 * 1024,
                log_level="warning")


if __name__ == "__main__":
    main()
