"""Fallback server (safety net): same protocol as the C++ server, onnxruntime-gpu instead of custom CUDA.

Pipeline per frame: JPEG decode -> RTMW keypoints (full-frame box, training/rtmw_ref.py) -> 57-point features
(training/signfeat.py) -> 30-frame window resampled to 10 fps by the client timestamps -> SignNet (ONNX) ->
decision logic (server/fallback/decision.py). Models come from server/config/server.json (profile primary or fallback).
Run: python server/fallback/fallback_server.py [--profile fallback]   (needs SIGNLANG_TOKEN in the environment)
"""
import argparse
import asyncio
import bisect
import hmac
import json
import logging
import os
import struct
import sys
import time
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


class Engine:
    """Shared ORT sessions. Per-connection state lives in ConnState so clients never mix windows."""

    def __init__(self, profile):
        m = CFG["models"][profile]
        self.kp = ort_cuda.make_session(str(ROOT / m["keypoint"]))      # raises if CPU-only
        self.temporal = ort_cuda.make_session(str(ROOT / m["temporal"]), tf32=False)  # exact FP32: the logits match PyTorch to 1e-5
        self.decision_cfg = {**CFG["decision"], **m.get("decision", {})}  # per-profile overrides (selected on val per model)
        self.wh, self.thr, self.present_thr = tuple(m["keypoint_input_wh"]), m["keypoint_thr"], m["hand_present_thr"]
        log.info("profile %s, onnxruntime providers: %s | %s", profile, self.kp.get_providers(), self.temporal.get_providers())
        blank = np.zeros((480, 640, 3), np.uint8)
        for _ in range(2):  # warm-up
            self.process(ConnState(self.decision_cfg), cv2.imencode(".jpg", blank)[1].tobytes(), 0)

    def process(self, st: "ConnState", jpeg: bytes, ts_ms: float) -> dict:
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
        st.push(ts_ms, pts, flags)
        t2 = time.perf_counter()
        wp, wf = st.window()
        feats = S.window_features(torch.from_numpy(wp)[None], torch.from_numpy(wf)[None]).numpy()
        logits = self.temporal.run(None, {"x": feats})[0][0]
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        t3 = time.perf_counter()
        label, conf, state, committed = st.decision.step(probs.tolist(), bool(flags[3] or flags[4]), st.t_last)  # strict presence drives the idle gate
        return {
            "type": "result", "label": label, "confidence": conf, "state": state, "committed_label": committed,
            "server_ms": {"decode": round((t1 - t0) * 1e3, 2), "keypoints": round((t2 - t1) * 1e3, 2),
                          "temporal": round((t3 - t2) * 1e3, 2), "total": round((t3 - t0) * 1e3, 2)},
        }


class ConnState:
    """Per connection: recent keypoint rows (kept by client timestamp) and the decision state machine."""

    def __init__(self, decision_cfg):
        self.dropped = 0
        self.decision = Decision(LABELS, decision_cfg)
        self.reset()

    def reset(self):
        self.times, self.pts, self.flags, self.t_last = [], [], [], -1.0
        self.decision.reset()

    def push(self, ts_ms, pts, flags):
        self.t_last = max(ts_ms, self.t_last + 1.0)  # timestamps must increase even if a client sends duplicates
        self.times.append(self.t_last), self.pts.append(pts), self.flags.append(flags)
        del self.times[:-120], self.pts[:-120], self.flags[:-120]

    def window(self):
        """WINDOW frames at step_ms spacing ending now; each slot holds the newest frame at or before its time (the first frame before the stream began)."""
        step = CFG["window"]["step_ms"]
        idx = [max(0, bisect.bisect_right(self.times, self.t_last - (S.WINDOW - 1 - k) * step) - 1) for k in range(S.WINDOW)]
        return np.stack([self.pts[i] for i in idx]), np.stack([self.flags[i] for i in idx])


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
            fid, payload, eos, ts_ms = item
            try:
                res = await loop.run_in_executor(None, engine.process, st, payload, ts_ms)
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
                slot = (fid, payload, bool(flags & 1), ts_ms)
                wake.set()
    finally:
        task.cancel()


def main():
    global engine
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default=CFG["models"]["default_profile"], choices=[k for k in CFG["models"] if k != "default_profile"])
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    profile = ap.parse_args().profile
    token = os.environ.get("SIGNLANG_TOKEN")
    if not token:
        sys.exit("SIGNLANG_TOKEN is not set. Set it with setx, then open a NEW terminal.")
    # uvicorn logs "returned without completing handshake" after every correct 401 denial (it never marks the
    # handshake done for denial responses, then skips its 500 because a response was already sent).
    # Verified in uvicorn/protocols/websockets/websockets_sansio_impl.py; drop only that exact message.
    logging.getLogger("uvicorn.error").addFilter(lambda rec: "without completing handshake" not in rec.getMessage())
    engine = Engine(profile)
    net = CFG["net"]
    log.info("fallback server v%s listening on %s:%d", VERSION, net["host"], net["port"])
    # ws_max_size is far above MAX_BYTES so oversized frames get a too_large reply instead of a closed socket.
    uvicorn.run(AuthWS(app, token), host=net["host"], port=net["port"], ws_max_size=16 * 1024 * 1024,
                log_level="warning")


if __name__ == "__main__":
    main()
