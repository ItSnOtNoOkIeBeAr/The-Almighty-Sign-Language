"""Fallback server (safety net): same protocol as the C++ server, onnxruntime-gpu instead of custom CUDA.

PLACEHOLDER (Phase 2): the keypoint step is a dummy (8x12 grayscale thumbnail = 96 features) and the
temporal model has random weights (training/make_placeholder_model.py). The decision step only maps the
smoothed label to idle/signing and never commits (real logic arrives in Phase 8).
Run: python server/fallback/fallback_server.py   (needs SIGNLANG_TOKEN in the environment)
"""
import asyncio
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
import ort_cuda  # noqa: E402  (must be imported before onnxruntime; registers CUDA DLL folders)

import uvicorn  # noqa: E402
from fastapi import FastAPI, WebSocket  # noqa: E402

VERSION = "0.1.0"
MAGIC, PROTO_VERSION = 0x534C4E47, 1
HEADER = struct.Struct("<IHHIQ")  # magic, version, flags, frame_id, client_ts_us = 20 bytes
CFG = json.loads((ROOT / "server" / "config" / "server.json").read_text())
LABELS = json.loads((ROOT / "protocol" / "labels.json").read_text())["labels"]
MAX_BYTES = CFG["net"]["max_message_bytes"]
FRAMES, FEAT = CFG["window"]["frames"], CFG["window"]["feature_dim"]
log = logging.getLogger("fallback")


class DecodeError(Exception):
    pass


class Engine:
    """Shared ORT session. Per-connection state lives in ConnState so clients never mix windows."""

    def __init__(self):
        self.sess = ort_cuda.make_session(str(ROOT / CFG["models"]["temporal"]))  # raises if CPU-only
        log.info("onnxruntime active providers: %s", self.sess.get_providers())
        self.sess.run(None, {"x": np.zeros((1, FRAMES, FEAT), np.float32)})  # warm-up

    def process(self, st: "ConnState", jpeg: bytes) -> dict:
        t0 = time.perf_counter()
        img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise DecodeError("not a decodable JPEG")
        t1 = time.perf_counter()
        # PLACEHOLDER: dummy keypoint step, a 12x8 grayscale thumbnail instead of real keypoints.
        feats = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (12, 8), interpolation=cv2.INTER_AREA)
        st.ring[:-1] = st.ring[1:]
        st.ring[-1] = feats.astype(np.float32).ravel() / 255.0 - 0.5
        st.count += 1
        t2 = time.perf_counter()
        if st.probs is None or st.count % CFG["window"]["run_every_k_frames"] == 0:
            logits = self.sess.run(None, {"x": st.ring[None]})[0][0]
            p = np.exp(logits - logits.max())
            p /= p.sum()
            a = CFG["decision"]["ema_alpha"]
            st.probs = p if st.probs is None else a * p + (1 - a) * st.probs
        t3 = time.perf_counter()
        k = int(st.probs.argmax())
        return {
            "type": "result", "label": LABELS[k], "confidence": round(float(st.probs[k]), 4),
            # PLACEHOLDER decision: never commits until Phase 8.
            "state": "idle" if k == 0 else "signing", "committed_label": None,
            "server_ms": {"decode": round((t1 - t0) * 1e3, 2), "keypoints": round((t2 - t1) * 1e3, 2),
                          "temporal": round((t3 - t2) * 1e3, 2), "total": round((t3 - t0) * 1e3, 2)},
        }


class ConnState:
    def __init__(self):
        self.reset()
        self.dropped = 0

    def reset(self):
        self.ring = np.zeros((FRAMES, FEAT), np.float32)
        self.count = 0
        self.probs = None


def parse(data: bytes):
    """Returns (error_code, detail, frame_id, flags, payload). frame_id is None if the header is unusable."""
    fid = None
    if len(data) >= HEADER.size:
        magic, ver, flags, fid, _ts = HEADER.unpack_from(data)
        if magic != MAGIC:
            fid = None
    if len(data) > MAX_BYTES:
        return "too_large", f"{len(data)} bytes > {MAX_BYTES}", fid, 0, b""
    if len(data) < HEADER.size:
        return "too_short", f"{len(data)} bytes < {HEADER.size}-byte header", None, 0, b""
    if magic != MAGIC:
        return "bad_magic", f"magic 0x{magic:08X}", None, 0, b""
    if ver != PROTO_VERSION:
        return "bad_version", f"version {ver}", fid, 0, b""
    payload = data[HEADER.size:]
    if not payload and not flags & 1:
        return "empty_payload", "no JPEG bytes", fid, flags, b""
    return None, "", fid, flags, payload


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
    st, slot, wake, send_lock = ConnState(), None, asyncio.Event(), asyncio.Lock()
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
            fid, payload, eos = item
            try:
                res = await loop.run_in_executor(None, engine.process, st, payload)
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
            code, detail, fid, flags, payload = parse(data)
            if code:
                await send({"type": "error", "code": code, "detail": detail,
                            **({"frame_id": fid} if fid is not None else {})})
            elif flags & 1 and not payload:
                slot = None
                st.reset()  # empty end_of_stream: no result, fresh decision state
            else:
                if slot is not None:
                    st.dropped += 1  # latest-only: the waiting frame is replaced, never queued
                slot = (fid, payload, bool(flags & 1))
                wake.set()
    finally:
        task.cancel()


def main():
    global engine
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    token = os.environ.get("SIGNLANG_TOKEN")
    if not token:
        sys.exit("SIGNLANG_TOKEN is not set. Set it with setx, then open a NEW terminal.")
    # uvicorn logs "returned without completing handshake" after every correct 401 denial (it never marks the
    # handshake done for denial responses, then skips its 500 because a response was already sent).
    # Verified in uvicorn/protocols/websockets/websockets_sansio_impl.py; drop only that exact message.
    logging.getLogger("uvicorn.error").addFilter(lambda rec: "without completing handshake" not in rec.getMessage())
    engine = Engine()
    net = CFG["net"]
    log.info("fallback server v%s listening on %s:%d", VERSION, net["host"], net["port"])
    # ws_max_size is far above MAX_BYTES so oversized frames get a too_large reply instead of a closed socket.
    uvicorn.run(AuthWS(app, token), host=net["host"], port=net["port"], ws_max_size=16 * 1024 * 1024,
                log_level="warning")


if __name__ == "__main__":
    main()
