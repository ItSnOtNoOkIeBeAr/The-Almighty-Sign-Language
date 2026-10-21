"""Protocol conformance test. Runs against ANY server URL; both the C++ and fallback servers must pass.

  python protocol/conformance_test.py --url ws://127.0.0.1:8765/ws [--token ...]
Token defaults to the SIGNLANG_TOKEN environment variable (never print it, never commit it).
Needs only: websockets, numpy, opencv-python (the client venv has all three).
"""
import argparse
import json
import os
import struct
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np
from websockets.exceptions import InvalidStatus
from websockets.sync.client import connect

MAGIC, HEADER = 0x534C4E47, struct.Struct("<IHHIQ")
MAX_BYTES = 2 * 1024 * 1024
LABELS = json.loads((Path(__file__).parent / "labels.json").read_text())["labels"]
STATES = {"idle", "signing", "committed"}
results = []


def jpeg(seed=0, w=640, h=360):
    rng = np.random.default_rng(seed)
    img = cv2.GaussianBlur(rng.integers(0, 255, (h, w, 3), dtype=np.uint8), (0, 0), 6)
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()


def msg(fid, payload, flags=0, magic=MAGIC, ver=1):
    return HEADER.pack(magic, ver, flags, fid, int(time.time() * 1e6)) + payload


def recv_json(ws, timeout=10):
    return json.loads(ws.recv(timeout=timeout))


def check(name):
    def deco(fn):
        def run(ctx):
            try:
                fn(ctx)
                results.append((name, True, ""))
            except Exception as e:  # noqa: BLE001 - report every failure, then continue
                results.append((name, False, f"{type(e).__name__}: {e}"))
        run.__name__ = fn.__name__
        return run
    return deco


def expect_result(r, fid):
    assert r.get("type") == "result", f"expected result, got {r}"
    assert r["frame_id"] == fid, f"frame_id {r['frame_id']} != {fid}"
    assert r["label"] in LABELS, f"unknown label {r['label']}"
    assert 0.0 <= r["confidence"] <= 1.0
    assert r["state"] in STATES
    assert r["committed_label"] is None or r["committed_label"] in LABELS
    assert {"decode", "keypoints", "temporal", "total"} <= set(r["server_ms"]), r["server_ms"]
    assert isinstance(r["dropped_frames"], int)


def expect_error(r, code):
    assert r.get("type") == "error", f"expected error, got {r}"
    assert r["code"] == code, f"expected {code}, got {r['code']} ({r.get('detail')})"


def open_ws(ctx, token=None):
    hdr = {"Authorization": f"Bearer {token}"} if token else {}
    return connect(ctx["url"], additional_headers=hdr, max_size=None, open_timeout=10)


@check("health: GET /health works without auth")
def t_health(ctx):
    # Explicit User-Agent: Cloudflare blocks urllib's default "Python-urllib" signature (error 1010) at the edge.
    req = urllib.request.Request(ctx["health"], headers={"User-Agent": "signlang-conformance/1.0"})
    with urllib.request.urlopen(req, timeout=10) as r:
        body = json.loads(r.read())
    assert body["ok"] is True and body["server"] in ("cpp", "fallback") and body["version"], body


@check("auth: missing token rejected with HTTP 401 before upgrade")
def t_auth_missing(ctx):
    try:
        open_ws(ctx).close()
    except InvalidStatus as e:
        assert e.response.status_code == 401, f"status {e.response.status_code}"
        return
    raise AssertionError("connection was accepted without a token")


@check("auth: wrong token rejected with HTTP 401")
def t_auth_wrong(ctx):
    try:
        open_ws(ctx, "definitely-not-the-token").close()
    except InvalidStatus as e:
        assert e.response.status_code == 401, f"status {e.response.status_code}"
        return
    raise AssertionError("connection was accepted with a wrong token")


@check("round trip: valid frame gets a well-formed result")
def t_roundtrip(ctx):
    with open_ws(ctx, ctx["token"]) as ws:
        for fid in (7, 8):
            ws.send(msg(fid, jpeg(fid)))
            expect_result(recv_json(ws), fid)


@check("malformed: every bad message gets the right error and the connection survives")
def t_malformed(ctx):
    good = jpeg(1)
    cases = [
        ("bad_type", "text message"),
        ("too_short", b"SLNG"),
        ("bad_magic", msg(1, good, magic=0xDEADBEEF)),
        ("bad_version", msg(1, good, ver=99)),
        ("empty_payload", msg(1, b"")),
        ("decode_failed", msg(1, b"this is not a jpeg" * 20)),
    ]
    with open_ws(ctx, ctx["token"]) as ws:
        for code, payload in cases:
            ws.send(payload)
            try:
                expect_error(recv_json(ws), code)
            except AssertionError as e:
                raise AssertionError(f"[{code}] {e}") from None
        ws.send(msg(2, good))  # still alive?
        expect_result(recv_json(ws), 2)


@check("oversized: > 2 MB gets too_large and the connection survives")
def t_oversized(ctx):
    with open_ws(ctx, ctx["token"]) as ws:
        for size in (MAX_BYTES + 1, 5 * 1024 * 1024):
            ws.send(msg(3, os.urandom(size - HEADER.size)))
            expect_error(recv_json(ws, 20), "too_large")
        ws.send(msg(4, jpeg(4)))
        expect_result(recv_json(ws), 4)


@check("end_of_stream: empty EOS gets no result and does not break the stream")
def t_eos(ctx):
    with open_ws(ctx, ctx["token"]) as ws:
        ws.send(msg(5, b"", flags=1))
        ws.send(msg(6, jpeg(6)))
        r = recv_json(ws)
        expect_result(r, 6)  # first reply belongs to frame 6, so the EOS produced nothing


@check("rapid fire: latest-only, last frame always answered, processed + dropped == sent")
def t_rapid(ctx):
    n, frame = 60, jpeg(9, 1280, 720)
    with open_ws(ctx, ctx["token"]) as ws:
        for fid in range(100, 100 + n):
            ws.send(msg(fid, frame))
        got, deadline = [], time.time() + 30
        while time.time() < deadline:
            r = recv_json(ws, 30)
            expect_result(r, r["frame_id"])
            got.append(r)
            if r["frame_id"] == 100 + n - 1:
                break
        ids = [r["frame_id"] for r in got]
        assert ids == sorted(set(ids)), f"frame ids not strictly increasing: {ids}"
        assert ids[-1] == 100 + n - 1, "most recent frame was never answered"
        assert len(got) <= n
        assert len(got) + got[-1]["dropped_frames"] == n, \
            f"processed {len(got)} + dropped {got[-1]['dropped_frames']} != sent {n}"
        ctx["rapid"] = f"sent {n}, processed {len(got)}, dropped {got[-1]['dropped_frames']}"


@check("reconnect: a fresh connection works after the previous one closed")
def t_reconnect(ctx):
    for i in range(3):
        with open_ws(ctx, ctx["token"]) as ws:
            ws.send(msg(i, jpeg(i)))
            expect_result(recv_json(ws), i)
    with open_ws(ctx, ctx["token"]) as ws:
        ws.send(msg(50, jpeg(50)))  # abrupt TCP drop mid-stream (no close handshake), server must survive
        ws.socket.close()
    time.sleep(0.5)
    with open_ws(ctx, ctx["token"]) as ws2:
        ws2.send(msg(51, jpeg(51)))
        expect_result(recv_json(ws2), 51)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True, help="ws:// or wss:// URL ending in /ws")
    ap.add_argument("--token", default=os.environ.get("SIGNLANG_TOKEN"))
    a = ap.parse_args()
    if not a.token:
        sys.exit("No token: pass --token or set SIGNLANG_TOKEN")
    u = urlparse(a.url)
    ctx = {"url": a.url, "token": a.token,
           "health": f"{'https' if u.scheme == 'wss' else 'http'}://{u.netloc}/health"}
    print(f"conformance: {a.url}")
    for t in (t_health, t_auth_missing, t_auth_wrong, t_roundtrip, t_malformed, t_oversized, t_eos, t_rapid,
              t_reconnect):
        t(ctx)
    for name, ok, why in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"\n         {why}" if why else ""))
    if "rapid" in ctx:
        print(f"  rapid fire: {ctx['rapid']}")
    failed = [r for r in results if not r[1]]
    print(f"{len(results) - len(failed)}/{len(results)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
