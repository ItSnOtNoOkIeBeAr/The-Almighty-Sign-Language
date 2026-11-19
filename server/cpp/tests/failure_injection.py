"""Failure injection against the C++ server (real processes, real GPU). Prints one PASS/FAIL line per case.

  client\.venv\Scripts\python server\cpp\tests\failure_injection.py [--build dev]
Uses ports 8767-8770 (never 8765), so a live server is not disturbed. Needs SIGNLANG_TOKEN and the built exe.
Cases: missing engine, missing degrade engine, low VRAM at startup, odd/garbage JPEG frames, client dying mid-frame,
server restart under a running client (reconnect), tier-1 degrade (injected slowness), --mode safe.
"""
import argparse
import json
import os
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from websockets.sync.client import connect

ROOT = Path(__file__).resolve().parents[3]
MAGIC, HEADER = 0x534C4E47, struct.Struct("<IHHIQ")
TOKEN = os.environ.get("SIGNLANG_TOKEN", "")
PATH = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin\x64;C:\dev\TensorRT-10.16.1.11\bin;" + os.environ["PATH"]
results = []


def exe(build):
    return ROOT / "server" / "cpp" / "build" / build / "signlang_server.exe"


def spawn(build, args, root=None):
    a = [str(exe(build))] + (["--root", str(root)] if root else []) + args
    return subprocess.Popen(a, env={**os.environ, "PATH": PATH}, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def wait_health(port, proc, timeout=60):
    end = time.time() + timeout
    while time.time() < end:
        if proc.poll() is not None:
            return False
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2).read()
            return True
        except Exception:  # noqa: BLE001
            time.sleep(0.5)
    return False


def stop(proc):
    if proc.poll() is None:
        proc.kill()
    try:
        return proc.communicate(timeout=10)[0]
    except Exception:  # noqa: BLE001
        return ""


def case(name):
    def deco(fn):
        def run(*a):
            try:
                detail = fn(*a)
                results.append((name, True, detail or ""))
            except Exception as e:  # noqa: BLE001
                results.append((name, False, f"{type(e).__name__}: {e}"))
        return run
    return deco


def frame_bytes(fid, jpeg, flags=0, ts_ms=None):
    return HEADER.pack(MAGIC, 1, flags, fid, int((fid * 100 if ts_ms is None else ts_ms) * 1000)) + jpeg


def good_jpeg(w=640, h=480, seed=1):
    rng = np.random.default_rng(seed)
    img = cv2.GaussianBlur(rng.integers(0, 255, (h, w, 3), dtype=np.uint8), (0, 0), 6)
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()


def open_ws(port):
    return connect(f"ws://127.0.0.1:{port}/ws", additional_headers={"Authorization": f"Bearer {TOKEN}"}, max_size=None, open_timeout=10)


def temp_root(mutate):
    """A throw-away repo root: labels.json + a modified server.json (absolute engine paths so the real engines still load)."""
    d = Path(tempfile.mkdtemp(prefix="fi_root_"))
    (d / "protocol").mkdir()
    shutil.copy(ROOT / "protocol" / "labels.json", d / "protocol" / "labels.json")
    (d / "server" / "config").mkdir(parents=True)
    cfg = json.loads((ROOT / "server" / "config" / "server.json").read_text())
    for prof in ("primary", "fallback"):
        for k in ("keypoint_engine", "temporal_engine"):
            cfg["models"][prof][k] = str(ROOT / cfg["models"][prof][k])
    mutate(cfg)
    (d / "server" / "config" / "server.json").write_text(json.dumps(cfg))
    return d


@case("missing keypoint engine: refuses to start, names the rebuild command")
def t_missing_engine(build):
    d = temp_root(lambda c: c["models"]["primary"].update(keypoint_engine=str(d_missing)))
    p = spawn(build, ["--port", "8767"], d)
    out = p.communicate(timeout=60)[0]
    assert p.returncode not in (0, None), f"exit code {p.returncode}"
    assert "engine file not found" in out and "build_engines.ps1" in out, out[-300:]
    return f"exit {p.returncode}, message names build_engines.ps1"


@case("missing DEGRADE engine: server still starts and answers, warns that tier-1 degrade is off")
def t_missing_alt_engine(build):
    d = temp_root(lambda c: c["models"]["fallback"].update(keypoint_engine=str(d_missing)))
    p = spawn(build, ["--port", "8767"], d)
    try:
        assert wait_health(8767, p), "did not start"
        with open_ws(8767) as ws:
            ws.send(frame_bytes(1, good_jpeg()))
            r = json.loads(ws.recv(timeout=10))
            assert r["type"] == "result" and r["profile"] == "primary", r
    finally:
        out = stop(p)
    assert "degrade target 'fallback' is not available" in out, out[-300:]
    return "started on primary, warning logged"


@case("low VRAM at startup: refuses to start with a clear message")
def t_low_vram(build):
    d = temp_root(lambda c: c["vram"].update(min_free_gb=1000))
    p = spawn(build, ["--port", "8767"], d)
    out = p.communicate(timeout=60)[0]
    assert p.returncode not in (0, None) and "not enough free VRAM" in out, out[-300:]
    return f"exit {p.returncode}: " + [l for l in out.splitlines() if "not enough free VRAM" in l][0].split("fatal: ")[-1][:90]


@case("odd frames: every kind gets a result or an error reply, the connection and the next frame are fine")
def t_odd_frames(build):
    p = spawn(build, ["--port", "8768"])
    try:
        assert wait_health(8768, p), "did not start"
        rng = np.random.default_rng(3)
        real = good_jpeg()
        gray1 = cv2.imencode(".jpg", rng.integers(0, 255, (480, 640), dtype=np.uint8))[1].tobytes()  # 1-channel JPEG
        prog = cv2.imencode(".jpg", cv2.GaussianBlur(rng.integers(0, 255, (480, 640, 3), dtype=np.uint8), (0, 0), 4), [cv2.IMWRITE_JPEG_PROGRESSIVE, 1])[1].tobytes()
        cases = {
            "random bytes": rng.bytes(5000), "SOI + noise": b"\xff\xd8\xff\xe0" + rng.bytes(3000), "truncated real JPEG": real[: len(real) // 2],
            "real JPEG with 100 corrupted bytes": bytes(b ^ 0xFF if 600 < i < 700 else b for i, b in enumerate(real)),
            "PNG sent as JPEG": cv2.imencode(".png", np.zeros((48, 64, 3), np.uint8))[1].tobytes(), "1x1 JPEG": cv2.imencode(".jpg", np.zeros((1, 1, 3), np.uint8))[1].tobytes(),
            "8x8 JPEG": cv2.imencode(".jpg", np.zeros((8, 8, 3), np.uint8))[1].tobytes(), "grayscale JPEG": gray1, "progressive JPEG": prog,
            "3840x2160 JPEG (limit)": good_jpeg(3840, 2160), "6000x4000 JPEG (over the limit)": good_jpeg(6000, 4000),
            "1x2000 JPEG": cv2.imencode(".jpg", np.zeros((2000, 1, 3), np.uint8))[1].tobytes(), "1-byte payload": b"\xff", "zero-filled 100 KB": bytes(100000),
        }
        out = []
        with open_ws(8768) as ws:
            fid = 0
            for name, payload in cases.items():
                fid += 1
                ws.send(frame_bytes(fid, payload))
                r = json.loads(ws.recv(timeout=30))
                assert r["type"] in ("result", "error"), r
                out.append(f"{name}: {r['type'] if r['type'] == 'result' else r['code']}")
                fid += 1
                ws.send(frame_bytes(fid, real))  # the stream must still work
                ok = json.loads(ws.recv(timeout=30))
                assert ok["type"] == "result" and ok["frame_id"] == fid, f"after '{name}': {ok}"
        assert p.poll() is None, "server died"
        return "; ".join(out)
    finally:
        stop(p)


@case("client dies mid-frame (20 x): server survives, the next client works, no connection stays stuck")
def t_midframe(build):
    p = spawn(build, ["--port", "8768"])
    try:
        assert wait_health(8768, p), "did not start"
        big = frame_bytes(1, good_jpeg(1280, 720))
        for i in range(20):
            s = socket.create_connection(("127.0.0.1", 8768))
            s.sendall((f"GET /ws HTTP/1.1\r\nHost: x\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n"
                       f"Sec-WebSocket-Version: 13\r\nAuthorization: Bearer {TOKEN}\r\n\r\n").encode())
            s.recv(1024)
            n = len(big)
            hdr = bytes([0x82, 0x80 | 126]) + struct.pack(">H", n) if n < 65536 else bytes([0x82, 0x80 | 127]) + struct.pack(">Q", n)
            s.sendall(hdr + b"\x01\x02\x03\x04" + bytes(b ^ (1, 2, 3, 4)[j & 3] for j, b in enumerate(big[: n // (2 + i % 3)])))  # part of the frame
            if i % 2:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))  # RST instead of FIN
            s.close()
        time.sleep(1.0)
        with open_ws(8768) as ws:
            ws.send(frame_bytes(7, good_jpeg()))
            assert json.loads(ws.recv(timeout=10))["type"] == "result"
        assert p.poll() is None, "server died"
        return "20 aborted connections (half FIN, half RST), then a normal client got a result"
    finally:
        stop(p)


@case("server restarts under a running client: the client reconnects by itself and results resume")
def t_reconnect(build):
    p = spawn(build, ["--port", "8769"])
    assert wait_health(8769, p), "did not start"
    client = subprocess.Popen([str(ROOT / "client/.venv/Scripts/python.exe"), str(ROOT / "client/client.py"), "--headless", "--duration", "40", "--width", "640", "--height", "480",
                               "--url", "ws://127.0.0.1:8769/ws"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=os.environ)
    time.sleep(10)
    stop(p)  # the server vanishes (like a crash or a tunnel drop)
    time.sleep(6)
    p2 = spawn(build, ["--port", "8769"])
    try:
        assert wait_health(8769, p2), "did not restart"
        out = client.communicate(timeout=90)[0]
    finally:
        stop(p2)
    lines = [l for l in out.splitlines() if l.startswith("#")]
    summ = [l for l in out.splitlines() if l.startswith("frames sent")]
    assert any("reconnecting" in l or "connect" in l for l in out.splitlines()), "no reconnect messages in the client log"
    m = summ[0] if summ else ""
    results_n = int(m.split("results:")[1].split()[0]) if "results:" in m else 0
    assert results_n > 100, f"only {results_n} results overall: {m}"
    final_state = [l for l in out.splitlines() if l.startswith("state:")]
    assert final_state and "connected" in final_state[0] and "reconnecting" not in final_state[0], final_state
    return f"{m.strip()} | {final_state[0]}"


@case("tier-1 degrade: injected slowness -> primary -> fallback keypoint model after 30 slow frames, stream keeps answering")
def t_degrade(build):
    p = spawn(build, ["--port", "8770", "--inject-delay-ms", "120"])
    try:
        assert wait_health(8770, p), "did not start"
        profiles = []
        with open_ws(8770) as ws:
            for fid in range(1, 46):
                ws.send(frame_bytes(fid, good_jpeg(seed=fid % 5)))
                r = json.loads(ws.recv(timeout=20))
                assert r["type"] == "result", r
                profiles.append(r["profile"])
        first_fb = profiles.index("fallback") if "fallback" in profiles else -1
        assert profiles[0] == "primary" and first_fb == 30, f"switch at frame index {first_fb}: {profiles}"
        assert set(profiles[first_fb:]) == {"fallback"}, "went back"
    finally:
        out = stop(p)
    assert "TIER-1 DEGRADE" in out, out[-300:]
    return f"frames 1-30 on primary, frame 31 on fallback ({profiles.count('primary')} primary + {profiles.count('fallback')} fallback), warning logged"


@case("--mode safe: starts on the fallback keypoint model, conformance 9/9")
def t_safe(build):
    p = spawn(build, ["--port", "8770", "--mode", "safe"])
    try:
        assert wait_health(8770, p), "did not start"
        with open_ws(8770) as ws:
            ws.send(frame_bytes(1, good_jpeg()))
            r = json.loads(ws.recv(timeout=10))
            assert r["profile"] == "fallback", r
        c = subprocess.run([str(ROOT / "client/.venv/Scripts/python.exe"), str(ROOT / "protocol/conformance_test.py"), "--url", "ws://127.0.0.1:8770/ws"], capture_output=True, text=True, env=os.environ)
        assert "9/9 passed" in c.stdout, c.stdout[-400:]
        return "profile fallback, conformance 9/9"
    finally:
        stop(p)


d_missing = Path(tempfile.gettempdir()) / "definitely_missing.engine"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", default="release")
    a = ap.parse_args()
    if not TOKEN:
        sys.exit("SIGNLANG_TOKEN is not set")
    print(f"failure injection with {exe(a.build)}")
    for t in (t_missing_engine, t_missing_alt_engine, t_low_vram, t_odd_frames, t_midframe, t_degrade, t_safe, t_reconnect):
        t(a.build)
    for name, ok, detail in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}\n         {detail}")
    bad = [r for r in results if not r[1]]
    print(f"{len(results) - len(bad)}/{len(results)} passed")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
