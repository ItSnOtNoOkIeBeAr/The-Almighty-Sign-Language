"""Soak test: streams 640x480 JPEG frames at 10 fps for N minutes (default 10) and reports, per minute, round-trip latency,
server time, errors, drops, GPU memory in use and the server process's memory. Verdict at the end: no errors, GPU memory flat,
no latency creep. Works through the tunnel or locally.
  client\.venv\Scripts\python server\scripts\soak.py [--url wss://signlang.dheynorlanda.me/ws] [--minutes 10] [--keypoints]
Needs SIGNLANG_TOKEN in the environment (never printed). GPU / process memory are read on THIS PC (run it here, not on the laptop).
"""
import argparse
import csv
import io
import json
import os
import struct
import subprocess
import sys
import time

import cv2
import numpy as np
from websockets.sync.client import connect

MAGIC, HEADER = 0x534C4E47, struct.Struct("<IHHIQ")


def gpu_used_mib():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout
        return int(out.strip().splitlines()[0])
    except Exception:  # noqa: BLE001
        return -1


def server_mem_mb(name="signlang_server.exe"):
    try:
        out = subprocess.run(["tasklist", "/FI", f"IMAGENAME eq {name}", "/FO", "CSV", "/NH"], capture_output=True, text=True, timeout=10).stdout
        row = next(csv.reader(io.StringIO(out)))
        return int(row[4].replace(",", "").replace(" K", "").replace(" ", "").strip()) / 1024.0  # "Mem Usage" in K
    except Exception:  # noqa: BLE001
        return -1.0


def make_frames(n=24, w=640, h=480):
    rng = np.random.default_rng(0)
    frames = []
    for i in range(n):  # different content every frame so decode/keypoint work varies like a real stream
        img = cv2.GaussianBlur(rng.integers(0, 255, (h, w, 3), dtype=np.uint8), (0, 0), 5 + i % 4)
        cv2.circle(img, (80 + 20 * i, 200), 60, (30, 120, 220), -1)
        frames.append(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])[1].tobytes())
    return frames


def pct(v, q):
    return float(np.percentile(v, q)) if len(v) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="wss://signlang.dheynorlanda.me/ws")
    ap.add_argument("--minutes", type=float, default=10)
    ap.add_argument("--keypoints", action="store_true", help="also request the debug keypoints (bigger replies)")
    a = ap.parse_args()
    token = os.environ.get("SIGNLANG_TOKEN")
    if not token:
        sys.exit("SIGNLANG_TOKEN is not set")
    frames = make_frames()
    print(f"soak: {a.url}, {a.minutes:g} min at 10 fps" + (", with debug keypoints" if a.keypoints else ""), flush=True)
    rows, errors, drops_last, fid = [], 0, 0, 0
    cur = {"rtt": [], "srv": [], "err": 0}
    t_start = time.time()
    end = t_start + a.minutes * 60
    minute = 0
    base = (gpu_used_mib(), server_mem_mb())
    reconnects = 0
    ws = None
    next_t = time.time()
    while time.time() < end:
        try:
            if ws is None:
                ws = connect(a.url, additional_headers={"Authorization": f"Bearer {token}"}, max_size=None, open_timeout=15)
            fid += 1
            ts_us = int((time.time() - t_start) * 1e6)
            msg = HEADER.pack(MAGIC, 1, 2 if a.keypoints else 0, fid, ts_us) + frames[fid % len(frames)]
            t0 = time.perf_counter()
            ws.send(msg)
            r = json.loads(ws.recv(timeout=20))
            rtt = (time.perf_counter() - t0) * 1000
            if r.get("type") == "result":
                cur["rtt"].append(rtt), cur["srv"].append(r["server_ms"]["total"])
                drops_last = r.get("dropped_frames", 0)
            else:
                cur["err"] += 1
                errors += 1
        except Exception as e:  # noqa: BLE001
            errors += 1
            cur["err"] += 1
            reconnects += 1
            print(f"  ! {type(e).__name__}: {e} (reconnecting)", flush=True)
            try:
                ws and ws.close()
            except Exception:  # noqa: BLE001
                pass
            ws = None
            time.sleep(1)
        next_t += 0.1
        time.sleep(max(0.0, next_t - time.time()))
        if int(time.time() - t_start) // 60 > minute or time.time() >= end:
            minute += 1
            row = (minute, len(cur["rtt"]), pct(cur["rtt"], 50), pct(cur["rtt"], 95), max(cur["rtt"], default=float("nan")), pct(cur["srv"], 50), pct(cur["srv"], 95),
                   cur["err"], gpu_used_mib(), server_mem_mb())
            rows.append(row)
            print(f"  min {row[0]:2d}: n={row[1]:4d}  rtt p50 {row[2]:6.1f} p95 {row[3]:6.1f} max {row[4]:6.1f} ms | server p50 {row[5]:5.2f} p95 {row[6]:5.2f} ms | errors {row[7]} | GPU {row[8]} MiB | server RSS {row[9]:.0f} MB", flush=True)
            cur = {"rtt": [], "srv": [], "err": 0}
    if ws:
        ws.close()
    first, last = rows[0], rows[-1]
    total = sum(r[1] for r in rows)
    gpu_span = max(r[8] for r in rows) - min(r[8] for r in rows)
    mem_span = max(r[9] for r in rows) - min(r[9] for r in rows)
    print(f"\nframes answered {total} of {fid} sent, errors/reconnects {errors}, server-reported drops {drops_last}")
    print(f"GPU memory in use: before {base[0]} MiB, per-minute min..max {min(r[8] for r in rows)}..{max(r[8] for r in rows)} MiB (spread {gpu_span} MiB)")
    print(f"server process memory: before {base[1]:.0f} MB, per-minute min..max {min(r[9] for r in rows):.0f}..{max(r[9] for r in rows):.0f} MB (spread {mem_span:.0f} MB)")
    print(f"latency creep: rtt p50 minute 1 {first[2]:.1f} ms -> last minute {last[2]:.1f} ms; p95 {first[3]:.1f} -> {last[3]:.1f} ms; server p50 {first[5]:.2f} -> {last[5]:.2f} ms")
    ok = errors == 0 and gpu_span <= 64 and mem_span <= 100 and last[2] <= first[2] * 1.25 + 5 and last[5] <= first[5] * 1.25 + 0.5
    print("SOAK", "PASS" if ok else "CHECK (see numbers)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
