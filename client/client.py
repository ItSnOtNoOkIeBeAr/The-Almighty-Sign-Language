"""Laptop client: webcam -> JPEG -> WebSocket -> label overlay. CPU only (opencv, websockets, numpy).

  python client/client.py [--url wss://<host>/ws]       live view (q or Esc quits); token from SIGNLANG_TOKEN
  python client/client.py --headless --duration 10      no window, prints results + latency summary
  python client/client.py --record hello --signer s1    record a 3 s clip (see recorder.py)
  python client/client.py --review [--label hello]      replay / delete recorded clips
Capture and display run in the main thread; encoding+sending and receiving each have their own thread,
so a slow network never freezes the preview.
"""
import argparse
import json
import os
import statistics
import sys
import threading
import time
from collections import Counter, deque

import cv2
from websockets.exceptions import InvalidStatus
from websockets.sync.client import connect

import recorder
from common import (add_common_args, encode_jpeg, load_config, open_camera, pack_frame, prepare_frame, put_text)


class Link:
    """Owns the connection: sender thread (reconnects with backoff) + one receiver thread per connection."""

    def __init__(self, cfg, token):
        self.cfg, self.token = cfg, token
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.frame = None            # newest raw camera frame, written by the main thread
        self.state, self.error = "connecting", ""
        self.result = None           # newest server result
        self.history = deque(maxlen=5)
        self.sent, self.fid = {}, 0
        self.rtts, self.server_totals, self.labels = [], [], Counter()
        self.sent_times = deque(maxlen=30)
        self.n_results = 0

    def start(self):
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        backoff = 1.0
        while not self.stop.is_set():
            try:
                self.state = "connecting"
                with connect(self.cfg["url"], additional_headers={"Authorization": f"Bearer {self.token}"},
                             max_size=None, open_timeout=10) as ws:
                    self.state, self.error, backoff = "connected", "", 1.0
                    rx = threading.Thread(target=self._recv, args=(ws,), daemon=True)
                    rx.start()
                    self._send_loop(ws, rx)
            except InvalidStatus as e:
                self.error = f"HTTP {e.response.status_code}" + (" (check SIGNLANG_TOKEN)" if e.response.status_code == 401 else "")
            except Exception as e:  # noqa: BLE001 - any network failure just means reconnect
                self.error = type(e).__name__
            if self.stop.is_set():
                break
            self.state = f"reconnecting in {backoff:.0f}s"
            self.stop.wait(backoff)
            backoff = min(backoff * 2, 10.0)

    def _send_loop(self, ws, rx):
        period, due = 1.0 / self.cfg["fps"], time.time()
        while not self.stop.is_set() and rx.is_alive():
            if self.stop.wait(max(0.0, due - time.time())):
                return
            due = max(due + period, time.time() - period)  # if we fall behind, skip ahead instead of bursting
            with self.lock:
                raw = self.frame
            if raw is None:
                continue
            jpeg = encode_jpeg(prepare_frame(raw, self.cfg), self.cfg["jpeg_quality"])
            self.sent[self.fid] = time.time()
            ws.send(pack_frame(self.fid, jpeg))
            self.fid += 1
            self.sent_times.append(time.time())
            for old in [k for k in self.sent if k < self.fid - 300]:
                del self.sent[old]

    def _recv(self, ws):
        try:
            for raw in ws:
                m = json.loads(raw)
                if m.get("type") == "result":
                    t0 = self.sent.pop(m["frame_id"], None)
                    rtt = (time.time() - t0) * 1000 if t0 else None
                    self.result, self.n_results = {**m, "rtt_ms": rtt}, self.n_results + 1
                    if rtt is not None:
                        self.rtts.append(rtt)
                    self.server_totals.append(m["server_ms"]["total"])
                    self.labels[m["label"]] += 1
                    if m.get("committed_label"):
                        self.history.append(m["committed_label"])
                elif m.get("type") == "error":
                    self.error = f"server error: {m['code']}"
        except Exception:  # noqa: BLE001 - closed socket ends the thread; sender notices rx is dead
            pass

    def send_fps(self):
        t = list(self.sent_times)
        return (len(t) - 1) / (t[-1] - t[0]) if len(t) > 2 and t[-1] > t[0] else 0.0


def draw_overlay(img, link, disp_fps):
    r = link.result or {}
    ok = link.state == "connected"
    put_text(img, f"{link.state}" + (f"  [{link.error}]" if link.error else ""), (10, 24), 0.6,
             (0, 255, 0) if ok else (0, 165, 255))
    put_text(img, f"{r.get('label', '-')}", (10, 70), 1.4, (255, 255, 255), 3)
    conf = float(r.get("confidence", 0.0))
    cv2.rectangle(img, (10, 85), (260, 105), (60, 60, 60), -1)
    cv2.rectangle(img, (10, 85), (10 + int(250 * conf), 105), (0, 200, 0), -1)
    put_text(img, f"{conf * 100:.0f}%  state: {r.get('state', '-')}", (270, 102), 0.55)
    put_text(img, "committed: " + (" > ".join(link.history) or "-"), (10, 135), 0.6, (0, 255, 255))
    sm = r.get("server_ms", {})
    rtt = r.get("rtt_ms")
    put_text(img, f"RTT {rtt:.0f} ms" if rtt is not None else "RTT -", (10, 165), 0.6)
    put_text(img, f"server {sm.get('total', 0):.1f} ms (dec {sm.get('decode', 0):.1f} kp {sm.get('keypoints', 0):.1f} "
                  f"temp {sm.get('temporal', 0):.1f})", (10, 190), 0.55)
    put_text(img, f"send {link.send_fps():.1f} fps | display {disp_fps:.0f} fps | dropped {r.get('dropped_frames', 0)}",
             (10, 215), 0.55)


def run_live(cfg, token, headless, duration):
    link, cap = Link(cfg, token), open_camera(cfg)
    link.start()
    t_start, last_n, shown, disp_fps, t_prev = time.time(), 0, 0, 0.0, time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("camera read failed")
            with link.lock:
                link.frame = frame
            if headless:
                if link.n_results != last_n and link.result:
                    r, last_n = link.result, link.n_results
                    print(f"#{r['frame_id']:>4} {r['label']:<14} conf {r['confidence']:.2f} state {r['state']:<9} "
                          f"rtt {(r['rtt_ms'] or 0):6.1f} ms  server {r['server_ms']['total']:5.1f} ms  dropped {r['dropped_frames']}",
                          flush=True)
                if time.time() - t_start >= duration:
                    break
                time.sleep(0.005)
            else:
                now = time.time()
                disp_fps = 0.9 * disp_fps + 0.1 / max(now - t_prev, 1e-6)
                t_prev = now
                img = prepare_frame(frame, cfg).copy()
                if img.shape[1] > 960:  # keep the preview a sane size on small screens
                    img = cv2.resize(img, (960, int(img.shape[0] * 960 / img.shape[1])))
                draw_overlay(img, link, disp_fps)
                cv2.imshow("The Almighty Sign Language", img)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
    finally:
        link.stop.set()
        cap.release()
        cv2.destroyAllWindows()
    if headless:
        rt = sorted(link.rtts)
        print("--- summary ---")
        print(f"state: {link.state} {link.error}".strip())
        print(f"frames sent: {link.fid}   results: {link.n_results}   labels: {dict(link.labels)}")
        if rt:
            print(f"round-trip ms: p50 {statistics.median(rt):.1f}  p95 {rt[int(0.95 * (len(rt) - 1))]:.1f}  max {rt[-1]:.1f}"
                  f"   | server total ms p50 {statistics.median(link.server_totals):.1f}")
        return 0 if link.n_results else 1
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--duration", type=float, default=10.0, help="headless run time in seconds")
    ap.add_argument("--record", metavar="LABEL")
    ap.add_argument("--signer")
    ap.add_argument("--review", action="store_true")
    ap.add_argument("--label", help="with --review: only this label")
    a = ap.parse_args()
    cfg = load_config(a)
    if a.review:
        return recorder.review(label=a.label)
    if a.record:
        if not a.signer:
            sys.exit("--record needs --signer <id>")
        return recorder.record_clip(a.record, a.signer, cfg, a.headless)
    token = os.environ.get("SIGNLANG_TOKEN")
    if not token:
        sys.exit("SIGNLANG_TOKEN is not set (setx SIGNLANG_TOKEN <value>, then open a NEW terminal).")
    sys.exit(run_live(cfg, token, a.headless, a.duration))


if __name__ == "__main__":
    main()
