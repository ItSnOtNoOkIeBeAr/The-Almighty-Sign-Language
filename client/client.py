"""Laptop client: webcam -> JPEG -> WebSocket -> label overlay. CPU only (opencv, websockets, numpy).

  python client/client.py [--url wss://<host>/ws]       Home screen: pick a word to see how it is signed, or Free cam (click a word for its picture, h = hide it, m = back to Home, q quits;
                                                        in Free cam k = toggle skeleton (on by default), d = details); token from SIGNLANG_TOKEN
  python client/client.py --free-cam                    skip the Home screen
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

import menu
import recorder
from common import FakeCamera, add_common_args, encode_jpeg, load_config, open_camera, pack_frame, prepare_frame
from overlay import BANNER_SECONDS, MAX_BANNER_WORDS, Smoother, draw_keypoints, draw_overlay

WINDOW = "The Almighty Sign Language"


class Link:
    """Owns the connection: sender thread (reconnects with backoff) + one receiver thread per connection."""

    def __init__(self, cfg, token):
        self.cfg, self.token = cfg, token
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.cond = threading.Condition(self.lock)  # signalled by the main thread for every new camera frame
        self.frame = None            # newest raw camera frame, written by the main thread
        self.frame_seq, self.frame_t, self.sent_seq = 0, 0.0, -1  # camera frame counter, its capture time, last frame sent
        self.state, self.error = "connecting", ""
        self.result = None           # newest server result
        self.history = deque(maxlen=5)
        self.sent, self.fid = {}, 0
        self.rtts, self.server_totals, self.labels = [], [], Counter()
        self.sent_times = deque(maxlen=30)
        self.n_results = 0
        # Flow control: never queue frames the link can't carry. If more than max_inflight frames are
        # unanswered we skip sending (0 disables). Without this a slow uplink builds a queue and the
        # round-trip time grows without bound (seen on laptop Wi-Fi: 0.4 s -> 2.8 s in 20 s).
        self.max_inflight = int(cfg.get("max_inflight", 3))
        self.last_ack, self.last_progress, self.skipped = -1, time.time(), 0
        self.want_kp = False         # ask the server for debug keypoints (key k); costs ~1.5 kB per reply
        self.kp = None               # newest keypoints message: {"w", "h", "thr", "pts": [[index, x, y, score], ...]}
        self.phrase = ((), 0.0)      # (committed words/sentences shown in the big banner, time of the newest one)

    def start(self):
        threading.Thread(target=self._run, daemon=True).start()

    def push_frame(self, frame):
        """Hand the newest camera frame to the sender thread."""
        with self.cond:
            self.frame, self.frame_t = frame, time.time()
            self.frame_seq += 1
            self.cond.notify_all()

    def _run(self):
        backoff = 1.0
        while not self.stop.is_set():
            try:
                self.state = "connecting"
                with connect(self.cfg["url"], additional_headers={"Authorization": f"Bearer {self.token}"},
                             max_size=None, open_timeout=10) as ws:
                    self.state, self.error, backoff = "connected", "", 1.0
                    self.last_ack, self.last_progress = self.fid - 1, time.time()
                    rx =threading.Thread(target=self._recv, args=(ws,), daemon=True)
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
        # Event-driven: wake for every new camera frame (a timer sleep on Windows rounds up to ~15 ms and lost 15% of a 30 fps stream) and send
        # the frame when it is due for the target rate. A 30 fps camera sending 20 fps gives gaps of 33 and 67 ms: fine, the server resamples
        # by timestamp. A camera slower than the target is simply sent as fast as it delivers: a frame is never sent twice.
        period, due, cam_dt, last_t = 1.0 / self.cfg["fps"], None, 1.0 / self.cfg["fps"], None
        while not self.stop.is_set() and rx.is_alive():
            with self.cond:
                self.cond.wait_for(lambda: self.frame_seq != self.sent_seq or self.stop.is_set(), timeout=0.2)
                raw, seq, t_cap = self.frame, self.frame_seq, self.frame_t
            if raw is None or seq == self.sent_seq:
                continue
            self.sent_seq = seq  # this frame is handled (sent below, or skipped because it is too early)
            if last_t is not None:
                cam_dt = 0.9 * cam_dt + 0.1 * min(max(t_cap - last_t, 1e-3), 0.5)  # running average of the camera's frame interval
            last_t = t_cap
            if due is None:
                due = t_cap
            if t_cap < due - 0.5 * cam_dt:  # too early for the target rate: wait for the next camera frame
                continue
            due = max(due + period, t_cap - period)  # if we fall behind, skip ahead instead of bursting
            unanswered = self.fid - 1 - self.last_ack
            # The 2 s escape hatch keeps a lost reply from freezing the stream forever.
            if self.max_inflight and unanswered >= self.max_inflight and time.time() - self.last_progress < 2.0:
                self.skipped += 1
                continue
            jpeg = encode_jpeg(prepare_frame(raw, self.cfg), self.cfg["jpeg_quality"])
            self.sent[self.fid] = time.time()
            ws.send(pack_frame(self.fid, jpeg, flags=2 if self.want_kp else 0, ts=t_cap))  # flags bit1 = send debug keypoints; ts = capture time
            self.fid += 1
            self.sent_times.append(time.time())
            for old in [k for k in self.sent if k < self.fid - 300]:
                del self.sent[old]

    def _recv(self, ws):
        try:
            for raw in ws:
                m = json.loads(raw)
                if "frame_id" in m:  # results AND per-frame errors both count as an answer
                    self.last_ack, self.last_progress = max(self.last_ack, m["frame_id"]), time.time()
                if m.get("type") == "result":
                    t0 = self.sent.pop(m["frame_id"], None)
                    rtt = (time.time() - t0) * 1000 if t0 else None
                    self.result, self.n_results = {**m, "rtt_ms": rtt}, self.n_results + 1
                    if rtt is not None:
                        self.rtts.append(rtt)
                    self.server_totals.append(m["server_ms"]["total"])
                    self.labels[m["label"]] += 1
                    self.kp = m.get("keypoints") if self.want_kp else None
                    if m.get("committed_label"):
                        self.history.append(m["committed_label"])
                        words, t = self.phrase
                        now = time.time()
                        # words pile up while the banner is still showing; once it has expired, start again from zero
                        words = (words if now - t <= BANNER_SECONDS else ())[-(MAX_BANNER_WORDS - 1):]
                        self.phrase = (words + (m["committed_label"],), now)
                elif m.get("type") == "error":
                    self.error = f"server error: {m['code']}"
        except Exception:  # noqa: BLE001 - closed socket ends the thread; sender notices rx is dead
            pass

    def send_fps(self):
        t = list(self.sent_times)
        return (len(t) - 1) / (t[-1] - t[0]) if len(t) > 2 and t[-1] > t[0] else 0.0


def open_cap(cfg):
    return FakeCamera(cfg, cfg["fake_camera_fps"]) if cfg.get("fake_camera_fps") else open_camera(cfg)


class CamSession:
    """One visit to the Free cam screen: camera + connection + the live view (mirrored preview, skeleton, banner)."""

    def __init__(self, cfg, token):
        self.cfg, self.link, self.cap = cfg, Link(cfg, token), open_cap(cfg)
        self.link.want_kp = True  # the skeleton shows from the first frame (k toggles)
        self.link.start()
        self.details, self.smoother, self.disp_fps, self.t_prev = False, Smoother(), 0.0, time.time()

    def frame(self):
        """Read one camera frame, send it, return the picture to show."""
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError("camera read failed")
        link = self.link
        link.push_frame(frame)
        now = time.time()
        self.disp_fps = 0.9 * self.disp_fps + 0.1 / max(now - self.t_prev, 1e-6)
        self.t_prev = now
        img = prepare_frame(frame, self.cfg).copy()
        if img.shape[1] > 960:  # keep the preview a sane size on small screens
            img = cv2.resize(img, (960, int(img.shape[0] * 960 / img.shape[1])))
        # Mirror the preview only (the frames sent to the server stay unflipped: hand side matters in ASL).
        # Keypoints are in camera coordinates, so draw them before the flip; text after, so it reads normally.
        if link.want_kp:
            draw_keypoints(img, link.kp, self.smoother)
        img = cv2.flip(img, 1)
        draw_overlay(img, link, self.disp_fps, self.details)
        return img

    def key(self, key):
        if key == ord("k"):
            self.link.want_kp = not self.link.want_kp  # the server only sends keypoints while this is on
            self.link.kp = None
        elif key == ord("d"):
            self.details = not self.details

    def close(self):
        self.link.stop.set()
        self.cap.release()


def run_ui(cfg, token, free_cam):
    """The window: Home (word cards + Free cam) -> how-to guide -> Free cam. q quits, m returns to Home, h hides the pinned picture."""
    mouse = menu.Mouse()
    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, mouse.on_event)
    screen, word, session, pinned = ("cam", None, None, None) if free_cam else ("home", None, None, None)
    try:
        while True:
            if screen == "cam":
                session = session or CamSession(cfg, token)
                img, hit = menu.cam_view(session.frame(), mouse, pinned, menu.recognised(session.link, time.time()))
                action = None
                if hit:  # clicking a word pins its how-to picture; clicking it again hides it
                    pinned = None if hit == pinned else hit
            elif screen == "guide":
                img, action = menu.render_guide(mouse, word)
            else:
                img, action = menu.render_home(mouse)
            cv2.imshow(WINDOW, img)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or (key == 27 and screen != "guide"):
                break
            if screen == "cam":
                session.key(key)
                if key == ord("h"):
                    pinned = None
                elif key == ord("m"):
                    action = ("home", None)
            elif key == 27:
                action = ("home", None)
            if action:
                if screen == "cam":
                    session.close()
                    session = None
                screen, word = action
                pinned = word if screen == "cam" else None
    finally:
        if session:
            session.close()
        cv2.destroyAllWindows()
    return 0


def run_live(cfg, token, duration, quiet=False):
    """Headless timed run: no window; returns a stats dict (also printed unless quiet)."""
    link, cap = Link(cfg, token), open_cap(cfg)
    link.want_kp = bool(cfg.get("keypoints"))  # headless stays lean unless asked
    link.start()
    t_start, last_n, cam_n = time.time(), 0, 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("camera read failed")
            link.push_frame(frame)
            cam_n += 1
            if link.n_results != last_n and link.result and not quiet:
                r, last_n = link.result, link.n_results
                print(f"#{r['frame_id']:>4} {r['label']:<14} conf {r['confidence']:.2f} state {r['state']:<9} "
                      f"rtt {(r['rtt_ms'] or 0):6.1f} ms  server {r['server_ms']['total']:5.1f} ms  dropped {r['dropped_frames']}",
                      flush=True)
            if time.time() - t_start >= duration:
                break
            time.sleep(0.005)
    finally:
        link.stop.set()
        cap.release()
    rt, elapsed = sorted(link.rtts), max(time.time() - t_start, 1e-6)
    stats = {"fps_req": float(cfg["fps"]), "cam_fps": cam_n / elapsed, "sent": link.fid, "send_fps": link.fid / elapsed, "results": link.n_results,
             "skipped": link.skipped, "dropped": (link.result or {}).get("dropped_frames", 0), "state": f"{link.state} {link.error}".strip(),
             "rtt": (statistics.median(rt), rt[int(0.95 * (len(rt) - 1))], rt[-1]) if rt else (0.0, 0.0, 0.0),
             "server_p50": statistics.median(link.server_totals) if link.server_totals else 0.0}
    if not quiet:
        print("--- summary ---")
        print(f"state: {stats['state']}")
        print(f"camera delivered {stats['cam_fps']:.1f} fps   requested send rate {stats['fps_req']:.0f} fps   actually sent {stats['send_fps']:.1f} fps")
        print(f"frames sent: {link.fid}   results: {link.n_results}   skipped by flow control: {link.skipped}   server-dropped: {stats['dropped']}   "
              f"labels: {dict(link.labels)}")
        if rt:
            print(f"round-trip ms: p50 {stats['rtt'][0]:.1f}  p95 {stats['rtt'][1]:.1f}  max {stats['rtt'][2]:.1f}   | server total ms p50 {stats['server_p50']:.1f}")
    return stats


def pick_fps(rows, rtt_limit=100.0):
    """Highest tested rate that the camera, the link and the server all carry: median round trip under rtt_limit, at most 5% of the frames
    skipped by flow control or dropped by the server, and the camera/send rate within 10% of the request. None if no rate qualifies."""
    ok = [r for r in rows if r["results"] and r["rtt"][0] <= rtt_limit and r["skipped"] <= 0.05 * (r["sent"] + r["skipped"])
          and r["dropped"] <= 0.05 * max(r["sent"], 1) and r["send_fps"] >= 0.9 * min(r["fps_req"], r["cam_fps"])]
    return max(ok, key=lambda r: r["fps_req"])["fps_req"] if ok else None


def run_sweep(cfg, token, rates, duration):
    rows = []
    for fps in rates:
        print(f"--- {fps:g} fps for {duration:g} s ...", flush=True)
        rows.append(run_live({**cfg, "fps": fps}, token, duration, quiet=True))
        time.sleep(2.0)  # let the previous connection close
    print(f"\n{'send fps':>9} {'camera':>7} {'sent':>6} {'answered':>8} {'skipped':>8} {'dropped':>8} {'rtt p50':>8} {'p95':>6} {'max':>6} {'server':>7}")
    for r in rows:
        print(f"{r['fps_req']:9g} {r['cam_fps']:7.1f} {r['sent']:6d} {r['results']:8d} {r['skipped']:8d} {r['dropped']:8d} "
              f"{r['rtt'][0]:8.0f} {r['rtt'][1]:6.0f} {r['rtt'][2]:6.0f} {r['server_p50']:6.1f}ms")
    best = pick_fps(rows)
    print("\nrecommended send rate: " + (f"{best:g} fps (highest with round trip p50 <= 100 ms, <= 5% skipped or dropped, camera keeps up)" if best else
          "none qualifies (round trip above 100 ms or frames skipped/dropped at every rate): use the lowest rate, or check Wi-Fi / camera"))
    return 0 if any(r["results"] for r in rows) else 1


def probe_camera(cfg, seconds):
    """What does the camera really deliver at the configured size? Tries the default, an explicit 30 fps request and MJPG + 30 fps."""
    best = None
    for name, extra in (("default", {}), ("30 fps requested", {"camera_fps": 30}), ("MJPG + 30 fps", {"camera_fourcc": "MJPG", "camera_fps": 30})):
        c = {**cfg, **extra}
        try:
            cap = open_camera(c)
        except RuntimeError as e:
            print(f"{name:<18} {e}")
            continue
        for _ in range(10):  # let auto exposure settle
            cap.read()
        n, t0, last = 0, time.time(), None
        while time.time() - t0 < seconds:
            ok, f = cap.read()
            if ok:
                n, last = n + 1, f
        fps = n / (time.time() - t0)
        four = int(cap.get(cv2.CAP_PROP_FOURCC))
        shape = "x".join(map(str, last.shape[1::-1])) if last is not None else "-"
        print(f"{name:<18} delivered {fps:5.1f} fps at {shape}   (camera reports {cap.get(cv2.CAP_PROP_FPS):.0f} fps, format "
              f"{''.join(chr((four >> 8 * i) & 255) for i in range(4)).strip() or '?'})")
        cap.release()
        if best is None or fps > best[0] + 1.0:
            best = (fps, name, extra)
    if best:
        flags = " ".join({"camera_fps": f"--camera-fps {v}", "camera_fourcc": f"--fourcc {v}"}[k] for k, v in best[2].items())
        print(f"\nbest: {best[1]} ({best[0]:.1f} fps)" + (f"; use: client.py {flags}" if flags else "") +
              "\nA camera in dim light lowers its frame rate by itself: probe in the light you will demo in.")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--duration", type=float, default=10.0, help="headless run time in seconds (per rate with --sweep, per camera setting with --probe-camera)")
    ap.add_argument("--sweep", metavar="FPS,FPS,..", help="headless: run each send rate in turn (e.g. 10,15,20,30), print round trip / drops per rate and the recommended one")
    ap.add_argument("--probe-camera", action="store_true", help="measure the frame rate the camera really delivers at the configured size")
    ap.add_argument("--free-cam", action="store_true", help="skip the Home screen and open the camera view directly")
    ap.add_argument("--record", metavar="LABEL")
    ap.add_argument("--signer")
    ap.add_argument("--review", action="store_true")
    ap.add_argument("--label", help="with --review: only this label")
    a = ap.parse_args()
    cfg = load_config(a)
    if a.probe_camera:
        sys.exit(probe_camera(cfg, a.duration if a.duration != 10.0 else 5.0))
    if a.review:
        return recorder.review(label=a.label)
    if a.record:
        if not a.signer:
            sys.exit("--record needs --signer <id>")
        return recorder.record_clip(a.record, a.signer, cfg, a.headless)
    token = os.environ.get("SIGNLANG_TOKEN")
    if not token:
        sys.exit("SIGNLANG_TOKEN is not set (setx SIGNLANG_TOKEN <value>, then open a NEW terminal).")
    if a.sweep:
        sys.exit(run_sweep(cfg, token, [float(x) for x in a.sweep.split(",")], a.duration if a.duration != 10.0 else 20.0))
    if not a.headless:
        sys.exit(run_ui(cfg, token, a.free_cam))
    sys.exit(0 if run_live(cfg, token, a.duration)["results"] else 1)


if __name__ == "__main__":
    main()
