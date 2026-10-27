"""Shared by client.py (live streaming) and recorder.py (clips).

encode_jpeg() and prepare_frame() are the ONE place frames are resized and compressed, so recorded
training clips are byte-for-byte what the server sees at runtime.
"""
import argparse
import json
import struct
import threading
import time
from pathlib import Path

import cv2

CLIENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CLIENT_DIR.parent
MAGIC, PROTO_VERSION = 0x534C4E47, 1
HEADER = struct.Struct("<IHHIQ")  # see protocol/PROTOCOL.md


def add_common_args(ap: argparse.ArgumentParser):
    ap.add_argument("--config", default=str(CLIENT_DIR / "config.json"))
    ap.add_argument("--url")
    ap.add_argument("--width", type=int)
    ap.add_argument("--height", type=int)
    ap.add_argument("--fps", type=float)
    ap.add_argument("--quality", type=int, help="JPEG quality 1-100")
    ap.add_argument("--camera", type=int, help="camera index")
    ap.add_argument("--max-inflight", type=int, help="flow control: max unanswered frames before skipping (0 = off)")
    ap.add_argument("--headless", action="store_true", help="no window; log to the console")


def load_config(args) -> dict:
    cfg = json.loads(Path(args.config).read_text())
    for key, val in (("url", args.url), ("width", args.width), ("height", args.height), ("fps", args.fps),
                     ("jpeg_quality", args.quality), ("camera_index", args.camera),
                     ("max_inflight", args.max_inflight)):
        if val is not None:
            cfg[key] = val
    return cfg


def open_camera(cfg) -> cv2.VideoCapture:
    # CAP_DSHOW opens faster than the default MSMF backend on Windows and honours resolution requests.
    cap = cv2.VideoCapture(cfg["camera_index"], cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open camera {cfg['camera_index']} (is another app using it?)")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg["height"])
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


class Grabber:
    """Reads the camera continuously in a thread so callers always get the NEWEST frame without blocking.

    A webcam in dim light can deliver far fewer frames than the target FPS; callers that sample at the
    target rate (live streaming, recording) then see repeated frames, exactly as the server would.
    """

    def __init__(self, cap):
        self.cap, self.frame, self.seq = cap, None, 0
        self._lock, self._stop = threading.Lock(), threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            ok, f = self.cap.read()
            if ok:
                with self._lock:
                    self.frame, self.seq = f, self.seq + 1

    def latest(self):
        with self._lock:
            return self.frame, self.seq

    def close(self):
        self._stop.set()
        self._thread.join(timeout=2)
        self.cap.release()


def prepare_frame(frame, cfg):
    h, w = cfg["height"], cfg["width"]
    if frame.shape[0] != h or frame.shape[1] != w:
        frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA if frame.shape[1] > w else cv2.INTER_LINEAR)
    return frame


def encode_jpeg(frame, quality: int) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return buf.tobytes()


def pack_frame(frame_id: int, jpeg: bytes, flags: int = 0) -> bytes:
    return HEADER.pack(MAGIC, PROTO_VERSION, flags, frame_id, int(time.time() * 1e6)) + jpeg


def put_text(img, text, org, scale=0.6, color=(255, 255, 255), thick=1):
    """Text with a dark outline so it stays readable on any background."""
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)
