"""On-screen display of the live client: debug keypoints, the big committed word/sentence, a compact latency readout.
Pure drawing (opencv + time), so it also works on the CPU-only laptop. Keys (set in client.py): k keypoints, d details, q/Esc quit.
"""
import math
import time

import cv2

from common import put_text

BANNER_SECONDS = 5.0   # the banner stays this long after the newest word; a new word restarts the timer
MAX_BANNER_WORDS = 6   # older words scroll off so the line stays readable
SENTENCE_TEXT = {"how_are_you": "HOW ARE YOU?", "where_are_you": "WHERE ARE YOU?"}
# COCO-WholeBody indices the server sends: body 0-12 and both hands 91-132. Only the arms are drawn:
# shoulders 5/6, elbows 7/8, wrists 9/10 (no face points 0-4, no hips 11/12).
ARM_IDS = [5, 6, 7, 8, 9, 10]
BODY_BONES = [(0, 1), (0, 2), (2, 4), (1, 3), (3, 5)]  # positions in ARM_IDS (like HAND_BONES: positions in a hand's 21 ids)
HAND_BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (11, 12),
              (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)]
WHITE, TIP_RED = (255, 255, 255), (60, 60, 255)  # BGR: clean white skeleton, red fingertips
FINGERTIPS = (4, 8, 12, 16, 20)                  # positions in a hand's 21 ids
SMOOTH_TAU = 0.05    # seconds: smoothing time constant for a point that is nearly still (kills jitter)
CATCHUP_PX = 12      # the further a point is behind its target, the faster it moves: tau / (1 + gap / CATCHUP_PX)
SNAP_PX = 90         # a point that jumps further than this (keypoint-image pixels) is a new detection: no gliding
GHOST_OVERLAP = 0.6  # hands closer than this fraction of a hand size are one hand detected twice
GRACE = 0.25         # seconds a point may be missing before it is dropped (hides single-frame flicker)


def label_text(label):
    """committed_label -> what to show: sentences as a question, words in capitals."""
    return SENTENCE_TEXT.get(label, label.replace("_", " ").upper())


def drop_ghost_hand(pts, scores):
    """The keypoint model sometimes reports BOTH hands on one real hand. If the two detected hands sit on top of each
    other (centre distance < GHOST_OVERLAP x the bigger hand size), keep only the one with the higher mean score."""
    hands = []
    for first in (91, 112):
        ids = [i for i in range(first, first + 21) if i in pts]
        if len(ids) >= 5:
            xs, ys = [pts[i][0] for i in ids], [pts[i][1] for i in ids]
            size = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
            hands.append((ids, (sum(xs) / len(xs), sum(ys) / len(ys)), size, sum(scores[i] for i in ids) / len(ids)))
    if len(hands) == 2:
        (ia, ca, sa, ma), (ib, cb, sb, mb) = hands
        if math.hypot(ca[0] - cb[0], ca[1] - cb[1]) < GHOST_OVERLAP * max(sa, sb):
            for i in (ib if ma >= mb else ia):
                del pts[i]
    return pts


class Smoother:
    """Per-point exponential smoothing in keypoint-image coordinates. Fed with each new server message, queried every
    display frame, so the skeleton glides between the ~10 Hz results instead of stepping."""

    def __init__(self):
        self.pos, self.seen, self.msg, self.target, self.last = {}, {}, None, {}, None

    def update(self, kp, now):
        if not kp:
            self.pos.clear(), self.seen.clear()
            self.msg = None
            return {}, 1, 1
        thr = kp.get("thr", 0.0)
        if kp is not self.msg:  # a new message: refresh targets
            self.msg = kp
            self.target = {int(i): (x, y) for i, x, y, s in kp["pts"] if s >= thr and x >= 0}
            self.target = drop_ghost_hand(self.target, {int(i): s for i, _, _, s in kp["pts"]})
            for i in self.target:
                self.seen[i] = now
        last, self.last = (now if self.last is None else self.last), now
        dt = max(now - last, 0.0)
        for i, (tx, ty) in self.target.items():
            px, py = self.pos.get(i, (tx, ty))
            gap = math.hypot(tx - px, ty - py)
            if gap > SNAP_PX:
                px, py, gap = tx, ty, 0.0
            alpha = 1 - math.exp(-dt * (1 + gap / CATCHUP_PX) / SMOOTH_TAU)  # fast hands -> big gap -> little lag
            self.pos[i] = (px + (tx - px) * alpha, py + (ty - py) * alpha)
        for i in [i for i, t in self.seen.items() if now - t > GRACE]:
            self.pos.pop(i, None), self.seen.pop(i)
        return self.pos, kp["w"], kp["h"]


def draw_keypoints(img, kp, smoother=None):
    """kp: the server's keypoints message. Points scoring below its threshold are hidden (they are guesses).
    Clean style: white arm and hand skeleton, red fingertip dots. With a Smoother the motion is smoothed."""
    if smoother is None:
        smoother = Smoother()
    pos, w, h = smoother.update(kp, time.time())
    if not pos:
        return
    sx, sy = img.shape[1] / w, img.shape[0] / h
    pts = {i: (int(x * sx), int(y * sy)) for i, (x, y) in pos.items()}

    def bones(ids, pairs, r, tips=()):
        for a, b in pairs:
            if ids[a] in pts and ids[b] in pts:
                cv2.line(img, pts[ids[a]], pts[ids[b]], WHITE, 2, cv2.LINE_AA)
        for n, i in enumerate(ids):
            if i in pts:
                cv2.circle(img, pts[i], r + 2 if n in tips else r, TIP_RED if n in tips else WHITE, -1, cv2.LINE_AA)

    bones(ARM_IDS, BODY_BONES, 4)
    bones(list(range(91, 112)), HAND_BONES, 2, FINGERTIPS)
    bones(list(range(112, 133)), HAND_BONES, 2, FINGERTIPS)


def draw_banner(img, link):
    """Big, centred, bottom-third banner with the newest committed word or sentence (words accumulate; fades out BANNER_SECONDS after the newest)."""
    words, t = link.phrase
    age = time.time() - t
    if not words or age > BANNER_SECONDS:
        return
    text, h, w = " ".join(label_text(x) for x in words), img.shape[0], img.shape[1]
    scale = 3.0
    while scale > 0.8 and cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 6)[0][0] > 0.9 * w:
        scale -= 0.2
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 6)
    alpha = 0.75 if age < BANNER_SECONDS - 1 else 0.75 * (BANNER_SECONDS - age)  # last second: fade
    y0, y1 = int(h * 0.68), int(h * 0.68) + th + 50
    box = img.copy()
    cv2.rectangle(box, (0, y0), (w, y1), (30, 30, 30), -1)
    cv2.addWeighted(box, alpha, img, 1 - alpha, 0, img)
    org = ((w - tw) // 2, y1 - 25)
    put_text(img, text, org, scale, (255, 255, 255), 6)  # same-thickness outline (see common.put_text)


def draw_overlay(img, link, disp_fps, details):
    r = link.result or {}
    ok = link.state == "connected"
    h, w = img.shape[:2]
    draw_banner(img, link)
    # top-left: connection (small) and the current prediction with its confidence
    put_text(img, link.state + (f"  [{link.error}]" if link.error else ""), (10, 24), 0.55, (0, 255, 0) if ok else (0, 165, 255))
    put_text(img, f"{r.get('label', '-')}", (10, 62), 1.0, (255, 255, 255), 2)
    conf = float(r.get("confidence", 0.0))
    cv2.rectangle(img, (10, 72), (190, 86), (60, 60, 60), -1)
    cv2.rectangle(img, (10, 72), (10 + int(180 * conf), 86), (0, 200, 0), -1)
    if r.get("profile") == "fallback":
        put_text(img, "SAFE MODE: small keypoint model", (10, 110), 0.5, (0, 165, 255))
    # bottom-right: latency readout, small
    rtt, sm = r.get("rtt_ms"), r.get("server_ms", {})
    lat = (f"{rtt:.0f} ms round trip" if rtt is not None else "- ms") + f"  |  server {sm.get('total', 0):.1f} ms"
    (tw, _), _ = cv2.getTextSize(lat, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    put_text(img, lat, (w - tw - 10, h - 12), 0.5, (200, 255, 200))
    put_text(img, "k skeleton   d details   q quit", (10, h - 12), 0.45, (190, 190, 190))
    if details:
        put_text(img, "committed: " + (" > ".join(link.history) or "-"), (10, 140), 0.55, (0, 255, 255))
        put_text(img, f"state {r.get('state', '-')}  conf {conf * 100:.0f}%  server dec {sm.get('decode', 0):.1f} kp {sm.get('keypoints', 0):.1f} "
                      f"temp {sm.get('temporal', 0):.1f} ms", (10, 165), 0.5)
        put_text(img, f"send {link.send_fps():.1f} fps | display {disp_fps:.0f} fps | server-dropped {r.get('dropped_frames', 0)} | skipped {link.skipped}",
                 (10, 190), 0.5)
