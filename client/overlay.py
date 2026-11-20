"""On-screen display of the live client: debug keypoints, the big committed word/sentence, a compact latency readout.
Pure drawing (opencv + time), so it also works on the CPU-only laptop. Keys (set in client.py): k keypoints, d details, q/Esc quit.
"""
import math
import time

import cv2

from common import put_text, text_size
from textfx import rounded_box

PANEL, ACCENT, GOOD, WARN = (24, 22, 20), (170, 215, 60), (110, 210, 90), (0, 165, 255)  # BGR: HUD panels, accents
BANNER_SECONDS = 5.0   # the banner stays this long after the newest word; a new word restarts the timer
MAX_BANNER_WORDS = 6   # older words scroll off so the line stays readable
SENTENCE_TEXT = {"how_are_you": "HOW ARE YOU?", "where_are_you": "WHERE ARE YOU?"}
# COCO-WholeBody indices the server sends: body 0-12 and both hands 91-132. Drawn: face dots, plus the arms:
# shoulders 5/6, elbows 7/8, wrists 9/10 (no hips 11/12).
FACE_IDS = [0, 1, 2, 3, 4]   # nose, eyes, ears: the only face points the server sends (no mouth/brows/contour)
ARM_IDS = [5, 6, 7, 8, 9, 10]
ARM_SIDES = [(5, 7, 9), (6, 8, 10)]  # (shoulder, elbow, wrist) per side; each side is drawn whole or not at all
BODY_BONES = [(0, 1), (0, 2), (2, 4), (1, 3), (3, 5)]  # positions in ARM_IDS (like HAND_BONES: positions in a hand's 21 ids)
HAND_BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (11, 12),
              (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)]
WHITE, TIP_RED = (255, 255, 255), (60, 60, 255)  # BGR: clean white skeleton, red fingertips
FINGERTIPS = (4, 8, 12, 16, 20)                  # positions in a hand's 21 ids
SMOOTH_TAU = 0.05    # seconds: smoothing time constant for a point that is nearly still (kills jitter)
CATCHUP_PX = 12      # the further a point is behind its target, the faster it moves: tau / (1 + gap / CATCHUP_PX)
SNAP_PX = 90         # a point that jumps further than this (keypoint-image pixels) is a new detection: no gliding
GHOST_OVERLAP = 0.6  # hands closer than this fraction of a hand size are one hand detected twice
ARM_MIN_SCORE = 0.5  # arm points are guesses when the arm is out of view: demand more than the hand threshold
EDGE_PX = 12         # arm points this close to (or beyond) the frame edge are clamped guesses, not seen joints
HAND_MIN_SCORE = 0.5   # a real hand is detected confidently: mean score of its points
HAND_MIN_POINTS = 14   # ... and most of its 21 points are seen
HAND_ARM_REACH = 1.3   # ... and it is at the end of a detected arm: its wrist is within this x its own size (+ a margin) of an arm wrist,
HAND_BODY_REACH = 5.0  # ... or within this many face widths of your nose (arm out of view)
GRACE = 0.25         # seconds a point may be missing before it is dropped (hides single-frame flicker)


def label_text(label):
    """committed_label -> what to show: sentences as a question, words in capitals."""
    return SENTENCE_TEXT.get(label, label.replace("_", " ").upper())


def face_anchor(pts):
    """(nose position, face width) from the face dots, or None. Face width = ear to ear (else eye to eye x 2): the person's own scale."""
    if 0 not in pts:
        return None
    dist = lambda a, b: math.hypot(pts[a][0] - pts[b][0], pts[a][1] - pts[b][1]) if a in pts and b in pts else None  # noqa: E731
    width = dist(3, 4) or (2 * dist(1, 2) if dist(1, 2) else None)
    return (pts[0], width) if width else None


def drop_stray_hands(pts, scores, face):
    """Keep a hand only if it is plausibly YOURS: confident, mostly complete, and either at the end of a detected arm or within arm's
    reach of your face (the forearm is often out of the picture when a hand is raised). `face` = face_anchor() taken before the arm filter.
    Objects and background clutter the model labels as a hand fail these tests and are dropped."""
    arms = [pts[i] for i in (9, 10) if i in pts]
    for first in (91, 112):
        ids = [i for i in range(first, first + 21) if i in pts]
        if not ids:
            continue
        xs, ys = [pts[i][0] for i in ids], [pts[i][1] for i in ids]
        size = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
        wrist = pts.get(first, (sum(xs) / len(xs), sum(ys) / len(ys)))
        on_arm = any(math.hypot(wrist[0] - ax, wrist[1] - ay) <= HAND_ARM_REACH * size + 20 for ax, ay in arms)
        near_face = bool(face) and math.hypot(wrist[0] - face[0][0], wrist[1] - face[0][1]) <= HAND_BODY_REACH * face[1]
        if not (len(ids) >= HAND_MIN_POINTS and sum(scores[i] for i in ids) / len(ids) >= HAND_MIN_SCORE and (on_arm or near_face)):
            for i in ids:
                del pts[i]
    return pts


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
    display frame, so the skeleton glides between the server results (one per sent frame, 10-30 Hz) instead of stepping."""

    def __init__(self):
        self.pos, self.seen, self.msg, self.target, self.last = {}, {}, None, {}, None

    def _seen_arm(self, i, scores, w, h):
        """True if the arm joint i has a confident position that is inside the frame (not a clamped guess)."""
        if i not in self.target:
            return False
        x, y = self.target[i]
        return scores[i] >= ARM_MIN_SCORE and EDGE_PX < x < w - EDGE_PX and EDGE_PX < y < h - EDGE_PX

    def update(self, kp, now):
        if not kp:
            self.pos.clear(), self.seen.clear()
            self.msg = None
            return {}, 1, 1
        thr = kp.get("thr", 0.0)
        if kp is not self.msg:  # a new message: refresh targets
            self.msg = kp
            self.target = {int(i): (x, y) for i, x, y, s in kp["pts"] if s >= thr and x >= 0}
            scores, w, h = {int(i): s for i, _, _, s in kp["pts"]}, kp["w"], kp["h"]
            face = face_anchor(self.target)
            for side in ARM_SIDES:  # a whole side is shown or hidden: an arm that is out of view gets guessed joints
                if not all(self._seen_arm(i, scores, w, h) for i in side[1:]):  # elbow and wrist must really be seen
                    for i in side:
                        self.target.pop(i, None)
            self.target = drop_ghost_hand(drop_stray_hands(self.target, scores, face), scores)
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

    bones(FACE_IDS, [], 3)
    bones(ARM_IDS, BODY_BONES, 4)
    bones(list(range(91, 112)), HAND_BONES, 2, FINGERTIPS)
    bones(list(range(112, 133)), HAND_BONES, 2, FINGERTIPS)


def draw_banner(img, link):
    """Centred rounded banner (bottom third) with every word/sentence signed in the last BANNER_SECONDS. A thin bar under the text
    shows how long it stays: it shrinks to nothing, then the next sign starts a fresh line."""
    words, t = link.phrase
    age = time.time() - t
    if not words or age > BANNER_SECONDS:
        return
    text, h, w = " ".join(label_text(x) for x in words), img.shape[0], img.shape[1]
    scale = 3.0
    while scale > 0.8 and text_size(text, scale, 2)[0] > 0.84 * w:
        scale -= 0.2
    tw, th = text_size(text, scale, 2)
    pad_x, pad_y = 36, 24
    bw, bh = tw + 2 * pad_x, th + 2 * pad_y + 12
    x, y = (w - bw) // 2, int(h * 0.64)
    fade = 1.0 if age < BANNER_SECONDS - 1 else BANNER_SECONDS - age  # last second: fade out
    rounded_box(img, (x, y, bw, bh), PANEL, 0.74 * fade, radius=24)
    left = max(6, int(tw * (1 - age / BANNER_SECONDS)))
    rounded_box(img, (x + pad_x, y + bh - 16, left, 5), ACCENT, fade, radius=2)
    put_text(img, text, (x + pad_x, y + pad_y + th), scale, WHITE, 2)


def draw_pill(img, text, x, y, scale=0.5, color=WHITE, dot=None, right=False):
    """Small rounded label. (x, y) = top-left, or top-right with right=True. Returns its width."""
    tw, th = text_size(text, scale)
    w = tw + 24 + (16 if dot else 0)
    x = x - w if right else x
    rounded_box(img, (x, y, w, th + 16), PANEL, 0.62, radius=(th + 16) // 2)
    if dot:
        cv2.circle(img, (x + 15, y + (th + 16) // 2), 5, dot, -1, cv2.LINE_AA)
    put_text(img, text, (x + 12 + (16 if dot else 0), y + 8 + th), scale, color)
    return w


def draw_overlay(img, link, disp_fps, details):
    r = link.result or {}
    ok = link.state == "connected"
    h, w = img.shape[:2]
    draw_banner(img, link)
    # top-left: connection chip, then a card with the current prediction and its confidence
    draw_pill(img, link.state + (f"  [{link.error}]" if link.error else ""), 10, 10, 0.5, WHITE, dot=GOOD if ok else WARN)
    conf = float(r.get("confidence", 0.0))
    rounded_box(img, (10, 48, 236, 70), PANEL, 0.62, radius=16)
    put_text(img, str(r.get("label", "-")).replace("_", " ").upper(), (24, 84), 0.95, WHITE, 2)
    rounded_box(img, (24, 98, 208, 8), (84, 80, 76), 1.0, radius=4)
    rounded_box(img, (24, 98, max(8, int(208 * conf)), 8), GOOD if conf >= 0.5 else WARN, 1.0, radius=4)
    if r.get("profile") == "fallback":
        draw_pill(img, "SAFE MODE: small keypoint model", 10, 128, 0.45, WARN)
    # bottom: key hints (left) and latency (right)
    rtt, sm = r.get("rtt_ms"), r.get("server_ms", {})
    lat = (f"{rtt:.0f} ms round trip" if rtt is not None else "- ms") + f"  |  server {sm.get('total', 0):.1f} ms"
    draw_pill(img, "k skeleton   d details   m menu   q quit", 10, h - 38, 0.45, (205, 205, 205))
    draw_pill(img, lat, w - 10, h - 38, 0.45, (200, 255, 200), right=True)
    if details:
        rounded_box(img, (10, 164, min(w - 20, 640), 92), PANEL, 0.62, radius=14)
        put_text(img, "committed: " + (" > ".join(link.history) or "-"), (24, 190), 0.5, (0, 255, 255))
        put_text(img, f"state {r.get('state', '-')}  conf {conf * 100:.0f}%  server dec {sm.get('decode', 0):.1f} kp {sm.get('keypoints', 0):.1f} "
                      f"temp {sm.get('temporal', 0):.1f} ms", (24, 218), 0.45)
        put_text(img, f"send {link.send_fps():.1f} fps | display {disp_fps:.0f} fps | server-dropped {r.get('dropped_frames', 0)} | skipped {link.skipped}",
                 (24, 244), 0.45)
