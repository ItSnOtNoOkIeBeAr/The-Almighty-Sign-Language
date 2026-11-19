"""On-screen display of the live client: debug keypoints, the big committed word/sentence, a compact latency readout.
Pure drawing (opencv + time), so it also works on the CPU-only laptop. Keys (set in client.py): k keypoints, d details, q/Esc quit.
"""
import time

import cv2

from common import put_text

BANNER_SECONDS = 4.0   # how long a committed word/sentence stays on screen
SENTENCE_TEXT = {"how_are_you": "HOW ARE YOU?", "where_are_you": "WHERE ARE YOU?"}
# COCO-WholeBody indices the server sends: body 0-12 (5..12 = shoulders, elbows, wrists, hips) and both hands 91-132
BODY_BONES = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12)]
HAND_BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (11, 12),
              (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)]
COLORS = {"body": (200, 160, 90), "left": (230, 120, 0), "right": (0, 140, 255)}  # BGR: left hand blue, right hand orange


def label_text(label):
    """committed_label -> what to show: sentences as a question, words in capitals."""
    return SENTENCE_TEXT.get(label, label.replace("_", " ").upper())


def draw_keypoints(img, kp):
    """kp: the server's keypoints message. Points scoring below its threshold are hidden (they are guesses)."""
    if not kp:
        return
    sx, sy, thr = img.shape[1] / kp["w"], img.shape[0] / kp["h"], kp.get("thr", 0.0)
    pts = {int(i): (int(x * sx), int(y * sy)) for i, x, y, s in kp["pts"] if s >= thr and x >= 0}

    def bones(ids, pairs, color, r):
        for a, b in pairs:
            if ids[a] in pts and ids[b] in pts:
                cv2.line(img, pts[ids[a]], pts[ids[b]], color, 2, cv2.LINE_AA)
        for i in ids:
            if i in pts:
                cv2.circle(img, pts[i], r, color, -1, cv2.LINE_AA)

    bones(list(range(13)), BODY_BONES, COLORS["body"], 4)
    bones(list(range(91, 112)), HAND_BONES, COLORS["left"], 3)
    bones(list(range(112, 133)), HAND_BONES, COLORS["right"], 3)


def draw_banner(img, link):
    """Big, centred, bottom-third banner with the newest committed word or sentence (fades out after BANNER_SECONDS)."""
    if not link.last_commit:
        return
    word, t = link.last_commit
    age = time.time() - t
    if age > BANNER_SECONDS:
        return
    text, h, w = label_text(word), img.shape[0], img.shape[1]
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
    if link.want_kp:
        draw_keypoints(img, link.kp)
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
    put_text(img, "k keypoints   d details   q quit", (10, h - 12), 0.45, (190, 190, 190))
    if details:
        put_text(img, "committed: " + (" > ".join(link.history) or "-"), (10, 140), 0.55, (0, 255, 255))
        put_text(img, f"state {r.get('state', '-')}  conf {conf * 100:.0f}%  server dec {sm.get('decode', 0):.1f} kp {sm.get('keypoints', 0):.1f} "
                      f"temp {sm.get('temporal', 0):.1f} ms", (10, 165), 0.5)
        put_text(img, f"send {link.send_fps():.1f} fps | display {disp_fps:.0f} fps | server-dropped {r.get('dropped_frames', 0)} | skipped {link.skipped}",
                 (10, 190), 0.5)
