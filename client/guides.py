"""How-to-sign pictures for the menu, drawn with OpenCV/numpy (no image files, no extra dependencies).

  guide_image(word) -> BGR card (CARD_W x CARD_H): title, 1-3 numbered frames with arrows, one caption per frame.
  python client/guides.py --out <folder>        write every card as a PNG (to look at them)

A real picture wins: if client/guides/<word>.png exists it is shown instead of the drawing, so better images can be dropped in later.
The hands reuse the skeleton look of overlay.py (white bones, red fingertips) so the guide matches what the camera view draws.
"""
import argparse
import math
from pathlib import Path

import cv2
import numpy as np

import overlay
from common import CLIENT_DIR, put_text, text_size

CARD_W, CARD_H = 1020, 400
PW, PH = 300, 250                       # one frame (panel) inside the card
ORDER = ["hello", "what", "how", "why", "where", "you", "how_are_you", "where_are_you"]
TITLES = {w: overlay.label_text(w) for w in ORDER}
OVERRIDE_DIR = CLIENT_DIR / "guides"    # optional hand-made pictures: client/guides/<word>.png

BG, PANEL, GREY, ARROW = (32, 30, 28), (46, 44, 42), (150, 150, 150), (0, 190, 255)

# --- a hand as 21 points (same indices as overlay.HAND_BONES), built in "hand units": wrist at 0,0, y up, ~1 unit = palm length ---
FINGERS = {  # name: (first point index, base x, base y, segment lengths, spread when the fingers are apart)
    "index": (5, -0.36, 1.00, (0.42, 0.26, 0.22), -0.28), "middle": (9, -0.12, 1.04, (0.46, 0.29, 0.24), -0.06),
    "ring": (13, 0.12, 0.98, (0.43, 0.27, 0.22), 0.14), "pinky": (17, 0.34, 0.90, (0.34, 0.20, 0.18), 0.36)}
# shape -> (extended fingers, fingers spread apart, thumb direction in radians from "up")
SHAPES = {"flat": ({"index", "middle", "ring", "pinky"}, False, -0.25), "open": ({"index", "middle", "ring", "pinky"}, True, -0.9),
          "fist": (set(), False, 0.9), "point": ({"index"}, False, 0.7), "Y": ({"pinky"}, False, -1.2)}


def hand_points(shape):
    ext, spread, thumb = SHAPES[shape]
    pts = [(0.0, 0.0)] * 21
    for name, (first, bx, by, segs, sp) in FINGERS.items():
        if name in ext:
            x, y, ang = bx, by, (sp if spread else 0.0)
            pts[first] = (x, y)
            for n, length in enumerate(segs):
                x, y = x + length * math.sin(ang), y + length * math.cos(ang)
                pts[first + 1 + n] = (x, y)
        else:  # folded: the finger is seen end-on, so its joints collapse onto the knuckle
            pts[first:first + 4] = [(bx, by), (bx, by + 0.14), (bx + 0.02, by + 0.06), (bx + 0.02, by - 0.10)]
    x, y, ang = -0.30, 0.22, thumb
    pts[1] = (x, y)
    for n, length in enumerate((0.30, 0.26, 0.22)):
        ang += 0.0 if shape in ("flat", "open", "Y") or n == 0 else 0.5
        x, y = x + length * math.sin(ang), y + length * math.cos(ang)
        pts[2 + n] = (x, y)
    return pts


def draw_hand(img, wrist, shape, angle_deg=0.0, size=44, mirror=False, arm=True):
    """wrist: pixel position. angle_deg: clockwise rotation from fingers-up. size: pixels per palm length."""
    a = math.radians(angle_deg)
    pts = []
    for x, y in hand_points(shape):
        x = -x if mirror else x
        pts.append((int(wrist[0] + size * (x * math.cos(a) + y * math.sin(a))), int(wrist[1] - size * (-x * math.sin(a) + y * math.cos(a)))))
    if arm:  # forearm continues straight back from the wrist
        back = (wrist[0] - int(size * 1.7 * math.sin(a)), wrist[1] + int(size * 1.7 * math.cos(a)))
        cv2.line(img, wrist, back, GREY, 3, cv2.LINE_AA)
        cv2.circle(img, back, 5, GREY, -1, cv2.LINE_AA)
    if shape in ("fist", "point", "Y"):  # folded fingers: outline the closed part of the hand
        block = [(-0.50, 0.15), (-0.50, 1.28), (0.50, 1.28), (0.50, 0.15)]
        cv2.polylines(img, [np.array([(int(wrist[0] + size * (x * (-1 if mirror else 1) * math.cos(a) + y * math.sin(a))),
                                       int(wrist[1] - size * (-x * (-1 if mirror else 1) * math.sin(a) + y * math.cos(a)))) for x, y in block])],
                      True, GREY, 2, cv2.LINE_AA)
    for a_, b_ in overlay.HAND_BONES + [(5, 9), (9, 13), (13, 17)]:
        cv2.line(img, pts[a_], pts[b_], overlay.WHITE, 2, cv2.LINE_AA)
    for n, p in enumerate(pts):
        tip = n in overlay.FINGERTIPS
        cv2.circle(img, p, 5 if tip else 3, overlay.TIP_RED if tip else overlay.WHITE, -1, cv2.LINE_AA)


def draw_head(img, centre=(150, 70), r=36):
    cx, cy = centre
    cv2.circle(img, centre, r, GREY, 2, cv2.LINE_AA)
    for dx in (-r // 3, r // 3):
        cv2.circle(img, (cx + dx, cy - r // 6), 3, GREY, -1, cv2.LINE_AA)
    cv2.line(img, (cx, cy + r), (cx, cy + r + r // 2), GREY, 2, cv2.LINE_AA)
    cv2.line(img, (cx - int(r * 2.2), cy + int(r * 1.9)), (cx + int(r * 2.2), cy + int(r * 1.9)), GREY, 2, cv2.LINE_AA)


def arrow(img, a, b, both=False):
    cv2.arrowedLine(img, a, b, ARROW, 3, cv2.LINE_AA, tipLength=0.22)
    if both:
        cv2.arrowedLine(img, b, a, ARROW, 3, cv2.LINE_AA, tipLength=0.22)


# --- the frames. Each one draws into a PW x PH panel. PLACEHOLDER: these are schematic drawings of the standard ASL signs; which
# --- variant of each sign the ASL Citizen training clips show has NOT been checked against the data (see governance/STATUS.md).
def hello_1(img): draw_head(img); draw_hand(img, (236, 138), "flat", -30, 52)
def hello_2(img): draw_head(img); draw_hand(img, (210, 170), "flat", 35, 48); arrow(img, (192, 38), (275, 26))
def what_1(img): draw_head(img, (150, 40), 24); draw_hand(img, (92, 240), "open", -12, 46); draw_hand(img, (208, 240), "open", 12, 46, mirror=True)
def what_2(img): what_1(img); arrow(img, (150, 128), (250, 128)); arrow(img, (150, 128), (50, 128))
def how_1(img): draw_head(img, (150, 40), 24); draw_hand(img, (96, 238), "open", -30, 46); draw_hand(img, (204, 238), "open", 30, 46, mirror=True)
def how_2(img):
    draw_head(img, (150, 40), 24); draw_hand(img, (120, 236), "fist", 6, 46); draw_hand(img, (180, 236), "fist", -6, 46, mirror=True)
    arrow(img, (88, 108), (126, 140)); arrow(img, (212, 108), (174, 140))
def why_1(img): hello_1(img)
def why_2(img): draw_head(img); draw_hand(img, (225, 178), "Y", 15, 50); arrow(img, (186, 56), (232, 96))
def where_1(img): draw_head(img, (150, 40), 24); draw_hand(img, (150, 240), "point", -4, 52)
def where_2(img): where_1(img); arrow(img, (90, 100), (210, 100), both=True)
def you_1(img):
    draw_head(img, (240, 80), 20); put_text(img, "other person", (190, 190), 0.5, GREY)
    draw_hand(img, (80, 205), "point", 55, 46); arrow(img, (150, 150), (196, 118))


FRAMES = {"hello": [hello_1, hello_2], "what": [what_1, what_2], "how": [how_1, how_2], "why": [why_1, why_2],
          "where": [where_1, where_2], "you": [you_1]}
FRAMES["how_are_you"] = FRAMES["how"] + FRAMES["you"]
FRAMES["where_are_you"] = FRAMES["where"] + FRAMES["you"]
STEPS = {"hello": ["Flat hand at your forehead", "Move it outward, like a salute"],
         "what": ["Both palms up, fingers open", "Shake the hands side to side"],
         "how": ["Open hands, angled outward", "Flip them in and close into fists, knuckles together"],
         "why": ["Fingertips touch your forehead", "Pull away, ending in a Y hand"],
         "where": ["Index finger up", "Shake it side to side"],
         "you": ["Point at the other person (diagonal is fine)"],
         "how_are_you": ["HOW: open hands, angled out", "...flip in and close into fists", "YOU: point at them, diagonal"],
         "where_are_you": ["WHERE: index finger up", "...shake it side to side", "YOU: point at them, diagonal"]}

_cache = {}


def render(word):
    card = np.full((CARD_H, CARD_W, 3), BG, np.uint8)
    put_text(card, TITLES[word], (30, 52), 1.2, (255, 255, 255), 2)
    frames, steps = FRAMES[word], STEPS[word]
    x0 = (CARD_W - (len(frames) * PW + (len(frames) - 1) * 40)) // 2
    for n, (draw, step) in enumerate(zip(frames, steps)):
        x = x0 + n * (PW + 40)
        panel = np.full((PH, PW, 3), PANEL, np.uint8)
        draw(panel)
        cv2.rectangle(panel, (0, 0), (PW - 1, PH - 1), (90, 88, 86), 1)
        card[75:75 + PH, x:x + PW] = panel
        cv2.circle(card, (x + 18, 93), 13, ARROW, -1, cv2.LINE_AA)
        cv2.putText(card, str(n + 1), (x + 12, 99), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
        tw, _ = text_size(step, 0.55)
        put_text(card, step, (x + (PW - tw) // 2, 75 + PH + 30), 0.55, (230, 230, 230))
        if n:
            cv2.arrowedLine(card, (x - 34, 75 + PH // 2), (x - 8, 75 + PH // 2), GREY, 2, cv2.LINE_AA, tipLength=0.4)
    put_text(card, "Schematic guide - confirm the sign with an ASL reference", (30, CARD_H - 14), 0.45, GREY)
    return card


def guide_image(word):
    """The how-to card for a word (cached). A file client/guides/<word>.png overrides the drawing."""
    if word not in _cache:
        custom = cv2.imread(str(OVERRIDE_DIR / f"{word}.png")) if (OVERRIDE_DIR / f"{word}.png").exists() else None
        _cache[word] = custom if custom is not None else render(word)
    return _cache[word]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="folder for the PNGs")
    out = Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    for w in ORDER:
        cv2.imwrite(str(out / f"{w}.png"), render(w))
    print(f"wrote {len(ORDER)} cards to {out}")
