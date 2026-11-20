"""Text and rounded shapes for the live window.

Text uses a real font (Pillow + a Windows system font: Segoe UI Semibold / Bold, else Bahnschrift) with a crisp outline.
If Pillow or every font is missing it falls back to OpenCV's built-in font, so nothing breaks and the API is the same.
Sizes keep OpenCV's `scale` meaning (scale 1.0 ~ 30 px font), so existing callers did not change.
"""
import functools
import os
from pathlib import Path

import cv2
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # the laptop may not have installed Pillow yet
    Image = None

FONT_DIRS = [Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"]
REGULAR = ("seguisb.ttf", "bahnschrift.ttf", "segoeui.ttf")   # semibold reads best at small sizes on video
BOLD = ("segoeuib.ttf", "bahnschrift.ttf", "seguisb.ttf")
PX_PER_SCALE = 30


@functools.lru_cache(maxsize=None)
def _font(px, bold):
    if Image is None:
        return None
    for name in (BOLD if bold else REGULAR):
        for d in FONT_DIRS:
            if (d / name).exists():
                return ImageFont.truetype(str(d / name), px)
    return None


def available():
    """True when text is drawn with a real font (False = OpenCV fallback)."""
    return _font(PX_PER_SCALE, False) is not None


def _px(scale):
    return max(8, round(scale * PX_PER_SCALE))


def text_size(text, scale, thick=1):
    """(width, cap height) in pixels of `text`, like cv2.getTextSize()[0]; the baseline is at the bottom of the cap height."""
    font = _font(_px(scale), thick >= 2)
    if font is None:
        return cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)[0]
    return int(round(font.getlength(text))), -font.getbbox("H", anchor="ls")[1]


@functools.lru_cache(maxsize=256)
def _masks(text, px, bold, outline):
    """(fill alpha, outline alpha, baseline x, baseline y) of the rendered text, as float arrays 0..1 (cached: HUD text barely changes)."""
    font = _font(px, bold)
    pad = outline + 2
    l, t, r, b = font.getbbox(text, anchor="ls", stroke_width=outline)
    size, origin = (r - l + 2 * pad, b - t + 2 * pad), (pad - l, pad - t)
    out = []
    for stroke in (0, outline):
        m = Image.new("L", size, 0)
        ImageDraw.Draw(m).text(origin, text, font=font, fill=255, anchor="ls", stroke_width=stroke)
        out.append(np.asarray(m, np.float32)[:, :, None] / 255.0)
    return out[0], out[1], origin[0], origin[1]


def draw_text(img, text, org, scale=0.6, color=(255, 255, 255), thick=1):
    """Text with a dark outline so it stays readable on any background. org = left end of the baseline. thick >= 2 = bold."""
    font = _font(_px(scale), thick >= 2)
    if font is None:
        _hershey(img, text, org, scale, color, thick)
        return
    px = _px(scale)
    fill, edge, ox, oy = _masks(text, px, thick >= 2, max(1, px // 14))
    h, w = fill.shape[:2]
    x0, y0 = org[0] - ox, org[1] - oy
    ax, ay, bx, by = max(x0, 0), max(y0, 0), min(x0 + w, img.shape[1]), min(y0 + h, img.shape[0])
    if ax >= bx or ay >= by:
        return
    sl = (slice(ay - y0, by - y0), slice(ax - x0, bx - x0))
    roi = img[ay:by, ax:bx].astype(np.float32)
    roi *= 1.0 - 0.85 * edge[sl]                                    # dark outline
    roi = roi * (1.0 - fill[sl]) + np.array(color, np.float32) * fill[sl]  # text colour
    img[ay:by, ax:bx] = roi.astype(np.uint8)


def _hershey(img, text, org, scale, color, thick):
    # The outline is the same text at the same thickness, shifted by 1-2 px in 8 directions. (A thicker black stroke looked
    # simpler but OpenCV 5 spaces glyphs differently per thickness, so outline and fill did not line up and the text looked doubled.)
    r = 2 if scale >= 0.9 else 1
    for dx in (-r, 0, r):
        for dy in (-r, 0, r):
            if dx or dy:
                cv2.putText(img, text, (org[0] + dx, org[1] + dy), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


@functools.lru_cache(maxsize=128)
def _box_mask(w, h, r):
    """(coverage 0..1 as float, border-pixel boolean) of a w x h rounded rectangle; cached because HUD boxes repeat every frame."""
    mask = np.zeros((h, w), np.uint8)
    cv2.rectangle(mask, (r, 0), (w - 1 - r, h - 1), 255, -1)
    cv2.rectangle(mask, (0, r), (w - 1, h - 1 - r), 255, -1)
    for cx, cy in ((r, r), (w - 1 - r, r), (r, h - 1 - r), (w - 1 - r, h - 1 - r)):
        cv2.circle(mask, (cx, cy), r, 255, -1, cv2.LINE_AA)
    padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)  # so the straight sides count as edges too
    edge = cv2.morphologyEx(padded, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))[1:-1, 1:-1] > 0
    return (mask.astype(np.float32) / 255.0)[:, :, None], edge & (mask > 0)


def rounded_box(img, rect, color, alpha=1.0, radius=10, border=None):
    """Filled rounded rectangle (x, y, w, h), blended with `alpha`; optional 1 px border colour. Clipped at the image edges."""
    x, y, w, h = rect
    ax, ay, bx, by = max(x, 0), max(y, 0), min(x + w, img.shape[1]), min(y + h, img.shape[0])
    if ax >= bx or ay >= by:
        return
    cover, edge = _box_mask(w, h, max(0, min(radius, w // 2, h // 2)))
    m = (cover[ay - y:by - y, ax - x:bx - x] * (alpha * 256)).astype(np.uint16)  # 0..256, integer blend: ~3x faster than float
    roi = img[ay:by, ax:bx]
    roi[...] = ((roi.astype(np.uint16) * (256 - m) + np.array(color, np.uint16) * m) >> 8).astype(np.uint8)
    if border is not None:
        roi[edge[ay - y:by - y, ax - x:bx - x]] = border
