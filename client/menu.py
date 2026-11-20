"""Clickable screens of the live window: Home (word cards + Free cam) and the how-to-sign guide.
Pure drawing + hit-testing on numpy images (opencv only), so they are unit-tested without a camera or a window.

Each render_* function takes the Mouse, draws a frame and returns (image, action); action is None until something was clicked.
"""
import cv2
import numpy as np

import guides
from common import put_text

W, H = 960, 540                         # size of the Home / guide screens
BG, CARD, CARD_HOT, BORDER = (32, 30, 28), (62, 58, 54), (96, 88, 80), (110, 106, 100)
GREEN, GREEN_HOT, WHITE = (60, 160, 60), (80, 200, 80), (255, 255, 255)


class Mouse:
    """Filled in by OpenCV's mouse callback (cv2.setMouseCallback(window, mouse.on_event))."""

    def __init__(self):
        self.x, self.y, self.clicked = -1, -1, None

    def on_event(self, event, x, y, flags, param):
        self.x, self.y = x, y
        if event == cv2.EVENT_LBUTTONUP:
            self.clicked = (x, y)

    def take(self):
        """The newest click, once (None if there was none since the last call)."""
        c, self.clicked = self.clicked, None
        return c


class Button:
    def __init__(self, key, label, rect, color=CARD, hot=CARD_HOT):
        self.key, self.label, self.rect, self.color, self.hot = key, label, rect, color, hot

    def hit(self, p):
        x, y, w, h = self.rect
        return p is not None and x <= p[0] < x + w and y <= p[1] < y + h

    def centre(self):
        x, y, w, h = self.rect
        return x + w // 2, y + h // 2


def draw_button(img, b, mouse, active=False, scale=0.7):
    x, y, w, h = b.rect
    fill = GREEN_HOT if active else (b.hot if b.hit((mouse.x, mouse.y)) else b.color)
    cv2.rectangle(img, (x, y), (x + w, y + h), fill, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), BORDER, 1)
    (tw, th), _ = cv2.getTextSize(b.label, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    put_text(img, b.label, (x + (w - tw) // 2, y + (h + th) // 2), scale, WHITE, 2)


def clicked(buttons, mouse):
    """Key of the button under the newest click, or None."""
    p = mouse.take()
    return next((b.key for b in buttons if b.hit(p)), None) if p else None


def home_buttons():
    cols, cw, ch, gap, x0, y0 = 4, 210, 84, 20, 30, 170
    words = [Button(w, guides.TITLES[w], (x0 + (i % cols) * (cw + gap), y0 + (i // cols) * (ch + gap), cw, ch))
             for i, w in enumerate(guides.ORDER)]
    return words + [Button("cam", "Free cam", (280, 410, 400, 80), GREEN, GREEN_HOT)]


def render_home(mouse):
    img = np.full((H, W, 3), BG, np.uint8)
    put_text(img, "The Almighty Sign Language", (30, 70), 1.4, WHITE, 2)
    put_text(img, "Pick a word to see how it is signed, or open the free camera and sign any of them.", (30, 118), 0.6, (200, 200, 200))
    buttons = home_buttons()
    for b in buttons:
        draw_button(img, b, mouse, scale=0.8 if b.key == "cam" else 0.65)
    put_text(img, "q  quit", (30, H - 16), 0.5, (170, 170, 170))
    key = clicked(buttons, mouse)
    return img, (None if key is None else ("cam", None) if key == "cam" else ("guide", key))


def guide_buttons():
    return [Button("home", "Back", (30, 450, 170, 60)), Button("cam", "Try it with the camera", (560, 450, 370, 60), GREEN, GREEN_HOT)]


def render_guide(mouse, word):
    img = np.full((H, W, 3), BG, np.uint8)
    card = guides.guide_image(word)
    card = cv2.resize(card, (900, int(card.shape[0] * 900 / card.shape[1])), interpolation=cv2.INTER_AREA)
    img[40:40 + card.shape[0], 30:30 + 900] = card
    buttons = guide_buttons()
    for b in buttons:
        draw_button(img, b, mouse)
    key = clicked(buttons, mouse)
    return img, (None if key is None else ("home", None) if key == "home" else ("cam", word))
