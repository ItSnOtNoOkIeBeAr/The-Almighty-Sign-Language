"""Tests for the menu screens and the how-to guides. No camera, window or server needed.

  client\\.venv\\Scripts\\python -m unittest client/test_ui.py -v
"""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import guides  # noqa: E402
import menu  # noqa: E402

LABELS = json.loads((Path(__file__).resolve().parent.parent / "protocol" / "labels.json").read_text())


def click(mouse, button):
    mouse.clicked = button.centre()


class GuideTests(unittest.TestCase):
    def test_every_trained_word_and_sentence_has_a_guide(self):
        trained = [w for w in LABELS["labels"] if w != "none"] + LABELS["sentences"]
        self.assertEqual(sorted(trained), sorted(guides.ORDER))

    def test_every_card_renders(self):
        for w in guides.ORDER:
            card = guides.guide_image(w)
            self.assertEqual(card.shape, (guides.CARD_H, guides.CARD_W, 3), w)
            self.assertGreater(int(card.std()), 10, f"{w}: card looks empty")
            self.assertEqual(len(guides.FRAMES[w]), len(guides.STEPS[w]), w)

    def test_custom_image_overrides_the_drawing(self):
        import cv2
        import numpy as np
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            cv2.imwrite(str(Path(d) / "hello.png"), np.full((50, 60, 3), 200, np.uint8))
            old, guides.OVERRIDE_DIR = guides.OVERRIDE_DIR, Path(d)
            guides._cache.pop("hello", None)
            try:
                self.assertEqual(guides.guide_image("hello").shape, (50, 60, 3))
            finally:
                guides.OVERRIDE_DIR = old
                guides._cache.pop("hello", None)


class MenuTests(unittest.TestCase):
    def test_button_hit_test(self):
        b = menu.Button("x", "X", (10, 20, 100, 50))
        self.assertTrue(b.hit((10, 20)) and b.hit((109, 69)))
        self.assertFalse(b.hit((110, 20)) or b.hit((9, 20)) or b.hit((50, 70)) or b.hit(None))

    def test_no_click_no_action(self):
        _, action = menu.render_home(menu.Mouse())
        self.assertIsNone(action)

    def test_home_word_cards_open_their_guide(self):
        for b in menu.home_buttons():
            if b.key == "cam":
                continue
            m = menu.Mouse()
            click(m, b)
            img, action = menu.render_home(m)
            self.assertEqual(action, ("guide", b.key))
            self.assertEqual(img.shape, (menu.H, menu.W, 3))

    def test_home_free_cam(self):
        m = menu.Mouse()
        click(m, next(b for b in menu.home_buttons() if b.key == "cam"))
        self.assertEqual(menu.render_home(m)[1], ("cam", None))

    def test_click_on_empty_space_does_nothing_and_is_consumed(self):
        m = menu.Mouse()
        m.clicked = (5, 5)
        self.assertIsNone(menu.render_home(m)[1])
        self.assertIsNone(m.clicked)

    def test_guide_buttons(self):
        back, cam = menu.guide_buttons()
        m = menu.Mouse()
        click(m, back)
        self.assertEqual(menu.render_guide(m, "hello")[1], ("home", None))
        click(m, cam)
        self.assertEqual(menu.render_guide(m, "where_are_you")[1], ("cam", "where_are_you"))

    def test_buttons_do_not_overlap_or_leave_the_screen(self):
        bs = menu.home_buttons()
        for b in bs:
            x, y, w, h = b.rect
            self.assertTrue(0 <= x and x + w <= menu.W and 0 <= y and y + h <= menu.H, b.key)
        for i, a in enumerate(bs):
            for b in bs[i + 1:]:
                ax, ay, aw, ah = a.rect
                bx, by, bw, bh = b.rect
                self.assertFalse(ax < bx + bw and bx < ax + aw and ay < by + bh and by < ay + ah, f"{a.key} overlaps {b.key}")


class Link:  # just what menu.recognised reads from client.Link
    def __init__(self, result=None, phrase=((), 0.0)):
        self.result, self.phrase = result, phrase


class CamViewTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        self.preview = np.full((480, 640, 3), 90, np.uint8)

    def test_canvas_has_the_side_panel_and_grows_for_a_pinned_guide(self):
        plain, _ = menu.cam_view(self.preview, menu.Mouse(), None, None)
        pinned, _ = menu.cam_view(self.preview, menu.Mouse(), "hello", None)
        self.assertEqual(plain.shape, (480, 640 + menu.PANEL_W, 3))
        self.assertEqual(pinned.shape[1], plain.shape[1])
        self.assertGreater(pinned.shape[0], plain.shape[0])

    def test_clicking_each_word_button_returns_that_word(self):
        for b in menu.cam_buttons(640):
            m = menu.Mouse()
            click(m, b)
            self.assertEqual(menu.cam_view(self.preview, m, None, None)[1], b.key)

    def test_click_on_the_preview_is_not_a_word(self):
        m = menu.Mouse()
        m.clicked = (100, 100)
        self.assertIsNone(menu.cam_view(self.preview, m, None, None)[1])

    def test_recognised_word_button_is_green(self):
        b = next(b for b in menu.cam_buttons(640) if b.key == "what")
        x, y, w, h = b.rect
        lit, _ = menu.cam_view(self.preview, menu.Mouse(), None, "what")
        off, _ = menu.cam_view(self.preview, menu.Mouse(), None, None)
        self.assertEqual(tuple(lit[y + 3, x + 3]), menu.GREEN_HOT)
        self.assertNotEqual(tuple(off[y + 3, x + 3]), menu.GREEN_HOT)

    def test_recognised(self):
        import time
        now = time.time()
        self.assertEqual(menu.recognised(Link(phrase=(("hello", "how_are_you"), now - 0.5)), now), "how_are_you")
        self.assertEqual(menu.recognised(Link({"label": "where", "confidence": 0.8}), now), "where")
        self.assertIsNone(menu.recognised(Link({"label": "where", "confidence": 0.3}), now))
        self.assertIsNone(menu.recognised(Link({"label": "none", "confidence": 0.9}), now))
        self.assertIsNone(menu.recognised(Link(phrase=(("hello",), now - 10), result=None), now))


class HandFilterTests(unittest.TestCase):
    """Only hands at the end of a detected arm, confident and mostly complete, are drawn."""

    @staticmethod
    def arm(first, ex=300, score=0.9):  # shoulder / elbow / wrist on the image
        s, e, w = (5, 7, 9) if first == 91 else (6, 8, 10)
        return [[s, ex, 200, score], [e, ex, 300, score], [w, ex, 400, score]]

    @staticmethod
    def hand(first, cx, cy, score=0.9, n=21):
        return [[first + k, cx + (k % 5) * 6, cy - (k // 5) * 8, score] for k in range(n)]

    def kept(self, pts):
        import overlay
        sm = overlay.Smoother()
        sm.update({"w": 640, "h": 480, "thr": 0.3, "pts": pts}, 1.0)
        return {i for i in sm.target if i >= 91}

    def test_hand_on_a_detected_arm_is_kept(self):
        self.assertEqual(len(self.kept(self.arm(91) + self.hand(91, 300, 400))), 21)

    def test_hand_with_no_arm_is_dropped(self):
        self.assertEqual(self.kept(self.hand(91, 300, 400)), set())

    def test_hand_far_from_the_arm_is_dropped(self):
        self.assertEqual(self.kept(self.arm(91) + self.hand(91, 80, 100)), set())

    def test_low_confidence_and_incomplete_hands_are_dropped(self):
        self.assertEqual(self.kept(self.arm(91) + self.hand(91, 300, 400, score=0.4)), set())
        self.assertEqual(self.kept(self.arm(91) + self.hand(91, 300, 400, n=8)), set())

    def test_arm_off_screen_means_no_hand(self):
        off = [[5, 300, 200, 0.9], [7, 300, 300, 0.9], [9, 300, 479, 0.9]]  # wrist clamped at the bottom edge: that side is hidden
        self.assertEqual(self.kept(off + self.hand(91, 300, 440)), set())

    def test_two_real_hands_both_kept(self):
        self.assertEqual(len(self.kept(self.arm(91, 200) + self.hand(91, 200, 400) + self.arm(112, 440) + self.hand(112, 440, 400))), 42)


if __name__ == "__main__":
    unittest.main()
