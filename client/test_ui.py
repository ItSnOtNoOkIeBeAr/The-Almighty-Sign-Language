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
import textfx  # noqa: E402

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
        self.assertEqual(tuple(lit[y + h // 2, x + 12]), menu.GREEN_HOT)
        self.assertNotEqual(tuple(off[y + h // 2, x + 12]), menu.GREEN_HOT)

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

    FACE = [[0, 320, 150, 0.9], [1, 300, 140, 0.9], [2, 340, 140, 0.9], [3, 270, 150, 0.9], [4, 370, 150, 0.9]]  # ear to ear = 100 px

    def test_raised_hand_with_the_arm_out_of_frame_is_kept_near_the_face(self):
        off = [[5, 300, 200, 0.9], [7, 300, 300, 0.9], [9, 300, 479, 0.9]]  # forearm leaves the picture: that arm side is hidden
        self.assertEqual(len(self.kept(self.FACE + off + self.hand(91, 330, 330))), 21)

    def test_hand_far_from_the_face_and_with_no_arm_is_dropped(self):
        small = [[0, 320, 150, 0.9], [1, 310, 145, 0.9], [2, 330, 145, 0.9], [3, 300, 150, 0.9], [4, 340, 150, 0.9]]  # ears 40 px apart: reach 200 px
        self.assertEqual(self.kept(small + self.hand(91, 40, 440)), set())  # ~400 px away = 10 face widths

    def test_hand_without_face_or_arm_is_dropped(self):
        self.assertEqual(self.kept(self.hand(91, 330, 330)), set())

    def test_two_real_hands_both_kept(self):
        self.assertEqual(len(self.kept(self.arm(91, 200) + self.hand(91, 200, 400) + self.arm(112, 440) + self.hand(112, 440, 400))), 42)


class TextFxTests(unittest.TestCase):
    def blank(self):
        import numpy as np
        return np.full((120, 300, 3), 40, np.uint8)

    def test_text_is_drawn_inside_its_size_and_nowhere_else(self):
        import numpy as np
        img = self.blank()
        textfx.draw_text(img, "Hello", (20, 80), 1.0, (255, 255, 255), 2)
        w, h = textfx.text_size("Hello", 1.0, 2)
        ys, xs = np.nonzero((img != 40).any(axis=2))
        self.assertTrue(len(xs) > 100)
        self.assertGreaterEqual(xs.min(), 20 - 6)
        self.assertLessEqual(xs.max(), 20 + w + 6)
        self.assertLessEqual(ys.max(), 80 + 12)            # descenders only a little below the baseline
        self.assertGreaterEqual(ys.min(), 80 - h - 8)

    def test_bigger_scale_is_bigger(self):
        self.assertGreater(textfx.text_size("Hello", 2.0)[0], textfx.text_size("Hello", 1.0)[0])
        self.assertGreater(textfx.text_size("Hello again", 1.0)[0], textfx.text_size("Hello", 1.0)[0])

    def test_drawing_past_the_edges_does_not_crash(self):
        img = self.blank()
        for org in ((-50, 50), (280, 50), (20, -10), (20, 400), (-500, -500)):
            textfx.draw_text(img, "Edge", org, 1.0)
            textfx.rounded_box(img, (org[0], org[1], 80, 40), (0, 0, 255), 0.5)

    def test_falls_back_to_opencv_without_a_font(self):
        import numpy as np
        saved = textfx._font
        textfx._font = lambda px, bold: None
        try:
            self.assertFalse(textfx.available())
            img = self.blank()
            textfx.draw_text(img, "Hi", (20, 80), 1.0)
            self.assertTrue((img != 40).any())
            self.assertGreater(textfx.text_size("Hi", 1.0)[0], 0)
        finally:
            textfx._font = saved

    def test_rounded_box_blends_and_rounds(self):
        img = self.blank()
        textfx.rounded_box(img, (50, 30, 100, 60), (200, 200, 200), 0.5, radius=20, border=(0, 0, 255))
        self.assertTrue(abs(int(img[60, 100, 0]) - 120) <= 1)       # 40 + (200 - 40) / 2 in the middle
        self.assertEqual(tuple(img[30, 50]), (40, 40, 40))           # the corner is cut off
        self.assertEqual(tuple(img[30, 100]), (0, 0, 255))           # border on the straight top side
        self.assertEqual(tuple(img[10, 10]), (40, 40, 40))           # nothing outside the box


if __name__ == "__main__":
    unittest.main()
