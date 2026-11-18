"""Keypoints -> temporal-model features. Used by training AND the fallback server, so both see identical numbers
(the C++ server must reproduce this, Phase 6).

Per frame the keypoint stage gives 133 RTMW keypoints (x, y in live-frame pixels, score). We keep 57 points:
  0-1 shoulders, 2-5 elbows+wrists, 6 nose, 7-14 eyebrow/mouth points, 15-35 left hand, 36-56 right hand
plus 5 flags per frame: left/right hand present (loose: keypoints are kept), shoulders reliable, left/right hand present (strict: really
visible; drives the idle gate). The loose rule alone calls ~75% of resting frames 'hand present' (hallucinated points).
A window of WINDOW frames (3 s at 10 fps) is normalised by its own median mid-shoulder and shoulder width, then
features = normalised xy (114) + frame-to-frame velocity (114) + the 5 flags = 233 per frame.
"""
import torch

WINDOW = 30
N_PTS = 57
N_FLAGS = 5
F_DIM = N_PTS * 2 * 2 + N_FLAGS
BODY = [5, 6, 7, 8, 9, 10]
NOSE = [0]
# dlib-68 numbering inside the 68 face points that start at index 23: brows 19, 24, 21, 22; mouth corners 48, 54, top 51, bottom 57
FACE = [23 + i for i in (19, 24, 21, 22, 48, 54, 51, 57)]
LHAND = list(range(91, 112))
RHAND = list(range(112, 133))
SEL = BODY + NOSE + FACE + LHAND + RHAND
L0, R0 = 15, 36  # first hand point inside the 57
DEFAULT_CENTER, DEFAULT_SCALE = (320.0, 300.0), 170.0  # used only when no frame of a window has reliable shoulders


def frame_row(kpt, score, thr, present_thr, size=(640, 480)):
    """kpt [133,2], score [133] (numpy) -> pts [57,2] float32, flags [5] bool.
    thr: loose per-model keypoint threshold (HAND_THR); present_thr: strict one (HAND_PRESENT_THR), which also needs >= 90% of
    the hand's points inside the frame (a resting hand below the frame gets confident points clamped at the bottom edge)."""
    import numpy as np
    k = np.asarray(kpt, np.float32)
    sc = np.asarray(score, np.float32)

    def hand(idx, t, inside_needed):
        ok = int((sc[idx] >= t).sum()) >= 15
        if ok and inside_needed:
            h = k[idx]
            ok = float(((h[:, 0] >= 0) & (h[:, 0] < size[0]) & (h[:, 1] >= 0) & (h[:, 1] < size[1] - 20)).mean()) >= 0.9
        return ok

    flags = [hand(LHAND, thr, False), hand(RHAND, thr, False), bool(sc[5] >= thr and sc[6] >= thr),
             hand(LHAND, present_thr, True), hand(RHAND, present_thr, True)]
    return k[SEL], np.array(flags, bool)


def normalize_window(pts, flags):
    """pts [B,T,57,2] raw pixels, flags [B,T,3] bool -> normalised pts [B,T,57,2] (hands zeroed when absent)."""
    ok = flags[..., 2]
    mid = 0.5 * (pts[:, :, 0] + pts[:, :, 1])                                  # [B,T,2]
    width = (pts[:, :, 0] - pts[:, :, 1]).norm(dim=-1)                         # [B,T]
    nan = torch.tensor(float("nan"), device=pts.device)
    c = torch.where(ok[..., None], mid, nan).nanmedian(dim=1).values           # [B,2]
    s = torch.where(ok, width, nan).nanmedian(dim=1).values                    # [B]
    c = torch.where(torch.isnan(c), torch.tensor(DEFAULT_CENTER, device=pts.device), c)
    s = torch.where(torch.isnan(s) | (s < 20), torch.tensor(DEFAULT_SCALE, device=pts.device), s)
    out = (pts - c[:, None, None]) / s[:, None, None, None]
    hand = torch.ones_like(out[..., 0], dtype=torch.bool)
    hand[:, :, L0:R0] = flags[..., 0:1]
    hand[:, :, R0:] = flags[..., 1:2]
    return out * hand[..., None]


def to_features(norm, flags):
    """norm [B,T,57,2], flags [B,T,3] -> [B,T,F_DIM] float32."""
    vel = torch.zeros_like(norm)
    vel[:, 1:] = norm[:, 1:] - norm[:, :-1]
    both = torch.ones_like(norm[..., 0], dtype=torch.bool)  # velocity only where the point exists in both frames
    both[:, 1:, L0:R0] = flags[:, 1:, 0:1] & flags[:, :-1, 0:1]
    both[:, 1:, R0:] = flags[:, 1:, 1:2] & flags[:, :-1, 1:2]
    vel = vel * both[..., None]
    B, T = norm.shape[:2]
    return torch.cat([norm.reshape(B, T, -1), vel.reshape(B, T, -1), flags.float()], dim=-1)


def window_features(pts, flags):
    return to_features(normalize_window(pts, flags), flags)
