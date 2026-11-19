"""Writes tests/data/golden.json: reference outputs of the Python code the C++ port must reproduce.
  training\.venv\Scripts\python server\cpp\tests\gen_golden.py
Inputs are rounded to 2 decimals first, so Python and C++ start from identical float32 values."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "training"))
import rtmw_ref as R  # noqa: E402
import signfeat as S  # noqa: E402
import torch  # noqa: E402

rng = np.random.default_rng(7)
r4 = lambda a: np.round(np.asarray(a, np.float64), 2).astype(np.float32)  # noqa: E731
out = {}
short = lambda a, d: np.round(np.asarray(a, np.float64), d).ravel().tolist()  # noqa: E731  (short JSON numbers, exact after rounding)

# 1. features: 30 frames of raw RTMW output -> frame_row (flags) -> window_features
def feature_case(shoulders_ok_frames, hand_l_frames, hand_r_frames):
    kpts, scores, rows, flags = [], [], [], []
    for t in range(S.WINDOW):
        k = r4(rng.uniform(0, 640, (133, 2)) * [1, 0.75])
        k[5] = r4([300 + t, 200 + 0.5 * t]); k[6] = r4([380 - 0.3 * t, 205])
        sc = r4(rng.uniform(0, 8, 133))
        sc[5:7] = 5.0 if shoulders_ok_frames(t) else 0.5
        sc[91:112] = 5.0 if hand_l_frames(t) else 0.1
        sc[112:133] = 5.0 if hand_r_frames(t) else 0.1
        row, fl = S.frame_row(k, sc, 1.9, 3.44, (640, 480))
        kpts.append(k), scores.append(sc), rows.append(row), flags.append(fl)
    w = S.window_features(torch.from_numpy(np.stack(rows))[None], torch.from_numpy(np.stack(flags))[None])[0].numpy()
    return {"kpt": [short(k, 2) for k in kpts], "score": [short(s, 2) for s in scores],
            "flags": [f.astype(int).tolist() for f in flags], "features": short(w, 5)}

out["features"] = [
    feature_case(lambda t: True, lambda t: t > 5, lambda t: t % 3 != 0),        # both hands, flickering right hand
    feature_case(lambda t: t % 4 != 0, lambda t: t < 12, lambda t: False),      # some frames without shoulders
    feature_case(lambda t: False, lambda t: True, lambda t: True),              # no reliable shoulders: default centre/scale
]

# 2. geometry + preprocess: cv2 reference on a small smooth image
W, H, IW, IH = 64, 48, 24, 32
img = cv2.GaussianBlur(rng.integers(0, 255, (H, W, 3), dtype=np.uint8), (0, 0), 2)
blob, centre, scale = R.pose_crop(img, [0, 0, W, H], (IW, IH), True)
out["preprocess"] = {"w": W, "h": H, "in_w": IW, "in_h": IH, "img": img.ravel().tolist(), "blob": short(blob, 5),
                     "center": centre.tolist(), "scale": scale.tolist()}

# 3. SimCC decode (K=8) with a crop for that geometry
K, NX, NY = 8, IW * 2, IH * 2
sx, sy = r4(rng.normal(0, 1, (K, NX))), r4(rng.normal(0, 1, (K, NY)))
sx[2], sy[2] = -1.0, -1.0  # score <= 0 -> keypoint -1
sx[3, 10], sx[3, 20] = 5.0, 5.0  # tie: the lower index must win
kpt, score = R.simcc_decode(sx, sy, (IW, IH), centre, scale)
out["simcc"] = {"k": K, "nx": NX, "ny": NY, "sx": short(sx, 2), "sy": short(sy, 2),
                "kpt": short(kpt, 4), "score": short(score, 5)}

p = Path(__file__).parent / "data" / "golden.json"
p.write_text(json.dumps(out, separators=(",", ":")))
print("wrote", p, p.stat().st_size // 1024, "KB")
