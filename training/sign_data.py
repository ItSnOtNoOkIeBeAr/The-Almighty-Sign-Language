"""Phase 7 data: cached ASL Citizen keypoints -> GPU-resident clips -> 30-frame windows (training and evaluation).

Class ids come from protocol/labels.json (0 = none). Manifest class `other` (150 other glosses) is `none`.
Every clip has an "active segment" (first to last run of frames where a hand is found); frames before and after it are the
signer's resting pose, which become extra `none` windows. Windows are sampled by nearest frame with clamping at the segment
ends, i.e. a sign shorter than 3 s is padded by holding the resting frames, like the live stream before and after a sign.
"""
import csv
import json
from pathlib import Path

import numpy as np
import torch

import phase3_eval as E
import signfeat as S

ROOT = Path(__file__).resolve().parent.parent
KP_DIR = E.DATA / "keypoints"
LABELS = json.loads((ROOT / "protocol" / "labels.json").read_text())["labels"]
CLS_ID = {"other": 0, **{n: i for i, n in enumerate(LABELS) if i}}


def active_segment(flags, min_run=2):
    """flags [T,3] -> (a0, a1, found) first/last frame of the first/last run of >= min_run frames with a hand found."""
    h = flags[:, 3:5].any(1)  # strict presence
    runs, i, T = [], 0, len(h)
    while i < T:
        if h[i]:
            j = i
            while j + 1 < T and h[j + 1]:
                j += 1
            if j - i + 1 >= min_run:
                runs.append((i, j))
            i = j + 1
        else:
            i += 1
    if not runs:
        return T // 4, max(T // 4, 3 * T // 4 - 1), False  # no hand ever found: assume the middle half
    return runs[0][0], runs[-1][1], True


def load(model):
    """-> dict of numpy arrays for one keypoint model ('rtmw_xl_384x288' or 'rtmw_lm_256x192')."""
    thr, present = E.HAND_THR[model], E.HAND_PRESENT_THR[model]
    rows = list(csv.DictReader(open(KP_DIR / "index.csv", encoding="utf-8")))
    shards = sorted(KP_DIR.glob("shard_*.npz"))
    clip, kpt, sc = [], [], []
    for p in shards:
        z = np.load(p)
        clip.append(z["clip"]), kpt.append(z[f"{model}_kpt"]), sc.append(z[f"{model}_score"])
    clip, kpt, sc = np.concatenate(clip), np.concatenate(kpt), np.concatenate(sc)
    n = len(rows)
    starts = np.searchsorted(clip, np.arange(n), "left")
    ends = np.searchsorted(clip, np.arange(n), "right")
    pts = np.zeros((len(clip), S.N_PTS, 2), np.float32)
    flg = np.zeros((len(clip), S.N_FLAGS), bool)
    for i in range(len(clip)):
        pts[i], flg[i] = S.frame_row(kpt[i], sc[i], thr, present)
    seg = np.array([active_segment(flg[starts[c]:ends[c]]) if ends[c] > starts[c] else (0, 0, False) for c in range(n)])
    return {
        "pts": pts, "flg": flg, "start": starts, "len": ends - starts, "seg": seg[:, :2], "seg_found": seg[:, 2].astype(bool),
        "cls": np.array([CLS_ID[r["cls"]] for r in rows]), "split": np.array([r["split"] for r in rows]),
        "signer": np.array([r["signer"] for r in rows]), "gloss": np.array([r["gloss"] for r in rows]),
    }


class Bank:
    """All frames on the GPU; builds [B,30,57,2] windows for chosen clips."""

    def __init__(self, d, device="cuda"):
        self.d, self.dev = d, device
        self.pts = torch.from_numpy(d["pts"]).to(device)
        self.flg = torch.from_numpy(d["flg"]).to(device)
        self.start = torch.from_numpy(d["start"]).to(device)
        self.len = torch.from_numpy(d["len"]).to(device)

    def windows(self, clips, t0, rate, lo=None, hi=None):
        """clips [B] long, t0 [B] float (clip frame index of window frame 0), rate [B] float frames per step,
        lo/hi [B] clamp range inside the clip (default: whole clip)."""
        clips = torch.as_tensor(clips, device=self.dev)
        n = self.len[clips]
        lo = torch.zeros_like(n) if lo is None else torch.as_tensor(lo, device=self.dev)
        hi = (n - 1) if hi is None else torch.as_tensor(hi, device=self.dev)
        tau = torch.as_tensor(t0, device=self.dev, dtype=torch.float32)[:, None] + torch.arange(S.WINDOW, device=self.dev)[None] * torch.as_tensor(rate, device=self.dev, dtype=torch.float32)[:, None]
        idx = torch.minimum(torch.maximum(tau.round().long(), lo[:, None]), hi[:, None])
        g = self.start[clips][:, None] + idx
        return self.pts[g], self.flg[g]

    def windows2(self, clips, t0, rate, lo, hi, clip2, t02, k):
        """Like windows(), but frames before index k come from a second clip (an earlier sign); clip2 < 0 means no second clip.
        Used for training windows that hold two consecutive signs (the label is the later one)."""
        p, f = self.windows(clips, t0, rate, lo, hi)
        m = torch.as_tensor(np.asarray(clip2) >= 0, device=self.dev)
        if m.any():
            i = torch.where(m)[0]
            sel = lambda a: torch.as_tensor(np.asarray(a), device=self.dev)[i]  # noqa: E731
            p2, f2 = self.windows(sel(clip2), sel(t02), sel(rate))
            first = torch.arange(S.WINDOW, device=self.dev)[None] < sel(k)[:, None]
            p[i] = torch.where(first[..., None, None], p2, p[i])
            f[i] = torch.where(first[..., None], f2, f[i])
        return p, f
