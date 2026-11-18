"""Phase 7 step 1: ASL Citizen clips -> live format -> RTMW keypoints (primary AND fallback model), cached to disk.

  python training/extract_keypoints.py            # all clips in data/public/asl_citizen/manifest.csv
Each clip becomes 10 fps, 640x480, JPEG q70 frames (same encoder as the client), full-frame box, then
rtmw_xl_384x288 (primary) and rtmw_lm_256x192 (fallback) with the exact pre/post-processing of rtmw_ref.py.
Output (gitignored, license): data/public/asl_citizen/keypoints/shard_XXX.npz + index.csv.
The C++ `--extract` parity check against these files comes with Phase 6.
"""
import csv
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

import ort_cuda
import phase3_eval as E
import rtmw_ref as R

MODELS = ["rtmw_xl_384x288", "rtmw_lm_256x192"]  # primary, fallback
OUT = E.DATA / "keypoints"
SHARD = 500
FULL = [0, 0, E.LIVE_W, E.LIVE_H]


def extract(frames, sess, wh):
    crops = [R.pose_crop(f, FULL, wh, True) for f in frames]
    blob = np.concatenate([c[0] for c in crops])
    sx, sy = sess.run(None, {"input": blob})
    kp, sc = [], []
    for i, (_, c, s) in enumerate(crops):
        k, v = R.simcc_decode(sx[i], sy[i], wh, c, s)
        kp.append(k), sc.append(v)
    return np.stack(kp).astype(np.float32), np.stack(sc).astype(np.float16)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(E.DATA / "manifest.csv", encoding="utf-8")))
    sess = {m: ort_cuda.make_session(str(E.POSE[m][0])) for m in MODELS}
    index, t0 = [], time.time()
    def prefetch(pool, rows, ahead=8):
        """Decode + JPEG-encode up to `ahead` clips in worker threads while the GPU works (a plain map would hold every clip in RAM)."""
        futs = deque()
        for r in rows:
            futs.append(pool.submit(E.to_live_frames, E.DATA / "videos" / r["file"]))
            if len(futs) >= ahead:
                yield futs.popleft().result()
        while futs:
            yield futs.popleft().result()

    with ThreadPoolExecutor(4) as pool:
        frames_iter = prefetch(pool, rows)
        buf = {m: ([], []) for m in MODELS}
        ids, fidx, shard_no = [], [], 0
        for ci, (r, frames) in enumerate(zip(rows, frames_iter)):
            index.append({**r, "n_frames": len(frames)})
            if frames:
                for m in MODELS:
                    k, s = extract(frames, sess[m], E.POSE[m][1])
                    buf[m][0].append(k), buf[m][1].append(s)
                ids += [ci] * len(frames)
                fidx += list(range(len(frames)))
            if (ci + 1) % SHARD == 0 or ci + 1 == len(rows):
                np.savez(OUT / f"shard_{shard_no:03d}.npz", clip=np.array(ids), frame=np.array(fidx),
                         **{f"{m}_kpt": np.concatenate(buf[m][0]) for m in MODELS}, **{f"{m}_score": np.concatenate(buf[m][1]) for m in MODELS})
                buf, ids, fidx, shard_no = {m: ([], []) for m in MODELS}, [], [], shard_no + 1
                el = time.time() - t0
                print(f"{ci + 1}/{len(rows)} clips  {el / 60:.1f} min  eta {el / (ci + 1) * (len(rows) - ci - 1) / 60:.1f} min", flush=True)
    with open(OUT / "index.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, list(index[0]))
        w.writeheader()
        w.writerows(index)
    print("done", sum(i["n_frames"] for i in index), "frames,", sum(i["n_frames"] == 0 for i in index), "clips without frames", flush=True)


if __name__ == "__main__":
    main()
