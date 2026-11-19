"""Keypoint parity: C++ pipeline (nvJPEG + fused kernel + TensorRT FP16 + GPU decode) vs the Python reference
(cv2 decode + rtmw_ref.pose_crop + ONNX Runtime FP32 + simcc_decode) on test-split clips in the live format.

  training\.venv\Scripts\python server\cpp\tests\parity_keypoints.py [--clips 24] [--frames 3] [--profile primary]
Pixel errors are in live-frame (640x480) pixels. Needs a built signlang_server.exe; does not need the server to be running.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "training"))
import ort_cuda  # noqa: E402
import phase3_eval as E  # noqa: E402
import rtmw_ref as R  # noqa: E402
import sign_data as D  # noqa: E402
import signfeat as S  # noqa: E402

sys.path.insert(0, str(ROOT / "client"))
from common import encode_jpeg  # noqa: E402

EXE = ROOT / "server" / "cpp" / "build" / "release" / "signlang_server.exe"
CFG = json.loads((ROOT / "server" / "config" / "server.json").read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", type=int, default=24)
    ap.add_argument("--frames", type=int, default=3)
    ap.add_argument("--profile", default="primary")
    ap.add_argument("--model", default="rtmw_xl_384x288", help="keypoint cache name used to pick test clips")
    a = ap.parse_args()
    prof = CFG["models"][a.profile]
    wh, thr, present = tuple(prof["keypoint_input_wh"]), prof["keypoint_thr"], prof["hand_present_thr"]

    d = D.load(a.model)
    rows = list(__import__("csv").DictReader(open(E.DATA / "manifest.csv", encoding="utf-8")))
    rng = np.random.default_rng(11)
    pool = np.where(d["split"] == "test")[0]
    clips = rng.choice(pool, a.clips, replace=False)

    tmp = Path(tempfile.mkdtemp(prefix="parity_"))
    jpgs = tmp / "jpg"
    jpgs.mkdir()
    n = 0
    for c in clips:
        frames = E.to_live_frames(E.DATA / "videos" / rows[c]["file"])
        for f in np.linspace(len(frames) * 0.2, len(frames) * 0.8, a.frames).astype(int):
            (jpgs / f"{n:04d}_{rows[c]['gloss']}_{f:03d}.jpg").write_bytes(encode_jpeg(frames[f], E.LIVE_Q))
            n += 1
    print(f"{a.clips} test clips -> {n} live-format frames in {jpgs}")

    out_json, blob_f32 = tmp / "cpp.json", tmp / "blob.f32"
    env = {**os.environ, "PATH": r"C:\dev\TensorRT-10.16.1.11\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin\x64;" + os.environ["PATH"]}
    r = subprocess.run([str(EXE), "--profile", a.profile, "--extract", str(jpgs), str(out_json), "--blob", str(blob_f32)], env=env, capture_output=True, text=True)
    if r.returncode:
        sys.exit(f"C++ extract failed:\n{r.stdout}\n{r.stderr}")
    cpp = json.loads(out_json.read_text())

    sess = ort_cuda.make_session(str(ROOT / prof["keypoint"]))
    err_all, err_conf, err_sel, flag_diff, flag_n = [], [], [], 0, 0
    low_used = used_total = 0  # model points whose score is below the threshold (they are NOT masked)
    big = []  # (error px, keypoint index, python score, C++ score, used by the model?, error in shoulder widths)
    first_blob_diff = None
    for i, item in enumerate(cpp):
        img = cv2.imdecode(np.frombuffer((jpgs / item["name"]).read_bytes(), np.uint8), cv2.IMREAD_COLOR)
        h, w = img.shape[:2]
        assert (item["w"], item["h"]) == (w, h)
        blob, centre, scale = R.pose_crop(img, [0, 0, w, h], wh, True)
        sx, sy = sess.run(None, {"input": blob})
        kp, sc = R.simcc_decode(sx[0], sy[0], wh, centre, scale)
        ck = np.array(item["kpt"], np.float32).reshape(133, 3)
        e = np.linalg.norm(ck[:, :2] - kp, axis=1)
        conf = sc >= thr
        err_all.append(e), err_conf.append(e[conf]), err_sel.append(e[S.SEL])
        fr_py, fl_py = S.frame_row(kp, sc, thr, present, (w, h))
        fr_cp, fl_cp = S.frame_row(ck[:, :2], ck[:, 2], thr, present, (w, h))
        for k in S.SEL:
            if k < 91 or (fl_py[0] if k < 112 else fl_py[1]):
                used_total += 1
                low_used += int(sc[k] < thr)
        shoulder = float(np.linalg.norm(kp[5] - kp[6])) or 1.0
        for k in np.where(e > 20)[0]:
            used = k in S.SEL and (k < 91 or (fl_py[0] if k < 112 else fl_py[1]))  # hand points only count when that hand is "present"
            big.append((float(e[k]), int(k), float(sc[k]), float(ck[k, 2]), bool(used), float(e[k]) / shoulder))
        flag_diff += int((fl_py != fl_cp).sum())
        flag_n += len(fl_py)
        if i == 0:
            cb = np.fromfile(blob_f32, np.float32).reshape(blob.shape)
            first_blob_diff = np.abs(cb - blob) * np.array([58.395, 57.12, 57.375], np.float32).reshape(1, 3, 1, 1)  # in grey levels

    def stats(name, parts):
        v = np.concatenate(parts)
        print(f"  {name:<40} n={len(v):6d}  mean {v.mean():.3f}  p50 {np.percentile(v, 50):.3f}  p95 {np.percentile(v, 95):.3f}  p99 {np.percentile(v, 99):.3f}  max {v.max():.2f} px")

    print(f"\nkeypoint pixel error, C++ (TensorRT FP16) vs Python (ORT FP32), {len(cpp)} frames, profile {a.profile}:")
    stats("all 133 keypoints", err_all)
    stats(f"confident ones (score >= {thr})", err_conf)
    stats("the 57 points the temporal model uses", err_sel)
    print(f"  flags (5 per frame) that differ: {flag_diff} of {flag_n}")
    print(f"  first-frame network input |C++ - Python| in grey levels: mean {first_blob_diff.mean():.3f}, p99 {np.percentile(first_blob_diff, 99):.2f}, max {first_blob_diff.max():.1f}"
          " (includes nvJPEG vs libjpeg decode differences)")
    n_pts = len(cpp) * 133
    grp = lambda k: "body" if k < 17 else ("feet" if k < 23 else ("face" if k < 91 else ("left hand" if k < 112 else "right hand")))  # noqa: E731
    print(f"\nlarge errors (> 20 px): {len(big)} of {n_pts} points ({100 * len(big) / n_pts:.2f}%)")
    used = [b for b in big if b[4]]
    print(f"  of which fed to the model (in the 57 points, hand present): {len(used)} ({100 * len(used) / (len(cpp) * 57):.2f}% of the model's points)")
    if big:
        from collections import Counter
        print("  by group:", dict(Counter(grp(b[1]) for b in big)))
        print(f"  Python score of those points: median {np.median([b[2] for b in big]):.2f}, max {max(b[2] for b in big):.2f}  (threshold {thr})")
        print(f"  C++ score of those points:    median {np.median([b[3] for b in big]):.2f}, max {max(b[3] for b in big):.2f}")
    if used:
        print(f"  used ones, error in shoulder widths (features are in these units): median {np.median([b[5] for b in used]):.2f}, max {max(b[5] for b in used):.2f}")
        for b in sorted(used, reverse=True)[:6]:
            print(f"    {grp(b[1]):>10} point {b[1]:3d}: {b[0]:6.1f} px = {b[5]:.2f} shoulder widths, score py {b[2]:.2f} / C++ {b[3]:.2f}")
    print(f"  model points that reach the temporal model with a score below the threshold (not masked): {low_used} of {used_total} ({100 * low_used / max(used_total, 1):.1f}%)")
    ok = np.percentile(np.concatenate(err_conf), 95) < 2.0
    print("PARITY", "OK" if ok else "CHECK")


if __name__ == "__main__":
    main()
