"""Phase 3 keypoint-model comparison (ONNX Runtime part + TensorRT-vs-ORT part).

  python training/phase3_eval.py ort      # detector x pose grid on ~50 clips converted to the live format
  python training/phase3_eval.py trtprep  # crops for the TensorRT comparison (raw float32 files)
  python training/phase3_eval.py trtcmp   # compares trtexec outputs with ORT on the same crops
Everything derived from the dataset stays under data/public/ (gitignored; license: no redistribution).
"""
import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np

import ort_cuda  # noqa: F401  (must come before onnxruntime users)
import rtmw_ref as R

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "client"))
from common import encode_jpeg  # noqa: E402  same JPEG encoder as the live stream

DATA = REPO / "data" / "public" / "asl_citizen"
OUT = DATA / "phase3"
CAND = REPO / "server" / "models" / "candidates"
LIVE_W, LIVE_H, LIVE_FPS, LIVE_Q = 640, 480, 10, 70
TARGET_CLASSES = ["hello", "what", "how", "why", "where", "you"]

DET = {
    "yolox_m": (CAND / "patched" / "yolox_m_topk3840.onnx", 640),  # TopK K 5000 -> 3840: TensorRT limit, same boxes (DECISIONS.md)
    "yolox_tiny": (next(CAND.glob("yolox_tiny*/**/end2end.onnx")), 416),
}
POSE = {  # name -> (onnx, (W, H) of the model input)
    "rtmw_xl_384x288": (next(CAND.glob("rtmw-dw-x-l*384x288*/end2end.onnx")), (288, 384)),
    "rtmw_lm_256x192": (next(CAND.glob("rtmw-dw-l-m*256x192*/end2end.onnx")), (192, 256)),
    "rtmw_xl_256x192": (next(CAND.glob("rtmw-dw-x-l*256x192*/end2end.onnx")), (192, 256)),
    "dwpose_l_384x288": (next(CAND.glob("rtmpose-l*384x288*/**/end2end.onnx")), (288, 384)),
    "dwpose_m_256x192": (next(CAND.glob("rtmpose-m*256x192*/**/end2end.onnx")), (192, 256)),
}


# Keypoint confidence thresholds per model. SimCC scores are not on one scale (rtmw_xl_384x288 is ~6x the others),
# so each is 0.5 x the median hand score on signing frames of the Phase 3 test clips (see DECISIONS.md).
HAND_THR = {"rtmw_xl_384x288": 1.9, "rtmw_lm_256x192": 0.3, "rtmw_xl_256x192": 0.33, "dwpose_l_384x288": 0.28, "dwpose_m_256x192": 0.27}

# Strict 'hand really visible' thresholds (+ >= 90% of points inside the frame), fitted on signing vs resting frames of all 4,826
# clips: resting-frame false positives ~2% per hand. Used for the idle gate (see signfeat.frame_row, DECISIONS.md Phase 7).
HAND_PRESENT_THR = {"rtmw_xl_384x288": 3.44, "rtmw_lm_256x192": 0.56}


def to_live_frames(path):
    """Dataset clip -> frames exactly as the server would see them: 10 fps, 640x480 (letterboxed, never squashed), JPEG q70."""
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(fps / LIVE_FPS))
    frames, i = [], 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i % step == 0:
            h, w = f.shape[:2]
            s = min(LIVE_W / w, LIVE_H / h)
            nw, nh = int(round(w * s)), int(round(h * s))
            canvas = np.zeros((LIVE_H, LIVE_W, 3), np.uint8)
            canvas[(LIVE_H - nh) // 2:(LIVE_H - nh) // 2 + nh, (LIVE_W - nw) // 2:(LIVE_W - nw) // 2 + nw] = cv2.resize(f, (nw, nh), interpolation=cv2.INTER_AREA)
            frames.append(cv2.imdecode(np.frombuffer(encode_jpeg(canvas, LIVE_Q), np.uint8), cv2.IMREAD_COLOR))
        i += 1
    cap.release()
    return frames


def pick_clips(n_per_class=6, n_other=14, seed=7):
    rows = list(csv.DictReader(open(DATA / "manifest.csv", encoding="utf-8")))
    rng = random.Random(seed)
    chosen = []
    for cls in TARGET_CLASSES + ["other"]:
        pool = sorted((r for r in rows if r["cls"] == cls and r["split"] == "test"), key=lambda r: r["file"])
        chosen += rng.sample(pool, n_other if cls == "other" else n_per_class)
    return chosen


def pct(a, q):
    return float(np.percentile(a, q)) if len(a) else float("nan")


def load_set():
    clips = pick_clips()
    data = []
    for r in clips:
        fr = to_live_frames(DATA / "videos" / r["file"])
        data.append({"cls": r["cls"], "gloss": r["gloss"], "signer": r["signer"], "file": r["file"], "frames": fr})
    print(f"{len(data)} clips, {sum(len(c['frames']) for c in data)} frames at {LIVE_W}x{LIVE_H} q{LIVE_Q} {LIVE_FPS} fps", flush=True)
    return data


def run_pose(sess, frame, bbox, wh, to_rgb):
    t0 = time.perf_counter()
    blob, c, s = R.pose_crop(frame, bbox, wh, to_rgb)
    t1 = time.perf_counter()
    sx, sy = sess.run(None, {"input": blob})
    t2 = time.perf_counter()
    kpt, score = R.simcc_decode(sx[0], sy[0], wh, c, s)
    t3 = time.perf_counter()
    return kpt, score, (t1 - t0) * 1e3, (t2 - t1) * 1e3, (t3 - t2) * 1e3


def cmd_ort():
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_set()
    det_sess = {k: ort_cuda.make_session(str(p)) for k, (p, _) in DET.items()}
    pose_sess = {k: ort_cuda.make_session(str(p)) for k, (p, _) in POSE.items()}
    full = [0, 0, LIVE_W, LIVE_H]

    # detector pass: boxes + latency per frame
    boxes = {k: [] for k in DET}
    det_ms = {k: [] for k in DET}
    for k, (p, size) in DET.items():
        for c in data:
            bl = []
            for fi, f in enumerate(c["frames"]):
                t0 = time.perf_counter()
                blob, ratio = R.yolox_preprocess(f, size)
                dets, labels = det_sess[k].run(None, {"input": blob})
                res = R.yolox_best_box(dets, labels, ratio)
                if (c is not data[0]) or fi >= 3:  # skip warm-up
                    det_ms[k].append((time.perf_counter() - t0) * 1e3)
                bl.append(res[0] if res else None)
            boxes[k].append(bl)
        found = sum(b is not None for bl in boxes[k] for b in bl)
        total = sum(len(bl) for bl in boxes[k])
        print(f"detector {k}: person found {found}/{total}  p50 {pct(det_ms[k], 50):.1f} ms p95 {pct(det_ms[k], 95):.1f} ms", flush=True)
    boxes["full_frame"] = [[full] * len(c["frames"]) for c in data]

    results = []
    combos = [(p, b, True) for p in POSE for b in ("yolox_m", "yolox_tiny", "full_frame")]
    combos += [("rtmw_xl_384x288", "yolox_m", False)]  # BGR input instead of RGB (rtmlib behaviour) as an ablation
    store = {}
    for pname, bsrc, to_rgb in combos:
        wh = POSE[pname][1]
        pre, run, dec = [], [], []
        per_clip, kp_list, sc_list = [], [], []
        for ci, c in enumerate(data):
            n = len(c["frames"])
            lo, hi = n // 4, n - n // 4  # middle half = where the signing happens
            for fi, f in enumerate(c["frames"]):
                bb = boxes[bsrc][ci][fi] if boxes[bsrc][ci][fi] is not None else full  # no person box -> full frame
                kpt, score, a, b, d = run_pose(pose_sess[pname], f, bb, wh, to_rgb)
                if ci or fi > 2:
                    pre.append(a), run.append(b), dec.append(d)
                lf, rf = R.hand_found(score[R.LEFT_HAND]), R.hand_found(score[R.RIGHT_HAND])
                per_clip.append((ci, fi, lo <= fi < hi, lf, rf, float(score[R.LEFT_HAND].mean()), float(score[R.RIGHT_HAND].mean())))
                kp_list.append(kpt), sc_list.append(score)
        store[f"{pname}|{bsrc}|{int(to_rgb)}"] = (np.stack(kp_list), np.stack(sc_list))
        pc = np.array(per_clip, dtype=np.float64)
        tgt = np.array([data[int(r[0])]["cls"] != "other" for r in pc])
        mid = pc[:, 2] == 1
        sel = tgt & mid
        any_hand = (pc[:, 3] + pc[:, 4]) > 0
        both = (pc[:, 3] + pc[:, 4]) == 2
        row = {
            "pose": pname, "boxes": bsrc, "to_rgb": to_rgb,
            "any_hand_mid": round(float(any_hand[sel].mean()), 3), "both_hands_mid": round(float(both[sel].mean()), 3),
            "any_hand_all": round(float(any_hand[tgt].mean()), 3),
            "hand_score_mid": round(float(np.maximum(pc[sel, 5], pc[sel, 6]).mean()), 3),
            "pre_ms_p50": round(pct(pre, 50), 2), "run_ms_p50": round(pct(run, 50), 2), "run_ms_p95": round(pct(run, 95), 2),
            "dec_ms_p50": round(pct(dec, 50), 2),
        }
        results.append(row)
        print(json.dumps(row), flush=True)

    # per-class "any hand found" for the main candidates
    json.dump({"results": results, "det_ms": {k: {"p50": pct(v, 50), "p95": pct(v, 95)} for k, v in det_ms.items()},
               "det_found": {k: [sum(b is not None for b in bl) for bl in boxes[k]] for k in DET}}, open(OUT / "ort_results.json", "w"), indent=1)
    # keep keypoints of the main combos for drawings
    np.savez_compressed(OUT / "ort_keypoints.npz", **{f"{k}|{part}": v[i] for k, v in store.items() for i, part in enumerate(("kpt", "score"))},
                        clip_index=np.array([(ci, fi, len(c["frames"])) for ci, c in enumerate(data) for fi in range(len(c["frames"]))]))
    json.dump([{k: c[k] for k in ("cls", "gloss", "signer", "file")} | {"n": len(c["frames"])} for c in data], open(OUT / "clips.json", "w"), indent=1)
    print("ort stage done")


def cmd_trtprep():
    """64 yolox_m crops (middle frames of target clips) per pose input size, saved as raw float32 NCHW + reference ORT outputs."""
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_set()
    sess_det = ort_cuda.make_session(str(DET["yolox_m"][0]))
    rng = random.Random(3)
    samples = []
    for c in data:
        if c["cls"] == "other":
            continue
        n = len(c["frames"])
        for fi in rng.sample(range(n // 4, n - n // 4), 2):
            samples.append(c["frames"][fi])
    samples = samples[:64]
    bboxes = []
    for f in samples:
        blob, ratio = R.yolox_preprocess(f, 640)
        d, l = sess_det.run(None, {"input": blob})
        r = R.yolox_best_box(d, l, ratio)
        bboxes.append(r[0] if r else [0, 0, LIVE_W, LIVE_H])
    meta = {}
    for wh in sorted({v[1] for v in POSE.values()}):
        crops = [R.pose_crop(f, b, wh, True) for f, b in zip(samples, bboxes)]
        blob = np.concatenate([c[0] for c in crops])
        blob.tofile(OUT / f"crops_{wh[1]}x{wh[0]}.bin")
        meta[f"{wh[1]}x{wh[0]}"] = {"n": len(crops), "shape": list(blob.shape), "centers": [c[1].tolist() for c in crops], "scales": [c[2].tolist() for c in crops]}
    json.dump(meta, open(OUT / "crops_meta.json", "w"))
    # detector inputs (first 16 frames) for the detector TRT-vs-ORT check
    for name, (p, size) in DET.items():
        blobs = [R.yolox_preprocess(f, size) for f in samples[:16]]
        for i, (b, ratio) in enumerate(blobs):
            b.tofile(OUT / f"det_{name}_{i:02d}.bin")
        json.dump([r for _, r in blobs], open(OUT / f"det_{name}_ratios.json", "w"))
    print("trtprep done:", {k: v["shape"] for k, v in meta.items()})


def read_trt_json(path):
    """trtexec --exportOutput JSON. For data-dependent shapes (NMS) it inserts a 'Final shape is AxBxC ...' note inside the list."""
    import re
    text = open(path).read()
    shapes = re.findall(r"Final shape is (\d+(?:x\d+)*) reported", text)
    j = json.loads(re.sub(r"Final shape is \S+ reported by the output allocator\.", "", text))
    out, k = {}, 0
    for o in j:
        dims = o["dimensions"]
        if "-1" in dims:
            dims, k = shapes[k], k + 1
        out[o["name"]] = np.array(o["values"], np.float32).reshape([int(x) for x in dims.split("x")])
    return out


def cmd_trtcmp():
    meta = json.load(open(OUT / "crops_meta.json"))
    z = np.load(OUT / "ort_keypoints.npz")
    rows = []
    for pname, (onnx, wh) in POSE.items():
        sc = z[f"{pname}|yolox_m|1|score"]  # per-model keypoint threshold (score scales differ, see draw stage)
        thr = 0.5 * float(np.median(np.maximum(sc[:, 91:112].mean(1), sc[:, 112:133].mean(1))))
        key = f"{wh[1]}x{wh[0]}"
        jpath = OUT / f"trt_{pname}.json"
        if not jpath.exists():
            print("missing", jpath)
            continue
        trt = read_trt_json(jpath)
        blob = np.fromfile(OUT / f"crops_{key}.bin", np.float32).reshape(meta[key]["shape"])
        sess = ort_cuda.make_session(str(onnx))
        sx, sy = sess.run(None, {"input": blob})
        err_px, hand_err_px, both = [], [], []
        agree = []
        for i in range(blob.shape[0]):
            c, s = np.array(meta[key]["centers"][i], np.float32), np.array(meta[key]["scales"][i], np.float32)
            ko, so = R.simcc_decode(sx[i], sy[i], wh, c, s)
            kt, st = R.simcc_decode(trt["simcc_x"][i], trt["simcc_y"][i], wh, c, s)
            d = np.linalg.norm(ko - kt, axis=1)  # image pixels (640x480 live frame)
            conf = so >= thr
            err_px.append(d[conf])
            hs = np.zeros(133, bool)
            hs[91:133] = conf[91:133]
            hand_err_px.append(d[hs])
            agree.append((R.hand_found(so[R.LEFT_HAND]) == R.hand_found(st[R.LEFT_HAND])) and (R.hand_found(so[R.RIGHT_HAND]) == R.hand_found(st[R.RIGHT_HAND])))
        e, h = np.concatenate(err_px), np.concatenate(hand_err_px)
        rows.append({"pose": pname, "crops": blob.shape[0], "kpt_thr": round(thr, 3), "kpts_compared": int(len(e)), "frac_over_5px": float((e > 5).mean()),
                     "hand_frac_over_5px": float((h > 5).mean()) if len(h) else None, "simcc_max_abs_diff": float(max(np.abs(sx - trt["simcc_x"]).max(), np.abs(sy - trt["simcc_y"]).max())),
                     "kpt_err_px_mean": float(e.mean()), "kpt_err_px_p95": pct(e, 95), "kpt_err_px_max": float(e.max()),
                     "hand_err_px_mean": float(h.mean()) if len(h) else None, "hand_err_px_p95": pct(h, 95), "hand_err_px_max": float(h.max()) if len(h) else None,
                     "hand_found_agree": float(np.mean(agree))})
        print(json.dumps(rows[-1]), flush=True)
    json.dump(rows, open(OUT / "trt_vs_ort.json", "w"), indent=1)


def cmd_draw():
    """Contact sheets of keypoint drawings (middle frame of one test clip per target class) for several model combos."""
    z = np.load(OUT / "ort_keypoints.npz")
    clips = json.load(open(OUT / "clips.json"))
    idx = z["clip_index"]
    combos = ["rtmw_xl_384x288|yolox_m|1", "rtmw_lm_256x192|yolox_m|1", "rtmw_xl_256x192|full_frame|1", "dwpose_l_384x288|yolox_m|1"]
    mid_p50 = {}
    for k in combos:  # per-model keypoint threshold: score scales differ (xl_384 is ~6x the others)
        s = z[k + "|score"]
        h = np.maximum(s[:, 91:112].mean(1), s[:, 112:133].mean(1))
        mid_p50[k] = 0.5 * float(np.median(h))
    (OUT / "drawings").mkdir(exist_ok=True)
    for cls in TARGET_CLASSES:
        ci = next(i for i, c in enumerate(clips) if c["cls"] == cls)
        n = clips[ci]["n"]
        frames = to_live_frames(DATA / "videos" / clips[ci]["file"])
        rows = []
        for k in combos:
            cells = []
            for fi in (n // 3, n // 2, 2 * n // 3):
                row = int(np.where((idx[:, 0] == ci) & (idx[:, 1] == fi))[0][0])
                img = R.draw(frames[fi], z[k + "|kpt"][row], z[k + "|score"][row], thr=mid_p50[k])
                cv2.putText(img, k.split("|")[0], (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cells.append(img)
            rows.append(np.hstack(cells))
        sheet = np.vstack(rows)
        cv2.imwrite(str(OUT / "drawings" / f"{cls}.jpg"), cv2.resize(sheet, None, fx=0.6, fy=0.6, interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print("drawings in", OUT / "drawings")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["ort", "trtprep", "trtcmp", "draw"])
    {"ort": cmd_ort, "trtprep": cmd_trtprep, "trtcmp": cmd_trtcmp, "draw": cmd_draw}[ap.parse_args().stage]()
