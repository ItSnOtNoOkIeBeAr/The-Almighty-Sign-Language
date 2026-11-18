"""Reference (numpy/OpenCV) pre- and post-processing for the RTMW/DWPose keypoint stage.

Exact steps come from each model's own `pipeline.json` (mmdeploy export) and were cross-checked against rtmlib 0.0.16:
  detector (YOLOX, NMS inside the ONNX): keep-ratio resize, pad to square with 114 at bottom/right, BGR, mean 0 / std 1.
  pose (RTMW / DWPose): bbox -> center+scale (padding 1.25), fix aspect to the model input, affine warp,
                        (x - mean) / std with ImageNet mean/std in 0..255 units. pipeline.json says to_rgb=true.
  decode: SimCC argmax per axis / 2 (split ratio), mapped back through center+scale.
This file is also the CPU reference the Phase 5 CUDA kernels must match.
"""
import cv2
import numpy as np

MEAN = np.array([123.675, 116.28, 103.53], np.float32)
STD = np.array([58.395, 57.12, 57.375], np.float32)
# COCO-WholeBody 133: body 0-16, feet 17-22, face 23-90, left hand 91-111, right hand 112-132
LEFT_HAND, RIGHT_HAND = slice(91, 112), slice(112, 133)


def yolox_preprocess(img_bgr, size):
    """-> NCHW float32 blob (0..255, BGR) and the resize ratio."""
    h, w = img_bgr.shape[:2]
    ratio = min(size / h, size / w)
    nh, nw = int(h * ratio + 0.5), int(w * ratio + 0.5)
    canvas = np.full((size, size, 3), 114, np.uint8)
    canvas[:nh, :nw] = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return canvas.transpose(2, 0, 1)[None].astype(np.float32), ratio


def yolox_best_box(dets, labels, ratio, score_thr=0.5):
    """Highest-score person (label 0) box in original-image pixels, or None. dets: [1,N,5] x1 y1 x2 y2 score."""
    d, lab = dets[0], labels[0]
    keep = (d[:, 4] >= score_thr) & (lab == 0)
    if not keep.any():
        return None
    best = d[keep][np.argmax(d[keep][:, 4])]
    return best[:4] / ratio, float(best[4])


def pose_crop(img_bgr, bbox_xyxy, input_wh, to_rgb):
    """Top-down affine crop. Returns NCHW float32 blob, center, scale (scale already aspect-fixed)."""
    w, h = input_wh
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    center = np.array([(x1 + x2) / 2, (y1 + y2) / 2], np.float32)
    scale = np.array([x2 - x1, y2 - y1], np.float32) * 1.25
    aspect = w / h
    scale = np.array([scale[0], scale[0] / aspect] if scale[0] > scale[1] * aspect else [scale[1] * aspect, scale[1]], np.float32)
    # 3-point affine as in mmpose: center, a point above the center, and its 90-degree rotation
    src = np.zeros((3, 2), np.float32)
    src[0] = center
    src[1] = center + np.array([0.0, scale[0] * -0.5], np.float32)
    src[2] = src[1] + np.array([-(src[0] - src[1])[1], (src[0] - src[1])[0]], np.float32)
    dst = np.zeros((3, 2), np.float32)
    dst[0] = [w * 0.5, h * 0.5]
    dst[1] = dst[0] + np.array([0.0, w * -0.5], np.float32)
    dst[2] = dst[1] + np.array([-(dst[0] - dst[1])[1], (dst[0] - dst[1])[0]], np.float32)
    m = cv2.getAffineTransform(src, dst)
    crop = cv2.warpAffine(img_bgr, m, (w, h), flags=cv2.INTER_LINEAR)
    if to_rgb:
        crop = crop[:, :, ::-1]
    blob = ((crop.astype(np.float32) - MEAN) / STD).transpose(2, 0, 1)[None]
    return np.ascontiguousarray(blob), center, scale


def simcc_decode(simcc_x, simcc_y, input_wh, center, scale):
    """simcc_x [K, 2W], simcc_y [K, 2H] -> keypoints [K,2] in image pixels, score [K] (mean of the two peaks)."""
    xs, ys = simcc_x.argmax(-1), simcc_y.argmax(-1)
    score = 0.5 * (simcc_x.max(-1) + simcc_y.max(-1))
    kpt = np.stack([xs, ys], -1).astype(np.float32) / 2.0
    kpt = kpt / np.array(input_wh, np.float32) * scale + center - scale / 2
    kpt[score <= 0] = -1
    return kpt, score


def hand_found(score, thr=0.4, min_points=15):
    """A hand counts as found when at least `min_points` of its 21 keypoints score >= thr."""
    return int((score >= thr).sum()) >= min_points


HAND_BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (11, 12),
              (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)]
BODY_BONES = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12)]


def draw(img, kpt, score, thr=0.4):
    """Draw upper body (blue-grey) and both hands (left = blue, right = orange) on a copy of img."""
    out = img.copy()

    def bones(idx, pairs, color, r):
        for a, b in pairs:
            if score[idx[a]] >= thr and score[idx[b]] >= thr:
                cv2.line(out, tuple(int(v) for v in kpt[idx[a]]), tuple(int(v) for v in kpt[idx[b]]), color, 2, cv2.LINE_AA)
        for i in idx:
            if score[i] >= thr:
                cv2.circle(out, tuple(int(v) for v in kpt[i]), r, color, -1, cv2.LINE_AA)

    bones(list(range(0, 13)), BODY_BONES, (180, 140, 90), 4)
    bones(list(range(91, 112)), HAND_BONES, (230, 120, 0), 3)    # BGR: left hand blue
    bones(list(range(112, 133)), HAND_BONES, (0, 140, 255), 3)   # right hand orange
    return out
