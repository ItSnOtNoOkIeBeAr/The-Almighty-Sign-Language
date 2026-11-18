"""View-only live check of a keypoint model on this PC's webcam. Saves NOTHING (no frames, no keypoints).

  python training/phase3_live_view.py --pose rtmw_xl_384x288 --det yolox_m --seconds 25
Frames go through the same JPEG q70 round trip as the live stream. q or Esc quits. Prints hand-found rates and latency.
"""
import argparse
import time

import cv2
import numpy as np

import ort_cuda
import phase3_eval as E
import rtmw_ref as R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pose", default="rtmw_xl_384x288", choices=list(E.POSE))
    ap.add_argument("--det", default="yolox_m", choices=list(E.DET) + ["full_frame"])
    ap.add_argument("--seconds", type=float, default=25)
    ap.add_argument("--camera", type=int, default=0)
    a = ap.parse_args()
    pose = ort_cuda.make_session(str(E.POSE[a.pose][0]))
    wh = E.POSE[a.pose][1]
    det = ort_cuda.make_session(str(E.DET[a.det][0])) if a.det != "full_frame" else None
    cap = cv2.VideoCapture(a.camera, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, E.LIVE_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, E.LIVE_H)
    if not cap.isOpened():
        raise SystemExit("cannot open camera")
    t_end, n, any_hand, both, lat, person = time.time() + a.seconds, 0, 0, 0, [], 0
    while time.time() < t_end:
        ok, raw = cap.read()
        if not ok:
            continue
        t0 = time.perf_counter()
        frame = cv2.imdecode(np.frombuffer(E.encode_jpeg(cv2.resize(raw, (E.LIVE_W, E.LIVE_H)), E.LIVE_Q), np.uint8), cv2.IMREAD_COLOR)
        box = [0, 0, E.LIVE_W, E.LIVE_H]
        if det is not None:
            blob, ratio = R.yolox_preprocess(frame, E.DET[a.det][1])
            res = R.yolox_best_box(*det.run(None, {"input": blob}), ratio)
            person += res is not None
            box = res[0] if res else box
        kpt, score, *_ = E.run_pose(pose, frame, box, wh, True)
        lat.append((time.perf_counter() - t0) * 1e3)
        thr = E.HAND_THR[a.pose]
        lf, rf = R.hand_found(score[R.LEFT_HAND], thr), R.hand_found(score[R.RIGHT_HAND], thr)
        n += 1
        any_hand += lf or rf
        both += lf and rf
        img = R.draw(frame, kpt, score, thr)
        cv2.putText(img, f"{a.det} + {a.pose}  {np.median(lat[-30:]):.0f} ms  L:{int(lf)} R:{int(rf)}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("phase3 live check (nothing is saved)", img)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break
    cap.release()
    cv2.destroyAllWindows()
    if n:
        print(f"{n} frames | person box {person}/{n if det else 0} | any hand {any_hand / n:.0%} | both hands {both / n:.0%} | "
              f"end-to-end p50 {np.percentile(lat, 50):.1f} ms p95 {np.percentile(lat, 95):.1f} ms (ORT, incl. JPEG round trip + drawing-free)")


if __name__ == "__main__":
    main()
