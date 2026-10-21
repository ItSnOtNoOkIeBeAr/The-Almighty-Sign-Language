"""Record 3-second training clips and review/delete them.

  python client/recorder.py record hello --signer s1        (also: client.py --record hello --signer s1)
  python client/recorder.py review [--label hello]          (also: client.py --review)
Clip layout (BUILD_SPEC section 6):
  data/raw/<label>/<signer>_<YYYYmmdd-HHMMSS>/000.jpg ... 044.jpg + meta.json
Frames go through the same prepare_frame()/encode_jpeg() as live streaming.
"""
import argparse
import json
import shutil
import time
from datetime import datetime
from pathlib import Path

import cv2

from common import (REPO_ROOT, Grabber, add_common_args, encode_jpeg, load_config, open_camera, prepare_frame,
                    put_text)

DATA_RAW = REPO_ROOT / "data" / "raw"
CLIP_SECONDS, COUNTDOWN = 3.0, 3
LABELS_FILE = REPO_ROOT / "protocol" / "labels.json"
# (end time in seconds, message): the routine shown on screen while recording.
ROUTINE = [(0.75, "HANDS DOWN", (200, 200, 200)), (2.25, "SIGN NOW", (0, 255, 0)), (3.01, "HANDS DOWN", (200, 200, 200))]


def check_label(label):
    if LABELS_FILE.exists():
        labels = json.loads(LABELS_FILE.read_text())["labels"]
        if label not in labels:
            raise SystemExit(f"unknown label '{label}'. Valid labels: {', '.join(labels)}")


def _show(img, lines, color=(255, 255, 255)):
    for i, text in enumerate(lines):
        put_text(img, text, (20, 40 + 36 * i), 1.0, color, 2)
    cv2.imshow("record", img)
    return cv2.waitKey(1) & 0xFF


def record_clip(label, signer, cfg, headless=False, data_root=DATA_RAW) -> Path:
    check_label(label)
    fps, quality = cfg["fps"], cfg["jpeg_quality"]
    n_frames = round(CLIP_SECONDS * fps)
    grab = Grabber(open_camera(cfg))
    try:
        while grab.latest()[0] is None:  # wait for the first frame
            time.sleep(0.01)
        time.sleep(0.5)  # let auto-exposure settle
        for sec in range(COUNTDOWN, 0, -1):  # countdown, preview stays live
            end = time.time() + 1.0
            if headless:
                print(f"recording '{label}' in {sec}...", flush=True)
            while time.time() < end:
                if not headless:
                    img = prepare_frame(grab.latest()[0], cfg).copy()
                    if _show(img, [f"GET READY: {sec}", "hands down, then sign, then hands down"]) == 27:
                        raise SystemExit("cancelled")
                time.sleep(0.01)
        jpegs, seqs, t0 = [], [], time.time()
        if headless:
            print("RECORDING (hands down, sign, hands down)", flush=True)
        for i in range(n_frames):
            # Same as live streaming: at each tick take the NEWEST camera frame (repeats if the camera is slow).
            time.sleep(max(0.0, t0 + i / fps - time.time()))
            raw, seq = grab.latest()
            frame = prepare_frame(raw, cfg)
            jpegs.append(encode_jpeg(frame, quality))
            seqs.append(seq)
            if not headless:
                t = time.time() - t0
                msg, col = next((m, c) for end, m, c in ROUTINE if t < end)
                _show(frame.copy(), [f"REC {i + 1}/{n_frames}", msg], col)
        captured_s = time.time() - t0  # stop the clock before camera teardown (which can block ~0.5 s)
        h, w = frame.shape[:2]
    finally:
        grab.close()
        if not headless:
            cv2.destroyAllWindows()
    unique = len(set(seqs))
    if unique < 0.8 * n_frames:
        print(f"WARNING: camera delivered only {unique} distinct frames for {n_frames} slots "
              f"(~{unique / CLIP_SECONDS:.1f} fps). Add light or lower --fps, or this clip has repeated frames.")
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = Path(data_root) / label / f"{signer}_{stamp}"
    out.mkdir(parents=True, exist_ok=False)
    for i, jb in enumerate(jpegs):
        (out / f"{i:03d}.jpg").write_bytes(jb)  # the exact bytes that would be streamed
    meta = {"label": label, "signer": signer, "fps": fps, "width": w, "height": h, "jpeg_quality": quality,
            "created": datetime.now().isoformat(timespec="seconds"),
            "unique_camera_frames": unique}  # extra key (beyond the spec): lets training drop clips from a starved camera
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"saved {len(jpegs)} frames (captured in {captured_s:.2f}s, {unique} distinct camera frames) to {out}")
    return out


def list_clips(data_root=DATA_RAW, label=None):
    root = Path(data_root)
    dirs = [root / label] if label else sorted(p for p in root.glob("*") if p.is_dir())
    return sorted(c for d in dirs for c in d.glob("*") if (c / "meta.json").exists())


def review(data_root=DATA_RAW, label=None):
    """Replay clips in a loop. n/right = next, p/left = previous, d then y = delete, q/Esc = quit."""
    clips = list_clips(data_root, label)
    if not clips:
        print("no clips found")
        return
    i, confirm = 0, False
    while clips:
        i %= len(clips)
        clip = clips[i]
        meta = json.loads((clip / "meta.json").read_text())
        frames = sorted(clip.glob("*.jpg"))
        key = -1
        for fp in frames:
            img = cv2.imread(str(fp))
            title = f"[{i + 1}/{len(clips)}] {meta['label']} / {meta['signer']}  {clip.name}"
            put_text(img, title, (10, 24), 0.6)
            put_text(img, "DELETE this clip? press y to confirm, any other key cancels" if confirm
                     else "n next | p prev | d delete | q quit", (10, img.shape[0] - 12), 0.55,
                     (0, 0, 255) if confirm else (255, 255, 255))
            cv2.imshow("review", img)
            key = cv2.waitKey(int(1000 / meta["fps"])) & 0xFF
            if key != 255:
                break
        if key == 255:
            continue  # loop the clip
        if confirm:
            confirm = False
            if key == ord("y"):
                if Path(data_root).resolve() not in clip.resolve().parents:
                    raise RuntimeError(f"refusing to delete outside {data_root}: {clip}")
                shutil.rmtree(clip)
                print("deleted", clip)
                clips.pop(i)
            continue
        if key in (ord("q"), 27):
            break
        elif key in (ord("n"), 83):
            i += 1
        elif key in (ord("p"), 81):
            i -= 1
        elif key == ord("d"):
            confirm = True
    cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    rec = sub.add_parser("record")
    rec.add_argument("label")
    rec.add_argument("--signer", required=True)
    add_common_args(rec)
    rev = sub.add_parser("review")
    rev.add_argument("--label")
    a = ap.parse_args()
    if a.cmd == "record":
        record_clip(a.label, a.signer, load_config(a), a.headless)
    else:
        review(label=a.label)


if __name__ == "__main__":
    main()
