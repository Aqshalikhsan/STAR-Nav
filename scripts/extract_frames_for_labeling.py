"""Extract a DIVERSE set of frames from real flight video, ready to upload to
Roboflow for labeling.

Why not just take every Nth frame: at 30-60 fps consecutive frames are nearly
identical, and labeling near-duplicates wastes your time without teaching the
network anything new. So this samples on a TIME interval and then additionally
drops frames that are too similar to the last kept one (mean abs pixel diff).

    python scripts/extract_frames_for_labeling.py \
        --videos dataset/input_video.MP4 dataset/lv_0_20250811210939.mp4 \
        --out dataset/frames_to_label --every 1.5 --min-diff 12

Then: zip the folder -> upload to Roboflow -> label -> export as YOLO detection
-> feed to scripts/roboflow_to_sacr_npz.py (which already understands that format).
"""
from __future__ import annotations

import argparse
import os

import cv2
import numpy as np


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--videos", nargs="+", required=True)
    p.add_argument("--out", default="dataset/frames_to_label")
    p.add_argument("--every", type=float, default=1.5, help="Sample one frame every N seconds.")
    p.add_argument("--min-diff", type=float, default=12.0,
                   help="Skip a frame if its mean abs pixel diff to the last KEPT frame is below this (near-duplicate).")
    p.add_argument("--max-frames", type=int, default=150, help="Cap total frames (labeling budget).")
    p.add_argument("--width", type=int, default=1280, help="Output width (keeps aspect); 0 = native.")
    args = p.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    kept_total, prev_small = 0, None
    for vp in args.videos:
        cap = cv2.VideoCapture(vp)
        if not cap.isOpened():
            print(f"cannot open {vp}"); continue
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, int(round(args.every * fps)))
        tag = os.path.splitext(os.path.basename(vp))[0][:18]
        kept_v, skipped = 0, 0
        for fi in range(0, n, step):
            if kept_total >= args.max_frames:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frame = cap.read()
            if not ok:
                continue
            small = cv2.resize(frame, (64, 36)).astype(np.float32)
            if prev_small is not None and np.abs(small - prev_small).mean() < args.min_diff:
                skipped += 1
                continue                      # near-duplicate of the last kept frame
            prev_small = small
            if args.width:
                h = int(frame.shape[0] * args.width / frame.shape[1])
                frame = cv2.resize(frame, (args.width, h))
            cv2.imwrite(os.path.join(args.out, f"{tag}_{fi:06d}.jpg"), frame,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            kept_v += 1; kept_total += 1
        cap.release()
        print(f"{os.path.basename(vp)}: kept {kept_v}  (skipped {skipped} near-duplicates, "
              f"sampled every {args.every}s = {step} frames)", flush=True)

    print(f"\n{kept_total} frames -> {args.out}/", flush=True)
    print("Next: zip that folder, upload to Roboflow, and label these classes:", flush=True)
    print("   batang  (trunk)   <- the obstacle; label EVERY visible trunk", flush=True)
    print("   orang   (person)  <- fixes the person/motorbike false positives + gives CLASS_ACTOR", flush=True)
    print("   pelepah (frond)   <- optional", flush=True)
    print("Export as **YOLO (detection, boxes)** -- fast to label, and", flush=True)
    print("scripts/roboflow_to_sacr_npz.py already reads that format.", flush=True)


if __name__ == "__main__":
    main()
