"""Convert a Roboflow oil-palm export into the .npz format SACR trains on
(rgb + seg_mask), remapping the dataset's classes to STAR-Nav's 5-class taxonomy:
    0=sky  1=ground  2=trunk  3=canopy  4=actor

Supports BOTH Roboflow export flavours:
  * **YOLO detection** (`data.yaml` + `<split>/images` + `<split>/labels/*.txt`
    with `cls cx cy w h` normalized) -> boxes are painted as rectangular masks.
    This is what "Pokok kelapa sawit 2 (yolov5pytorch)" is.
  * **Semantic segmentation masks** (grayscale PNG per image) -> used directly.

Class names are mapped by keyword (case-insensitive), so Indonesian labels work:
    batang / trunk / stem      -> TRUNK   (2)   <- the obstacle STAR-Nav cares about
    pelepah / frond / leaf     -> CANOPY  (3)
    buah / masak / muda / kosong / fruit / ffb -> CANOPY (3)  (tree material)
    orang / person / worker    -> ACTOR   (4)
    everything else / background -> GROUND (1)
Override with e.g. --class-map "batang=2,Pelepah=3,Muda=1".

BOX->MASK CAVEAT: a bounding box is a *rectangle*, not the object's silhouette.
For oil-palm TRUNKS (roughly vertical cylinders) a box is a decent approximation,
which is why this is usable; for fronds it is coarse. Trunk boxes are painted LAST
so the obstacle class wins any overlap. If you want true silhouettes, refine the
boxes with SAM later -- but box-masks are enough to teach SACR what real bark and
real trunks *look like*, which is the point of this fine-tune.

Real static photos have NO metric depth and NO corridor-phi, so `theta_corr_gt` is
written as zeros and you fine-tune SEG-ONLY (see finetune_sacr_real.py).

    python scripts/roboflow_to_sacr_npz.py --root dataset/sawit_yolo --out data/real_sawit.npz
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PIL import Image

from star_nav.utils.config import load_config

SKY, GROUND, TRUNK, CANOPY, ACTOR = 0, 1, 2, 3, 4
KEYWORDS = [
    (TRUNK,  ("batang", "trunk", "stem", "log")),
    (ACTOR,  ("orang", "person", "worker", "human", "people")),
    (CANOPY, ("pelepah", "frond", "leaf", "canopy", "daun",
              "buah", "fruit", "ffb", "masak", "mengkal", "muda", "kosong", "tandan")),
    (SKY,    ("sky", "langit")),
    (GROUND, ("ground", "tanah", "soil", "floor", "road", "path")),
]
# paint order: background classes first, TRUNK last so the obstacle wins overlaps
PAINT_ORDER = [SKY, GROUND, CANOPY, ACTOR, TRUNK]


def name_to_star(name: str) -> int:
    n = name.strip().lower()
    for star, kws in KEYWORDS:
        if any(k in n for k in kws):
            return star
    return GROUND


def read_yaml_names(root):
    """Minimal 'names: [a, b, ...]' reader (avoids a yaml dependency)."""
    for cand in (os.path.join(root, "data.yaml"), os.path.join(root, "data.yml")):
        if not os.path.exists(cand):
            continue
        for line in open(cand):
            if line.strip().startswith("names:"):
                raw = line.split(":", 1)[1].strip()
                if raw.startswith("["):
                    return [s.strip().strip("'\"") for s in raw.strip("[]").split(",")]
    return None


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True, help="Export root (contains data.yaml + train/valid/test).")
    p.add_argument("--out", default="data/real_sawit.npz")
    p.add_argument("--config", default=None)
    p.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    p.add_argument("--class-map", default="", help='Override, e.g. "batang=2,Muda=1" (name or class index).')
    p.add_argument("--preview", default=None, help="Write a PNG sanity-check overlay of N samples here.")
    p.add_argument("--max-frames", type=int, default=0, help="Cap frames (0 = all); useful on low RAM.")
    p.add_argument("--min-trunks", type=int, default=0,
                   help="Keep only images with >= this many trunk boxes (2 = plantation view; 0 = no filter).")
    p.add_argument("--max-trunk-width", type=float, default=1.0,
                   help="Drop images whose mean trunk box is wider than this (1.0 = off; 0.35 drops close-ups).")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    W, H = cfg.env.image_size

    names = read_yaml_names(args.root)
    if not names:
        raise SystemExit(f"no data.yaml names found under {args.root}")
    lut = {i: name_to_star(n) for i, n in enumerate(names)}
    for kv in args.class_map.split(","):
        if "=" not in kv:
            continue
        k, v = kv.split("=")
        if k.strip().isdigit():
            lut[int(k)] = int(v)
        else:
            for i, n in enumerate(names):
                if k.strip().lower() in n.strip().lower():
                    lut[i] = int(v)
    star_name = {SKY: "sky", GROUND: "ground", TRUNK: "TRUNK", CANOPY: "canopy", ACTOR: "actor"}
    print("class mapping (dataset -> STAR-Nav):")
    for i, n in enumerate(names):
        print(f"   {i}: {n!r:24s} -> {star_name[lut[i]]}")

    # Collect paths first, then fill PRE-ALLOCATED arrays: a list-then-np.stack
    # doubles peak memory, and an int64 mask is 8 bytes/pixel (~3 GB for 1.2k
    # frames) -- that combination OOM-kills the process. uint8 mask + preallocate.
    # Optional PLANTATION-VIEW filter. This dataset is really a fruit-grading set:
    # ~450 of its images are close-up bunch photos with no trunk at all, which teach
    # SACR nothing about flying a corridor. Keeping only frames with several SLENDER
    # (i.e. distant) trunk boxes selects the corridor-like views that actually match
    # the drone's viewpoint.
    trunk_idx = [i for i, n in enumerate(names) if lut[i] == TRUNK]

    def trunk_stats(lp):
        w = []
        if os.path.exists(lp):
            for line in open(lp):
                f = line.split()
                if len(f) >= 5 and int(float(f[0])) in trunk_idx:
                    w.append(float(f[3]))
        return len(w), (float(np.mean(w)) if w else 0.0)

    pairs, skipped = [], 0
    for split in args.splits:
        idir = os.path.join(args.root, split, "images")
        ldir = os.path.join(args.root, split, "labels")
        if not os.path.isdir(idir):
            continue
        for ip in sorted(glob.glob(os.path.join(idir, "*"))):
            stem = os.path.splitext(os.path.basename(ip))[0]
            lp = os.path.join(ldir, stem + ".txt")
            if args.min_trunks or args.max_trunk_width < 1.0:
                ntr, mw = trunk_stats(lp)
                if ntr < args.min_trunks or (ntr and mw > args.max_trunk_width):
                    skipped += 1
                    continue
            pairs.append((ip, lp))
    if skipped:
        print(f"plantation-view filter: kept {len(pairs)}, skipped {skipped} "
              f"(min_trunks={args.min_trunks}, max_trunk_width={args.max_trunk_width})", flush=True)
    if args.max_frames:
        pairs = pairs[: args.max_frames]
    if not pairs:
        raise SystemExit("no images found -- check --root / --splits / filter")

    N = len(pairs)
    rgb = np.empty((N, H, W, 3), dtype=np.uint8)
    seg = np.empty((N, H, W), dtype=np.uint8)     # 5 classes fit in uint8
    k = 0
    for ip, lp in pairs:
        try:
            img = Image.open(ip).convert("RGB").resize((W, H), Image.BILINEAR)
        except Exception:
            continue
        mask = np.full((H, W), GROUND, dtype=np.uint8)
        boxes = []
        if os.path.exists(lp):
            for line in open(lp):
                f = line.split()
                if len(f) < 5:
                    continue
                c = int(float(f[0]))
                cx, cy, bw, bh = (float(x) for x in f[1:5])
                boxes.append((lut.get(c, GROUND), cx, cy, bw, bh))
        # paint background classes first, TRUNK last (obstacle wins overlaps)
        for star in PAINT_ORDER:
            for s, cx, cy, bw, bh in boxes:
                if s != star:
                    continue
                x0 = int(max(0, (cx - bw / 2) * W)); x1 = int(min(W, (cx + bw / 2) * W))
                y0 = int(max(0, (cy - bh / 2) * H)); y1 = int(min(H, (cy + bh / 2) * H))
                if x1 > x0 and y1 > y0:
                    mask[y0:y1, x0:x1] = star
        rgb[k] = np.asarray(img, dtype=np.uint8)
        seg[k] = mask
        k += 1
        if k % 200 == 0:
            print(f"  converted {k}/{N} ...", flush=True)
    rgb, seg = rgb[:k], seg[:k]
    if k == 0:
        raise SystemExit("no images converted -- check --root / --splits")
    theta = np.zeros((len(rgb), cfg.sacr.geom_dim), dtype=np.float32)  # dummy: seg-only fine-tune
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(args.out, rgb=rgb, seg_mask=seg, theta_corr_gt=theta)

    frac = {star_name[int(c)]: round(float((seg == c).mean()), 3) for c in np.unique(seg)}
    print(f"\nwrote {args.out}: {len(rgb)} frames {W}x{H}   pixel-fraction {frac}", flush=True)
    if (seg == TRUNK).mean() < 0.01:
        print("WARNING: almost no TRUNK pixels -- check the class mapping!", flush=True)

    if args.preview:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        colors = np.array([[135, 206, 235], [120, 100, 80], [200, 40, 40], [40, 160, 60], [230, 200, 40]])
        k = min(6, len(rgb))
        idx = np.linspace(0, len(rgb) - 1, k).astype(int)
        fig, axes = plt.subplots(2, k, figsize=(3 * k, 5))
        for j, i in enumerate(idx):
            axes[0, j].imshow(rgb[i]); axes[0, j].axis("off")
            over = (0.55 * rgb[i] + 0.45 * colors[seg[i]]).astype(np.uint8)
            axes[1, j].imshow(over); axes[1, j].axis("off")
        axes[0, 0].set_title("image", loc="left"); axes[1, 0].set_title("mask (red=TRUNK, green=canopy)", loc="left")
        plt.tight_layout(); plt.savefig(args.preview, dpi=90)
        print(f"sanity-check overlay -> {args.preview}", flush=True)


if __name__ == "__main__":
    main()
