"""Auto-label real flight frames WITHOUT human annotation: GroundingDINO finds the
objects from a TEXT prompt, SAM turns each box into a precise silhouette.

    frame --> GroundingDINO("palm tree trunk. person.") --> boxes
          --> SAM(box prompt)                           --> pixel-accurate masks
          --> seg_mask (TRUNK / ACTOR / GROUND)         --> SACR .npz

Why this and not "let the fine-tuned SACR label its own frames": SACR was trained
on BOX labels, so it systematically over-predicts trunk (its masks smear down onto
the ground). Self-labelling with it AMPLIFIES that bias. SAM decides boundaries
from IMAGE EVIDENCE instead, so it breaks the loop -- measured trunk pixel
fraction drops from 0.21 (SACR-labelled, bleeding) to ~0.12, which matches the
human-labelled Roboflow set (0.123).

TWO GATES that matter:
  * SHAPE GATE on DINO trunk boxes: a "tree trunk" box often swallows the whole
    crown. A real trunk is TALL and NARROW and doesn't fill the frame, so reject
    boxes that aren't at least `--min-aspect` times taller than wide, or that
    cover more than `--max-box-area` of the frame. Without this, SAM happily
    segments the whole tree and canopy gets labelled TRUNK.
  * PERSON is painted LAST so it overrides trunk -- this is what kills the
    person/motorbike-as-trunk false positive at its source, and it finally
    supplies the CLASS_ACTOR that CAMR needs.

HONEST LIMIT: it cannot label a trunk DINO never proposes (thin/far background
trunks are still missed). Hand-label a small held-out set if you want to MEASURE
these auto-labels rather than trust them.

    python scripts/autolabel_video.py --frames dataset/frames_auto \
        --out data/real_video_auto.npz --preview renders/autolabel_check.png
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from PIL import Image

from star_nav.utils.config import load_config
from star_nav.utils.seeding import get_device

SKY, GROUND, TRUNK, CANOPY, ACTOR = 0, 1, 2, 3, 4
IGNORE = 255   # excluded from the loss (see finetune_sacr_real.py ignore_index)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--frames", default="dataset/frames_auto")
    p.add_argument("--config", default=None)
    p.add_argument("--out", default="data/real_video_auto.npz")
    p.add_argument("--dino", default="IDEA-Research/grounding-dino-tiny")
    p.add_argument("--sam", default="facebook/sam-vit-base")
    p.add_argument("--text", default="palm tree trunk. tree trunk. person.")
    p.add_argument("--box-thr", type=float, default=0.15,
                   help="DINO score to accept a box as a confident TRUNK label.")
    p.add_argument("--ignore-below", type=float, default=0.08,
                   help="Boxes scoring between this and --box-thr are marked IGNORE (255): we are not "
                        "sure enough to call them trunk, but calling them GROUND would teach the net "
                        "that a trunk is ground -- a harmful false negative. Excluded from the loss.")
    p.add_argument("--min-aspect", type=float, default=1.5,
                   help="Trunk box must be >= this many times taller than wide (rejects whole-tree boxes).")
    p.add_argument("--max-box-area", type=float, default=0.25,
                   help="Reject trunk boxes covering more than this fraction of the frame.")
    p.add_argument("--person-thr", type=float, default=0.30,
                   help="Person needs a HIGH score: it overrides trunk, so a false person wrecks the mask.")
    p.add_argument("--sam-chunk", type=int, default=6, help="Boxes per SAM forward pass (4GB GPU: keep small).")
    p.add_argument("--max-boxes", type=int, default=24, help="Cap boxes per frame (keeps the highest scores).")
    p.add_argument("--preview", default=None)
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    device = get_device(cfg.device)
    W, H = cfg.env.image_size

    from transformers import (AutoProcessor, AutoModelForZeroShotObjectDetection,
                              SamModel, SamProcessor)
    dproc = AutoProcessor.from_pretrained(args.dino)
    dino = AutoModelForZeroShotObjectDetection.from_pretrained(args.dino).to(device).eval()
    sproc = SamProcessor.from_pretrained(args.sam)
    sam = SamModel.from_pretrained(args.sam).to(device).eval()
    print("GroundingDINO + SAM ready", flush=True)

    files = sorted(glob.glob(os.path.join(args.frames, "*.jpg")))
    if not files:
        raise SystemExit(f"no frames in {args.frames}")
    N = len(files)
    rgb = np.empty((N, H, W, 3), dtype=np.uint8)
    seg = np.empty((N, H, W), dtype=np.uint8)
    n_tr = n_ac = n_ig = 0

    def sam_masks(img, boxes):
        """boxes in full-res xyxy -> list of full-res bool masks.
        SAM is run in CHUNKS: at a low DINO threshold a frame can propose dozens of
        boxes, and feeding them all at once OOMs a 4 GB GPU."""
        out_masks = []
        for c in range(0, len(boxes), args.sam_chunk):
            chunk = boxes[c:c + args.sam_chunk]
            si = sproc(img, input_boxes=[chunk], return_tensors="pt").to(device)
            with torch.no_grad():
                so = sam(**si, multimask_output=False)
            m = sproc.image_processor.post_process_masks(
                so.pred_masks.cpu(), si["original_sizes"].cpu(), si["reshaped_input_sizes"].cpu())[0]
            out_masks += [mm.numpy() for mm in m.squeeze(1)]
            del si, so
        return out_masks

    for i, f in enumerate(files):
        img = Image.open(f).convert("RGB")
        FW, FH = img.size

        inp = dproc(images=img, text=args.text, return_tensors="pt").to(device)
        with torch.no_grad():
            out = dino(**inp)
        r = dproc.post_process_grounded_object_detection(
            out, inp.input_ids, threshold=args.ignore_below, text_threshold=args.ignore_below,
            target_sizes=[img.size[::-1]])[0]
        labs = r.get("text_labels", r["labels"])

        tboxes, pboxes, iboxes = [], [], []
        for b, l, sc in zip(r["boxes"].cpu().numpy(), labs, r["scores"].cpu().numpy()):
            x0, y0, x1, y1 = (float(v) for v in b)
            bw, bh = x1 - x0, y1 - y0
            if bw <= 1 or bh <= 1:
                continue
            if "person" in str(l).lower():
                # PERSON must clear the CONFIDENT threshold. The low `ignore_below`
                # tier exists only to mark unsure TRUNKS as IGNORE; letting weak
                # person boxes through here floods the mask with ACTOR (it is painted
                # last, so it overrides trunk) -- measured 13.7% ACTOR pixels, absurd.
                if float(sc) >= args.person_thr:
                    pboxes.append([x0, y0, x1, y1])
                continue
            if bh < args.min_aspect * bw:                  # not tall+narrow -> whole-tree box
                continue
            if bw * bh > args.max_box_area * FW * FH:      # fills the frame -> junk
                continue
            (tboxes if float(sc) >= args.box_thr else iboxes).append([x0, y0, x1, y1])

        tboxes, iboxes, pboxes = tboxes[:args.max_boxes], iboxes[:args.max_boxes], pboxes[:args.max_boxes]
        mask_full = np.full((FH, FW), GROUND, dtype=np.uint8)
        for m in sam_masks(img, iboxes):                   # unsure -> IGNORE (not "ground"!)
            mask_full[m] = IGNORE
        for m in sam_masks(img, tboxes):                   # SAM silhouettes -> TRUNK
            mask_full[m] = TRUNK
        for m in sam_masks(img, pboxes):                   # painted LAST -> overrides TRUNK
            mask_full[m] = ACTOR

        rgb[i] = np.asarray(img.resize((W, H), Image.BILINEAR), dtype=np.uint8)
        seg[i] = np.asarray(Image.fromarray(mask_full).resize((W, H), Image.NEAREST), dtype=np.uint8)
        n_tr += len(tboxes); n_ac += len(pboxes); n_ig += len(iboxes)
        torch.cuda.empty_cache()
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{N} ...", flush=True)

    theta = np.zeros((N, cfg.sacr.geom_dim), dtype=np.float32)   # real photos have no depth/phi
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(args.out, rgb=rgb, seg_mask=seg, theta_corr_gt=theta)
    frac = {int(c): round(float((seg == c).mean()), 3) for c in np.unique(seg)}
    print(f"\nwrote {args.out}: {N} frames   pixel-fraction {frac}  (2=TRUNK 4=ACTOR)")
    print(f"DINO: {n_tr} confident trunk boxes, {n_ac} person boxes, {n_ig} unsure -> IGNORE. SAM made the silhouettes.", flush=True)

    if args.preview:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        col = np.zeros((256, 3), dtype=np.uint8)   # 256 entries: IGNORE is 255, not 5!
        col[:5] = [[135, 206, 235], [120, 100, 80], [220, 30, 30], [40, 170, 60], [250, 220, 30]]
        col[255] = [255, 0, 255]                   # magenta = IGNORE (excluded from loss)
        k = 6
        idx = np.linspace(0, N - 1, k).astype(int)
        fig, ax = plt.subplots(2, k, figsize=(3.1 * k, 5.2))
        for j, i in enumerate(idx):
            ax[0, j].imshow(rgb[i]); ax[0, j].axis("off")
            ax[1, j].imshow((0.5 * rgb[i] + 0.5 * col[seg[i]]).astype(np.uint8)); ax[1, j].axis("off")
        ax[0, 0].set_title("your frame", loc="left", fontsize=10)
        ax[1, 0].set_title("AUTO label: DINO+SAM (red=TRUNK, yellow=ACTOR)", loc="left", fontsize=10)
        plt.tight_layout(); plt.savefig(args.preview, dpi=88)
        print(f"preview -> {args.preview}", flush=True)


if __name__ == "__main__":
    main()
