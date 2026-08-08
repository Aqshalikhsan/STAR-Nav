"""Adapt the SACR segmentation decoder to real oil-palm images.

Starts from the simulation-trained sacr.pt and trains ONLY ``sacr.seg_head``
with pixel-wise cross-entropy (lambda_geom = lambda_depth = 0). The encoder,
depth branch, geometry head and channel gate stay frozen, so z_struct_aug and
the metric range proxies read by AGSS are unchanged; only the reported masks
change. CAMR, the policy and AGSS are not touched.

Use labelled images from plantations other than the test corridor. Paper
setting: 20 epochs, learning rate 1e-5.

    # 1) convert the labelled download:
    python scripts/roboflow_to_sacr_npz.py --root data/roboflow_sawit --out data/real_sawit.npz
    # 2) adapt the decoder:
    python scripts/finetune_sacr_real.py --config configs/paper.yaml --data data/real_sawit.npz         --sacr-ckpt checkpoints/paper/sacr.pt --out checkpoints/paper/sacr_real.pt
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F

from star_nav.models.sacr import SACR
from star_nav.training.finetune_real import freeze_all_but_seg_decoder
from star_nav.utils.config import load_config
from star_nav.utils.seeding import get_device, set_seed


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", nargs="+", default=["data/real_sawit.npz"], help="npz(s) from roboflow_to_sacr_npz.py")
    p.add_argument("--config", default=None)
    p.add_argument("--sacr-ckpt", default="checkpoints/mock/sacr.pt", help="Sim SACR to start from.")
    p.add_argument("--out", default="checkpoints/real/sacr.pt")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-5, help="Decoder learning rate.")
    p.add_argument("--val-frac", type=float, default=0.15)
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    set_seed(cfg.seed)
    device = get_device(cfg.device)

    ds = [np.load(pth) for pth in args.data]
    rgb = np.concatenate([d["rgb"] for d in ds], axis=0)      # (N,H,W,3) uint8
    seg = np.concatenate([d["seg_mask"] for d in ds], axis=0)  # (N,H,W) int
    n = len(rgb)
    rng = np.random.default_rng(cfg.seed)
    perm = rng.permutation(n)
    nval = max(1, int(args.val_frac * n))
    val_idx, tr_idx = perm[:nval], perm[nval:]
    print(f"real dataset: {n} frames ({len(tr_idx)} train / {nval} val)  seg classes={np.unique(seg).tolist()}", flush=True)

    sacr = SACR(in_channels=cfg.sacr.in_channels, feature_channels=cfg.sacr.feature_channels,
                num_seg_classes=cfg.sacr.num_seg_classes, geom_dim=cfg.sacr.geom_dim,
                geom_hidden=cfg.sacr.geom_hidden, struct_dim=cfg.sacr.struct_dim,
                depth_pool_regions=cfg.sacr.depth_pool_regions,
                depth_uncertainty=getattr(cfg.sacr, "depth_uncertainty", False)).to(device)
    sacr.load_compatible_state_dict(torch.load(args.sacr_ckpt, map_location=device))
    print(f"loaded sim SACR from {args.sacr_ckpt}", flush=True)

    params = freeze_all_but_seg_decoder(sacr)
    print("training seg_head only; encoder, depth, geometry head and gate frozen", flush=True)
    optim = torch.optim.Adam(params, lr=args.lr)

    def batch(idx):
        x = torch.from_numpy(rgb[idx]).float().permute(0, 3, 1, 2).to(device) / 255.0
        y = torch.from_numpy(seg[idx]).long().to(device)
        return x, y

    def seg_loss(idx):
        x, y = batch(idx)
        out = sacr(x, need_seg=True)
        return F.cross_entropy(out.seg_logits, y, ignore_index=255)   # 255 = auto-label IGNORE region

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    best = 1e9
    for ep in range(args.epochs):
        sacr.eval(); sacr.seg_head.train()   # frozen BatchNorm statistics stay fixed
        rng.shuffle(tr_idx)
        tot = 0.0; nb = 0
        for b in range(0, len(tr_idx) - args.batch + 1, args.batch):
            loss = seg_loss(tr_idx[b:b + args.batch])
            optim.zero_grad(); loss.backward(); optim.step()
            tot += float(loss); nb += 1
        # validation
        sacr.eval()
        with torch.no_grad():
            vloss = 0.0; vb = 0
            for b in range(0, len(val_idx) - args.batch + 1, args.batch) or [0]:
                bi = val_idx[b:b + args.batch]
                if len(bi) == 0:
                    bi = val_idx
                vloss += float(seg_loss(bi)); vb += 1
            vloss /= max(1, vb)
        print(f"epoch {ep:2d}  train L_seg={tot/max(1,nb):.4f}  val L_seg={vloss:.4f}", flush=True)
        if vloss < best:
            best = vloss
            torch.save(sacr.state_dict(), args.out)
            print(f"  new best -> saved {args.out}", flush=True)

    print(f"\ndone. adapted SACR at {args.out} (best val L_seg={best:.4f}).", flush=True)
    print("Deploy with it: point vision_deploy.py --sacr-ckpt at this file (CAMR/policy unchanged).", flush=True)


if __name__ == "__main__":
    main()
