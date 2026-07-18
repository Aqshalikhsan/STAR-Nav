"""Phase-1 SACR pretraining with per-epoch checkpointing + best-model tracking
+ resume. Trains on a captured perception dataset (an .npz with rgb / seg_mask
/ theta_corr_gt -- e.g. the real-Gazebo dataset from
ros_gazebo_bridge/perception_capture/collect_dataset.py, or any dataset in that
same format).

Checkpoints (written to --out-dir, default `checkpoints/gazebo/`):
  sacr_last.pt   -- rewritten EVERY epoch (full training state: model + optim +
                    epoch + best_val + rng), so a crash/kill loses at most one
                    epoch and training can always resume.
  sacr_best.pt   -- the epoch with the lowest VALIDATION L_SACR so far.
  sacr.pt        -- plain model weights of the best epoch (what run_train_all /
                    downstream CAMR+PPO load; same shape as train_perception's).

Resume:
  python scripts/train_sacr.py --resume checkpoints/gazebo/sacr_last.pt  # or sacr_best.pt
restores model + optimizer + epoch counter + best-so-far and continues.

Examples:
  python scripts/train_sacr.py --data data/sacr_gazebo_dataset.npz --epochs 40
  python scripts/train_sacr.py --resume checkpoints/gazebo/sacr_best.pt --epochs 80
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from star_nav.models.sacr import SACR, sacr_loss
from star_nav.utils.config import load_config
from star_nav.utils.seeding import get_device, set_seed


def build_sacr(cfg, device):
    return SACR(
        in_channels=cfg.sacr.in_channels, feature_channels=cfg.sacr.feature_channels,
        num_seg_classes=cfg.sacr.num_seg_classes, geom_dim=cfg.sacr.geom_dim,
        geom_hidden=cfg.sacr.geom_hidden, struct_dim=cfg.sacr.struct_dim,
        depth_pool_regions=cfg.sacr.depth_pool_regions,
        depth_uncertainty=getattr(cfg.sacr, "depth_uncertainty", False),
    ).to(device)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", nargs="+", default=["data/sacr_gazebo_dataset.npz"],
                   help="One or more npz datasets (rgb (N,H,W,3 uint8), seg_mask (N,H,W), "
                        "theta_corr_gt (N,4)); concatenated. Pass the static + dynamic "
                        "(actor-containing) sets together to train an actor-aware SACR.")
    p.add_argument("--config", default=None, help="YAML config (defaults to configs/default.yaml).")
    p.add_argument("--epochs", type=int, default=40, help="Total epochs to reach (not additional).")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--out-dir", default="checkpoints/gazebo")
    p.add_argument("--resume", default=None, help="Checkpoint to resume from (e.g. checkpoints/gazebo/sacr_last.pt).")
    p.add_argument("--eval-every", type=int, default=1, help="Run validation every N epochs (checkpoints are still every epoch).")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    os.makedirs(args.out_dir, exist_ok=True)

    ds = [np.load(p) for p in args.data]
    rgb = np.concatenate([d["rgb"] for d in ds], axis=0)
    seg = np.concatenate([d["seg_mask"] for d in ds], axis=0)
    theta = np.concatenate([d["theta_corr_gt"] for d in ds], axis=0)
    # Full metric depth map (novelty): supervises the depth net (L_depth) and the
    # per-region aleatoric log-variance head (L_unc). Only used if EVERY dataset
    # carries it -- older datasets without `depth` fall back to the seg+geom path.
    has_depth = all("depth" in d.files for d in ds)
    depth = np.concatenate([d["depth"] for d in ds], axis=0) if has_depth else None
    N = len(rgb)
    if len(args.data) > 1:
        print("datasets:", {p: len(d["rgb"]) for p, d in zip(args.data, ds)}, flush=True)
    rng = np.random.default_rng(cfg.seed)
    perm = rng.permutation(N)
    nval = max(args.batch, int(N * args.val_frac))
    val_idx, tr_idx = perm[:nval], perm[nval:]
    unc_on = getattr(cfg.sacr, "depth_uncertainty", False)
    print(f"device={device}  data={args.data}  frames={N} (train={len(tr_idx)} val={len(val_idx)})  "
          f"seg classes={np.unique(seg).tolist()}  depth={'yes' if has_depth else 'no'}  "
          f"uncertainty_head={unc_on}", flush=True)
    if unc_on and not has_depth:
        print("WARNING: depth_uncertainty=True but dataset has no depth map -- "
              "L_depth/L_unc will be 0 (uncertainty head trains only via z_struct_aug grads).", flush=True)

    sacr = build_sacr(cfg, device)
    optim = torch.optim.Adam(sacr.parameters(), lr=cfg.sacr.lr)

    start_epoch, best_val = 0, float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        sacr.load_state_dict(ckpt["model"])
        if "optim" in ckpt:
            optim.load_state_dict(ckpt["optim"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val = ckpt.get("best_val", float("inf"))
        print(f"resumed from {args.resume}: start_epoch={start_epoch} best_val={best_val:.4f}", flush=True)

    def batches(idx, shuffle):
        idx = idx.copy()
        if shuffle:
            rng.shuffle(idx)
        for b in range(0, len(idx) - args.batch + 1, args.batch):
            bi = idx[b:b + args.batch]
            dt = torch.from_numpy(depth[bi]).float().to(device) if depth is not None else None
            yield (torch.from_numpy(rgb[bi]).float().permute(0, 3, 1, 2).to(device) / 255.0,
                   torch.from_numpy(seg[bi]).long().to(device),
                   torch.from_numpy(theta[bi]).float().to(device),
                   dt)

    lambda_depth = getattr(cfg.sacr, "lambda_depth", 0.2)
    lambda_unc = getattr(cfg.sacr, "lambda_unc", 0.5)

    @torch.no_grad()
    def evaluate(idx):
        sacr.eval()
        lseg, lgeom, ldep, lunc, sc, st = [], [], [], [], 0, 0
        for r, s, th, dt in batches(idx, False):
            out = sacr(r, need_seg=True)
            L = sacr_loss(out, s, th, depth_target=dt, lambda_geom=cfg.sacr.lambda_geom,
                          lambda_depth=lambda_depth, lambda_unc=lambda_unc, mu_smooth=cfg.sacr.mu_smooth)
            lseg.append(L["L_seg"].item()); lgeom.append(L["L_geom"].item())
            ldep.append(L["L_depth"].item()); lunc.append(L["L_unc"].item())
            sc += (out.seg_logits.argmax(1) == s).sum().item(); st += s.numel()
        vseg, vgeom = float(np.mean(lseg)), float(np.mean(lgeom))
        vdep, vunc = float(np.mean(ldep)), float(np.mean(lunc))
        vscore = vseg + cfg.sacr.lambda_geom * vgeom + lambda_depth * vdep + lambda_unc * vunc
        return vseg, vgeom, vdep, vunc, vscore, sc / st

    def save(path, epoch, model_only=False):
        if model_only:
            torch.save(sacr.state_dict(), path)
        else:
            torch.save({"model": sacr.state_dict(), "optim": optim.state_dict(),
                        "epoch": epoch, "best_val": best_val}, path)

    last_pt = os.path.join(args.out_dir, "sacr_last.pt")
    best_pt = os.path.join(args.out_dir, "sacr_best.pt")
    plain_pt = os.path.join(args.out_dir, "sacr.pt")

    t0 = time.time()
    for epoch in range(start_epoch, args.epochs):
        sacr.train()
        for r, s, th, dt in batches(tr_idx, True):
            out = sacr(r, need_seg=True)
            L = sacr_loss(out, s, th, depth_target=dt, lambda_geom=cfg.sacr.lambda_geom,
                          lambda_depth=lambda_depth, lambda_unc=lambda_unc, mu_smooth=cfg.sacr.mu_smooth)
            optim.zero_grad(); L["L_SACR"].backward(); optim.step()

        # per-epoch checkpoint (always), so a kill loses at most one epoch
        save(last_pt, epoch)

        if epoch % args.eval_every == 0 or epoch == args.epochs - 1:
            vseg, vgeom, vdep, vunc, vscore, vacc = evaluate(val_idx)
            improved = vscore < best_val
            if improved:
                best_val = vscore
                save(best_pt, epoch)
                save(plain_pt, epoch, model_only=True)  # plain weights of the best epoch
                # keep best_val current inside sacr_last.pt too
                save(last_pt, epoch)
            print(f"epoch {epoch:2d}  VAL L_seg={vseg:.3f} L_geom={vgeom:.3f} L_depth={vdep:.3f} "
                  f"L_unc={vunc:.3f} L_SACR={vscore:.3f} acc={vacc:.3f}  best={best_val:.3f}"
                  f"{'  <- new best' if improved else ''}  ({time.time()-t0:.0f}s)", flush=True)

    print(f"done. last={last_pt}  best={best_pt} (val L_SACR={best_val:.3f})  weights={plain_pt}", flush=True)


if __name__ == "__main__":
    main()
