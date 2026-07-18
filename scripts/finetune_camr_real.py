"""Train CAMR on REAL flight video -- with NO annotation at all.

This works because CAMR is SELF-SUPERVISED (see camr_loss):
    L_pred = || f_pred(h_t) - z_struct_aug_{t+1} ||^2   <- predict the NEXT frame's
                                                            SACR features: needs only
                                                            consecutive frames
    L_temp = || h_t - h_{t-1} ||^2                      <- temporal smoothness
    L_occ  = BCE(...)                                    <- the ONLY labelled term;
                                                            DISABLED here (real ACTOR
                                                            labels are unreliable)
So raw video + a frozen SACR is enough. No masks, no boxes, no human effort.

POSE-BLIND BY NECESSITY. CAMR's input is x_t = [z_struct_aug ; pose ; imu]. The real
drone (iNav, no PX4/VIO/mocap) has NO position source, in training OR in flight, so
pose/imu are fed as ZEROS here -- exactly what vision_deploy.py does at deploy time.
Train/deploy are therefore consistent.

!!! THE CATCH -- read before you fly !!!
A pose-blind CAMR produces a DIFFERENT belief h_t than the Mock-trained CAMR (which
saw real pose). The PPO policy is an MLP over that belief, so the sim policy will be
OUT OF DISTRIBUTION unless it is ALSO retrained pose-blind in Mock. Perception alone
does not close the loop -- see the project deploy notes (this is exactly why the Mock
policy wandered in Gazebo).

    python scripts/finetune_camr_real.py \
        --videos dataset/input_video.MP4 dataset/lv_0_20250811210939.mp4 \
        --sacr-ckpt checkpoints/real/sacr_full.pt --out checkpoints/real/camr.pt
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import torch

from star_nav.models.sacr import SACR
from star_nav.models.camr import CAMR, CausalWindowBuffer, camr_loss
from star_nav.utils.config import load_config
from star_nav.utils.seeding import get_device, set_seed


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--videos", nargs="+", required=True)
    p.add_argument("--config", default=None)
    p.add_argument("--sacr-ckpt", default="checkpoints/real/sacr_full.pt", help="FROZEN real SACR.")
    p.add_argument("--camr-init", default="checkpoints/mock/camr.pt",
                   help="Warm-start from the Mock CAMR (keeps the architecture/scale); '' = from scratch.")
    p.add_argument("--out", default="checkpoints/real/camr.pt")
    p.add_argument("--dt", type=float, default=0.2,
                   help="Seconds between sampled frames. MUST match the sim/control step (Mock dt=0.2 -> 5 Hz), "
                        "or CAMR learns the wrong temporal scale.")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=16, help="Windows per update.")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--val-frac", type=float, default=0.15)
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    W, H = cfg.env.image_size

    # --- frozen real SACR ---
    sacr = SACR(in_channels=cfg.sacr.in_channels, feature_channels=cfg.sacr.feature_channels,
                num_seg_classes=cfg.sacr.num_seg_classes, geom_dim=cfg.sacr.geom_dim,
                geom_hidden=cfg.sacr.geom_hidden, struct_dim=cfg.sacr.struct_dim,
                depth_pool_regions=cfg.sacr.depth_pool_regions,
                depth_uncertainty=getattr(cfg.sacr, "depth_uncertainty", False)).to(device)
    sacr.load_state_dict(torch.load(args.sacr_ckpt, map_location=device))
    sacr.eval()
    for prm in sacr.parameters():
        prm.requires_grad_(False)

    # --- encode every sampled video frame to z_struct_aug (one episode per video) ---
    episodes = []
    for vp in args.videos:
        cap = cv2.VideoCapture(vp)
        if not cap.isOpened():
            print(f"cannot open {vp}"); continue
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = max(1, int(round(args.dt * fps)))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        zs = []
        for fi in range(0, n, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, fr = cap.read()
            if not ok:
                continue
            rgb = cv2.cvtColor(cv2.resize(fr, (W, H)), cv2.COLOR_BGR2RGB)
            x = torch.from_numpy(rgb).float().permute(2, 0, 1)[None].to(device) / 255.0
            with torch.no_grad():
                zs.append(sacr.encode(x)[0].cpu())
        cap.release()
        if len(zs) > cfg.camr.window_size + 1:
            episodes.append(torch.stack(zs))          # (T, z_dim), contiguous in time
            print(f"{os.path.basename(vp)}: {len(zs)} frames @ dt={args.dt}s "
                  f"({step}-frame stride, {fps:.0f} fps source)", flush=True)
    if not episodes:
        raise SystemExit("no usable episodes")
    z_dim = episodes[0].shape[1]
    print(f"z_struct_aug dim = {z_dim};  {len(episodes)} episodes, "
          f"{sum(len(e) for e in episodes)} frames total", flush=True)

    # --- CAMR (occupancy head OFF: no trustworthy real ACTOR labels) ---
    camr = CAMR(z_struct_aug_dim=z_dim, pose_dim=cfg.camr.pose_dim, imu_dim=cfg.camr.imu_dim,
                window_size=cfg.camr.window_size, hidden_dim=cfg.camr.hidden_dim,
                predict_occupancy=False).to(device)
    if args.camr_init and os.path.exists(args.camr_init):
        sd = torch.load(args.camr_init, map_location=device)
        missing = camr.load_state_dict(sd, strict=False)
        print(f"warm-started CAMR from {args.camr_init} "
              f"(dropped: {len(missing.unexpected_keys)} occupancy keys)", flush=True)
    optim = torch.optim.Adam(camr.parameters(), lr=args.lr)

    # pose/imu are ZERO: the real vehicle has no position source (see docstring)
    zero_pose = torch.zeros(cfg.camr.pose_dim, device=device)
    zero_imu = torch.zeros(cfg.camr.imu_dim, device=device)

    # every valid window start: needs window_size history AND a t+1 target
    Wn = cfg.camr.window_size
    idx = [(e, t) for e in range(len(episodes)) for t in range(Wn + 1, len(episodes[e]) - 1)]
    rng = np.random.default_rng(cfg.seed)
    perm = rng.permutation(len(idx))
    nval = max(1, int(args.val_frac * len(idx)))
    val_i, tr_i = perm[:nval], perm[nval:]
    print(f"{len(idx)} windows ({len(tr_i)} train / {nval} val)", flush=True)

    # VECTORIZED. The naive version stepped the CausalWindowBuffer frame-by-frame
    # (B * window_size tiny forwards per batch); Python overhead pinned the GPU at
    # 7% and would have taken hours. CAMR.forward already accepts a whole window
    # (B, Wn, input_dim), so build the windows as one tensor and do ONE forward.
    eps_dev = [e.to(device) for e in episodes]

    def make_windows(indices, end_offset=0):
        """Causal windows ending at (t - end_offset), stacked: (B, Wn, input_dim)."""
        z = torch.stack([eps_dev[idx[j][0]][idx[j][1] - end_offset - Wn + 1:
                                            idx[j][1] - end_offset + 1] for j in indices])
        B = z.shape[0]
        pose = zero_pose.expand(B, Wn, cfg.camr.pose_dim)
        imu = zero_imu.expand(B, Wn, cfg.camr.imu_dim)
        return camr.fuse(z, pose, imu)

    def batch_loss(indices):
        out = camr(make_windows(indices, 0))
        pred = camr.predict_next(out.h_t)
        z_next = torch.stack([eps_dev[idx[j][0]][idx[j][1] + 1] for j in indices])
        with torch.no_grad():                       # h_{t-1} for the temporal term
            prev = camr(make_windows(indices, 1)).h_t
        return camr_loss(out, z_next, pred, prev_h_t=prev,
                         beta_temp=getattr(cfg.camr, "beta_temp", 0.5))["L_CAMR"]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    best = 1e9
    for ep_i in range(args.epochs):
        camr.train()
        rng.shuffle(tr_i)
        tot, nb = 0.0, 0
        for b in range(0, len(tr_i) - args.batch + 1, args.batch):
            loss = batch_loss(tr_i[b:b + args.batch])
            optim.zero_grad(); loss.backward(); optim.step()
            tot += float(loss); nb += 1
        camr.eval()
        with torch.no_grad():
            vt, vb = 0.0, 0
            for b in range(0, len(val_i), args.batch):
                bi = val_i[b:b + args.batch]
                if len(bi):
                    vt += float(batch_loss(bi)); vb += 1
            vloss = vt / max(1, vb)
        print(f"epoch {ep_i:2d}  train L_CAMR={tot/max(1,nb):.4f}  val L_CAMR={vloss:.4f}", flush=True)
        if vloss < best:
            best = vloss
            torch.save(camr.state_dict(), args.out)
            print(f"  new best -> saved {args.out}", flush=True)

    print(f"\ndone. real CAMR at {args.out} (best val L_CAMR={best:.4f}).", flush=True)
    print("REMINDER: this belief is POSE-BLIND. Retrain the PPO policy pose-blind in Mock "
          "or it will be out of distribution.", flush=True)


if __name__ == "__main__":
    main()
