"""Phase-1 CAMR pretraining (Section 3.3) with per-epoch checkpointing +
best-model tracking + resume, mirroring scripts/train_sacr.py.

CAMR fuses, per frame in a *temporal sequence*:
    x_t = [ z_struct_aug (frozen SACR, from rgb) ; pose_t ; imu_t ]
runs a forward + reverse LSTM over the causal window W_t, and is trained to
predict the next frame's z_struct_aug (L_pred) with a temporal-smoothness term
on the belief (L_temp):  L_CAMR = L_pred + beta * L_temp.

Data: the same captured perception npz used for SACR (rgb / seg_mask /
theta_corr_gt / pose), which is stored in temporal capture order. This script:
  * loads the FROZEN SACR (checkpoints/gazebo/sacr.pt) and precomputes z_struct_aug
    for every frame (no grad),
  * splits the stream into episodes at large backward jumps in pose-x (each
    glide pass is one episode),
  * synthesizes a 6-D imu = [body-frame accel (+gravity) ; gyro] from the pose
    trajectory (the rig carries no IMU sensor; this is the standard IMU model
    for a level, yaw-steered platform),
  * trains CAMR on causal windows, keeping L_temp valid within contiguous
    per-episode blocks.

Checkpoints (in --out-dir, default checkpoints/gazebo/): camr_last.pt (full state,
every epoch), camr_best.pt (lowest val L_CAMR), camr.pt (plain best weights for
PPO/AGSS). Resume: --resume checkpoints/gazebo/camr_last.pt.

Examples:
  python scripts/train_camr.py --data data/sacr_gazebo_dataset.npz --epochs 60
  python scripts/train_camr.py --resume checkpoints/gazebo/camr_best.pt --epochs 120
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from star_nav.models.sacr import SACR
from star_nav.models.camr import CAMR, camr_loss
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


# Anticipatory-occupancy ground truth, derived directly from the captured
# CLASS_ACTOR segmentation (no world-frame actor coordinates needed). CLASS_ACTOR
# is 4 (mirrors star_nav/envs/mock_env.py). A worker is "present left/right" in a
# frame when its actor-pixel count in that image half exceeds MIN_ACTOR_PX (a
# nearer/larger worker => more pixels, so this doubles as a rough proximity
# gate). The occupancy target at step t is the OR of that presence over the next
# OCC_HORIZON frames -- so the belief learns to ANTICIPATE where a crossing
# worker will be, matching CAMR.predict_occupancy's [left, right] head.
CLASS_ACTOR = 4
OCC_HORIZON = 8
MIN_ACTOR_PX = 30


def seg_future_occupancy(seg, episodes, horizon=OCC_HORIZON, min_px=MIN_ACTOR_PX):
    """(N, 2) float32 future [left, right] actor occupancy from seg masks.
    Future OR stays within each episode (never looks across a glide boundary)."""
    N, H, W = seg.shape
    a = (seg == CLASS_ACTOR)
    left_now = a[:, :, : W // 2].sum(axis=(1, 2)) > min_px      # (N,)
    right_now = a[:, :, W // 2 :].sum(axis=(1, 2)) > min_px
    now = np.stack([left_now, right_now], axis=1).astype(np.float32)  # (N, 2)
    occ = np.zeros((N, 2), dtype=np.float32)
    for (lo, hi) in episodes:
        for t in range(lo, hi):
            fut_hi = min(hi, t + 1 + horizon)
            if fut_hi > t + 1:
                occ[t] = now[t + 1 : fut_hi].max(axis=0)
    return occ


def split_episodes(pose, x_jump=-5.0):
    """Split the temporally-ordered stream into episodes at large backward
    jumps in x (each glide pass restarts near the corridor start)."""
    x = pose[:, 0]
    cut = np.where(np.diff(x) < x_jump)[0] + 1
    bounds = [0, *cut.tolist(), len(pose)]
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]


def synth_imu(pose, dt=0.11, g=9.81):
    """6-D imu = [accel_body(3) ; gyro_body(3)] synthesized from the pose
    trajectory for a level, yaw-steered rig. accel includes the accelerometer's
    reaction to gravity (+g on body z); gyro is dominated by yaw rate.
    """
    pos = pose[:, :3]
    qx, qy, qz, qw = pose[:, 3], pose[:, 4], pose[:, 5], pose[:, 6]
    yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    yaw_u = np.unwrap(yaw)

    v = np.gradient(pos, dt, axis=0)          # world-frame velocity
    a_world = np.gradient(v, dt, axis=0)      # world-frame acceleration
    yaw_rate = np.gradient(yaw_u, dt)         # about body z

    c, s = np.cos(yaw), np.sin(yaw)
    ax_b = c * a_world[:, 0] + s * a_world[:, 1]      # rotate world->body (yaw only)
    ay_b = -s * a_world[:, 0] + c * a_world[:, 1]
    az_b = a_world[:, 2] + g                          # + reaction to gravity
    gyro = np.stack([np.zeros_like(yaw_rate), np.zeros_like(yaw_rate), yaw_rate], axis=1)
    accel = np.stack([ax_b, ay_b, az_b], axis=1)
    return np.concatenate([accel, gyro], axis=1).astype(np.float32)


@torch.no_grad()
def precompute_z(sacr, rgb, device, batch=8):
    """z_struct_aug for every frame (frozen SACR, no grad)."""
    sacr.eval()
    zs = []
    for b in range(0, len(rgb), batch):
        r = torch.from_numpy(rgb[b:b + batch]).float().permute(0, 3, 1, 2).to(device) / 255.0
        zs.append(sacr(r, need_seg=False).z_struct_aug.cpu())
    return torch.cat(zs, 0).numpy()


def episode_windows(x_seq, z, T, occ=None):
    """All causal windows for one episode (oldest->newest) + their next-z
    targets. windows: (M, T, input_dim); targets: (M, z_dim). Contiguous, so
    L_temp over consecutive rows is valid within this block. If ``occ`` (L, 2) is
    given, also returns the future-occupancy target at each window's ENDING frame
    (M, 2) -- the frame whose belief h_t drives predict_occupancy.
    """
    L = len(x_seq)
    if L < T + 1:
        return None, None, None
    idx = np.arange(T - 1, L - 1)
    win = np.stack([x_seq[t - T + 1:t + 1] for t in idx], 0)
    tgt = z[idx + 1]
    occ_tgt = occ[idx] if occ is not None else None
    return win, tgt, occ_tgt


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data/sacr_gazebo_dataset.npz")
    p.add_argument("--sacr-ckpt", default="checkpoints/gazebo/sacr.pt", help="Frozen SACR weights.")
    p.add_argument("--config", default=None)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--out-dir", default="checkpoints/gazebo")
    p.add_argument("--resume", default=None)
    p.add_argument("--val-frac", type=float, default=0.15,
                   help="Fraction of each episode's windows (contiguous tail) held out for validation.")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    os.makedirs(args.out_dir, exist_ok=True)
    T = cfg.camr.window_size

    d = np.load(args.data)
    rgb, pose = d["rgb"], d["pose"]
    seg = d["seg_mask"] if "seg_mask" in d.files else None
    N = len(rgb)

    sacr = build_sacr(cfg, device)
    sacr.load_state_dict(torch.load(args.sacr_ckpt, map_location=device))
    for pm in sacr.parameters():
        pm.requires_grad_(False)
    print(f"device={device}  loaded frozen SACR from {args.sacr_ckpt}", flush=True)

    z = precompute_z(sacr, rgb, device)                     # (N, z_dim)
    imu = synth_imu(pose)                                    # (N, 6)
    x_all = np.concatenate([z, pose.astype(np.float32), imu], axis=1)  # (N, input_dim)
    eps = split_episodes(pose)

    # Anticipatory occupancy: build the future [left,right] actor target from the
    # captured seg masks (novelty CAMR head). Off if the head is disabled or the
    # dataset has no seg / no actor pixels.
    occ_on = getattr(cfg.camr, "predict_occupancy", False)
    occ_full = None
    if occ_on and seg is not None and (seg == CLASS_ACTOR).any():
        occ_full = seg_future_occupancy(seg, eps)
        pos = occ_full.sum(axis=0)
        print(f"occupancy GT: frames-with-future-actor L={int(pos[0])} R={int(pos[1])} / {N}", flush=True)
    elif occ_on:
        print("WARNING: predict_occupancy=True but no actor pixels in seg -- L_occ will be 0.", flush=True)
    print(f"frames={N}  episodes={eps}  z_dim={z.shape[1]}  input_dim={x_all.shape[1]}  "
          f"occupancy_head={occ_on}", flush=True)

    # Build contiguous train/val window blocks per episode (val = tail fraction).
    train_blocks, val_blocks = [], []
    for (a, b) in eps:
        occ_ep = occ_full[a:b] if occ_full is not None else None
        win, tgt, occ_tgt = episode_windows(x_all[a:b], z[a:b], T, occ_ep)
        if win is None:
            continue
        nval = max(1, int(len(win) * args.val_frac))
        ot_tr = occ_tgt[:-nval] if occ_tgt is not None else None
        ot_va = occ_tgt[-nval:] if occ_tgt is not None else None
        train_blocks.append((win[:-nval], tgt[:-nval], ot_tr))
        val_blocks.append((win[-nval:], tgt[-nval:], ot_va))
    n_tr = sum(len(w) for w, _, _ in train_blocks)
    n_va = sum(len(w) for w, _, _ in val_blocks)
    print(f"windows: train={n_tr} val={n_va} (T={T})", flush=True)

    camr = CAMR(z_struct_aug_dim=sacr.z_struct_aug_dim, pose_dim=cfg.camr.pose_dim,
                imu_dim=cfg.camr.imu_dim, window_size=T, hidden_dim=cfg.camr.hidden_dim,
                predict_occupancy=occ_on, occ_dim=getattr(cfg.camr, "occ_dim", 2)).to(device)
    optim = torch.optim.Adam(camr.parameters(), lr=cfg.camr.lr)
    beta = cfg.camr.beta_temp
    lambda_occ = getattr(cfg.camr, "lambda_occ", 0.5)

    start_epoch, best_val = 0, float("inf")
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        camr.load_state_dict(ck["model"])
        if "optim" in ck:
            optim.load_state_dict(ck["optim"])
        start_epoch = ck.get("epoch", 0) + 1
        best_val = ck.get("best_val", float("inf"))
        print(f"resumed from {args.resume}: start_epoch={start_epoch} best_val={best_val:.4f}", flush=True)

    # L_pred + L_temp (+ lambda_occ * L_occ) for one contiguous window block
    # (L_temp over consecutive windows is valid because the block is contiguous
    # within an episode; occ_np is the future-occupancy target at each window's
    # ending frame).
    def block_losses(win_np, tgt_np, occ_np):
        win = torch.from_numpy(win_np).float().to(device)
        tgt = torch.from_numpy(tgt_np).float().to(device)
        out = camr(win)
        pred = camr.predict_next(out.h_t)
        l_pred = torch.nn.functional.mse_loss(pred, tgt)
        if out.h_t.shape[0] > 1:
            l_temp = torch.nn.functional.mse_loss(out.h_t[1:], out.h_t[:-1].detach())
        else:
            l_temp = torch.zeros((), device=device)
        l_occ = torch.zeros((), device=device)
        if camr.use_occupancy and occ_np is not None:
            occ_logits = camr.predict_occupancy(out.h_t)
            occ_t = torch.from_numpy(occ_np).float().to(device)
            l_occ = torch.nn.functional.binary_cross_entropy_with_logits(occ_logits, occ_t)
        return l_pred, l_temp, l_occ, l_pred + beta * l_temp + lambda_occ * l_occ

    def evaluate():
        camr.eval()
        with torch.no_grad():
            lp, lt, lo, lc = [], [], [], []
            for win, tgt, occ_tgt in val_blocks:
                a, b, o, c = block_losses(win, tgt, occ_tgt)
                lp.append(a.item()); lt.append(b.item()); lo.append(o.item()); lc.append(c.item())
        return float(np.mean(lp)), float(np.mean(lt)), float(np.mean(lo)), float(np.mean(lc))

    def save(path, epoch, model_only=False):
        if model_only:
            torch.save(camr.state_dict(), path)
        else:
            torch.save({"model": camr.state_dict(), "optim": optim.state_dict(),
                        "epoch": epoch, "best_val": best_val}, path)

    last_pt = os.path.join(args.out_dir, "camr_last.pt")
    best_pt = os.path.join(args.out_dir, "camr_best.pt")
    plain_pt = os.path.join(args.out_dir, "camr.pt")

    t0 = time.time()
    order = list(range(len(train_blocks)))
    for epoch in range(start_epoch, args.epochs):
        camr.train()
        np.random.shuffle(order)
        for i in order:
            win, tgt, occ_tgt = train_blocks[i]
            _, _, _, loss = block_losses(win, tgt, occ_tgt)
            optim.zero_grad(); loss.backward(); optim.step()

        save(last_pt, epoch)
        vpred, vtemp, vocc, vscore = evaluate()
        improved = vscore < best_val
        if improved:
            best_val = vscore
            save(best_pt, epoch); save(plain_pt, epoch, model_only=True); save(last_pt, epoch)
        print(f"epoch {epoch:3d}  VAL L_pred={vpred:.4f} L_temp={vtemp:.4f} L_occ={vocc:.4f} "
              f"L_CAMR={vscore:.4f} best={best_val:.4f}{'  <- new best' if improved else ''}  "
              f"({time.time()-t0:.0f}s)", flush=True)

    print(f"done. last={last_pt}  best={best_pt} (val L_CAMR={best_val:.4f})  weights={plain_pt}", flush=True)


if __name__ == "__main__":
    main()
