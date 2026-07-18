"""Phase 3 (Algorithm tab:method-4): real-world adaptation. Fine-tunes
SACR then CAMR on logged real-world flight data to close the
appearance-domain gap D_app, while the PPO policy and AGSS remain frozen
(policy transfer relies on h_t staying geometrically meaningful, not on
re-optimizing navigation behavior in the field).

Expects real-world logs already parsed into the same `Episode` structure
used by `collect_data.py` (rgb frames, pose, imu, and a geometry target
`theta_corr_gt` -- e.g. derived offline from a ground-truth LiDAR
trajectory / corridor survey, since dense pixel-level segmentation labels
are rarely available for real flights). Write your own log parser to
produce `Episode` objects from your flight-log format (rosbag, MAVLink
`.ulg`, etc.) and pass them to `finetune_real`.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ..models.camr import CAMR, camr_loss
from ..models.sacr import SACR
from ..utils.logger import CSVLogger
from .collect_data import Episode, episode_to_tensors


def finetune_sacr_real(sacr: SACR, episodes: list[Episode], cfg, device, logger: CSVLogger) -> SACR:
    sacr.train()
    optim = torch.optim.Adam(sacr.parameters(), lr=cfg.finetune_real.sacr_lr)

    step = 0
    for epoch in range(cfg.finetune_real.epochs):
        for ep in episodes:
            tensors = episode_to_tensors(ep, device)
            batch_size = cfg.finetune_real.batch_size
            for start in range(0, tensors["rgb"].shape[0], batch_size):
                rgb = tensors["rgb"][start:start + batch_size]
                theta_gt = tensors["theta_corr_gt"][start:start + batch_size]
                if rgb.shape[0] == 0:
                    continue

                out = sacr(rgb, need_seg=False)
                # Algorithm tab:method-4 fine-tunes SACR with Loss = L_geom
                # only (no L_seg): real flights rarely have dense per-pixel
                # segmentation ground truth, only geometry/pose references.
                l_geom = F.mse_loss(out.theta_corr, theta_gt)

                optim.zero_grad()
                l_geom.backward()
                optim.step()

                if step % cfg.training.log_every == 0:
                    logger.log(step, {"L_geom_real": l_geom.item()})
                step += 1

    sacr.eval()
    for p in sacr.parameters():
        p.requires_grad_(False)
    return sacr


def finetune_camr_real(sacr: SACR, camr: CAMR, episodes: list[Episode], cfg, device, logger: CSVLogger) -> CAMR:
    camr.train()
    optim = torch.optim.Adam(camr.parameters(), lr=cfg.finetune_real.camr_lr)
    window_size = cfg.camr.window_size

    step = 0
    for epoch in range(cfg.finetune_real.epochs):
        for ep in episodes:
            if len(ep.rgb) < window_size + 1:
                continue
            tensors = episode_to_tensors(ep, device)
            with torch.no_grad():
                z_struct_aug = sacr(tensors["rgb"], need_seg=False).z_struct_aug

            x_seq = torch.cat([z_struct_aug, tensors["pose"], tensors["imu"]], dim=-1)
            L = x_seq.shape[0]

            for t in range(window_size - 1, L - 1):
                window = x_seq[t - window_size + 1: t + 1].unsqueeze(0)
                out = camr(window)
                predicted_next = camr.predict_next(out.h_t)
                target_next = z_struct_aug[t + 1].unsqueeze(0)

                # Algorithm tab:method-4: Fine-Tune CAMR, Loss = L_pred only.
                l_pred = F.mse_loss(predicted_next, target_next)

                optim.zero_grad()
                l_pred.backward()
                optim.step()

                if step % cfg.training.log_every == 0:
                    logger.log(step, {"L_pred_real": l_pred.item()})
                step += 1

    camr.eval()
    for p in camr.parameters():
        p.requires_grad_(False)
    return camr


def finetune_real(sacr: SACR, camr: CAMR, episodes: list[Episode], cfg, device, logger: CSVLogger):
    """Sequential fine-tuning exactly as ordered in Algorithm tab:method-4:
    freeze perception -> adapt to real observations -> (policy/AGSS stay
    frozen throughout, so they are simply not passed into this function).
    """
    sacr = finetune_sacr_real(sacr, episodes, cfg, device, logger)
    camr = finetune_camr_real(sacr, camr, episodes, cfg, device, logger)
    return sacr, camr
