"""Phase 1 (perception-first pretraining, Section 3.5 / Algorithms
tab:method-1 and tab:method-2): train SACR on framewise supervision, then
train CAMR on sequences of (frozen) SACR outputs. PPO/AGSS are not
involved in this phase at all.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from ..models.sacr import SACR, sacr_loss
from ..models.camr import CAMR, camr_loss
from ..utils.logger import CSVLogger
from .collect_data import Episode, episode_to_tensors


class _FrameDataset(Dataset):
    """Flattens all episodes into individual (frame, seg, theta_corr_gt)
    samples for SACR's i.i.d. supervised pretraining step.
    """

    def __init__(self, episodes: list[Episode]):
        self.rgb, self.seg, self.depth, self.theta = [], [], [], []
        for ep in episodes:
            self.rgb.extend(ep.rgb)
            self.seg.extend(ep.seg_mask)
            self.depth.extend(ep.depth)
            self.theta.extend(ep.theta_corr_gt)

    def __len__(self):
        return len(self.rgb)

    def __getitem__(self, idx):
        rgb = torch.from_numpy(self.rgb[idx]).float().permute(2, 0, 1) / 255.0
        seg = torch.from_numpy(self.seg[idx]).long()
        depth = torch.from_numpy(np.asarray(self.depth[idx])).float()
        theta = torch.from_numpy(self.theta[idx]).float()
        return rgb, seg, depth, theta


def train_sacr(sacr: SACR, episodes: list[Episode], cfg, device, logger: CSVLogger) -> SACR:
    dataset = _FrameDataset(episodes)
    loader = DataLoader(dataset, batch_size=cfg.training.perception_batch_size, shuffle=True, drop_last=True)
    optim = torch.optim.Adam(sacr.parameters(), lr=cfg.sacr.lr)

    sacr.train()
    step = 0
    for epoch in range(cfg.training.perception_epochs):
        for rgb, seg, depth_gt, theta_gt in loader:
            rgb, seg, depth_gt, theta_gt = rgb.to(device), seg.to(device), depth_gt.to(device), theta_gt.to(device)

            out = sacr(rgb, need_seg=True)
            losses = sacr_loss(out, seg, theta_gt, depth_target=depth_gt,
                               lambda_geom=cfg.sacr.lambda_geom,
                               lambda_depth=getattr(cfg.sacr, "lambda_depth", 0.2),
                               lambda_unc=getattr(cfg.sacr, "lambda_unc", 0.5),
                               mu_smooth=cfg.sacr.mu_smooth)

            optim.zero_grad()
            losses["L_SACR"].backward()
            optim.step()

            if step % cfg.training.log_every == 0:
                logger.log(step, {k: v.item() for k, v in losses.items()})
            step += 1
    return sacr


def train_camr(sacr: SACR, camr: CAMR, episodes: list[Episode], cfg, device, logger: CSVLogger) -> CAMR:
    sacr.eval()
    for p in sacr.parameters():
        p.requires_grad_(False)

    optim = torch.optim.Adam(camr.parameters(), lr=cfg.camr.lr)
    window_size = cfg.camr.window_size
    step = 0

    camr.train()
    for epoch in range(cfg.training.perception_epochs):
        for ep in episodes:
            if len(ep.rgb) < window_size + 1:
                continue
            tensors = episode_to_tensors(ep, device)
            with torch.no_grad():
                z_struct_aug = sacr(tensors["rgb"], need_seg=False).z_struct_aug  # (L, d_s+d_d)

            x_seq = torch.cat([z_struct_aug, tensors["pose"], tensors["imu"]], dim=-1)  # (L, input_dim)
            occ_seq = tensors.get("occ")                                       # (L, 2) or None
            L = x_seq.shape[0]

            prev_h_t = None
            for t in range(window_size - 1, L - 1):
                window = x_seq[t - window_size + 1: t + 1].unsqueeze(0)  # (1, T, input_dim)
                out = camr(window)
                predicted_next = camr.predict_next(out.h_t)
                target_next = z_struct_aug[t + 1].unsqueeze(0)

                # Anticipatory occupancy target for the window ending at step t.
                occ_logits = occ_target = None
                if camr.use_occupancy and occ_seq is not None:
                    occ_logits = camr.predict_occupancy(out.h_t)
                    occ_target = occ_seq[t].unsqueeze(0)

                losses = camr_loss(out, target_next, predicted_next, prev_h_t,
                                   beta_temp=cfg.camr.beta_temp,
                                   occ_logits=occ_logits, occ_target=occ_target,
                                   lambda_occ=getattr(cfg.camr, "lambda_occ", 0.5))

                optim.zero_grad()
                losses["L_CAMR"].backward()
                optim.step()

                prev_h_t = out.h_t.detach()

                if step % cfg.training.log_every == 0:
                    logger.log(step, {k: v.item() for k, v in losses.items()})
                step += 1

    for p in sacr.parameters():
        p.requires_grad_(True)
    return camr
