"""SACR ablation variants S1-S7 (paper Section 5.2, Table 9).

Each variant toggles internal SACR components; mIoU and geometry MAE are measured
directly against the simulator's ground-truth segmentation / corridor geometry.

  S1: ENet encoder only
  S2: + geometry head, no attention
  S3: + attention (random, no geometry conditioning)
  S4: + attention (geometry-conditioned)
  S5: + depth pooling (no attention)
  S6: + depth pooling (with attention)
  S7: SACR full (S6 + temporal smoothness L_smooth, a training-time term)

NOTE ON SR: mIoU / Geo. MAE here are genuine perception measurements. The
Success_Rate column of Table 9 requires a *policy trained per variant* (the
paper trains the ablation as a separate campaign); this module measures
perception, and the SR column is filled only when a variant's representation is
dimension-compatible with the loaded policy (documented in the exporter).
"""
from __future__ import annotations

import numpy as np
import torch

from ..models.sacr import SACR

# variant -> SACR ablation kwargs (+ 'smooth' training flag for S7)
CONFIGS: dict[str, dict] = {
    "S1": dict(use_geometry=False, attention_mode="none",     use_depth=False),
    "S2": dict(use_geometry=True,  attention_mode="none",     use_depth=False),
    "S3": dict(use_geometry=True,  attention_mode="random",   use_depth=False),
    "S4": dict(use_geometry=True,  attention_mode="geometry", use_depth=False),
    "S5": dict(use_geometry=True,  attention_mode="none",     use_depth=True),
    "S6": dict(use_geometry=True,  attention_mode="geometry", use_depth=True),
    "S7": dict(use_geometry=True,  attention_mode="geometry", use_depth=True, smooth=True),
}


def build_variant(cfg, variant: str, base_sacr: SACR, device) -> SACR:
    """Construct a SACR with the variant's toggles, copying every weight whose
    shape matches the base (trained) SACR -- so the ablation runs on the trained
    encoder/heads with the selected components disabled."""
    kw = {k: v for k, v in CONFIGS[variant].items() if k != "smooth"}
    m = SACR(
        in_channels=cfg.sacr.in_channels, feature_channels=cfg.sacr.feature_channels,
        num_seg_classes=cfg.sacr.num_seg_classes, geom_dim=cfg.sacr.geom_dim,
        geom_hidden=cfg.sacr.geom_hidden, struct_dim=cfg.sacr.struct_dim,
        depth_pool_regions=cfg.sacr.depth_pool_regions, **kw,
    ).to(device)
    base = base_sacr.state_dict()
    tgt = m.state_dict()
    copied = {k: v for k, v in base.items() if k in tgt and tgt[k].shape == v.shape}
    tgt.update(copied); m.load_state_dict(tgt)
    m.eval()
    return m


@torch.no_grad()
def perception_metrics(sacr: SACR, env, scenario: str, weather: str,
                       num_classes: int, n_frames: int = 20, device=None):
    """Measure mIoU (segmentation vs GT seg_mask) and Geo. MAE (theta_corr vs
    GT theta_corr) over n_frames of a scenario. Returns (mIoU_percent, geo_mae).
    Geo. MAE is NaN when the variant has no geometry head (S1)."""
    inter = np.zeros(num_classes); union = np.zeros(num_classes)
    geo_errs = []
    obs = env.reset(scenario=scenario, weather=weather)
    for _ in range(n_frames):
        rgb = torch.as_tensor(obs.rgb, dtype=torch.float32, device=device).permute(2, 0, 1).unsqueeze(0) / 255.0
        out = sacr(rgb, need_seg=True)
        info = env.step(np.array([0.5, 0.0, 0.0, 0.0], dtype=np.float32)).info
        if out.seg_logits is not None:
            pred = out.seg_logits.argmax(1).squeeze(0).cpu().numpy()
            gt = np.asarray(info.seg_mask)
            if pred.shape != gt.shape:  # align if head resizes
                continue
            for c in range(num_classes):
                p, g = pred == c, gt == c
                inter[c] += np.logical_and(p, g).sum()
                union[c] += np.logical_or(p, g).sum()
        if out.theta_corr is not None:
            gt_theta = np.asarray(info.theta_corr_gt, dtype=np.float32)
            geo_errs.append(float(np.mean(np.abs(out.theta_corr.squeeze(0).cpu().numpy() - gt_theta))))
        obs = env.reset(scenario=scenario, weather=weather)
    valid = union > 0
    miou = float(np.mean(inter[valid] / union[valid]) * 100.0) if valid.any() else 0.0
    geo_mae = float(np.mean(geo_errs)) if geo_errs else float("nan")
    return miou, geo_mae
