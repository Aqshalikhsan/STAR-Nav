"""SACR: Structure-Aware Corridor Representation (paper Section 3.2).

Implements, in order:

  z_t          = f_enc(I_t)                                   (encoder)
  theta_corr   = f_geom(GAP(z_t))                              (geometry head)
  alpha        = sigmoid(W_alpha theta_corr + b_alpha)         (channel gate)
  z_struct     = GAP(alpha (*) z_t)                            (gated pooling)
  d_t          = MiDaS(I_t)                                    (depth branch)
  pool(d_t)    = [d_L_bar, d_C_bar, d_R_bar]                   (region pooling)
  z_struct_aug = [z_struct ; pool(d_t)]                        (output to CAMR)

Consistent with Invariant I2, everything except z_struct_aug (i.e. z_t,
theta_corr, d_t) is intermediate and never returned to callers outside
this module in the RL loop -- see `SACR.encode()` vs. `SACR.forward()`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .enet import ENetEncoder, SegmentationHead
from .depth_net import build_depth_net, region_aware_pool


@dataclass
class SACROutput:
    z_struct_aug: torch.Tensor          # (B, d_s + d_d [+ d_d]) -- the only value CAMR sees
    theta_corr: Optional[torch.Tensor] = None   # (B, k) -- internal only (I2)
    seg_logits: Optional[torch.Tensor] = None   # (B, num_classes, H, W) -- training only
    depth: Optional[torch.Tensor] = None        # (B, 1, H, W) -- internal only (I2)
    depth_logvar: Optional[torch.Tensor] = None # (B, regions) -- aleatoric log-var per depth pool region (uncertainty-aware variant)


class GeometryHead(nn.Module):
    """f_geom: 2-layer MLP mapping GAP(z_t) -> theta_corr in R^k."""

    def __init__(self, in_dim: int, hidden: list[int], out_dim: int):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(inplace=True)]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, g_t: torch.Tensor) -> torch.Tensor:
        return self.net(g_t)


class SACR(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        feature_channels: int = 128,
        num_seg_classes: int = 5,
        geom_dim: int = 4,
        geom_hidden: list[int] = (512, 256),
        struct_dim: int = 128,
        depth_pool_regions: int = 3,
        depth_backend: str = "lightweight",
        depth_uncertainty: bool = False,
    ):
        super().__init__()
        self.feature_channels = feature_channels
        self.depth_pool_regions = depth_pool_regions
        self.depth_uncertainty = depth_uncertainty

        self.f_enc = ENetEncoder(in_channels, feature_channels)
        self.seg_head = SegmentationHead(feature_channels, num_seg_classes)
        self.geom_head = GeometryHead(feature_channels, list(geom_hidden), geom_dim)

        # Channel-wise attention gate: alpha = sigmoid(W_alpha theta_corr + b_alpha)
        self.attn_proj = nn.Linear(geom_dim, feature_channels)

        # Project GAP(alpha * z_t) -> z_struct (d_s). Kept as identity-sized
        # linear so struct_dim is configurable independently of feature_channels.
        self.struct_proj = nn.Linear(feature_channels, struct_dim)

        self.depth_net = build_depth_net(depth_backend)

        # Aleatoric-uncertainty head: predicts a per-region log-variance for the
        # pooled depth [d_L, d_C, d_R]. Fed to AGSS so the safety margin can
        # widen where SACR is unsure about lateral clearance (novelty: the
        # shield reacts to *perception confidence*, not just corridor
        # complexity). Appended to z_struct_aug so it travels the same encode()
        # path AGSS reads d_L/d_R from. Off by default -> layout unchanged.
        if depth_uncertainty:
            self.depth_logvar_head = nn.Sequential(
                nn.Linear(feature_channels, feature_channels),
                nn.ReLU(inplace=True),
                nn.Linear(feature_channels, depth_pool_regions),
            )

        self.z_struct_aug_dim = struct_dim + depth_pool_regions * (2 if depth_uncertainty else 1)

    def forward(self, image: torch.Tensor, need_seg: bool = False) -> SACROutput:
        """image: (B, 3, H, W) in [0, 1]."""
        b, _, h, w = image.shape

        z_t = self.f_enc(image)                                   # (B, C, h', w')
        g_t = F.adaptive_avg_pool2d(z_t, 1).flatten(1)             # GAP(z_t) -> (B, C)
        theta_corr = self.geom_head(g_t)                           # (B, k)

        alpha = torch.sigmoid(self.attn_proj(theta_corr))          # (B, C)
        z_gated = z_t * alpha.unsqueeze(-1).unsqueeze(-1)           # channel-wise modulation
        pooled = F.adaptive_avg_pool2d(z_gated, 1).flatten(1)       # GAP(alpha (*) z_t)
        z_struct = self.struct_proj(pooled)                        # (B, d_s)

        depth = self.depth_net(image)                               # (B, 1, H, W)
        pooled_depth = region_aware_pool(depth, self.depth_pool_regions)  # (B, d_d)

        depth_logvar = None
        if self.depth_uncertainty:
            depth_logvar = self.depth_logvar_head(g_t)              # (B, d_d)
            z_struct_aug = torch.cat([z_struct, pooled_depth, depth_logvar], dim=1)  # (B, d_s + 2*d_d)
        else:
            z_struct_aug = torch.cat([z_struct, pooled_depth], dim=1)   # (B, d_s + d_d)

        seg_logits = self.seg_head(z_t, (h, w)) if need_seg else None

        return SACROutput(
            z_struct_aug=z_struct_aug,
            theta_corr=theta_corr,
            seg_logits=seg_logits,
            depth=depth,
            depth_logvar=depth_logvar,
        )

    @torch.no_grad()
    def encode(self, image: torch.Tensor) -> torch.Tensor:
        """Inference-time convenience: returns only z_struct_aug, the sole
        quantity that may cross the SACR module boundary (Invariant I2).
        """
        return self.forward(image, need_seg=False).z_struct_aug


def sacr_loss(
    output: SACROutput,
    seg_target: torch.Tensor,
    theta_corr_gt: torch.Tensor,
    depth_target: Optional[torch.Tensor] = None,
    prev_theta_corr: Optional[torch.Tensor] = None,
    lambda_geom: float = 1.0,
    lambda_depth: float = 0.2,
    lambda_unc: float = 0.5,
    mu_smooth: float = 0.1,
) -> dict[str, torch.Tensor]:
    """L_SACR = L_seg + lambda_geom * L_geom + lambda_depth * L_depth,
       L_geom  = ||theta_corr - theta_corr_gt||^2 + mu * L_smooth
       L_smooth = ||theta_corr_t - theta_corr_{t-1}||^2  (temporal consistency)
       L_depth  = |depth_pred - depth_gt|  (metric depth regression)

    L_depth supervises the MiDaS/lightweight depth branch against the
    simulator's ground-truth *metric* depth map. Without it the depth net is
    left at its random init, so ``z_struct_aug``'s pooled depth (which AGSS
    reads as d_L/d_R for the safety shield) is garbage -- pass ``depth_target``
    in metres to make it real geometry. Backward compatible: omit
    ``depth_target`` and L_depth is 0.
    """
    l_seg = F.cross_entropy(output.seg_logits, seg_target)

    geom_fit = F.mse_loss(output.theta_corr, theta_corr_gt)
    if prev_theta_corr is not None:
        l_smooth = F.mse_loss(output.theta_corr, prev_theta_corr.detach())
    else:
        l_smooth = torch.zeros((), device=output.theta_corr.device)
    l_geom = geom_fit + mu_smooth * l_smooth

    l_unc = torch.zeros((), device=output.theta_corr.device)
    if depth_target is not None and output.depth is not None:
        pred = output.depth
        if pred.shape[-2:] != depth_target.shape[-2:]:
            pred = F.interpolate(pred, size=depth_target.shape[-2:], mode="bilinear", align_corners=False)
        l_depth = F.l1_loss(pred.squeeze(1), depth_target)

        # Heteroscedastic (aleatoric) supervision on the *pooled* depth AGSS
        # actually reads: pool predicted mean and ground-truth into regions and
        # fit a per-region Gaussian NLL, 0.5*e^{-s}*(mu-y)^2 + 0.5*s, so the
        # predicted log-var s calibrates to how wrong each region's clearance
        # tends to be. Clamped for numerical safety.
        if output.depth_logvar is not None:
            regions = output.depth_logvar.shape[-1]
            mu_r = region_aware_pool(pred, regions)                       # (B, R)
            y_r = region_aware_pool(depth_target.unsqueeze(1), regions)   # (B, R)
            s_r = output.depth_logvar.clamp(-6.0, 6.0)                    # (B, R)
            l_unc = (0.5 * torch.exp(-s_r) * (mu_r - y_r) ** 2 + 0.5 * s_r).mean()
    else:
        l_depth = torch.zeros((), device=output.theta_corr.device)

    l_sacr = l_seg + lambda_geom * l_geom + lambda_depth * l_depth + lambda_unc * l_unc
    return {"L_SACR": l_sacr, "L_seg": l_seg, "L_geom": l_geom, "L_depth": l_depth, "L_unc": l_unc}
