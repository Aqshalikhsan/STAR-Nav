"""Monocular depth branch used inside SACR (Section 3.2, Eq. d_t = MiDaS(I_t)).

Two backends are supported:

  * "midas_hub"  -- loads the real pretrained MiDaS small model via
                    ``torch.hub`` (requires internet access on first run).
                    Use this for results you intend to compare against the
                    paper's reported numbers.
  * "lightweight" (default) -- a small trainable encoder-decoder with the
                    same input/output contract, so the full pipeline is
                    runnable offline/without downloading external weights.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class LightweightDepthNet(nn.Module):
    """Small encoder-decoder regressing a single-channel depth map from an
    RGB frame. Not a substitute for MiDaS in accuracy, only in interface.
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 32):
        super().__init__()

        def down(c_in, c_out):
            return nn.Sequential(
                nn.Conv2d(c_in, c_out, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
            )

        def up(c_in, c_out):
            return nn.Sequential(
                nn.ConvTranspose2d(c_in, c_out, 4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
            )

        self.enc1 = down(in_channels, base_channels)
        self.enc2 = down(base_channels, base_channels * 2)
        self.enc3 = down(base_channels * 2, base_channels * 4)
        self.dec3 = up(base_channels * 4, base_channels * 2)
        self.dec2 = up(base_channels * 2, base_channels)
        self.dec1 = up(base_channels, base_channels)
        self.out_conv = nn.Conv2d(base_channels, 1, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        d3 = self.dec3(e3)
        d2 = self.dec2(d3)
        d1 = self.dec1(d2)
        depth = torch.relu(self.out_conv(d1))
        return torch.nn.functional.interpolate(depth, size=(h, w), mode="bilinear", align_corners=False)


def build_depth_net(backend: str = "lightweight") -> nn.Module:
    if backend == "lightweight":
        return LightweightDepthNet()
    if backend == "midas_hub":
        model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        return model
    raise ValueError(f"Unknown depth backend: {backend}")


def region_aware_pool(depth: torch.Tensor, regions: int = 3) -> torch.Tensor:
    """pool(d_t) = [d_L_bar, d_C_bar, d_R_bar] (Eq. in Section 3.2):
    mean-pool the depth map over `regions` equal vertical bands, ordered
    left to right.

    Args:
        depth: (B, 1, H, W) or (B, H, W) dense depth map.
    Returns:
        (B, regions) pooled depth statistics.
    """
    if depth.dim() == 3:
        depth = depth.unsqueeze(1)
    b, _, h, w = depth.shape
    band = w // regions
    pooled = []
    for i in range(regions):
        start = i * band
        end = (i + 1) * band if i < regions - 1 else w
        pooled.append(depth[:, :, :, start:end].mean(dim=(1, 2, 3)))
    return torch.stack(pooled, dim=1)  # (B, regions)
