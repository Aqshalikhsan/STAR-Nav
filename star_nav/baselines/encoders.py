"""Observation encoders for the baseline agents. Each maps the shared
observation (monocular RGB + VIO pose + raw IMU) to a feature vector. Baselines
do NOT use SACR/CAMR -- these are their own perception front-ends, matching each
method's original formulation (CNN for PPO/TD3/Mem-DRL/NavRL, a small ViT for
ViT-PPO).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ObsSpec:
    rgb_hw: tuple = (64, 64)     # RGB is resized to this before the encoder
    pose_dim: int = 7
    imu_dim: int = 6


def preprocess(obs, spec: ObsSpec, device):
    """EnvObservation -> (rgb (1,3,H,W) in [0,1], proprio (1, pose+imu))."""
    rgb = torch.as_tensor(np.asarray(obs.rgb), dtype=torch.float32, device=device)
    rgb = rgb.permute(2, 0, 1).unsqueeze(0) / 255.0
    rgb = F.interpolate(rgb, size=spec.rgb_hw, mode="bilinear", align_corners=False)
    proprio = torch.as_tensor(
        np.concatenate([np.asarray(obs.pose, np.float32), np.asarray(obs.imu, np.float32)]),
        dtype=torch.float32, device=device).unsqueeze(0)
    return rgb, proprio


class CNNEncoder(nn.Module):
    """Small 3-conv CNN over RGB, fused with proprioception -> feature_dim."""

    def __init__(self, spec: ObsSpec, feature_dim: int = 256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU(inplace=True),
        )
        with torch.no_grad():
            n = self.conv(torch.zeros(1, 3, *spec.rgb_hw)).flatten(1).shape[1]
        self.proj = nn.Sequential(nn.Linear(n + spec.pose_dim + spec.imu_dim, feature_dim), nn.ReLU(inplace=True))
        self.feature_dim = feature_dim

    def forward(self, rgb, proprio):
        v = self.conv(rgb).flatten(1)
        return self.proj(torch.cat([v, proprio], dim=1))


class PatchViTEncoder(nn.Module):
    """Compact ViT: non-overlapping patch embedding + a few transformer blocks,
    fused with proprioception. Represents the ViT-PPO baseline's front-end."""

    def __init__(self, spec: ObsSpec, feature_dim: int = 256, patch: int = 8,
                 dim: int = 128, depth: int = 2, heads: int = 4):
        super().__init__()
        h, w = spec.rgb_hw
        assert h % patch == 0 and w % patch == 0, "rgb_hw must be divisible by patch"
        self.n_patch = (h // patch) * (w // patch)
        self.embed = nn.Conv2d(3, dim, patch, stride=patch)
        self.pos = nn.Parameter(torch.zeros(1, self.n_patch, dim))
        layer = nn.TransformerEncoderLayer(dim, heads, dim * 2, batch_first=True, activation="gelu")
        self.tf = nn.TransformerEncoder(layer, depth)
        self.proj = nn.Sequential(nn.Linear(dim + spec.pose_dim + spec.imu_dim, feature_dim), nn.ReLU(inplace=True))
        self.feature_dim = feature_dim

    def forward(self, rgb, proprio):
        t = self.embed(rgb).flatten(2).transpose(1, 2) + self.pos   # (B, N, dim)
        t = self.tf(t).mean(dim=1)                                  # global token
        return self.proj(torch.cat([t, proprio], dim=1))


class FrameCNN(nn.Module):
    """RGB -> single embedding vector (no proprio). Per-frame front-end for the
    temporal baselines (Mem-DRL, DTPPO)."""

    def __init__(self, spec: ObsSpec, dim: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU(inplace=True),
        )
        with torch.no_grad():
            n = self.conv(torch.zeros(1, 3, *spec.rgb_hw)).flatten(1).shape[1]
        self.proj = nn.Sequential(nn.Linear(n, dim), nn.ReLU(inplace=True))
        self.dim = dim

    def forward(self, rgb):
        return self.proj(self.conv(rgb).flatten(1))


class SpatialPatchTransformer(nn.Module):
    """RGB -> single embedding via a patch-token transformer (the *spatial*
    transformer of the dual-transformer DTPPO baseline)."""

    def __init__(self, spec: ObsSpec, dim: int = 128, patch: int = 8, depth: int = 2, heads: int = 4):
        super().__init__()
        h, w = spec.rgb_hw
        assert h % patch == 0 and w % patch == 0
        self.embed = nn.Conv2d(3, dim, patch, stride=patch)
        self.pos = nn.Parameter(torch.zeros(1, (h // patch) * (w // patch), dim))
        layer = nn.TransformerEncoderLayer(dim, heads, dim * 2, batch_first=True, activation="gelu")
        self.tf = nn.TransformerEncoder(layer, depth)
        self.dim = dim

    def forward(self, rgb):
        t = self.embed(rgb).flatten(2).transpose(1, 2) + self.pos
        return self.tf(t).mean(dim=1)


def build_encoder(kind: str, spec: ObsSpec, feature_dim: int = 256) -> nn.Module:
    if kind == "cnn":
        return CNNEncoder(spec, feature_dim)
    if kind == "vit":
        return PatchViTEncoder(spec, feature_dim)
    raise ValueError(f"unknown encoder {kind!r}")
