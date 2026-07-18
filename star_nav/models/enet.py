"""Lightweight ENet-style semantic-segmentation encoder.

This is the f_enc backbone referenced in Section 3.2 ("SACR employs a
lightweight segmentation encoder f_enc based on the ENet architecture").
It reproduces ENet's defining ideas -- an early downsampling initial
block, bottleneck residual units, and a mix of dilated / asymmetric
convolutions to keep the receptive field large while the parameter count
stays small -- without pulling in a full third-party implementation.

Output: z_t in R^{C x h x w} with h = H/8, w = W/8 (three downsampling
stages), matching the encoder-only role SACR requires: everything after
this file (segmentation head, geometry head, attention gate) consumes
z_t and z_t never leaves SACR (Invariant I2).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class InitialBlock(nn.Module):
    """ENet's initial block: a 3x3/stride-2 conv concatenated with a
    parallel max-pool branch, halving spatial resolution once.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        conv_channels = out_channels - in_channels
        self.conv = nn.Conv2d(in_channels, conv_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.PReLU(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([self.conv(x), self.pool(x)], dim=1)
        return self.act(self.bn(out))


class Bottleneck(nn.Module):
    """ENet bottleneck: 1x1 projection -> main conv (regular / dilated /
    asymmetric / downsampling) -> 1x1 expansion, with a residual skip.
    """

    def __init__(
        self,
        channels: int,
        internal_ratio: int = 4,
        kernel_size: int = 3,
        dilation: int = 1,
        asymmetric: bool = False,
        downsample: bool = False,
        dropout_prob: float = 0.1,
    ):
        super().__init__()
        internal = channels // internal_ratio
        self.downsample = downsample
        out_channels = channels * 2 if downsample else channels

        proj_stride = 2 if downsample else 1
        self.proj = nn.Sequential(
            nn.Conv2d(channels, internal, kernel_size=proj_stride, stride=proj_stride, bias=False)
            if downsample
            else nn.Conv2d(channels, internal, kernel_size=1, bias=False),
            nn.BatchNorm2d(internal),
            nn.PReLU(internal),
        )

        if asymmetric:
            pad = kernel_size // 2
            main = nn.Sequential(
                nn.Conv2d(internal, internal, kernel_size=(kernel_size, 1), padding=(pad, 0), bias=False),
                nn.Conv2d(internal, internal, kernel_size=(1, kernel_size), padding=(0, pad), bias=False),
            )
        else:
            pad = dilation * (kernel_size - 1) // 2
            main = nn.Conv2d(internal, internal, kernel_size=kernel_size, padding=pad, dilation=dilation, bias=False)
        self.main = nn.Sequential(main, nn.BatchNorm2d(internal), nn.PReLU(internal))

        self.expand = nn.Sequential(
            nn.Conv2d(internal, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.dropout = nn.Dropout2d(dropout_prob)
        self.out_act = nn.PReLU(out_channels)

        if downsample:
            self.skip_pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=False)
            self.skip_proj = nn.Conv2d(channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        main = self.expand(self.main(self.proj(x)))
        main = self.dropout(main)

        if self.downsample:
            skip = self.skip_proj(self.skip_pool(x))
        else:
            skip = x

        return self.out_act(main + skip)


class ENetEncoder(nn.Module):
    """Encoder-only ENet: initial block + two downsampling stages, each
    followed by a small stack of regular/dilated/asymmetric bottlenecks.
    """

    def __init__(self, in_channels: int = 3, feature_channels: int = 128):
        super().__init__()
        stage0_channels = 16
        stage1_channels = stage0_channels * 2  # Bottleneck(downsample=True) doubles channel count

        self.initial = InitialBlock(in_channels, stage0_channels)

        self.stage1 = nn.ModuleList([
            Bottleneck(stage0_channels, downsample=True),
        ])
        self.stage1 += [Bottleneck(stage1_channels) for _ in range(3)]

        self.stage2_down = Bottleneck(stage1_channels, downsample=True)
        stage2_channels = stage1_channels * 2
        self.stage2 = nn.ModuleList([
            Bottleneck(stage2_channels, dilation=2),
            Bottleneck(stage2_channels, kernel_size=5, asymmetric=True),
            Bottleneck(stage2_channels, dilation=4),
            Bottleneck(stage2_channels, dilation=8),
        ])

        self.out_proj = nn.Conv2d(stage2_channels, feature_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.initial(x)
        for block in self.stage1:
            out = block(out)
        out = self.stage2_down(out)
        for block in self.stage2:
            out = block(out)
        return self.out_proj(out)  # z_t: (B, C, H/8, W/8)


class SegmentationHead(nn.Module):
    """Lightweight decoder producing per-pixel class logits from z_t, used
    only to compute L_seg during SACR pretraining. Not part of the
    downstream navigation path.
    """

    def __init__(self, feature_channels: int, num_classes: int):
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(feature_channels, feature_channels // 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(feature_channels // 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(feature_channels // 2, feature_channels // 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(feature_channels // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(feature_channels // 4, num_classes, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, z_t: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
        logits = self.up(z_t)
        return F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)
