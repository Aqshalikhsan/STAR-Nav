"""ViT-PPO baseline as a **dual-transformer** policy (after DTPPO, Wei et al.
[66]), trained with PPO.

Transformer #1 (spatial) attends over image patches to produce a per-frame
embedding; transformer #2 (temporal) attends over the last T frame embeddings to
produce a motion-aware feature, which is fused with proprioception. Best-effort
reimplementation of the dual-transformer architecture at comparable capacity.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .encoders import SpatialPatchTransformer
from .ppo import PPOBase


class DTPPOAgent(PPOBase):
    feedforward = False

    def __init__(self, obs_spec, action_dim, device, name="ViTPPO",
                 dim=128, window=8, t_depth=2, heads=4, **kw):
        self._dim, self._win_len, self._t_depth, self._heads = dim, window, t_depth, heads
        super().__init__(obs_spec, action_dim, device, name=name, **kw)
        self.finish_init()

    def build_encoder(self):
        s = self.spec
        self.spatial = SpatialPatchTransformer(s, self._dim, heads=self._heads)
        self.tpos = nn.Parameter(torch.zeros(1, self._win_len, self._dim))
        layer = nn.TransformerEncoderLayer(self._dim, self._heads, self._dim * 2,
                                           batch_first=True, activation="gelu")
        self.temporal = nn.TransformerEncoder(layer, self._t_depth)
        self.proj = nn.Sequential(
            nn.Linear(self._dim + s.pose_dim + s.imu_dim, self.feature_dim), nn.ReLU(inplace=True))
        self._win = []

    def reset_state(self):
        self._win = []

    def _temporal_feat(self, emb_window, proprio):
        seq = torch.stack(emb_window, dim=1)                 # (1, L, dim)
        L = seq.shape[1]
        seq = seq + self.tpos[:, :L]
        out = self.temporal(seq)[:, -1]                      # most-recent token
        return self.proj(torch.cat([out, proprio], dim=-1))

    def encode_step(self, rgb, proprio):
        emb = self.spatial(rgb)
        self._win.append(emb.detach())
        self._win = self._win[-self._win_len:]
        return self._temporal_feat(self._win, proprio)

    def encode_seq(self, rgb_seq, proprio_seq, dones):
        win, feats = [], []
        for t in range(rgb_seq.shape[0]):
            if t > 0 and dones[t - 1] > 0.5:
                win = []
            win.append(self.spatial(rgb_seq[t:t + 1]))
            win = win[-self._win_len:]
            feats.append(self._temporal_feat(win, proprio_seq[t:t + 1]))
        return torch.cat(feats, dim=0)
