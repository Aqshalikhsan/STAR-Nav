"""Mem-DRL baseline: memory-augmented deep RL with a **dual-memory** recurrent
architecture (after Haddad & Khudher, Dual-Memory Architecture [65]), trained
with PPO.

Two stacked recurrent memories over the frame-embedding stream: a fast
working memory (per-step LSTM) and a slower contextual memory whose input is the
fast memory's hidden state, giving a short-horizon + longer-horizon pair. Their
hidden states plus proprioception form the policy feature. This is a best-effort
reimplementation of the method's architecture at comparable capacity.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .encoders import FrameCNN
from .ppo import PPOBase


class MemDRLAgent(PPOBase):
    feedforward = False

    def __init__(self, obs_spec, action_dim, device, name="MemDRL",
                 frame_dim=128, mem_hidden=128, **kw):
        self._frame_dim, self._mem = frame_dim, mem_hidden
        super().__init__(obs_spec, action_dim, device, name=name, **kw)
        self.finish_init()

    def build_encoder(self):
        s = self.spec
        self.frame = FrameCNN(s, self._frame_dim)
        self.fast = nn.LSTMCell(self._frame_dim, self._mem)          # working memory
        self.slow = nn.LSTMCell(self._mem, self._mem)                # contextual memory
        self.proj = nn.Sequential(
            nn.Linear(2 * self._mem + s.pose_dim + s.imu_dim, self.feature_dim), nn.ReLU(inplace=True))
        self._state = None

    def reset_state(self):
        self._state = None

    def _zero(self, b):
        z = lambda: torch.zeros(b, self._mem, device=self.device)
        return (z(), z(), z(), z())        # fh, fc, sh, sc

    def _cell(self, emb, proprio, state):
        fh, fc, sh, sc = state
        fh, fc = self.fast(emb, (fh, fc))
        sh, sc = self.slow(fh, (sh, sc))
        feat = self.proj(torch.cat([fh, sh, proprio], dim=-1))
        return feat, (fh, fc, sh, sc)

    def encode_step(self, rgb, proprio):
        if self._state is None:
            self._state = self._zero(rgb.shape[0])
        feat, st = self._cell(self.frame(rgb), proprio, self._state)
        self._state = tuple(s.detach() for s in st)
        return feat

    def encode_seq(self, rgb_seq, proprio_seq, dones):
        state = self._zero(1)
        feats = []
        for t in range(rgb_seq.shape[0]):
            if t > 0 and dones[t - 1] > 0.5:
                state = self._zero(1)
            emb = self.frame(rgb_seq[t:t + 1])
            feat, state = self._cell(emb, proprio_seq[t:t + 1], state)
            feats.append(feat)
        return torch.cat(feats, dim=0)
