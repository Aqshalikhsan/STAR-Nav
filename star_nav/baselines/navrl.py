"""NavRL baseline: PPO coupled with a **safety shield** that projects the
commanded velocity away from obstacles (after Xu et al., NavRL: Learning Safe
Flight in Dynamic Environments [68]).

NavRL's safety module reasons about free directions (velocity obstacles) and
hard-projects the policy's velocity onto the safe set. Here a small clearance
head predicts left / right / frontal clearance from the CNN feature (supervised
with an auxiliary loss against the simulator's corridor geometry during
training, so at inference the shield uses only the monocular input), and a
velocity-obstacle-inspired projection reduces forward speed near a frontal
obstacle and biases the lateral velocity toward the more open side. Best-effort
reimplementation of NavRL's PPO + safety-shield structure.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .ppo import PPOBase


class NavRLAgent(PPOBase):
    feedforward = True

    def __init__(self, obs_spec, action_dim, device, name="NavRL",
                 d_front=1.5, side_margin=1.0, aux_coef=0.1, **kw):
        self.d_front, self.side_margin, self.aux_coef = d_front, side_margin, aux_coef
        super().__init__(obs_spec, action_dim, device, name=name, **kw)
        self.finish_init()

    def build_encoder(self):
        super().build_encoder()                      # CNNEncoder
        self.clearance_head = nn.Sequential(
            nn.Linear(self.feature_dim, 64), nn.ReLU(inplace=True), nn.Linear(64, 3))  # [d_L, d_R, d_F]
        self._last_feat = None

    def encode_step(self, rgb, proprio):
        f = self.encoder(rgb, proprio)
        self._last_feat = f.detach()
        return f

    # ---- auxiliary clearance supervision (train-time only) ----
    def collect_aux(self, info):
        g = getattr(info, "theta_corr_gt", None)     # [phi, d_L, d_R, d_F]
        if g is None:
            return None
        g = np.asarray(g, dtype=np.float32)
        return [float(g[1]), float(g[2]), float(g[3])]

    def aux_loss(self, rgb_b, proprio_b, aux_b):
        if aux_b is None:
            return torch.zeros((), device=self.device)
        pred = self.clearance_head(self.encode_batch(rgb_b, proprio_b))
        return self.aux_coef * F.mse_loss(pred, aux_b)

    # ---- velocity-obstacle-inspired safety shield (inference) ----
    def shield(self, action, obs):
        if self._last_feat is None:
            return action
        with torch.no_grad():
            c = self.clearance_head(self._last_feat).squeeze(0)
        dL, dR, dF = c[0], c[1], c[2]
        a = action.clone()
        if dF < self.d_front:                         # brake near a frontal obstacle
            a[0] = a[0] * torch.clamp(dF / self.d_front, 0.1, 1.0)
        prox = torch.clamp(1.0 - torch.min(dL, dR) / self.side_margin, 0.0, 1.0)
        steer = torch.tanh(dR - dL)                    # >0: right is more open
        if a.shape[0] > 1:
            a[1] = torch.clamp(a[1] + 0.5 * prox * steer, -1.0, 1.0)
        return a
