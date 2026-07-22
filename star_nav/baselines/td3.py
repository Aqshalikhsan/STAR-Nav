"""TD3 baseline agent (Twin Delayed DDPG), off-policy. Deterministic actor +
twin critics with target smoothing and delayed policy updates, over the shared
RGB+pose+IMU observation. Actions are tanh-bounded velocities. A compact,
faithful reimplementation of the TD3 baseline used in the paper (Sec 5.5).
"""
from __future__ import annotations

import copy
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders import ObsSpec, build_encoder, preprocess


class _Actor(nn.Module):
    def __init__(self, feat_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(feat_dim, 128), nn.ReLU(inplace=True),
                                 nn.Linear(128, action_dim), nn.Tanh())

    def forward(self, f):
        return self.net(f)


class _Critic(nn.Module):
    def __init__(self, feat_dim, action_dim):
        super().__init__()
        self.q = nn.Sequential(nn.Linear(feat_dim + action_dim, 128), nn.ReLU(inplace=True),
                               nn.Linear(128, 1))

    def forward(self, f, a):
        return self.q(torch.cat([f, a], dim=-1)).squeeze(-1)


class TD3Agent(nn.Module):
    def __init__(self, obs_spec: ObsSpec, action_dim, device, name="TD3",
                 encoder="cnn", feature_dim=256, lr=3e-4, gamma=0.99, tau=0.005,
                 policy_noise=0.2, noise_clip=0.5, policy_delay=2,
                 expl_noise=0.1, replay_size=20000, batch=256):
        super().__init__()
        self.name, self.device, self.action_dim = name, device, action_dim
        self.gamma, self.tau = gamma, tau
        self.policy_noise, self.noise_clip, self.policy_delay = policy_noise, noise_clip, policy_delay
        self.expl_noise, self.batch = expl_noise, batch
        self.spec = obs_spec

        self.encoder = build_encoder(encoder, obs_spec, feature_dim)
        self.actor = _Actor(feature_dim, action_dim)
        self.c1, self.c2 = _Critic(feature_dim, action_dim), _Critic(feature_dim, action_dim)
        self.to(device)
        self.actor_t = copy.deepcopy(self.actor)
        self.c1_t, self.c2_t = copy.deepcopy(self.c1), copy.deepcopy(self.c2)
        self.enc_t = copy.deepcopy(self.encoder)
        self.opt_a = torch.optim.Adam(list(self.actor.parameters()), lr=lr)
        self.opt_c = torch.optim.Adam(list(self.encoder.parameters()) +
                                      list(self.c1.parameters()) + list(self.c2.parameters()), lr=lr)
        self.buf = deque(maxlen=replay_size)
        self._it = 0

    def _prep(self, obs):
        return preprocess(obs, self.spec, self.device)

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        rgb, pr = self._prep(obs)
        a = self.actor(self.encoder(rgb, pr))
        if not deterministic:
            a = a + torch.randn_like(a) * self.expl_noise
        return a.clamp(-1, 1).squeeze(0).cpu().numpy()

    def _store(self, o, a, r, no, d):
        rgb, pr = self._prep(o); nrgb, npr = self._prep(no)
        self.buf.append((rgb.squeeze(0).cpu(), pr.squeeze(0).cpu(), a,
                         r, nrgb.squeeze(0).cpu(), npr.squeeze(0).cpu(), float(d)))

    def _sample(self):
        batch = random.sample(self.buf, self.batch)
        rgb, pr, a, r, nrgb, npr, d = zip(*batch)
        to = lambda xs: torch.stack(xs).to(self.device)
        return (to(rgb), to(pr), torch.tensor(np.array(a), dtype=torch.float32, device=self.device),
                torch.tensor(r, dtype=torch.float32, device=self.device), to(nrgb), to(npr),
                torch.tensor(d, dtype=torch.float32, device=self.device))

    def _update(self):
        rgb, pr, a, r, nrgb, npr, d = self._sample()
        with torch.no_grad():
            nf = self.enc_t(nrgb, npr)
            noise = (torch.randn_like(a) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            na = (self.actor_t(nf) + noise).clamp(-1, 1)
            q_t = torch.min(self.c1_t(nf, na), self.c2_t(nf, na))
            target = r + self.gamma * (1 - d) * q_t
        f = self.encoder(rgb, pr)
        loss_c = F.mse_loss(self.c1(f, a), target) + F.mse_loss(self.c2(f, a), target)
        self.opt_c.zero_grad(); loss_c.backward(); self.opt_c.step()

        loss_a = 0.0
        self._it += 1
        if self._it % self.policy_delay == 0:
            f2 = self.encoder(rgb, pr).detach()      # actor doesn't move the encoder
            la = -self.c1(f2, self.actor(f2)).mean()
            self.opt_a.zero_grad(); la.backward(); self.opt_a.step()
            loss_a = float(la.item())
            for net, net_t in [(self.actor, self.actor_t), (self.c1, self.c1_t),
                               (self.c2, self.c2_t), (self.encoder, self.enc_t)]:
                for p, pt in zip(net.parameters(), net_t.parameters()):
                    pt.data.mul_(1 - self.tau).add_(self.tau * p.data)
        return {"loss": float(loss_c.item()), "actor_loss": loss_a}

    def train_iteration(self, env, scenario, weather, rollout_steps=1024, warmup=1000):
        self.train()
        obs = env.reset(scenario=scenario, weather=weather)
        ep_returns, ep_succ, ep_ret, last = [], [], 0.0, {}
        for _ in range(rollout_steps):
            a = (np.random.uniform(-1, 1, self.action_dim).astype(np.float32)
                 if len(self.buf) < warmup else self.act(obs, deterministic=False))
            res = env.step(a)
            self._store(obs, a, float(res.reward), res.obs, res.done)
            ep_ret += float(res.reward); obs = res.obs
            if len(self.buf) >= max(self.batch, warmup):
                last = self._update()
            if res.done:
                ep_returns.append(ep_ret); ep_succ.append(float(res.success)); ep_ret = 0.0
                obs = env.reset(scenario=scenario, weather=weather)
        return {"loss": last.get("loss", 0.0),
                "mean_return": float(np.mean(ep_returns)) if ep_returns else ep_ret,
                "return_var": float(np.var(ep_returns)) if ep_returns else 0.0,
                "success_prob": float(np.mean(ep_succ)) if ep_succ else 0.0}

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
        self.eval()
