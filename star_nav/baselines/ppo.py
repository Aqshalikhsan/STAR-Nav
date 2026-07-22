"""PPO baseline and the shared on-policy trainer the other baselines build on.

`PPOBase` is a faithful PPO (Schulman et al. 2017 [59]): clipped surrogate,
GAE(lambda), value + entropy terms, and -- unlike the first draft -- it
**re-encodes observations in the update so the vision encoder trains end to
end** (feed-forward encoders via mini-batches, recurrent/temporal encoders via a
sequential pass that back-propagates through time).

`PPOAgent` = PPOBase with a CNN encoder = the plain PPO baseline. Mem-DRL,
ViT-PPO (DTPPO) and NavRL subclass PPOBase and swap in their own encoder / safety
module (see their files). These are best-effort reimplementations of the cited
source works from their published formulations (no original code was available
offline); PPO and TD3 match their algorithms, the others match each method's
architecture and mechanism at comparable capacity.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders import CNNEncoder, ObsSpec, preprocess


class PPOBase(nn.Module):
    feedforward = True                      # subclasses set False for recurrent/temporal encoders

    def __init__(self, obs_spec: ObsSpec, action_dim, device, name="PPO",
                 feature_dim=256, lr=3e-4, gamma=0.99, lam=0.95, clip=0.2,
                 epochs=4, minibatch=256, entropy_coef=0.01, value_coef=0.5):
        super().__init__()
        self.name, self.device, self.spec = name, device, obs_spec
        self.action_dim, self.feature_dim = action_dim, feature_dim
        self.gamma, self.lam, self.clip = gamma, lam, clip
        self.epochs, self.minibatch = epochs, minibatch
        self.entropy_coef, self.value_coef = entropy_coef, value_coef
        self.actor_mean = nn.Sequential(nn.Linear(feature_dim, 128), nn.ReLU(inplace=True),
                                        nn.Linear(128, action_dim))
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))
        self.critic = nn.Sequential(nn.Linear(feature_dim, 128), nn.ReLU(inplace=True), nn.Linear(128, 1))

    # ---- encoder hooks (subclasses implement) ----
    def build_encoder(self):
        self.encoder = CNNEncoder(self.spec, self.feature_dim)

    def reset_state(self):
        pass

    def encode_step(self, rgb, proprio):        # rollout time, stateful, (1, feat)
        return self.encoder(rgb, proprio)

    def encode_batch(self, rgb, proprio):       # feed-forward update, (B, feat) with grad
        return self.encoder(rgb, proprio)

    def encode_seq(self, rgb_seq, proprio_seq, dones):  # sequential update (recurrent/temporal)
        raise NotImplementedError

    # ---- safety shield + aux perception hooks (NavRL overrides) ----
    def shield(self, action, obs):
        return action

    def collect_aux(self, info):
        return None

    def aux_loss(self, rgb_b, proprio_b, aux_b):
        return torch.zeros((), device=self.device)

    # ---- policy ----
    def _dist(self, f):
        return torch.distributions.Normal(self.actor_mean(f), self.log_std.exp())

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        rgb, pr = preprocess(obs, self.spec, self.device)
        f = self.encode_step(rgb, pr)
        mean = self.actor_mean(f)
        a = mean if deterministic else mean + torch.randn_like(mean) * self.log_std.exp()
        return self.shield(a.squeeze(0), obs).cpu().numpy()

    def finish_init(self):
        self.build_encoder()
        self.to(self.device)
        self.opt = torch.optim.Adam(self.parameters(), lr=3e-4)

    # ---- training ----
    def train_iteration(self, env, scenario, weather, rollout_steps=1024):
        self.train(); self.reset_state()
        rgbs, prs, acts, old_logps, vals, rews, dones, auxs = [], [], [], [], [], [], [], []
        ep_returns, ep_succ, ep_ret = [], [], 0.0
        obs = env.reset(scenario=scenario, weather=weather)
        for _ in range(rollout_steps):
            rgb, pr = preprocess(obs, self.spec, self.device)
            with torch.no_grad():
                f = self.encode_step(rgb, pr)
                dist = self._dist(f); a = dist.sample()
                logp = dist.log_prob(a).sum(-1).reshape(())
                v = self.critic(f).reshape(())
            action = self.shield(a.squeeze(0), obs).cpu().numpy()
            res = env.step(np.asarray(action, dtype=np.float32))
            rgbs.append(rgb.squeeze(0)); prs.append(pr.squeeze(0)); acts.append(a.squeeze(0))
            old_logps.append(logp); vals.append(v); rews.append(float(res.reward)); dones.append(bool(res.done))
            auxs.append(self.collect_aux(res.info))
            ep_ret += float(res.reward); obs = res.obs
            if res.done:
                ep_returns.append(ep_ret); ep_succ.append(float(res.success)); ep_ret = 0.0
                obs = env.reset(scenario=scenario, weather=weather); self.reset_state()

        rgb_t, pr_t, act_t = torch.stack(rgbs), torch.stack(prs), torch.stack(acts)
        old_logp = torch.stack(old_logps); vals_t = torch.stack(vals)
        rews_t = torch.tensor(rews, device=self.device); dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device)
        adv = torch.zeros_like(rews_t); last = 0.0
        for t in reversed(range(len(rews))):
            nonterm = 1.0 - dones_t[t]
            nextv = vals_t[t + 1] if t + 1 < len(rews) else 0.0
            delta = rews_t[t] + self.gamma * nextv * nonterm - vals_t[t]
            last = delta + self.gamma * self.lam * nonterm * last
            adv[t] = last
        ret = adv + vals_t
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        aux_ok = all(x is not None for x in auxs)
        aux_t = torch.tensor(np.array(auxs), dtype=torch.float32, device=self.device) if aux_ok else None

        n = len(rews); last_loss = 0.0
        for _ in range(self.epochs):
            if self.feedforward:
                idx = np.random.permutation(n)
                for s in range(0, n, self.minibatch):
                    b = idx[s:s + self.minibatch]
                    f = self.encode_batch(rgb_t[b], pr_t[b])
                    last_loss = self._ppo_step(f, act_t[b], old_logp[b], adv[b], ret[b],
                                               rgb_t[b], pr_t[b], None if aux_t is None else aux_t[b])
            else:
                f = self.encode_seq(rgb_t, pr_t, dones_t)
                last_loss = self._ppo_step(f, act_t, old_logp, adv, ret, rgb_t, pr_t, aux_t)
        return {"loss": last_loss,
                "mean_return": float(np.mean(ep_returns)) if ep_returns else ep_ret,
                "return_var": float(np.var(ep_returns)) if ep_returns else 0.0,
                "success_prob": float(np.mean(ep_succ)) if ep_succ else 0.0}

    def _ppo_step(self, f, actions, old_logp, adv, ret, rgb_b, pr_b, aux_b):
        dist = self._dist(f)
        logp = dist.log_prob(actions).sum(-1)
        ent = dist.entropy().sum(-1)
        value = self.critic(f).squeeze(-1)
        ratio = (logp - old_logp).exp()
        surr = torch.min(ratio * adv, torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * adv)
        loss = (-surr.mean() + self.value_coef * F.mse_loss(value, ret)
                - self.entropy_coef * ent.mean() + self.aux_loss(rgb_b, pr_b, aux_b))
        self.opt.zero_grad(); loss.backward(); self.opt.step()
        return float(loss.item())

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path, map_location=self.device, weights_only=True)); self.eval()


class PPOAgent(PPOBase):
    """Plain PPO baseline (CNN encoder)."""

    def __init__(self, obs_spec, action_dim, device, name="PPO", **kw):
        super().__init__(obs_spec, action_dim, device, name=name, **kw)
        self.finish_init()
