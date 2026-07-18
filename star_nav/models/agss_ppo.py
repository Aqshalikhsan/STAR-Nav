"""AGSS-PPO: PPO actor-critic policy + Adaptive Geometric Safety Shield
(paper Section 3.4).

Actor:  Linear(2d_h,256) -> LayerNorm -> ReLU -> Linear(256,128) ->
        LayerNorm -> ReLU -> Linear(128,4) -> tanh                 (mu_t)
        a_t ~ N(mu_t, diag(sigma^2))
Critic: identical trunk, scalar head                                 V(h_t)

AGSS (deterministic, no gradient back to the actor -- Invariant I5):
  c_t       = sigmoid(w_c^T h_t + b_c)
  d_safe    = d_0 + alpha * c_t
  v_y_min   = -(d_L_bar - d_safe)
  v_y_max   =  d_R_bar - d_safe
  v_y_safe  = clip(v_y, v_y_min, v_y_max)
  a_safe    = (v_x, v_y_safe, v_z, omega)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch.distributions import Normal


def _mlp_trunk(in_dim: int, hidden: list[int]) -> nn.Sequential:
    layers = []
    prev = in_dim
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.ReLU(inplace=True)]
        prev = h
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(self, belief_dim: int, action_dim: int = 4, hidden: list[int] = (256, 128), init_log_std: float = -0.5):
        super().__init__()
        self.trunk = _mlp_trunk(belief_dim, list(hidden))
        self.mu_head = nn.Linear(hidden[-1], action_dim)
        self.log_std = nn.Parameter(torch.full((action_dim,), init_log_std))

    def forward(self, h_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu = torch.tanh(self.mu_head(self.trunk(h_t)))
        std = self.log_std.exp().expand_as(mu)
        return mu, std

    def distribution(self, h_t: torch.Tensor) -> Normal:
        mu, std = self.forward(h_t)
        return Normal(mu, std)


class Critic(nn.Module):
    def __init__(self, belief_dim: int, hidden: list[int] = (256, 128)):
        super().__init__()
        self.trunk = _mlp_trunk(belief_dim, list(hidden))
        self.value_head = nn.Linear(hidden[-1], 1)

    def forward(self, h_t: torch.Tensor) -> torch.Tensor:
        return self.value_head(self.trunk(h_t)).squeeze(-1)


@dataclass
class ActionSample:
    action: torch.Tensor        # a_t, raw PPO candidate action (used for log-prob / gradients)
    log_prob: torch.Tensor
    value: torch.Tensor
    mu: torch.Tensor
    std: torch.Tensor


class ActorCritic(nn.Module):
    """Bundles Actor + Critic. AGSS is intentionally a separate module
    (`AGSSShield` below) with no parameters shared with the actor, so that
    no gradient can flow from safety projection back into the policy.
    """

    def __init__(self, belief_dim: int, action_dim: int = 4, actor_hidden=(256, 128), critic_hidden=(256, 128), init_log_std: float = -0.5):
        super().__init__()
        self.actor = Actor(belief_dim, action_dim, list(actor_hidden), init_log_std)
        self.critic = Critic(belief_dim, list(critic_hidden))

    @torch.no_grad()
    def act(self, h_t: torch.Tensor, deterministic: bool = False) -> ActionSample:
        dist = self.actor.distribution(h_t)
        action = dist.mean if deterministic else dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        value = self.critic(h_t)
        mu, std = self.actor(h_t)
        return ActionSample(action=action, log_prob=log_prob, value=value, mu=mu, std=std)

    def evaluate_actions(self, h_t: torch.Tensor, actions: torch.Tensor):
        """Used during PPO updates: recompute log-prob, entropy, and value
        for previously-collected (h_t, a_t) pairs under the current policy.
        """
        dist = self.actor.distribution(h_t)
        log_prob = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        value = self.critic(h_t)
        return log_prob, entropy, value


class AGSSShield:
    """Deterministic post-policy projection layer. Stateless w.r.t. the
    computational graph: everything here runs under `torch.no_grad()`
    semantics at the call site (train_ppo.py detaches a_t before AGSS).
    """

    def __init__(self, d0: float, alpha: float, complexity_dim: int, device: torch.device,
                 beta: float = 0.0, gamma: float = 0.0):
        self.d0 = d0
        self.alpha = alpha
        # beta:  weight on SACR's per-side depth uncertainty (sigma) -- widens
        #        the margin where perception is unsure (uncertainty-aware).
        # gamma: weight on CAMR's predicted future actor occupancy per side --
        #        widens the margin toward a side a worker is about to occupy
        #        (anticipatory). Both 0 -> identical to the original shield.
        self.beta = beta
        self.gamma = gamma
        # w_c, b_c: complexity estimator c_t = sigmoid(w_c^T h_t + b_c).
        # Kept as a small trainable-free linear map (fixed random projection)
        # since the paper treats c_t as a scalar summary of h_t, not a policy
        # parameter subject to the PPO objective (Invariant I5).
        gen = torch.Generator(device="cpu").manual_seed(0)
        self.w_c = torch.randn(complexity_dim, generator=gen) / (complexity_dim ** 0.5)
        self.w_c = self.w_c.to(device)
        self.b_c = torch.zeros(1, device=device)

    def complexity(self, h_t: torch.Tensor) -> torch.Tensor:
        """c_t = sigmoid(w_c^T h_t + b_c) in (0, 1)."""
        return torch.sigmoid(h_t @ self.w_c + self.b_c)

    def safety_margin(self, c_t: torch.Tensor, sigma: Optional[torch.Tensor] = None,
                      occ: Optional[torch.Tensor] = None) -> torch.Tensor:
        """d_safe = d_0 + alpha * c_t [+ beta * sigma] [+ gamma * occ]

        sigma/occ are optional per-side terms; when omitted the margin reduces
        to the original d_0 + alpha * c_t.
        """
        d_safe = self.d0 + self.alpha * c_t
        if sigma is not None:
            d_safe = d_safe + self.beta * sigma
        if occ is not None:
            d_safe = d_safe + self.gamma * occ
        return d_safe

    def project(self, action: torch.Tensor, h_t: torch.Tensor, d_left: torch.Tensor, d_right: torch.Tensor,
                sigma_left: Optional[torch.Tensor] = None, sigma_right: Optional[torch.Tensor] = None,
                occ_left: Optional[torch.Tensor] = None, occ_right: Optional[torch.Tensor] = None) -> dict[str, torch.Tensor]:
        """Projects the candidate action's lateral velocity (index 1) onto
        the nearest geometrically feasible interval; all other action
        dimensions (v_x, v_z, omega) pass through unmodified.

        The safety margin is computed *per side* so uncertainty/occupancy on
        the left need not equal the right (adaptive, asymmetric shield). With
        all optional args None and beta=gamma=0 this is exactly the original
        symmetric d_0 + alpha * c_t shield.

        Args:
            action: (B, 4) candidate action (v_x, v_y, v_z, omega) from PPO.
            h_t: (B, 2*d_h) belief state.
            d_left, d_right: (B,) pooled left/right corridor depth (from
                SACR's z_struct_aug depth-pool slice).
            sigma_left, sigma_right: (B,) SACR per-side depth uncertainty
                (std, i.e. exp(0.5*logvar)) from the uncertainty-aware variant.
            occ_left, occ_right: (B,) CAMR predicted future actor-occupancy
                probability on each side from the anticipatory head.
        """
        c_t = self.complexity(h_t)
        d_safe_left = self.safety_margin(c_t, sigma_left, occ_left)
        d_safe_right = self.safety_margin(c_t, sigma_right, occ_right)

        v_y_min = -(d_left - d_safe_left)
        v_y_max = d_right - d_safe_right
        v_y_max = torch.maximum(v_y_max, v_y_min)  # guard against infeasible/degenerate geometry

        v_y = action[:, 1]
        v_y_safe = torch.max(torch.min(v_y, v_y_max), v_y_min)  # elementwise per-sample clamp

        safe_action = action.clone()
        safe_action[:, 1] = v_y_safe

        return {
            "safe_action": safe_action,
            "c_t": c_t,
            "d_safe": 0.5 * (d_safe_left + d_safe_right),
            "d_safe_left": d_safe_left,
            "d_safe_right": d_safe_right,
            "v_y_min": v_y_min,
            "v_y_max": v_y_max,
            "intervened": (v_y_safe - v_y).abs() > 1e-6,
            "correction_magnitude": (v_y_safe - v_y).abs(),
        }
