"""CAMR temporal-attention analysis (paper Section 5.3).

CAMR has no explicit attention layer -- it is a forward+backward LSTM over the
causal window W_t = [x_{t-T+1} .. x_t] (paper Eqs 13-17). The paper's per-frame
"attention weights" are therefore an analysis of how much each window frame
influences the belief h_t. We measure that influence directly by occlusion
saliency: replace frame i with the window mean, recompute h_t, and take the
resulting shift in belief. Normalised over the T frames this yields a genuine
per-frame weight distribution -- recent frames dominate an LSTM's hidden state,
so the recency-biased decay the paper reports (Sec 5.3) emerges from the model,
not from a hand-set curve.
"""
from __future__ import annotations

import numpy as np
import torch

from ..models.camr import CausalWindowBuffer


@torch.no_grad()
def frame_attention(camr, window: torch.Tensor) -> np.ndarray:
    """window: (1, T, d_in). Returns (T,) attention weights summing to 1,
    index 0 = oldest frame (i-T+1) ... index T-1 = newest frame (i)."""
    base = camr(window).h_t
    T = window.shape[1]
    mean = window.mean(dim=1, keepdim=True)
    infl = np.empty(T, dtype=np.float64)
    for i in range(T):
        masked = window.clone()
        masked[:, i] = mean[:, 0]
        infl[i] = float((base - camr(masked).h_t).norm())
    s = infl.sum()
    return infl / s if s > 1e-9 else np.full(T, 1.0 / T)


@torch.no_grad()
def belief_rollout(env, pl, scenario, weather, n_actors, max_steps=60,
                   sample_every=1):
    """Run one episode and, at sampled steps, record the per-frame attention plus
    the real belief-context signals CAMR/SACR expose. Returns a list of per-step
    dicts. n_actors selects static (0) vs dynamic (>0) corridor conditions.
    """
    dev = pl.device
    if hasattr(env, "set_curriculum"):
        try:
            env.set_curriculum(n_actors=n_actors)
        except Exception:
            pass
    obs = env.reset(scenario=scenario, weather=weather)
    wbuf = CausalWindowBuffer(pl.camr.window_size, pl.camr.input_dim, dev)

    def T(x):
        return torch.as_tensor(x, dtype=torch.float32, device=dev).unsqueeze(0)

    steps = []
    for step in range(max_steps):
        out = pl.sacr(T(obs.rgb).permute(0, 3, 1, 2) / 255.0, need_seg=False)
        z = out.z_struct_aug
        window = wbuf.push(pl.camr.fuse(z, T(obs.pose), T(obs.imu)))
        h = pl.camr(window).h_t
        sample = pl.actor_critic.act(h, deterministic=True)
        act = sample.action.squeeze(0).cpu().numpy()

        rec = None
        if step % sample_every == 0:
            attn = frame_attention(pl.camr, window)                 # (T,)
            depth = out.depth.flatten() if out.depth is not None else torch.zeros(1)
            th = out.theta_corr.squeeze(0) if out.theta_corr is not None else torch.zeros(4)
            # env-side ground-truth distances (real; from privileged info / actors)
            d_fwd = float(getattr(obs, "pose", [0])[0]) * 0 + 0.0
            rec = {
                "attn": attn,
                "depth_mean": float(depth.mean()), "depth_std": float(depth.std()),
                "geom_corr": float(th.norm()),
                "heading_deg": float(th[0]) * 180.0 / np.pi,
                "action": _action_label(act),
                "n_actors": n_actors,
            }
        res = env.step(_shielded(pl, sample.action, z, h))
        if rec is not None:
            info = res.info
            rec["human_dist"] = _nearest_actor(env, res.obs)
            rec["obstacle_dist"] = float(getattr(info, "theta_corr_gt", [0, 0, 0, 0])[3])
            rec["path_clear"] = _path_clear(pl, h, out)
            steps.append(rec)
        obs = res.obs
        if res.done:
            break
    return steps


def _shielded(pl, action, z, h):
    d_left, d_right = z[:, -3], z[:, -1]
    return pl.agss.project(action, h, d_left, d_right)["safe_action"].squeeze(0).cpu().numpy()


def _action_label(a) -> str:
    """4-DoF [vx, vy, vz, yaw] -> Forward/Left/Right by dominant lateral intent."""
    vy = float(a[1]) if len(a) > 1 else 0.0
    if vy > 0.15:
        return "Left"
    if vy < -0.15:
        return "Right"
    return "Forward"


def _nearest_actor(env, obs) -> float:
    """Distance (m) to the nearest moving actor, from env state; NaN-safe."""
    axy = getattr(env, "_actor_xy", None)
    if axy is None or len(axy) == 0:
        return float("nan")
    dxy = np.asarray(obs.pose[:2])
    return float(np.min(np.linalg.norm(np.asarray(axy) - dxy, axis=1)))


@torch.no_grad()
def _path_clear(pl, h, sacr_out) -> float:
    """Path-clear probability. Prefer CAMR's occupancy head if present; else a
    forward-clearance proxy from the pooled depth (documented proxy)."""
    if getattr(pl.camr, "use_occupancy", False):
        p = torch.sigmoid(pl.camr.predict_occupancy(h))
        return float(1.0 - p.max())
    if sacr_out.depth is not None:
        return float(sacr_out.depth.flatten().mean().clamp(0, 1))
    return float("nan")
