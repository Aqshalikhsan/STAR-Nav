"""Phase 2 (Section 3.4 / Algorithm tab:method-3): PPO policy optimization
with the Adaptive Geometric Safety Shield executing on top of it.

Perception (SACR, CAMR) is frozen here -- only the actor-critic is
updated, consistent with the "perception-first" strategy (Section 3.5):
train the representation, then optimize navigation behavior over the
now-fixed latent belief h_t.

Invariant I5 ("no gradient flows from AGSS to PPO") is implemented
literally: the log-probability used in the PPO ratio is always
log pi_theta(a_t | h_t) for the *raw* candidate action a_t, never for the
AGSS-projected a_safe. AGSS only changes what is actually executed in the
environment (and therefore the observed reward/next state), not what the
policy is credited/blamed for in the gradient.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from ..envs.base_env import BaseCorridorEnv
from ..models.agss_ppo import ActorCritic, AGSSShield
from ..models.camr import CAMR, CausalWindowBuffer
from ..models.sacr import SACR
from ..utils.logger import CSVLogger
from .buffers import RolloutBuffer, minibatches


def _sacr_camr_encode(sacr: SACR, camr: CAMR, window_buffer: CausalWindowBuffer, rgb, pose, imu):
    with torch.no_grad():
        z_struct_aug = sacr.encode(rgb)                       # (1, d_s+d_d)
        x_t = camr.fuse(z_struct_aug, pose, imu)               # (1, input_dim)
        window = window_buffer.push(x_t)                       # (1, T, input_dim)
        out = camr(window)
    return out.h_t, z_struct_aug


def train_ppo(
    env: BaseCorridorEnv,
    sacr: SACR,
    camr: CAMR,
    actor_critic: ActorCritic,
    agss: AGSSShield,
    cfg,
    device: torch.device,
    logger: CSVLogger,
) -> ActorCritic:
    sacr.eval()
    camr.eval()
    for p in sacr.parameters():
        p.requires_grad_(False)
    for p in camr.parameters():
        p.requires_grad_(False)

    optim = torch.optim.Adam(actor_critic.parameters(), lr=cfg.agss_ppo.lr)
    belief_dim = 2 * cfg.camr.hidden_dim
    buffer = RolloutBuffer(belief_dim, cfg.agss_ppo.action_dim, cfg.agss_ppo.rollout_steps, device)
    rng = np.random.default_rng(cfg.seed)

    scenarios = list(cfg.env.scenarios.to_dict().keys())
    weathers = cfg.env.weather_conditions

    window_buffer = CausalWindowBuffer(cfg.camr.window_size, camr.input_dim, device)
    episode_count = 0
    global_step = 0

    obs = env.reset(scenario=scenarios[episode_count % len(scenarios)], weather=weathers[0])
    window_buffer.reset()

    def to_tensor(x, dtype=torch.float32):
        return torch.as_tensor(x, dtype=dtype, device=device).unsqueeze(0)

    rgb_t = to_tensor(obs.rgb).permute(0, 3, 1, 2) / 255.0
    pose_t = to_tensor(obs.pose)
    imu_t = to_tensor(obs.imu)
    h_t, z_struct_aug = _sacr_camr_encode(sacr, camr, window_buffer, rgb_t, pose_t, imu_t)

    while episode_count < cfg.training.ppo_total_episodes:
        buffer.reset()
        for _ in range(cfg.agss_ppo.rollout_steps):
            sample = actor_critic.act(h_t)

            d_left = z_struct_aug[:, -3]   # region-pool order is [L, C, R]; see depth_net.region_aware_pool
            d_right = z_struct_aug[:, -1]
            projection = agss.project(sample.action, h_t, d_left, d_right)
            safe_action = projection["safe_action"].squeeze(0).cpu().numpy()

            result = env.step(safe_action)

            buffer.add(h_t.squeeze(0), sample.action.squeeze(0), sample.log_prob.squeeze(0),
                       sample.value.squeeze(0), result.reward, result.done)

            if result.done:
                episode_count += 1
                obs = env.reset(scenario=scenarios[episode_count % len(scenarios)],
                                 weather=weathers[episode_count % len(weathers)])
                window_buffer.reset()
            else:
                obs = result.obs

            rgb_t = to_tensor(obs.rgb).permute(0, 3, 1, 2) / 255.0
            pose_t = to_tensor(obs.pose)
            imu_t = to_tensor(obs.imu)
            h_t, z_struct_aug = _sacr_camr_encode(sacr, camr, window_buffer, rgb_t, pose_t, imu_t)
            global_step += 1

            if buffer.full():
                break

        with torch.no_grad():
            last_value = actor_critic.critic(h_t).squeeze(0)
        data = buffer.get(last_value, cfg.agss_ppo.gamma, cfg.agss_ppo.gae_lambda)

        for batch in minibatches(data, cfg.agss_ppo.minibatch_size, cfg.agss_ppo.ppo_epochs, rng):
            new_log_prob, entropy, value = actor_critic.evaluate_actions(batch["beliefs"], batch["actions"])
            ratio = torch.exp(new_log_prob - batch["log_probs"])

            surr1 = ratio * batch["advantages"]
            surr2 = torch.clamp(ratio, 1 - cfg.agss_ppo.clip_eps, 1 + cfg.agss_ppo.clip_eps) * batch["advantages"]
            l_clip = -torch.min(surr1, surr2).mean()

            l_value = F.mse_loss(value, batch["returns"])
            entropy_bonus = entropy.mean()

            loss = l_clip + cfg.agss_ppo.value_coef * l_value - cfg.agss_ppo.entropy_coef * entropy_bonus

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor_critic.parameters(), cfg.agss_ppo.max_grad_norm)
            optim.step()

        if episode_count % cfg.training.log_every == 0:
            logger.log(episode_count, {
                "L_CLIP": l_clip.item(),
                "L_value": l_value.item(),
                "entropy": entropy_bonus.item(),
                "mean_reward": buffer.rewards[:buffer.ptr if buffer.ptr else 1].mean().item(),
            })

    return actor_critic
