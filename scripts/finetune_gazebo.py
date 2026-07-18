"""Fine-tune the AGSS-PPO policy DIRECTLY against GazeboROSEnv, warm-started from
the Mock-trained policy, to close the zero-shot Mock->Gazebo transfer gap (the
Mock policy "wanders" in Gazebo -- see DEPLOY_NOVELTY.md).

Same PPO machinery as scripts/train_ppo.py, but:
  * env is GazeboROSEnv (real PX4+Gazebo flight), NOT MockCorridorEnv;
  * NO curriculum -- the Gazebo world is a fixed generated scene, so we can't
    re-randomize goal/lane/actors between episodes; we just fly the one corridor;
  * the policy is WARM-STARTED from checkpoints/mock/ppo.pt (not random) and
    nudged with a small LR + target-KL cap so the useful Mock behaviour isn't
    destroyed -- this is adaptation, not training from scratch;
  * perception (SACR/CAMR) is the FROZEN Gazebo-novelty checkpoints, identical to
    deploy_gazebo.py, so the belief the policy sees at fine-tune time is exactly
    the deploy-time belief;
  * rollouts are SHORT (Gazebo is real-time: ~0.2 s/step + arm/takeoff on reset),
    so default rollout-steps/iterations are small. Expect this to be slow.

Runs INSIDE the ros-bridge container (needs rclpy + a live PX4/MAVROS/bridge),
exactly like deploy_gazebo.py. Writes checkpoints/gazebo_ft/ (ppo.pt / ppo_best /
ppo_last), leaving checkpoints/mock/ppo.pt untouched. Resume with --resume
checkpoints/gazebo_ft/ppo_last.pt. Deploy the result with
  ./scripts/deploy_best.sh   (after pointing --policy-ckpt at gazebo_ft/ppo.pt)
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F

from star_nav.models.sacr import SACR
from star_nav.models.camr import CAMR, CausalWindowBuffer
from star_nav.models.agss_ppo import ActorCritic, AGSSShield
from star_nav.training.buffers import RolloutBuffer, minibatches
from star_nav.utils.config import load_config
from star_nav.utils.logger import CSVLogger
from star_nav.utils.seeding import get_device, set_seed


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None)
    p.add_argument("--iterations", type=int, default=30, help="PPO rollout+update cycles (keep small -- Gazebo is real-time).")
    p.add_argument("--rollout-steps", type=int, default=200, help="Experience steps per iter (~0.2 s each in Gazebo).")
    p.add_argument("--episode-max-steps", type=int, default=120)
    p.add_argument("--lr", type=float, default=1e-4, help="Fine-tune LR (< Mock's, to preserve the warm start).")
    p.add_argument("--target-kl", type=float, default=0.03, help="Early-stop the update at this KL (anti-collapse).")
    p.add_argument("--sacr-ckpt", default="checkpoints/gazebo/sacr.pt")
    p.add_argument("--camr-ckpt", default="checkpoints/gazebo/camr.pt")
    p.add_argument("--policy-init", default="checkpoints/mock/ppo.pt", help="Warm-start policy weights (Mock-trained).")
    p.add_argument("--out-dir", default="checkpoints/gazebo_ft")
    p.add_argument("--resume", default=None, help="Resume a gazebo_ft run (checkpoints/gazebo_ft/ppo_last.pt).")
    p.add_argument("--scenario", default="A")
    args = p.parse_args(argv)

    cfg = load_config(args.config, overrides={
        "env.name": "gazebo_ros",
        "env.episode_max_steps": args.episode_max_steps,
        "agss_ppo.rollout_steps": args.rollout_steps,
    })
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    os.makedirs(args.out_dir, exist_ok=True)

    from ros_gazebo_bridge.env import GazeboROSEnv
    env = GazeboROSEnv(cfg.env)

    # Frozen Gazebo-novelty perception (same build as deploy_gazebo.py).
    unc_on = getattr(cfg.sacr, "depth_uncertainty", False)
    sacr = SACR(in_channels=cfg.sacr.in_channels, feature_channels=cfg.sacr.feature_channels,
                num_seg_classes=cfg.sacr.num_seg_classes, geom_dim=cfg.sacr.geom_dim,
                geom_hidden=cfg.sacr.geom_hidden, struct_dim=cfg.sacr.struct_dim,
                depth_pool_regions=cfg.sacr.depth_pool_regions, depth_uncertainty=unc_on).to(device)
    camr = CAMR(z_struct_aug_dim=sacr.z_struct_aug_dim, pose_dim=cfg.camr.pose_dim,
                imu_dim=cfg.camr.imu_dim, window_size=cfg.camr.window_size, hidden_dim=cfg.camr.hidden_dim,
                predict_occupancy=getattr(cfg.camr, "predict_occupancy", False),
                occ_dim=getattr(cfg.camr, "occ_dim", 2)).to(device)
    sacr.load_state_dict(torch.load(args.sacr_ckpt, map_location=device))
    camr.load_state_dict(torch.load(args.camr_ckpt, map_location=device))
    for m in (sacr, camr):
        m.eval()
        for pm in m.parameters():
            pm.requires_grad_(False)
    R = cfg.sacr.depth_pool_regions

    belief_dim = 2 * cfg.camr.hidden_dim
    ac = ActorCritic(belief_dim=belief_dim, action_dim=cfg.agss_ppo.action_dim,
                     actor_hidden=cfg.agss_ppo.actor_hidden, critic_hidden=cfg.agss_ppo.critic_hidden,
                     init_log_std=cfg.agss_ppo.init_log_std).to(device)
    agss = AGSSShield(d0=cfg.agss_ppo.d0, alpha=cfg.agss_ppo.alpha, complexity_dim=belief_dim, device=device,
                      beta=getattr(cfg.agss_ppo, "beta_unc", 0.0), gamma=getattr(cfg.agss_ppo, "gamma_occ", 0.0))
    optim = torch.optim.Adam(ac.parameters(), lr=args.lr)
    buffer = RolloutBuffer(belief_dim, cfg.agss_ppo.action_dim, cfg.agss_ppo.rollout_steps, device)
    wbuf = CausalWindowBuffer(cfg.camr.window_size, camr.input_dim, device)
    rng = np.random.default_rng(cfg.seed)

    start_iter, best_reward = 0, -1e9
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        ac.load_state_dict(ck["model"]); optim.load_state_dict(ck["optim"])
        start_iter = ck.get("iter", 0) + 1; best_reward = ck.get("best_reward", -1e9)
        print(f"resumed gazebo_ft from {args.resume}: start_iter={start_iter} best_reward={best_reward:.3f}", flush=True)
    else:
        ac.load_state_dict(torch.load(args.policy_init, map_location=device))  # warm start from Mock
        print(f"warm-started policy from {args.policy_init} (Mock-trained)", flush=True)

    def to_t(x):
        return torch.as_tensor(x, dtype=torch.float32, device=device).unsqueeze(0)

    def encode(obs):
        with torch.no_grad():
            z = sacr.encode(to_t(obs.rgb).permute(0, 3, 1, 2) / 255.0)
            h = camr(wbuf.push(camr.fuse(z, to_t(obs.pose), to_t(obs.imu)))).h_t
        return h, z

    def shield_terms(z, h):
        if unc_on:
            d_left, d_right = z[:, -2 * R], z[:, -(R + 1)]
            sig_left = torch.exp(0.5 * z[:, -R].clamp(-6.0, 1.4))
            sig_right = torch.exp(0.5 * z[:, -1].clamp(-6.0, 1.4))
        else:
            d_left, d_right = z[:, -R], z[:, -1]
            sig_left = sig_right = None
        occ_left = occ_right = None
        if camr.use_occupancy:
            with torch.no_grad():
                pocc = torch.sigmoid(camr.predict_occupancy(h))
            occ_left, occ_right = pocc[:, 0], pocc[:, 1]
        return d_left, d_right, sig_left, sig_right, occ_left, occ_right

    logger = CSVLogger(cfg.training.log_dir, "ppo_gazebo_ft")
    print(f"device={device} belief_dim={belief_dim} rollout_steps={cfg.agss_ppo.rollout_steps} "
          f"iters={args.iterations} lr={args.lr} -> {args.out_dir}", flush=True)

    obs = env.reset(scenario=args.scenario); wbuf.reset()
    h_t, z = encode(obs)
    t0 = time.time()
    for it in range(start_iter, args.iterations):
        buffer.reset()
        ep_rewards, ep_success, ep_collide, cur_r = [], [], [], 0.0
        interventions = steps = 0
        for _ in range(cfg.agss_ppo.rollout_steps):
            s = ac.act(h_t)
            d_left, d_right, sig_left, sig_right, occ_left, occ_right = shield_terms(z, h_t)
            proj = agss.project(s.action, h_t, d_left, d_right,
                                sigma_left=sig_left, sigma_right=sig_right,
                                occ_left=occ_left, occ_right=occ_right)
            interventions += int(proj["intervened"].item()); steps += 1
            res = env.step(proj["safe_action"].squeeze(0).cpu().numpy())
            buffer.add(h_t.squeeze(0), s.action.squeeze(0), s.log_prob.squeeze(0),
                       s.value.squeeze(0), res.reward, res.done)
            cur_r += res.reward
            if res.done:
                ep_rewards.append(cur_r); cur_r = 0.0
                ep_success.append(float(res.success))
                ep_collide.append(float(getattr(res.info, "collided", False)))
                obs = env.reset(scenario=args.scenario); wbuf.reset()
            else:
                obs = res.obs
            h_t, z = encode(obs)
            if buffer.full():
                break

        with torch.no_grad():
            last_v = ac.critic(h_t).squeeze(0)
        data = buffer.get(last_v, cfg.agss_ppo.gamma, cfg.agss_ppo.gae_lambda)
        target_kl = args.target_kl if args.target_kl and args.target_kl > 0 else None
        l_clip = l_val = torch.zeros((), device=device)
        for batch in minibatches(data, cfg.agss_ppo.minibatch_size, cfg.agss_ppo.ppo_epochs, rng):
            lp, ent, val = ac.evaluate_actions(batch["beliefs"], batch["actions"])
            ratio = torch.exp(lp - batch["log_probs"])
            with torch.no_grad():
                approx_kl = (batch["log_probs"] - lp).mean().item()
            if target_kl is not None and approx_kl > 1.5 * target_kl:
                break
            s1 = ratio * batch["advantages"]
            s2 = torch.clamp(ratio, 1 - cfg.agss_ppo.clip_eps, 1 + cfg.agss_ppo.clip_eps) * batch["advantages"]
            l_clip = -torch.min(s1, s2).mean()
            l_val = F.mse_loss(val, batch["returns"])
            loss = l_clip + cfg.agss_ppo.value_coef * l_val - cfg.agss_ppo.entropy_coef * ent.mean()
            optim.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(ac.parameters(), cfg.agss_ppo.max_grad_norm)
            optim.step()

        mean_r = float(np.mean(ep_rewards)) if ep_rewards else cur_r
        succ = float(np.mean(ep_success)) if ep_success else 0.0
        coll = float(np.mean(ep_collide)) if ep_collide else 0.0
        interv = interventions / max(1, steps)

        torch.save({"model": ac.state_dict(), "optim": optim.state_dict(),
                    "iter": it, "best_reward": best_reward}, os.path.join(args.out_dir, "ppo_last.pt"))
        improved = mean_r > best_reward
        if improved:
            best_reward = mean_r
            torch.save({"model": ac.state_dict(), "optim": optim.state_dict(),
                        "iter": it, "best_reward": best_reward}, os.path.join(args.out_dir, "ppo_best.pt"))
            torch.save(ac.state_dict(), os.path.join(args.out_dir, "ppo.pt"))
        logger.log(it, {"mean_reward": mean_r, "success": succ, "collision": coll,
                        "agss_interv": interv, "L_clip": float(l_clip), "L_value": float(l_val)})
        print(f"iter {it:3d}  mean_reward={mean_r:7.2f}  success={succ:.2f} collision={coll:.2f} "
              f"agss_interv={interv:.2f}  eps={len(ep_rewards):2d}  best={best_reward:7.2f}"
              f"{'  <- new best' if improved else ''}  ({time.time()-t0:.0f}s)", flush=True)

    env.close()
    print(f"done. last=ppo_last.pt best=ppo_best.pt (mean_reward={best_reward:.2f}) weights={args.out_dir}/ppo.pt", flush=True)


if __name__ == "__main__":
    main()
