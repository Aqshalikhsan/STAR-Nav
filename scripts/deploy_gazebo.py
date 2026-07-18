"""Deploy / inference a trained AGSS-PPO policy DIRECTLY in the real Gazebo
env (GazeboROSEnv), with no retraining.

Why this "just works": the policy is an MLP over the CAMR belief h_t (dim
2*camr.hidden_dim = 256, fixed by the config), and AGSS reads SACR's
z_struct_aug depth slice (dim struct_dim+depth_pool_regions = 131, also fixed
by the config). Those dims are identical whether the features come from the
Mock env or from Gazebo, so the exact same `ppo.pt` weights load and run here
unchanged. The ONLY thing that must differ from training is the perception:
here we load the GAZEBO-trained, actor-aware SACR/CAMR (checkpoints/gazebo/
sacr.pt, camr.pt) so the belief is computed from real Gazebo imagery. The
policy weights come from checkpoints/mock/ppo.pt (Mock-trained). The structure-
aware design (SACR regresses the same corridor geometry + segments the same
classes in both domains) is what makes the Mock-trained policy transfer.

Run this INSIDE the ROS container (needs rclpy + the ros_gazebo_bridge package
+ a live bridge/MAVROS/PX4), e.g.:

    docker exec star_nav_ros_bridge bash -lc \
      'source /opt/ros/humble/setup.bash; source /ros2_ws/install/setup.bash; \
       python3 scripts/deploy_gazebo.py --episodes 3 --deterministic'

CAVEAT: this drives a *flying* vehicle. It only produces motion once the drone
actually arms + flies in the RL loop -- currently blocked by PX4 SITL's
GPS-drift auto-disarm (see project memory: the GPS-denied/VIO arming must be
closed first). The script itself is domain-agnostic and ready; it will fly the
policy the moment reset()/step() produce real motion.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from star_nav.models.sacr import SACR
from star_nav.models.camr import CAMR, CausalWindowBuffer
from star_nav.models.agss_ppo import ActorCritic, AGSSShield
from star_nav.utils.config import load_config
from star_nav.utils.seeding import get_device, set_seed


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None, help="YAML config (must set env.name=gazebo_ros + ros topics).")
    p.add_argument("--sacr-ckpt", default="checkpoints/gazebo/sacr.pt", help="Gazebo-trained (actor-aware) SACR.")
    p.add_argument("--camr-ckpt", default="checkpoints/gazebo/camr.pt", help="Gazebo-trained CAMR.")
    p.add_argument("--policy-ckpt", default="checkpoints/mock/ppo.pt", help="Mock-trained AGSS-PPO policy weights.")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--deterministic", action="store_true", help="Use the policy mean action (recommended for deploy).")
    p.add_argument("--scenario", default="A")
    args = p.parse_args(argv)

    cfg = load_config(args.config, overrides={"env.name": "gazebo_ros"})
    set_seed(cfg.seed)
    device = get_device(cfg.device)

    # Gazebo env lives in the ros_gazebo_bridge package (import lazily so this
    # file at least parses on a host without ROS installed).
    from ros_gazebo_bridge.env import GazeboROSEnv
    env = GazeboROSEnv(cfg.env)

    # Build perception honouring the same novelty flags as training so the
    # Gazebo-trained checkpoints (uncertainty + occupancy heads) load and the
    # adaptive shield gets sigma/occ. With the flags off this is the original
    # 131-dim path. NOTE: the Gazebo SACR/CAMR must have been trained with the
    # SAME flags (dims must match) -- retrain Gazebo perception with depth +
    # actor-pose GT when enabling these (see ros_gazebo_bridge occupancy GT).
    unc_on = getattr(cfg.sacr, "depth_uncertainty", False)
    sacr = SACR(in_channels=cfg.sacr.in_channels, feature_channels=cfg.sacr.feature_channels,
                num_seg_classes=cfg.sacr.num_seg_classes, geom_dim=cfg.sacr.geom_dim,
                geom_hidden=cfg.sacr.geom_hidden, struct_dim=cfg.sacr.struct_dim,
                depth_pool_regions=cfg.sacr.depth_pool_regions,
                depth_uncertainty=unc_on).to(device)
    camr = CAMR(z_struct_aug_dim=sacr.z_struct_aug_dim, pose_dim=cfg.camr.pose_dim,
                imu_dim=cfg.camr.imu_dim, window_size=cfg.camr.window_size,
                hidden_dim=cfg.camr.hidden_dim,
                predict_occupancy=getattr(cfg.camr, "predict_occupancy", False),
                occ_dim=getattr(cfg.camr, "occ_dim", 2)).to(device)
    sacr.load_state_dict(torch.load(args.sacr_ckpt, map_location=device))
    camr.load_state_dict(torch.load(args.camr_ckpt, map_location=device))
    sacr.eval(); camr.eval()
    R = cfg.sacr.depth_pool_regions

    belief_dim = 2 * cfg.camr.hidden_dim
    ac = ActorCritic(belief_dim=belief_dim, action_dim=cfg.agss_ppo.action_dim,
                     actor_hidden=cfg.agss_ppo.actor_hidden, critic_hidden=cfg.agss_ppo.critic_hidden,
                     init_log_std=cfg.agss_ppo.init_log_std).to(device)
    ac.load_state_dict(torch.load(args.policy_ckpt, map_location=device))  # Mock-trained weights load directly
    ac.eval()
    agss = AGSSShield(d0=cfg.agss_ppo.d0, alpha=cfg.agss_ppo.alpha, complexity_dim=belief_dim, device=device,
                      beta=getattr(cfg.agss_ppo, "beta_unc", 0.0), gamma=getattr(cfg.agss_ppo, "gamma_occ", 0.0))
    wbuf = CausalWindowBuffer(cfg.camr.window_size, camr.input_dim, device)

    def shield_terms(z, h):
        """Per-side (d_left, d_right, sigma_left, sigma_right, occ_left, occ_right)
        from frozen perception -- mirrors train_ppo.py so the deployed shield is
        identical to the one trained against."""
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
    print(f"loaded SACR={args.sacr_ckpt} CAMR={args.camr_ckpt} policy={args.policy_ckpt} "
          f"(belief_dim={belief_dim}) -> running in Gazebo", flush=True)

    def to_t(x):
        return torch.as_tensor(x, dtype=torch.float32, device=device).unsqueeze(0)

    def encode(obs):
        with torch.no_grad():
            z = sacr.encode(to_t(obs.rgb).permute(0, 3, 1, 2) / 255.0)
            h = camr(wbuf.push(camr.fuse(z, to_t(obs.pose), to_t(obs.imu)))).h_t
        return h, z

    for ep in range(args.episodes):
        obs = env.reset(scenario=args.scenario); wbuf.reset()
        h_t, z = encode(obs)
        total_r = 0.0; interv = 0
        for step in range(args.max_steps):
            with torch.no_grad():
                sample = ac.act(h_t, deterministic=args.deterministic)
                d_left, d_right, sig_left, sig_right, occ_left, occ_right = shield_terms(z, h_t)
                proj = agss.project(sample.action, h_t, d_left, d_right,
                                    sigma_left=sig_left, sigma_right=sig_right,
                                    occ_left=occ_left, occ_right=occ_right)
            interv += int(proj["intervened"].item())
            res = env.step(proj["safe_action"].squeeze(0).cpu().numpy())
            total_r += res.reward
            h_t, z = encode(res.obs if not res.done else res.obs)
            if res.done:
                print(f"episode {ep}: steps={step+1} reward={total_r:.1f} success={res.success} "
                      f"collided={getattr(res.info,'collided',None)} agss_interv={interv}", flush=True)
                break
        else:
            print(f"episode {ep}: reached max_steps reward={total_r:.1f} agss_interv={interv}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
