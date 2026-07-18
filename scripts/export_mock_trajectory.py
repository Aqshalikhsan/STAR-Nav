"""Roll out a Mock-trained AGSS-PPO policy in MockCorridorEnv and export its
trajectory as a inference list, PLUS a byte-matched Gazebo world so the two
environments are geometrically identical ("samakan env nya biar gampang").

Why this exists (the decoupled deploy). Running the Mock policy *live* in
Gazebo wanders: the zero-shot Mock->Gazebo belief-distribution gap breaks the
perception->policy loop (see project memory `gazebo-deploy`). But the policy's
*output* -- the path it steers through the plantation -- is good. So instead of
re-running perception+policy in Gazebo, we:

  1. roll the policy out ONCE in Mock (where its belief is in-distribution),
     recording the (x, y) trajectory it flies, and
  2. generate a Gazebo world whose trunks sit at the EXACT same positions as
     the Mock world the rollout happened in (matched layout),

so a downstream `fly_inference_gazebo.py` can just position-track the recorded
inference in Gazebo -- which PX4 offboard does reliably -- with the trunks in
the same places, so the path stays collision-free. Perception is decoupled from
control at deploy time.

Outputs (default under renders/deploy/):
  * <out>_inference.csv    -- t, x, y (Mock metres; start at ~x=2)
  * <out>.sdf              -- Gazebo world with trunks at the matched positions
  * <out>.world.json       -- ground-truth layout (load_world_layout format)

Example (best zigzag policy, matched zigzag world):
    python scripts/export_mock_trajectory.py --zigzag-amp 5 \
        --policy-ckpt checkpoints/mock_zigzag/ppo.pt \
        --sacr-ckpt   checkpoints/mock_zigzag/sacr.pt \
        --camr-ckpt   checkpoints/mock_zigzag/camr.pt \
        --out renders/deploy/zigzag
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
# The ros_gazebo_bridge PACKAGE lives one level down (ros_gazebo_bridge/ros_gazebo_bridge);
# add the outer dir so `import ros_gazebo_bridge.world_gen` resolves without a ROS install.
sys.path.insert(0, os.path.join(_ROOT, "ros_gazebo_bridge"))

import numpy as np
import torch

from star_nav.envs import MockCorridorEnv
from star_nav.models.sacr import SACR
from star_nav.models.camr import CAMR, CausalWindowBuffer
from star_nav.models.agss_ppo import ActorCritic, AGSSShield
from star_nav.utils.config import load_config
from star_nav.utils.seeding import get_device

from ros_gazebo_bridge.world_gen import CorridorWorld, write_world_bundle


def _load_policy(ac, path, device):
    """ppo.pt is a raw state_dict; ppo_best/ppo_last.pt are checkpoints with a
    'model' key -- accept either."""
    blob = torch.load(path, map_location=device)
    ac.load_state_dict(blob["model"] if isinstance(blob, dict) and "model" in blob else blob)


def rollout(env, sacr, camr, ac, agss, cfg, device, seed, max_steps):
    """One deterministic episode. Returns (traj Nx2, reached_x, success, collided)."""
    R = cfg.sacr.depth_pool_regions
    unc_on = getattr(cfg.sacr, "depth_uncertainty", False)
    wbuf = CausalWindowBuffer(cfg.camr.window_size, camr.input_dim, device)

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
                p = torch.sigmoid(camr.predict_occupancy(h))
            occ_left, occ_right = p[:, 0], p[:, 1]
        return d_left, d_right, sig_left, sig_right, occ_left, occ_right

    env.rng = np.random.default_rng(seed)     # env is otherwise unseeded
    obs = env.reset(scenario="A")
    wbuf.reset()
    h_t, z = encode(obs)
    traj = [env._pos.copy()]
    actors = [env._actor_xy.copy()]           # (n,2) per step -- moving people
    success = collided = False
    for _ in range(max_steps):
        with torch.no_grad():
            s = ac.act(h_t, deterministic=True)
            dl, dr, sl, sr, ol, orr = shield_terms(z, h_t)
            proj = agss.project(s.action, h_t, dl, dr, sigma_left=sl, sigma_right=sr,
                                occ_left=ol, occ_right=orr)
        res = env.step(proj["safe_action"].squeeze(0).cpu().numpy())
        traj.append(env._pos.copy())
        actors.append(env._actor_xy.copy())
        h_t, z = encode(res.obs)
        if res.done:
            success = bool(res.success)
            collided = bool(getattr(res.info, "collided", False))
            break
    return (np.array(traj, dtype=np.float32), np.array(actors, dtype=np.float32),
            float(env._pos[0]), success, collided)


def inject_workers(sdf_path, start_xy):
    """Insert one VelocityControl-driven person_worker per actor (at its start xy)
    into the generated world SDF, mirroring perception_capture/capture_scenario_a.sdf.
    Each worker exposes /model/workerN/cmd_vel (Twist in) + /model/workerN/odometry
    (Odometry out); the flyer drives cmd_vel to replay the Mock crowd. Returns the
    number of workers written."""
    blocks = []
    for i, (x, y) in enumerate(start_xy):
        blocks.append(f"""    <include>
      <uri>model://person_worker</uri>
      <name>worker{i}</name>
      <pose>{x:.3f} {y:.3f} 0 0 0 0</pose>
      <plugin filename="gz-sim-velocity-control-system" name="gz::sim::systems::VelocityControl">
        <topic>/model/worker{i}/cmd_vel</topic>
      </plugin>
      <plugin filename="gz-sim-odometry-publisher-system" name="gz::sim::systems::OdometryPublisher">
        <dimensions>3</dimensions>
        <odom_topic>/model/worker{i}/odometry</odom_topic>
      </plugin>
      <plugin filename="gz-sim-label-system" name="gz::sim::systems::Label">
        <label>4</label>
      </plugin>
    </include>
""")
    with open(sdf_path, "r", encoding="utf-8") as f:
        sdf = f.read()
    sdf = sdf.replace("  </world>", "".join(blocks) + "  </world>", 1)
    with open(sdf_path, "w", encoding="utf-8") as f:
        f.write(sdf)
    return len(start_xy)


def polyline_clearance(wps, tree_xy):
    """Min distance from the STRAIGHT-LINE inference polyline (what the Gazebo
    position tracker actually flies -- it cuts corners the smooth path rounds) to
    any trunk centre. This is the number that predicts a corner-cut collision."""
    if len(tree_xy) == 0:
        return 9e9
    mind = 9e9
    for a, b in zip(wps[:-1], wps[1:]):
        for t in np.linspace(0.0, 1.0, 16):
            p = a + t * (b - a)
            mind = min(mind, float(np.linalg.norm(tree_xy - p, axis=1).min()))
    return mind


def decimate(traj, spacing):
    """Keep the first point, then one point every `spacing` metres of path, plus
    the last -- so the flyer gets a sparse, monotone inference list."""
    keep = [traj[0]]
    for p in traj[1:]:
        if np.linalg.norm(p - keep[-1]) >= spacing:
            keep.append(p)
    if not np.array_equal(keep[-1], traj[-1]):
        keep.append(traj[-1])
    return np.array(keep, dtype=np.float32)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None)
    p.add_argument("--sacr-ckpt", default="checkpoints/mock_zigzag/sacr.pt")
    p.add_argument("--camr-ckpt", default="checkpoints/mock_zigzag/camr.pt")
    p.add_argument("--policy-ckpt", default="checkpoints/mock_zigzag/ppo.pt")
    p.add_argument("--zigzag-amp", type=float, default=5.0,
                   help="Match the amp the policy was trained with (5 for mock_zigzag).")
    p.add_argument("--n-actors", type=int, default=6,
                   help="Moving people in the Mock rollout. Their trajectories are logged and "
                        "replayed in Gazebo as VelocityControl-driven person_worker models synced "
                        "to the drone's progress, so the deployed corridor has the same moving "
                        "crowd the policy avoided. Set 0 for a trees-only static path.")
    p.add_argument("--seeds", type=int, default=40, help="Try this many layouts; pick the best reach.")
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--inference-spacing", type=float, default=1.0,
                   help="Metres between exported inference. Denser = the straight-line tracker cuts "
                        "corners less, staying closer to the collision-free smooth path.")
    p.add_argument("--out", default="renders/deploy/zigzag", help="Output prefix.")
    args = p.parse_args(argv)

    overrides = {"env.name": "mock", "env.n_actors": args.n_actors}
    if args.zigzag_amp > 0:
        overrides["env.zigzag_amp"] = args.zigzag_amp
    cfg = load_config(args.config, overrides=overrides)
    device = get_device(cfg.device)

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
    sacr.eval(); camr.eval()

    belief_dim = 2 * cfg.camr.hidden_dim
    ac = ActorCritic(belief_dim=belief_dim, action_dim=cfg.agss_ppo.action_dim,
                     actor_hidden=cfg.agss_ppo.actor_hidden, critic_hidden=cfg.agss_ppo.critic_hidden,
                     init_log_std=cfg.agss_ppo.init_log_std).to(device)
    _load_policy(ac, args.policy_ckpt, device)
    ac.eval()
    agss = AGSSShield(d0=cfg.agss_ppo.d0, alpha=cfg.agss_ppo.alpha, complexity_dim=belief_dim, device=device,
                      beta=getattr(cfg.agss_ppo, "beta_unc", 0.0), gamma=getattr(cfg.agss_ppo, "gamma_occ", 0.0))

    env = MockCorridorEnv(cfg.env)
    goal_x = cfg.env.area_size_m - 2.0
    print(f"searching {args.seeds} layouts for the best full-corridor run (goal x={goal_x:.0f})...", flush=True)
    best = None
    for seed in range(args.seeds):
        traj, actors, reach, success, collided = rollout(env, sacr, camr, ac, agss, cfg, device, seed, args.max_steps)
        # Clearance of the DECIMATED path (env._world holds this seed's trunks).
        wp_clear = polyline_clearance(decimate(traj, args.inference_spacing), env._world.tree_xy)
        wall = wp_clear - float(env._world.trunk_radius)
        tag = "SUCCESS" if success else ("collided" if collided else "stalled")
        # Prefer a clean goal-reaching run, then collision-free, then the largest
        # INFERENCE-PATH wall clearance (guards against corner-cut trunk hits in
        # Gazebo), then reach.
        score = (1 if success else 0, 0 if collided else 1, round(wall, 2), reach)
        if best is None or score > best[0]:
            best = (score, seed, traj, actors, reach, success, collided, wall)
        print(f"  seed {seed:2d}: reach x={reach:5.1f}  wp-wall={wall:4.2f}m  {tag}", flush=True)
    _, seed, traj, actors, reach, success, collided, wall = best
    print(f"\nbest: seed={seed} reach x={reach:.1f} success={success} collided={collided} "
          f"wp-wall-clearance={wall:.2f}m", flush=True)

    # Rebuild the winning world so we can export its EXACT trunk layout.
    env.rng = np.random.default_rng(seed)
    env.reset(scenario="A")
    world = env._world

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    wps = decimate(traj, args.inference_spacing)
    with open(args.out + "_inference.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["i", "x", "y"])
        for i, (x, y) in enumerate(wps):
            w.writerow([i, f"{x:.3f}", f"{y:.3f}"])
    # Full (undecimated) drone path + per-step actor positions, so the flyer can
    # replay the crowd synced to the drone's progress.
    np.save(args.out + "_traj.npy", traj)
    np.save(args.out + "_actors.npy", actors)          # (T, n, 2)

    matched = CorridorWorld(
        tree_xy=world.tree_xy.astype(np.float32), trunk_radius=float(world.trunk_radius),
        corridor_center_y=float(world.corridor_center_y), goal_xy=world.goal_xy.astype(np.float32),
        area_size_m=float(cfg.env.area_size_m), scenario="A")
    write_world_bundle(matched, args.out, world_name="oil_palm_corridor")
    n_workers = inject_workers(args.out + ".sdf", actors[0]) if len(actors[0]) else 0

    print(f"\nwrote:")
    print(f"  {args.out}_inference.csv  ({len(wps)} inference, start=({wps[0,0]:.1f},{wps[0,1]:.1f}))")
    print(f"  {args.out}_traj.npy       ({len(traj)} raw points)")
    print(f"  {args.out}_actors.npy     ({actors.shape[0]} steps x {actors.shape[1]} people)")
    print(f"  {args.out}.sdf            (Gazebo world, {len(world.tree_xy)} trunks + {n_workers} people -- matched)")
    print(f"  {args.out}.world.json     (ground-truth layout)")


if __name__ == "__main__":
    main()
