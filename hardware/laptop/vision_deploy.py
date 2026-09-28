"""vision_deploy.py -- field loop on the ground station:
camera stream -> SACR -> CAMR -> PPO (policy mean) -> AGSS -> velocity PID -> RC link.

Inputs per 33 ms control step
  * I_t      : decoded frame from the digital video link (UVC/RTSP/index).
  * pose_t   : VINS-Mono odometry on the same frames, expressed relative to the
               active surveyed waypoint ([p - w_active ; q], 7-D).
  * imu_raw_t: flight-controller IMU received over MAVLink (6-D); the same
               samples are republished for VINS-Mono.

The policy mean mu_t is clipped and scaled to the command limits, AGSS filters
the lateral component with the pooled clearances d_L, d_R and the trained
complexity head, and the ground-station PID turns the filtered velocity into
roll/pitch targets (+/- 30 deg) against the VINS-Mono velocity estimate.

SAFETY / SANITY:
  * --no-serial : dry run; prints channels instead of driving the Arduino.
  * --no-arm    : run the loop but never raise the arm channel.
  * Keep a safety pilot on the ELRS link with the iNav failsafe and a disarm switch.

    python hardware/laptop/vision_deploy.py --source 0 --config configs/paper.yaml \
        --waypoints survey/waypoints_vio.csv --no-serial      # dry run first
"""
from __future__ import annotations

import argparse
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", required=True, help="RTSP url, video device index, or file.")
    p.add_argument("--config", default=os.path.join(_ROOT, "configs/paper.yaml"))
    p.add_argument("--sacr-ckpt", default=os.path.join(_ROOT, "checkpoints/paper/sacr_real.pt"),
                   help="SACR with the adapted segmentation decoder (all other weights as trained in simulation).")
    p.add_argument("--camr-ckpt", default=os.path.join(_ROOT, "checkpoints/paper/camr.pt"))
    p.add_argument("--policy-ckpt", default=os.path.join(_ROOT, "checkpoints/paper/actor_critic.pt"))
    p.add_argument("--complexity-ckpt", default=os.path.join(_ROOT, "checkpoints/paper/complexity.pt"))
    p.add_argument("--waypoints", required=True,
                   help="CSV of surveyed waypoints (x,y[,z]) mapped into the VIO frame at take-off.")
    p.add_argument("--port", default="/dev/ttyUSB0", help="Arduino serial port.")
    p.add_argument("--hover-throttle", type=float, default=0.0, help="Normalized hover throttle [-1, 1].")
    p.add_argument("--vel-pid", type=float, nargs=3, default=None, help="kp ki kd (overrides deploy.vel_pid).")
    p.add_argument("--no-serial", action="store_true", help="DRY RUN: print channels, don't open the Arduino.")
    p.add_argument("--no-arm", action="store_true", help="Never raise the arm channel.")
    args = p.parse_args(argv)

    import numpy as np
    import cv2
    import torch
    from star_nav.models.sacr import SACR
    from star_nav.models.camr import CAMR, CausalWindowBuffer
    from star_nav.models.agss_ppo import ActorCritic, AGSSShield, ComplexityHead
    from star_nav.utils.config import load_config
    from star_nav.utils.seeding import get_device
    from star_nav.utils.vins import VinsOdometry, WaypointTracker, waypoint_relative_pose
    from rc_link import RCLink
    from mavlink_imu import MavlinkImu
    from policy_to_channels import VelocityToAttitude, make_velocity_sender, world_to_body_xy, yaw_from_quaternion

    cfg = load_config(args.config)
    dep = cfg.deploy
    device = get_device(cfg.device)
    W, H = cfg.env.image_size
    hz = getattr(cfg.env, "control_hz", 30.0)
    a_max = np.array([cfg.env.max_forward_speed, cfg.env.max_forward_speed,
                      cfg.env.max_vertical_speed, cfg.env.max_yaw_rate_deg], dtype=np.float32)

    # --- perception, belief, policy, shield (as trained in simulation) ---
    R = cfg.sacr.depth_pool_regions
    sacr = SACR(in_channels=cfg.sacr.in_channels, feature_channels=cfg.sacr.feature_channels,
                num_seg_classes=cfg.sacr.num_seg_classes, geom_dim=cfg.sacr.geom_dim,
                geom_hidden=cfg.sacr.geom_hidden, struct_dim=cfg.sacr.struct_dim,
                depth_pool_regions=R).to(device)
    camr = CAMR(z_struct_aug_dim=sacr.z_struct_aug_dim, pose_dim=cfg.camr.pose_dim,
                imu_dim=cfg.camr.imu_dim, window_size=cfg.camr.window_size, hidden_dim=cfg.camr.hidden_dim,
                use_attention=getattr(cfg.camr, "use_attention", False)).to(device)
    sacr.load_compatible_state_dict(torch.load(args.sacr_ckpt, map_location=device))
    camr.load_state_dict(torch.load(args.camr_ckpt, map_location=device))
    sacr.eval(); camr.eval()
    belief_dim = 2 * cfg.camr.hidden_dim
    ac = ActorCritic(belief_dim=belief_dim, action_dim=cfg.agss_ppo.action_dim,
                     actor_hidden=cfg.agss_ppo.actor_hidden, critic_hidden=cfg.agss_ppo.critic_hidden,
                     init_log_std=cfg.agss_ppo.init_log_std).to(device)
    blob = torch.load(args.policy_ckpt, map_location=device)
    ac.load_state_dict(blob["model"] if isinstance(blob, dict) and "model" in blob else blob)
    ac.eval()
    head = ComplexityHead(belief_dim, w_ref=cfg.agss_ppo.w_ref)
    head.load_state_dict(torch.load(args.complexity_ckpt, map_location=device, weights_only=True))
    agss = AGSSShield(d0=cfg.agss_ppo.d0, alpha=cfg.agss_ppo.alpha, complexity_dim=belief_dim, device=device,
                      tau=cfg.agss_ppo.tau, lateral_action_scale=cfg.agss_ppo.lateral_action_scale,
                      complexity_weights=head.shield_weights())
    wbuf = CausalWindowBuffer(cfg.camr.window_size, camr.input_dim, device, stride=cfg.camr.stride)

    def to_t(x):
        return torch.as_tensor(x, dtype=torch.float32, device=device).unsqueeze(0)

    # --- state sources ---
    vins = VinsOdometry(dep.vins_topic)
    imu_src = MavlinkImu(dep.mavlink_url)
    tracker = WaypointTracker(np.loadtxt(args.waypoints, delimiter=","), radius=dep.waypoint_radius)
    print("waiting for VINS-Mono initialization and MAVLink IMU ...", flush=True)
    while not vins.ready():
        time.sleep(0.1)

    # --- video + RC link ---
    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"cannot open video source: {args.source}")
    link = None if args.no_serial else RCLink(args.port, 115200)
    controller = VelocityToAttitude(args.vel_pid or dep.vel_pid, max_tilt_deg=dep.max_tilt_deg)
    send = make_velocity_sender(link, controller, hover_throttle=args.hover_throttle,
                                max_vertical_speed=cfg.env.max_vertical_speed,
                                max_yaw_rate_deg=cfg.env.max_yaw_rate_deg) if link else None

    dt = 1.0 / hz
    n, t_prev = 0, time.monotonic()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                if link:
                    link.disarm()
                print("no frame; retrying...", flush=True); time.sleep(0.1); continue
            rgb = cv2.cvtColor(cv2.resize(frame, (W, H)), cv2.COLOR_BGR2RGB).astype(np.float32)

            p_vio, q_vio, v_vio, _ = vins.latest()
            pose = waypoint_relative_pose(p_vio, q_vio, tracker.update(p_vio))
            imu = imu_src.latest()

            with torch.no_grad():
                z = sacr.encode(to_t(rgb).permute(0, 3, 1, 2) / 255.0)
                h_t = camr(wbuf.push(camr.fuse(z, to_t(pose), to_t(imu)))).h_t
                s = ac.act(h_t, deterministic=dep.deterministic)          # policy mean on the vehicle
                bounded = s.action.clamp(-1.0, 1.0)
                d_left, d_right = z[:, sacr.struct_dim], z[:, sacr.struct_dim + 2]
                proj = agss.project(bounded, h_t, d_left, d_right)
            a_safe = proj["safe_action"].squeeze(0).cpu().numpy() * a_max  # m/s, m/s, m/s, deg/s

            now = time.monotonic(); step_dt = max(now - t_prev, 1e-3); t_prev = now
            v_body = world_to_body_xy(v_vio[:2], yaw_from_quaternion(q_vio))
            if link:
                send(a_safe, v_body, step_dt, armed=not args.no_arm)
            n += 1
            if n % 30 == 0:
                print(f"\r wp={tracker.index} a_safe=[{a_safe[0]:+.2f} {a_safe[1]:+.2f} {a_safe[2]:+.2f} "
                      f"{a_safe[3]:+.1f}] agss={'on ' if bool(proj['intervened']) else 'off'} "
                      f"d_safe={float(proj['d_safe']):.2f} {'(dry)' if link is None else ''}   ",
                      end="", flush=True)
            time.sleep(max(0.0, dt - (time.monotonic() - now)))
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        imu_src.close()
        if link:
            link.close()
        print("\nstopped, disarmed.")


if __name__ == "__main__":
    main()
