"""vision_deploy.py -- STAGE 4 full loop: real camera -> SACR -> CAMR -> AGSS-PPO
policy -> rc_link -> Arduino -> radio -> drone.

This is where the vision feed (DJI goggles -> CosmoStream/Raspi -> laptop) meets
the STAR-Nav brain and the control bridge. It is the real-hardware twin of
scripts/deploy_gazebo.py: same SACR/CAMR/policy/AGSS, but the frames come from a
video stream instead of ROS /camera, and the action goes out through rc_link
(hardware/laptop/rc_link.py) instead of MAVROS.

VIDEO SOURCE (--source):
  * RTSP straight from the Pi/CosmoStream  (LOWEST latency -- recommended):
        --source rtsp://<pi-ip>:8554/cam
  * OBS Virtual Camera (if you route through OBS for overlay/recording):
        in OBS click "Start Virtual Camera", then pass the /dev/videoN index:
        --source 10
  * any file / URL OpenCV can open.
  Every extra hop (OBS, re-encode) adds latency to the CONTROL loop -- prefer RTSP.

SAFETY / SANITY:
  * --no-serial  : DRY RUN. Prints channels instead of driving the Arduino, so you
                   can verify camera->action on the bench with NO drone. Do this first.
  * --no-arm     : run the policy but never set the arm channel high.
  * Props off, trainer-switch override in hand, until you trust it.

!!! HONEST CAVEAT !!!
SACR/CAMR were trained on SIM images and the policy on body VELOCITIES for a
position controller. On a real DJI feed + angle-mode FPV radio, expect the
belief distribution and action space to be OFF (see project deploy notes):
plan to fine-tune perception on real frames and adapt the action mapping
(policy_to_channels.py). This file is the correct plumbing to iterate on that,
not a guaranteed zero-shot flight.

    pip install opencv-python torch pyyaml pyserial
    python vision_deploy.py --source rtsp://192.168.1.50:8554/cam \
        --sacr-ckpt ../../checkpoints/mock/sacr.pt \
        --camr-ckpt ../../checkpoints/mock/camr.pt \
        --policy-ckpt ../../checkpoints/mock/ppo.pt --no-serial   # dry run first
"""
from __future__ import annotations

import argparse
import os
import sys
import time

# repo root so `star_nav` imports resolve (two levels up from hardware/laptop/)
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # for rc_link / policy_to_channels


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", required=True, help="RTSP url (rtsp://...) or OBS virtual-cam index (int).")
    p.add_argument("--config", default=os.path.join(_ROOT, "configs/default.yaml"))
    p.add_argument("--sacr-ckpt", default=os.path.join(_ROOT, "checkpoints/mock/sacr.pt"))
    p.add_argument("--camr-ckpt", default=os.path.join(_ROOT, "checkpoints/mock/camr.pt"))
    p.add_argument("--policy-ckpt", default=os.path.join(_ROOT, "checkpoints/mock/ppo.pt"))
    p.add_argument("--port", default="/dev/ttyUSB0", help="Arduino serial port.")
    p.add_argument("--hover-throttle", type=float, default=0.0, help="Normalized hover throttle [-1,1] (measure it manually first!).")
    p.add_argument("--gains", type=float, nargs=4, default=(1.0, 1.0, 0.5, 1.0), help="roll pitch yaw vz scaling.")
    p.add_argument("--hz", type=float, default=30.0, help="Control loop rate cap.")
    p.add_argument("--no-serial", action="store_true", help="DRY RUN: print channels, don't open the Arduino.")
    p.add_argument("--no-arm", action="store_true", help="Never raise the arm channel.")
    p.add_argument("--deterministic", action="store_true", default=True)
    args = p.parse_args(argv)

    import numpy as np
    import cv2
    import torch
    from star_nav.models.sacr import SACR
    from star_nav.models.camr import CAMR, CausalWindowBuffer
    from star_nav.models.agss_ppo import ActorCritic, AGSSShield
    from star_nav.utils.config import load_config
    from star_nav.utils.seeding import get_device
    from rc_link import RCLink
    from policy_to_channels import make_sender

    cfg = load_config(args.config)
    device = get_device(cfg.device)
    W, H = cfg.env.image_size  # SACR input size the perception was trained at

    # --- build perception + policy exactly like the sim deploy ---
    unc_on = getattr(cfg.sacr, "depth_uncertainty", False)
    R = cfg.sacr.depth_pool_regions
    sacr = SACR(in_channels=cfg.sacr.in_channels, feature_channels=cfg.sacr.feature_channels,
                num_seg_classes=cfg.sacr.num_seg_classes, geom_dim=cfg.sacr.geom_dim,
                geom_hidden=cfg.sacr.geom_hidden, struct_dim=cfg.sacr.struct_dim,
                depth_pool_regions=R, depth_uncertainty=unc_on).to(device)
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
    blob = torch.load(args.policy_ckpt, map_location=device)
    ac.load_state_dict(blob["model"] if isinstance(blob, dict) and "model" in blob else blob)
    ac.eval()
    agss = AGSSShield(d0=cfg.agss_ppo.d0, alpha=cfg.agss_ppo.alpha, complexity_dim=belief_dim, device=device,
                      beta=getattr(cfg.agss_ppo, "beta_unc", 0.0), gamma=getattr(cfg.agss_ppo, "gamma_occ", 0.0))
    wbuf = CausalWindowBuffer(cfg.camr.window_size, camr.input_dim, device)

    def to_t(x):
        return torch.as_tensor(x, dtype=torch.float32, device=device).unsqueeze(0)

    # NOTE: real pose/imu would come from FC telemetry (attitude/IMU). Without a
    # position source they are incomplete; zeros are a placeholder to keep dims.
    zero_pose = np.zeros(cfg.camr.pose_dim, dtype=np.float32)
    zero_imu = np.zeros(cfg.camr.imu_dim, dtype=np.float32)

    def encode(rgb):
        with torch.no_grad():
            z = sacr.encode(to_t(rgb).permute(0, 3, 1, 2) / 255.0)
            h = camr(wbuf.push(camr.fuse(z, to_t(zero_pose), to_t(zero_imu)))).h_t
        return h, z

    def shield_terms(z, h):
        if unc_on:
            dl, dr = z[:, -2 * R], z[:, -(R + 1)]
            sl = torch.exp(0.5 * z[:, -R].clamp(-6.0, 1.4)); sr = torch.exp(0.5 * z[:, -1].clamp(-6.0, 1.4))
        else:
            dl, dr = z[:, -R], z[:, -1]; sl = sr = None
        ol = orr = None
        if camr.use_occupancy:
            with torch.no_grad():
                pocc = torch.sigmoid(camr.predict_occupancy(h))
            ol, orr = pocc[:, 0], pocc[:, 1]
        return dl, dr, sl, sr, ol, orr

    # --- video source ---
    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"cannot open video source: {args.source}")

    link = None if args.no_serial else RCLink(args.port, cfg.env.ros.baud if hasattr(cfg.env.ros, "baud") else 115200)
    send = make_sender(link, hover_throttle=args.hover_throttle, gains=tuple(args.gains)) if link else None
    print(f"vision deploy: source={args.source}  serial={'DRY-RUN' if link is None else args.port}  "
          f"belief_dim={belief_dim}  (props off!)", flush=True)

    dt = 1.0 / args.hz
    n = 0; t0 = time.monotonic()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                if link:
                    link.disarm()               # no frame -> stop commanding motion
                print("no frame; retrying...", flush=True); time.sleep(0.1); continue
            rgb = cv2.cvtColor(cv2.resize(frame, (W, H)), cv2.COLOR_BGR2RGB).astype(np.float32)
            h_t, z = encode(rgb)
            with torch.no_grad():
                s = ac.act(h_t, deterministic=args.deterministic)
                dl, dr, sl, sr, ol, orr = shield_terms(z, h_t)
                proj = agss.project(s.action, h_t, dl, dr, sigma_left=sl, sigma_right=sr, occ_left=ol, occ_right=orr)
            action = proj["safe_action"].squeeze(0).cpu().numpy()   # [vx, vy, vz, yaw]
            if link:
                send(action, armed=not args.no_arm)
            n += 1
            if n % 30 == 0:
                fps = n / (time.monotonic() - t0)
                print(f"\r fps~{fps:4.1f}  action=[{action[0]:+.2f} {action[1]:+.2f} "
                      f"{action[2]:+.2f} {action[3]:+.2f}]  {'(dry)' if link is None else ''}   ", end="", flush=True)
            time.sleep(max(0.0, dt - (time.monotonic() - t0) % dt))
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if link:
            link.close()
        print("\nstopped, disarmed.")


if __name__ == "__main__":
    main()
