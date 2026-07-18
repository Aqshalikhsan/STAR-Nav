"""Isolate the drone's BASIC flight stability from inference-path tracking: arm, climb,
then hold a single fixed position setpoint and log the attitude (roll/pitch/yaw
from gt_odom) so we can measure how much it twitches while simply hovering.

If roll/pitch oscillate here -- with NO horizontal motion commanded -- the shake
is the airframe/rate-controller tune (see px4_airframes/4013_gz_fpv5, which
documents the fpv5 as high-T/W and not bench-tuned), NOT the pure-pursuit path.

    docker exec star_nav_ros_bridge bash -lc '... python3 scripts/hover_test.py --secs 12'
"""
from __future__ import annotations
import argparse, os, sys, time
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT); sys.path.insert(0, os.path.join(_ROOT, "ros_gazebo_bridge"))
import numpy as np
from star_nav.utils.config import load_config


def quat_rpy(q):
    x, y, z, w = q
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.degrees([roll, pitch, yaw])


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    p.add_argument("--altitude", type=float, default=2.2)
    p.add_argument("--secs", type=float, default=12.0)
    p.add_argument("--hz", type=float, default=50.0)
    p.add_argument("--out", default=None, help="Optional CSV (t,roll,pitch,yaw,z).")
    args = p.parse_args(argv)

    cfg = load_config(args.config, overrides={"env.name": "gazebo_ros"})
    import rclpy
    from ros_gazebo_bridge.ros_bridge_node import ROSGazeboBridge
    if not rclpy.ok():
        rclpy.init()
    node = ROSGazeboBridge(cfg.env.ros, node_name=getattr(cfg.env.ros, "node_name", "star_nav_bridge"))
    node.wait_until_ready()
    try:
        node.disarm(); node.arm_and_offboard()
        t0 = time.monotonic()
        while time.monotonic() - t0 < 10.0:
            *_, gt = node.get_latest()
            if float(gt[2]) >= args.altitude - 0.15:
                break
            node.send_body_velocity_setpoint(0.0, 0.0, 1.0, 0.0); node.spin_for(0.1)
        node.send_body_velocity_setpoint(0.0, 0.0, 0.0, 0.0); node.spin_for(0.5)

        loc = node.local_xy(); gt = node.gt_xy()
        hold = loc.copy()
        print(f"holding local ({hold[0]:.2f},{hold[1]:.2f}) at {args.altitude} m for {args.secs}s ...", flush=True)
        dt = 1.0 / args.hz
        rows = []
        t_start = time.monotonic()
        while time.monotonic() - t_start < args.secs:
            node.send_local_position_setpoint(hold[0], hold[1], args.altitude, 0.0)
            node.spin_for(dt)
            *_, gtf = node.get_latest()
            rpy = quat_rpy(gtf[3:7])
            rows.append((time.monotonic() - t_start, rpy[0], rpy[1], rpy[2], float(gtf[2])))
        R = np.array(rows)
        # de-trend by subtracting the mean so we measure OSCILLATION, not offset
        roll, pitch, yaw, z = R[:, 1], R[:, 2], R[:, 3], R[:, 4]
        dr = np.diff(roll) * args.hz; dp = np.diff(pitch) * args.hz   # deg/s (attitude rate)
        print(f"\nHOVER attitude oscillation over {args.secs:.0f}s ({len(R)} samples):")
        print(f"  roll : std {roll.std():.2f} deg   rate std {dr.std():6.1f} deg/s   pk-pk {roll.max()-roll.min():.2f} deg")
        print(f"  pitch: std {pitch.std():.2f} deg   rate std {dp.std():6.1f} deg/s   pk-pk {pitch.max()-pitch.min():.2f} deg")
        print(f"  alt  : std {z.std()*100:.1f} cm   pk-pk {(z.max()-z.min())*100:.1f} cm")
        print(f"  -> attitude-rate std is the 'vibration' proxy; >~30 deg/s reads as visible camera shake.", flush=True)
        if args.out:
            import csv
            with open(args.out, "w", newline="") as f:
                w = csv.writer(f); w.writerow(["t", "roll", "pitch", "yaw", "z"]); w.writerows(R.tolist())
    finally:
        node.shutdown(); node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
