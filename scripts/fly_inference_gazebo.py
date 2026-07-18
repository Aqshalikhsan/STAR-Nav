"""Fly a Mock-exported trajectory as position inference in real PX4+Gazebo.

The decoupled deploy (see scripts/export_mock_trajectory.py for the "why"): the
Mock-trained policy WANDERS when run live in Gazebo (the zero-shot belief gap),
but the PATH it produces in Mock is good. So we exported that path to a inference
list plus a byte-matched Gazebo world (trunks at the same positions). This
script just position-tracks those inference -- which PX4 offboard does reliably
-- through the matched trunk field, so the flight stays collision-free WITHOUT
running perception or the policy at deploy time. Perception is decoupled from
control.

Frame handling ("samakan env nya"): the inference are in the Mock/world corridor
frame (start ~x=2). MAVROS publishes local_position in ENU with its origin at
the EKF init (spawn) point, while gt_odom reports true world xy. We measure the
constant offset `local - gt` once after takeoff and add it to every inference, so
a world/Mock inference maps to the correct MAVROS-local setpoint regardless of
where PX4 spawned the vehicle. Progress is judged in world (gt) frame against
the raw inference.

Run INSIDE the ROS container against a live bridge/MAVROS/PX4 that launched the
MATCHED world (renders/deploy/zigzag.sdf + .world.json):

    docker exec star_nav_ros_bridge bash -lc \
      'source /opt/ros/humble/setup.bash; source /ros2_ws/install/setup.bash; \
       python3 scripts/fly_inference_gazebo.py \
         --inference renders/deploy/zigzag_inference.csv'
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "ros_gazebo_bridge"))

import numpy as np

from star_nav.utils.config import load_config


def read_inference(path):
    wps = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            wps.append((float(r["x"]), float(r["y"])))
    return np.array(wps, dtype=np.float32)


def densify_smooth(wps, step=0.1, smooth_win=0.5):
    """Resample the inference polyline to `step` m and lightly boxcar-smooth it
    (window `smooth_win` m) so the corners are gently rounded -- the tracker then
    follows a continuous, kink-free path, which is what keeps the drone's attitude
    (and its onboard camera) steady instead of jerking at each inference point."""
    wps = wps.astype(float)
    P = [wps[0]]
    for a, b in zip(wps[:-1], wps[1:]):
        d = float(np.linalg.norm(b - a)); n = max(1, int(round(d / step)))
        for k in range(1, n + 1):
            P.append(a + (b - a) * (k / n))
    P = np.array(P)
    w = max(1, int(round(smooth_win / step)))
    if w > 1 and len(P) > 2 * w:
        ker = np.ones(w) / w
        Ps = np.stack([np.convolve(P[:, 0], ker, mode="same"),
                       np.convolve(P[:, 1], ker, mode="same")], axis=1)
        m = w // 2                       # convolve 'same' distorts the ends -> keep originals
        Ps[:m] = P[:m]; Ps[-m:] = P[-m:]
        P = Ps
    return P


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None)
    p.add_argument("--inference", default="renders/deploy/zigzag_inference.csv")
    p.add_argument("--altitude", type=float, default=2.2, help="Corridor flight altitude (m).")
    p.add_argument("--lookahead", type=float, default=1.2, help="Pure-pursuit carrot distance ahead on the path (m).")
    p.add_argument("--cruise", type=float, default=1.2, help="Cruise speed / velocity feed-forward (m/s).")
    p.add_argument("--path-smooth", type=float, default=0.5, help="Boxcar corner-rounding window on the path (m).")
    p.add_argument("--max-flight-time", type=float, default=120.0, help="Safety cap on the tracking phase (s).")
    p.add_argument("--climb-speed", type=float, default=1.0)
    p.add_argument("--hz", type=float, default=20.0, help="Control loop / setpoint update rate (Hz).")
    p.add_argument("--land", action="store_true", help="Descend + disarm at the end (default: hover at goal).")
    p.add_argument("--log-traj", default=None, help="Write the LIVE ground-truth flight path (t,x,y,z) to this CSV.")
    p.add_argument("--save-frames", default=None, help="Save onboard /camera frames (FPV) to this .npy stack for a video.")
    p.add_argument("--actors", default=None, help="Per-step actor positions .npy (T,n,2) to replay as moving people.")
    p.add_argument("--drone-path", default=None, help="Mock drone path .npy (T,2) used to sync actors to progress.")
    p.add_argument("--actor-gain", type=float, default=3.0, help="P-gain for the actor position servo.")
    p.add_argument("--actor-vmax", type=float, default=1.5, help="Max actor speed (m/s).")
    args = p.parse_args(argv)

    cfg = load_config(args.config, overrides={"env.name": "gazebo_ros"})
    wps = read_inference(args.inference)
    print(f"loaded {len(wps)} inference from {args.inference}: "
          f"start=({wps[0,0]:.1f},{wps[0,1]:.1f}) goal=({wps[-1,0]:.1f},{wps[-1,1]:.1f})", flush=True)

    import rclpy
    from ros_gazebo_bridge.ros_bridge_node import ROSGazeboBridge

    if not rclpy.ok():
        rclpy.init()
    node = ROSGazeboBridge(cfg.env.ros, node_name=getattr(cfg.env.ros, "node_name", "star_nav_bridge"))
    node.wait_until_ready()

    # --- moving people: VelocityControl-driven person_worker models, position-
    # servoed to the Mock actor trajectory and SYNCED to the drone's own progress
    # (nearest Mock step by drone-x). This is the faithful replay: drone + people
    # come from the SAME Mock rollout, so at each mock step the policy kept a real
    # >=0.97 m gap -- the drone NEVER overlaps a person. (Driving the crowd on a
    # free wall-clock instead desyncs it from that avoidance and makes the drone
    # appear to fly through people in the 2-D top-down, which it does not: it also
    # flies at 2.2 m, over their ~1.8 m heads -- see the altitude panel.) ---
    A = D = est = None
    worker_pubs = []
    if args.actors and os.path.exists(args.actors) and args.drone_path and os.path.exists(args.drone_path):
        from geometry_msgs.msg import Twist
        A = np.load(args.actors).astype(np.float64)       # (T, n, 2)
        D = np.load(args.drone_path).astype(np.float64)   # (T, 2) drone path, SAME mock steps
        nW = A.shape[1]
        worker_pubs = [node.create_publisher(Twist, f"/model/worker{i}/cmd_vel", 10) for i in range(nW)]
        est = A[0].copy()
        print(f"driving {nW} workers synced to drone progress (faithful avoidance) from {args.actors}", flush=True)

    def drive_workers(drone_xy, dt):
        if A is None:
            return
        from geometry_msgs.msg import Twist
        s = int(np.argmin(np.abs(D[:, 0] - drone_xy[0])))   # nearest Mock step by drone-x
        for i in range(A.shape[1]):
            v = np.clip(args.actor_gain * (A[s, i] - est[i]), -args.actor_vmax, args.actor_vmax)
            est[i] += v * dt
            m = Twist(); m.linear.x = float(v[0]); m.linear.y = float(v[1])
            worker_pubs[i].publish(m)

    try:
        # --- arm + climb to altitude (body-velocity, like GazeboROSEnv._takeoff) ---
        node.disarm()
        node.arm_and_offboard()
        print("armed + offboard; climbing...", flush=True)
        t0 = time.monotonic()
        while time.monotonic() - t0 < 10.0:
            *_, gt = node.get_latest()
            z = float(gt[2])
            if z >= args.altitude - 0.15:
                break
            node.send_body_velocity_setpoint(0.0, 0.0, args.climb_speed, 0.0)
            node.spin_for(0.1)
        node.send_body_velocity_setpoint(0.0, 0.0, 0.0, 0.0)
        node.spin_for(0.5)

        # --- constant world->local offset (see module docstring) ---
        loc = node.local_xy(); gt = node.gt_xy()
        if loc is None or gt is None:
            raise RuntimeError("no local/gt pose yet -- cannot map inference into the local frame")
        offset = loc - gt
        print(f"world->local offset = ({offset[0]:.2f},{offset[1]:.2f})  "
              f"(gt=({gt[0]:.1f},{gt[1]:.1f}) local=({loc[0]:.1f},{loc[1]:.1f}))", flush=True)

        # --- smooth pure-pursuit tracking ---
        # Instead of stop-and-go position setpoints per inference point (which make PX4
        # accelerate/decelerate + snap yaw at every point -> attitude twitch that
        # shakes the onboard camera), follow a continuous "carrot" a fixed
        # look-ahead ahead of the drone along a densified, lightly-smoothed path,
        # with a velocity feed-forward at the cruise speed and a kink-free yaw from
        # the path tangent. Constant velocity + smooth heading => steady attitude.
        dt = 1.0 / args.hz
        P = densify_smooth(wps, step=0.1, smooth_win=args.path_smooth)   # (N, 2)
        seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
        S = np.concatenate([[0.0], np.cumsum(seg)]); total = float(S[-1])
        tang = np.gradient(P, axis=0)
        tang /= (np.linalg.norm(tang, axis=1, keepdims=True) + 1e-9)

        def at(s):
            s = float(np.clip(s, 0.0, total))
            pos = np.array([np.interp(s, S, P[:, 0]), np.interp(s, S, P[:, 1])])
            th = float(np.arctan2(np.interp(s, S, tang[:, 1]), np.interp(s, S, tang[:, 0])))
            return pos, th

        traj_log = []                       # live ground-truth flight path (t,x,y,z)
        _frames = []; _frame_t = []         # onboard FPV frames + their timestamps
        t_start = time.monotonic(); last_print = 0.0
        print(f"pure-pursuit over {total:.1f} m path (lookahead {args.lookahead} m, "
              f"cruise {args.cruise} m/s, smooth {args.path_smooth} m)", flush=True)
        while True:
            gt = node.gt_xy()
            j = int(np.argmin(np.linalg.norm(P - gt, axis=1))); s_now = float(S[j])
            carrot, yaw = at(s_now + args.lookahead)
            tgt = carrot + offset
            ff = args.cruise * np.array([np.cos(yaw), np.sin(yaw)])       # velocity feed-forward
            node.send_local_position_setpoint(tgt[0], tgt[1], args.altitude, yaw, vx=ff[0], vy=ff[1])
            node.spin_for(dt)
            rgb = node.get_latest()[0]
            gt = node.gt_xy()
            drive_workers(gt, dt)
            *_, gtf = node.get_latest()
            now = time.monotonic() - t_start
            traj_log.append((now, float(gtf[0]), float(gtf[1]), float(gtf[2])))
            if args.save_frames and rgb is not None:
                _frames.append(np.asarray(rgb).copy()); _frame_t.append(now)
            if s_now - last_print >= 5.0:
                last_print = s_now
                print(f"  s={s_now:5.1f}/{total:.0f} m  pos=({gt[0]:5.1f},{gt[1]:5.1f})", flush=True)
            if s_now >= total - 0.4:
                break
            if time.monotonic() - t_start > args.max_flight_time:
                print(f"  flight timeout at s={s_now:.1f}/{total:.1f} m", flush=True)
                break

        if args.log_traj:
            os.makedirs(os.path.dirname(args.log_traj) or ".", exist_ok=True)
            with open(args.log_traj, "w", newline="") as f:
                w = csv.writer(f); w.writerow(["t", "x", "y", "z"])
                w.writerows([(f"{t:.3f}", f"{x:.3f}", f"{y:.3f}", f"{z:.3f}") for t, x, y, z in traj_log])
            print(f"logged {len(traj_log)} gt path samples -> {args.log_traj}", flush=True)
        if args.save_frames and _frames:
            os.makedirs(os.path.dirname(args.save_frames) or ".", exist_ok=True)
            np.save(args.save_frames, np.stack(_frames))
            np.save(os.path.splitext(args.save_frames)[0] + "_t.npy", np.array(_frame_t))
            print(f"saved {len(_frames)} FPV frames (+timestamps) -> {args.save_frames}", flush=True)

        gt = node.gt_xy()
        print(f"\nDONE. final world pos=({gt[0]:.1f},{gt[1]:.1f}) goal=({wps[-1,0]:.1f},{wps[-1,1]:.1f}) "
              f"dist={np.linalg.norm(gt - wps[-1]):.2f}m", flush=True)

        if args.land:
            print("landing...", flush=True)
            t0 = time.monotonic()
            while time.monotonic() - t0 < 8.0:
                node.send_body_velocity_setpoint(0.0, 0.0, -0.6, 0.0)
                node.spin_for(0.1)
            node.disarm()
        else:
            # hold at the goal so the vehicle doesn't drift out of OFFBOARD.
            tgt = wps[-1] + offset
            for _ in range(20):
                node.send_local_position_setpoint(tgt[0], tgt[1], args.altitude, 0.0)
                node.spin_for(0.1)
    finally:
        for pub in worker_pubs:                 # stop the crowd
            from geometry_msgs.msg import Twist
            pub.publish(Twist())
        node.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
