"""Minimal, policy-free smoke test of the live GazeboROSEnv backend.

Proves the Rung 3 backend actually runs end to end against a live
PX4 SITL + Gazebo + MAVROS + ROS 2 stack: construct GazeboROSEnv, reset()
to get a real observation, take a few step()s with a fixed forward action,
and check the observation/step contract. No trained policy or checkpoints
needed -- this tests the environment plumbing, not navigation quality.

Run INSIDE the ros-bridge container, with the sim already up (px4-gazebo +
the corridor_sim launch providing MAVROS + ros_gz_bridge), e.g.:

    docker exec star_nav_ros_bridge bash -lc \
      'source /opt/ros/humble/setup.bash; source /ros2_ws/install/setup.bash; \
       cd /workspace && python3 scripts/gazebo_env_smoke.py --steps 5'
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from star_nav.utils.config import load_config


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None, help="YAML config; env.name is forced to gazebo_ros.")
    p.add_argument("--scenario", default="A")
    p.add_argument("--steps", type=int, default=5)
    args = p.parse_args(argv)

    cfg = load_config(args.config, overrides={"env.name": "gazebo_ros"})

    # Import lazily so this file parses on a host without ROS installed.
    from ros_gazebo_bridge.env import GazeboROSEnv

    env = GazeboROSEnv(cfg.env)
    try:
        obs = env.reset(scenario=args.scenario)
        assert obs.rgb.ndim == 3, f"rgb should be HxWx3, got {obs.rgb.shape}"
        assert obs.pose.shape == (7,), f"pose should be (7,), got {obs.pose.shape}"
        assert obs.imu.shape == (6,), f"imu should be (6,), got {obs.imu.shape}"
        print(f"reset OK: rgb={obs.rgb.shape} pose={obs.pose.shape} imu={obs.imu.shape}", flush=True)

        for i in range(args.steps):
            res = env.step(np.array([0.5, 0.0, 0.0, 0.0], dtype=np.float32))
            assert isinstance(res.reward, float)
            assert isinstance(res.done, bool)
            print(f"step {i}: reward={res.reward:.3f} done={res.done}", flush=True)
            if res.done:
                break
    finally:
        env.close()

    print("GAZEBO ENV SMOKE OK", flush=True)


if __name__ == "__main__":
    main()
