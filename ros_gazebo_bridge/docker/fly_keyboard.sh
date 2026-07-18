#!/usr/bin/env bash
# Manual keyboard teleop (before any autonomy). MUST run in an interactive
# terminal -- this puts your TTY in raw key-read mode.
#
# Controls: t=takeoff  w/s=pitch fwd/back  a/d=roll left/right  q/e=yaw
#           r/f=climb/descend  space=hover  l=land  x/Ctrl-C=quit (auto-lands)
#
# Usage: ./fly_keyboard.sh [extra ros2-run args, e.g. --takeoff-alt 3.0]
set -euo pipefail

if [[ "$(sudo docker inspect -f '{{.State.Running}}' star_nav_ros_bridge 2>/dev/null)" != "true" ]]; then
    echo "star_nav_ros_bridge is not running -- run ./start_sim.sh first." >&2
    exit 1
fi

sudo docker exec -it star_nav_ros_bridge bash -lc "
  export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
  source /opt/ros/humble/setup.bash && source /ros2_ws/install/setup.bash
  ros2 run ros_gazebo_bridge keyboard_control $*"
