#!/usr/bin/env bash
# Open a camera view window (rqt_image_view). Run again for another window
# (e.g. once for rgb, once for depth).
#
# Usage:
#   ./camera.sh          # RGB camera (/camera)
#   ./camera.sh depth    # depth camera (/depth_camera)
#   ./camera.sh /some/other/topic
set -euo pipefail

if [[ "$(sudo docker inspect -f '{{.State.Running}}' star_nav_ros_bridge 2>/dev/null)" != "true" ]]; then
    echo "star_nav_ros_bridge is not running -- run ./start_sim.sh first." >&2
    exit 1
fi

case "${1:-rgb}" in
    rgb)   TOPIC=/camera ;;
    depth) TOPIC=/depth_camera ;;
    *)     TOPIC="$1" ;;
esac

sudo docker exec -d star_nav_ros_bridge bash -lc "
  export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
  export __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia
  source /opt/ros/humble/setup.bash && source /ros2_ws/install/setup.bash
  ros2 run rqt_image_view rqt_image_view $TOPIC"

echo "Opened rqt_image_view on $TOPIC (window should appear shortly)."
