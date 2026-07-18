#!/usr/bin/env bash
# Open an rqt diagnostic tool window. Run again for another one (same or
# different tool) -- each call opens one independent window.
#
# Usage:
#   ./rqt_tools.sh console    # ROS/MAVROS log messages, real-time
#   ./rqt_tools.sh plot       # plot a numeric topic field over time
#   ./rqt_tools.sh topic      # topic Hz/bandwidth monitor
#   ./rqt_tools.sh graph      # node <-> topic connection graph
#   ./rqt_tools.sh service    # call a ROS service from a GUI form
set -euo pipefail

if [[ "$(sudo docker inspect -f '{{.State.Running}}' star_nav_ros_bridge 2>/dev/null)" != "true" ]]; then
    echo "star_nav_ros_bridge is not running -- run ./start_sim.sh first." >&2
    exit 1
fi

case "${1:-}" in
    console) PKG=rqt_console ;;
    plot)    PKG=rqt_plot ;;
    topic)   PKG=rqt_topic ;;
    graph)   PKG=rqt_graph ;;
    service) PKG=rqt_service_caller ;;
    *)
        echo "Usage: $0 <console|plot|topic|graph|service>" >&2
        exit 1
        ;;
esac

sudo docker exec -d star_nav_ros_bridge bash -lc "
  export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
  export __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia
  source /opt/ros/humble/setup.bash && source /ros2_ws/install/setup.bash
  ros2 run $PKG $PKG"

echo "Opened $PKG (window should appear shortly)."
