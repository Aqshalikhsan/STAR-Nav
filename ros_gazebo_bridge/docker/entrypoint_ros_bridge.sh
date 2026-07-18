#!/usr/bin/env bash
# Modes:
#   bridge   (default) -- ros2 launch ros_gazebo_bridge corridor_sim.launch.py
#   world    -- generate a corridor world: entrypoint.sh world --scenario A --out ros_gazebo_bridge/worlds/scenario_a
#   train    -- python scripts/run_train_all.py --config configs/default.yaml
#   eval     -- python scripts/run_eval_all.py --config configs/default.yaml
# Not `set -u` -- ROS 2's own setup.bash scripts reference variables (e.g.
# AMENT_CURRENT_PREFIX) that are legitimately unset on first source, and
# aren't written to be nounset-safe.
set -eo pipefail

source /opt/ros/humble/setup.bash
source /ros2_ws/install/setup.bash
cd /workspace

MODE="${1:-bridge}"
shift || true

case "$MODE" in
  bridge)
    exec ros2 launch ros_gazebo_bridge corridor_sim.launch.py "$@"
    ;;
  world)
    exec python3 -m ros_gazebo_bridge.world_gen "$@"
    ;;
  train)
    exec python3 scripts/run_train_all.py "$@"
    ;;
  eval)
    exec python3 scripts/run_eval_all.py "$@"
    ;;
  *)
    echo "Unknown mode '$MODE' (expected: bridge|world|train|eval)" >&2
    exit 1
    ;;
esac
