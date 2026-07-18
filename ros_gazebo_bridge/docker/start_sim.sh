#!/usr/bin/env bash
# Launch PX4+Gazebo (GUI) + MAVROS/ros_gz_bridge, generating the world first
# if it doesn't exist yet. Idempotent: safe to re-run.
#
# Usage:
#   ./start_sim.sh              # launch (reuse containers if already up)
#   ./start_sim.sh --fresh      # force-recreate both containers (use this
#                                # if arming was denied after a previous
#                                # landing -- see MANUAL_FLIGHT_GUIDE.md)
#   PX4_SIM_MODEL=pavo_femto ./start_sim.sh   # fly a different vehicle
set -euo pipefail
cd "$(dirname "$0")"

FRESH=0
[[ "${1:-}" == "--fresh" ]] && FRESH=1

REPO_ROOT="$(cd ../.. && pwd)"
WORLD_SDF="../worlds/scenario_a.sdf"
export PX4_GZ_WORLD=/worlds/scenario_a.sdf
export PX4_SIM_MODEL="${PX4_SIM_MODEL:-fpv5}"
export PX4_GZ_MODEL_POSE="${PX4_GZ_MODEL_POSE:-1,24,0.3,0,0,0}"
export HEADLESS="${HEADLESS:-0}"
export DISPLAY="${DISPLAY:-:0}"

say() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }

xhost +local:docker >/dev/null 2>&1 || true

if [[ ! -f "$WORLD_SDF" ]]; then
    say "Generating world (scenario A)..."
    ( cd .. && source "$REPO_ROOT/.venv/bin/activate" && \
      python3 -m ros_gazebo_bridge.world_gen --config "$REPO_ROOT/configs/default.yaml" \
          --scenario A --out worlds/scenario_a )
fi

if [[ "$FRESH" == "1" ]]; then
    say "Stopping/removing existing containers (--fresh)..."
    sudo -E docker compose stop px4-gazebo ros-bridge 2>&1 || true
    sudo -E docker compose rm -f px4-gazebo ros-bridge 2>&1 || true
fi

say "Starting px4-gazebo (PX4_SIM_MODEL=$PX4_SIM_MODEL)..."
sudo -E docker compose up -d px4-gazebo

say "Waiting for PX4 'Ready for takeoff!'..."
for _ in $(seq 1 30); do
    sudo docker logs star_nav_px4_gazebo 2>&1 | grep -q "Ready for takeoff" && break
    sleep 1
done
sudo docker logs star_nav_px4_gazebo 2>&1 | grep -q "Ready for takeoff" \
    || { echo "PX4 did not report ready in time -- check: sudo docker logs star_nav_px4_gazebo"; exit 1; }

say "Starting ros-bridge (MAVROS + ros_gz_bridge)..."
sudo -E docker compose up -d ros-bridge
[[ "$FRESH" == "1" ]] && sudo docker restart star_nav_ros_bridge >/dev/null

say "Waiting for MAVROS heartbeat..."
for _ in $(seq 1 30); do
    sudo docker logs star_nav_ros_bridge 2>&1 | grep -q "CON: Got HEARTBEAT" && break
    sleep 1
done
sudo docker logs star_nav_ros_bridge 2>&1 | grep -q "CON: Got HEARTBEAT" \
    || { echo "MAVROS did not connect in time -- check: sudo docker logs star_nav_ros_bridge"; exit 1; }

say "Ready. Next: ./camera.sh (view camera) or ./fly_keyboard.sh (fly)."
