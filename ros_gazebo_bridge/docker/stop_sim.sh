#!/usr/bin/env bash
# Stop the simulation containers (Gazebo lockstep physics otherwise burns
# ~130% CPU continuously even when idle).
set -euo pipefail
cd "$(dirname "$0")"
export PX4_GZ_WORLD=/worlds/scenario_a.sdf
sudo -E docker compose stop px4-gazebo ros-bridge
