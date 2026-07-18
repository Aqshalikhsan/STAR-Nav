#!/usr/bin/env bash
# One-shot bootstrap for running the STAR-Nav sim stack on a NATIVE Ubuntu/
# Debian Linux box (where -- unlike WSL -- the GPU renders gz's camera, so
# the drone camera actually produces images and flight video renders fast).
#
# What it does (all idempotent -- safe to re-run):
#   1. installs Docker Engine + the compose plugin (if missing)
#   2. installs the NVIDIA Container Toolkit so containers can use the GPU
#      (only if an NVIDIA GPU + driver is detected)
#   3. builds both images via docker compose
#
# It does NOT install the NVIDIA *driver* itself (that is a host concern):
# on Ubuntu run `sudo ubuntu-drivers autoinstall && reboot` first if
# `nvidia-smi` doesn't work yet.
#
# Usage:
#   git clone https://github.com/Aqshalikhsan/star-nav.git
#   cd star-nav/code/ros_gazebo_bridge/docker
#   ./setup_linux.sh
set -euo pipefail

say() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m!! %s\033[0m\n' "$*"; }

if [ "$(uname -s)" != "Linux" ]; then
    warn "This script is for native Linux. On Windows use WSL2 (see README)."
    exit 1
fi

# --- 1. Docker Engine + compose plugin --------------------------------------
if ! command -v docker >/dev/null 2>&1; then
    say "Installing Docker Engine (get.docker.com)..."
    curl -fsSL https://get.docker.com | sudo sh
    sudo usermod -aG docker "$USER" || true
    warn "Added you to the 'docker' group -- log out/in (or run 'newgrp docker') for it to take effect."
else
    say "Docker already installed: $(docker --version)"
fi

if ! docker compose version >/dev/null 2>&1; then
    say "Installing docker compose plugin..."
    sudo apt-get update && sudo apt-get install -y docker-compose-plugin
fi

# --- 2. NVIDIA Container Toolkit (GPU in containers) ------------------------
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    if ! docker info 2>/dev/null | grep -qi nvidia; then
        say "Installing NVIDIA Container Toolkit (GPU rendering + CUDA in containers)..."
        curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
            | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
        curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
            | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
            | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
        sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
        sudo nvidia-ctk runtime configure --runtime=docker
        sudo systemctl restart docker
    else
        say "NVIDIA Container Toolkit already configured."
    fi
    export STAR_NAV_GPU=1
else
    warn "No working nvidia-smi -- building CPU-only (gz camera will fall back to software rendering)."
    warn "For GPU: install the driver first ('sudo ubuntu-drivers autoinstall && reboot'), then re-run."
fi

# --- 3. build images --------------------------------------------------------
cd "$(dirname "$0")"
say "Building images (px4-gazebo + ros-bridge) -- this takes a while the first time..."
docker compose build

say "Done. Next:"
cat <<'EOF'
  # allow the containers to draw on your X server (for the Gazebo GUI):
  xhost +local:docker

  # generate a world, launch the sim, bring up MAVROS+bridge, then fly:
  cd ..            # -> ros_gazebo_bridge/
  python3 -m ros_gazebo_bridge.world_gen --config ../configs/default.yaml \
      --scenario A --out worlds/scenario_a
  PX4_GZ_WORLD=$(pwd)/worlds/scenario_a.sdf PX4_SIM_MODEL=fpv5 \
      PX4_GZ_MODEL_POSE="1,24,0.3,0,0,0" HEADLESS=0 \
      docker compose -f docker/docker-compose.yml up -d px4-gazebo
  docker compose -f docker/docker-compose.yml up -d ros-bridge

  # keyboard-fly it (needs a TTY):
  docker exec -it star_nav_ros_bridge bash -lc '
    export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
    source /opt/ros/humble/setup.bash && source /ros2_ws/install/setup.bash
    python3 -m ros_gazebo_bridge.keyboard_control'
EOF
