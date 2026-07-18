#!/usr/bin/env bash
# ============================================================================
# Deploy the best Mock-trained STAR-Nav policy in real PX4+Gazebo, end to end.
#
# This wraps EVERY step (and every gotcha found the hard way) so a deploy is one
# command. It:
#   1. brings up the sim (px4-gazebo + ros-bridge) via docker compose,
#   2. FRESH-restarts PX4 so its EKF is clean -- PX4 denies arming after a
#      previous flight+land cycle (see docker/MANUAL_FLIGHT_GUIDE.md), and a
#      fresh spawn puts the drone at the corridor start (x=1),
#   3. syncs the LIVE repo code into the ros-bridge container (its baked image is
#      stale: pre-novelty star_nav, no scripts/, missing world layout json),
#   4. bridges the ground-truth odometry for the spawned model (the baked bridge
#      hard-codes the wrong model name, so gt_odom never arrives otherwise),
#   5. runs scripts/deploy_gazebo.py: Gazebo-novelty SACR+CAMR -> Mock policy ->
#      adaptive AGSS shield -> GazeboROSEnv (arm -> takeoff -> altitude-hold ->
#      fly). Logs to logs/deploy.log.
#
# Usage:
#   ./scripts/deploy_best.sh                 # 1 episode, 150 steps, deterministic
#   EPISODES=3 MAXSTEPS=200 ./scripts/deploy_best.sh
#   PX4_SIM_MODEL=pavo_femto ./scripts/deploy_best.sh
#
# Monitor:  ./scripts/monitor.sh deploy
# NOTE: needs the docker images built (see ros_gazebo_bridge/docker/setup_linux.sh).
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="$(pwd)"

COMPOSE="ros_gazebo_bridge/docker/docker-compose.yml"
PX4C="star_nav_px4_gazebo"
ROSC="star_nav_ros_bridge"
MODEL="${PX4_SIM_MODEL:-fpv5}"
EPISODES="${EPISODES:-1}"
MAXSTEPS="${MAXSTEPS:-150}"
PKG="/ros2_ws/install/ros_gazebo_bridge/lib/python3.10/site-packages/ros_gazebo_bridge"
WORLDS_PKG="/ros2_ws/install/ros_gazebo_bridge/lib/python3.10/ros_gazebo_bridge/worlds"

export PX4_GZ_WORLD=/worlds/scenario_a.sdf
export PX4_SIM_MODEL="${MODEL}"
export PX4_GZ_MODEL_POSE="${PX4_GZ_MODEL_POSE:-1,24,0.3,0,0,0}"
export HEADLESS="${HEADLESS:-1}"

say() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }

xhost +local:docker >/dev/null 2>&1 || true

say "1/5  Bringing up the sim (compose up)..."
docker compose -f "$COMPOSE" up -d px4-gazebo ros-bridge >/dev/null

say "2/5  Fresh-restarting PX4 (reset EKF so arming is allowed) ..."
docker restart "$PX4C" >/dev/null
for _ in $(seq 1 30); do
  docker logs "$PX4C" 2>&1 | grep -q "Ready for takeoff" && break; sleep 2
done
docker logs "$PX4C" 2>&1 | grep -q "Ready for takeoff" \
  || { echo "PX4 not ready -- check: docker logs $PX4C"; exit 1; }
docker restart "$ROSC" >/dev/null
sleep 8
for _ in $(seq 1 15); do
  docker exec "$ROSC" bash -lc 'source /opt/ros/humble/setup.bash; timeout 3 ros2 topic echo --once /mavros/state 2>/dev/null' \
    | grep -q 'connected: true' && break; sleep 3
done

say "3/5  Syncing live repo code into $ROSC (image is stale) ..."
docker cp "$REPO/star_nav"  "$ROSC:/workspace/" 2>/dev/null
docker cp "$REPO/scripts"   "$ROSC:/workspace/" 2>/dev/null
docker cp "$REPO/configs"   "$ROSC:/workspace/" 2>/dev/null
docker cp "$REPO/ros_gazebo_bridge/ros_gazebo_bridge/env.py" "$ROSC:$PKG/env.py" 2>/dev/null
docker exec "$ROSC" bash -lc "mkdir -p $WORLDS_PKG" 2>/dev/null
docker cp "$REPO/ros_gazebo_bridge/worlds/scenario_a.world.json" "$ROSC:$WORLDS_PKG/scenario_a.world.json" 2>/dev/null
docker cp "$REPO/ros_gazebo_bridge/worlds/scenario_a.sdf"        "$ROSC:$WORLDS_PKG/scenario_a.sdf" 2>/dev/null

say "4/5  Bridging ground-truth odometry for /model/${MODEL}_0 ..."
docker exec -d "$ROSC" bash -lc "source /opt/ros/humble/setup.bash; source /ros2_ws/install/setup.bash 2>/dev/null; \
  exec ros2 run ros_gz_bridge parameter_bridge \
    /model/${MODEL}_0/odometry@nav_msgs/msg/Odometry[gz.msgs.Odometry > /tmp/gt_odom_bridge.log 2>&1"
sleep 5

say "5/5  Flying the policy (episodes=${EPISODES} max_steps=${MAXSTEPS}) ..."
mkdir -p logs
docker exec "$ROSC" bash -lc "source /opt/ros/humble/setup.bash; source /ros2_ws/install/setup.bash 2>/dev/null; \
  cd /workspace && python3 scripts/deploy_gazebo.py \
    --episodes ${EPISODES} --max-steps ${MAXSTEPS} --deterministic 2>&1" 2>&1 \
  | grep -vE "FutureWarning|weights_only|RuntimeWarning|umr_sum" | tee logs/deploy.log

say "done. (perception <- checkpoints/gazebo/, policy <- checkpoints/mock/ppo.pt)"
