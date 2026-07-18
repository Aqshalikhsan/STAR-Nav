#!/usr/bin/env bash
# ============================================================================
# Live-monitor a STAR-Nav training or deploy run: prints a compact summary of
# the key metrics so far, then follows the log (tail -f).
#
# Usage:
#   ./scripts/monitor.sh              # Mock PPO training (default)
#   ./scripts/monitor.sh mock         # Mock PPO training  (logs/train_mock.log)
#   ./scripts/monitor.sh sacr         # Gazebo SACR retrain (logs/train_sacr_novelty.log)
#   ./scripts/monitor.sh camr         # Gazebo CAMR retrain (logs/train_camr_novelty.log)
#   ./scripts/monitor.sh deploy       # Gazebo deploy       (logs/deploy.log)
#   ./scripts/monitor.sh path/to.log  # any explicit log file
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

case "${1:-mock}" in
  mock)   LOG="logs/train_mock.log" ; KEY="^iter|success|advance|stage" ;;
  sacr)   LOG="logs/train_sacr_novelty.log" ; KEY="^epoch|L_SACR|new best" ;;
  camr)   LOG="logs/train_camr_novelty.log" ; KEY="^epoch|L_CAMR|occupancy|new best" ;;
  deploy) LOG="logs/deploy.log" ; KEY="episode|reward|agss|loaded|Traceback" ;;
  *)      LOG="$1" ; KEY="." ;;
esac

if [[ ! -f "$LOG" ]]; then
  echo "no log yet at: $LOG"
  echo "(is the run started? e.g. ./scripts/train_mock.sh  or  ./scripts/deploy_best.sh)"
  exit 1
fi

echo "=============================================================="
echo " monitoring: $LOG"
echo " last 15 key lines:"
echo "=============================================================="
grep -nE "$KEY" "$LOG" 2>/dev/null | tail -15 || tail -15 "$LOG"
echo "=============================================================="
echo " following live (Ctrl-C to stop)..."
echo "=============================================================="
tail -n 0 -f "$LOG"
