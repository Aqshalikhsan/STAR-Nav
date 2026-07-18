#!/usr/bin/env bash
# ============================================================================
# Train the STAR-Nav policy on MockCorridorEnv (dependency-free 2-D sim).
#
# One command does the whole Mock pipeline:
#   Phase 1 (perception) -- if checkpoints/mock/{sacr,camr}.pt are NOT cached,
#            train_ppo.py auto-collects Mock episodes and trains+freezes the
#            novelty SACR (depth-uncertainty) + CAMR (occupancy) on them.
#   Phase 2 (policy)     -- AGSS-PPO curriculum (12 stages, 10 m -> 48 m + up to
#            5 moving workers) over the FROZEN belief. Writes checkpoints/mock/:
#              ppo.pt       best-iter policy weights  (deploy this)
#              ppo_best.pt  best full state (model+optim+iter)
#              ppo_last.pt  every-iter full state (resume)
#
# Usage:
#   ./scripts/train_mock.sh                 # fresh curriculum run (2000 iters)
#   ITERS=800 ./scripts/train_mock.sh       # shorter run
#   RESUME=checkpoints/mock/ppo_last.pt ./scripts/train_mock.sh   # continue
#   START_STAGE=7 ./scripts/train_mock.sh   # jump into the curriculum
#   EXTRA="--no-occupancy" ./scripts/train_mock.sh   # ablation flags passthrough
#
# Monitor it live in another terminal with:  ./scripts/monitor.sh mock
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p logs
LOG="logs/train_mock.log"
ITERS="${ITERS:-2000}"
START_STAGE="${START_STAGE:-0}"
EXTRA="${EXTRA:-}"
RESUME_ARG=""
[[ -n "${RESUME:-}" ]] && RESUME_ARG="--resume ${RESUME}"

echo "==> Mock training -> checkpoints/mock/   (iters=${ITERS} start_stage=${START_STAGE})"
echo "==> logging to ${LOG}  (tail with: ./scripts/monitor.sh mock)"

python3 scripts/train_ppo.py \
  --curriculum \
  --iterations "${ITERS}" \
  --advance-at 0.5 --advance-streak 3 \
  --target-kl 0.03 \
  --start-stage "${START_STAGE}" \
  ${RESUME_ARG} ${EXTRA} 2>&1 | tee "${LOG}"

echo "==> done. policy: checkpoints/mock/ppo.pt  (deploy with ./scripts/deploy_best.sh)"
