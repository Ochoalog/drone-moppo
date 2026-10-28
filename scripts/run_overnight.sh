#!/usr/bin/env bash
# Overnight: PPO on {S0,S1,S2} x seeds {0,1,2} (resumable: runs with final_model.zip are skipped; a failed
# run does not stop the others), then full paired evaluation on the test region, north-region
# evaluation (H3) and REPORT.md.
# Usage: nohup bash scripts/run_overnight.sh > runs/overnight_$(date +%F).log 2>&1 &
set -uo pipefail
source "$(dirname "$0")/env.sh"
log() { echo "[$(date '+%F %T')] $*"; }
CFG=configs/overnight.yaml
ROOT_RUNS=runs/overnight
mkdir -p "$ROOT_RUNS"
DESC=$($PY scripts/best_descriptor.py) || { log "sanity gate failed: no RL training (see REPORT.md)"; exit 2; }
log "descriptor: $DESC; config: $CFG; timesteps: $($PY -c "from dal.config import load_config;print(load_config('$CFG')['ppo']['total_timesteps'])")"
FAILED=()
for SEED in 0 1 2; do          # seed-major order: every scenario gets seed 0 first
  for SCEN in S0 S1 S2; do
    NAME="${SCEN}_${DESC}_s${SEED}"
    if [ -f "$ROOT_RUNS/$NAME/final_model.zip" ]; then log "skip $NAME (final model exists)"; continue; fi
    log "train $NAME"
    T0=$(date +%s)
    if $PY -m dal.train --config $CFG --scenario $SCEN --seed $SEED --descriptor "$DESC" --out "$ROOT_RUNS" \
        > "$ROOT_RUNS/$NAME.train.log" 2>&1; then
      log "done $NAME in $(( $(date +%s) - T0 ))s"
    else
      log "FAILED $NAME (see $ROOT_RUNS/$NAME.train.log)"; FAILED+=("$NAME")
    fi
  done
done
log "training finished; failed: ${FAILED[*]:-none}"
log "evaluation on the TEST region (all methods, 1000 paired episodes)"
$PY -m dal.evaluate --config $CFG --descriptor "$DESC" --runs "$ROOT_RUNS" --split test \
  --out results/episodes_test.csv || log "test evaluation FAILED"
log "evaluation on the TRAIN region (H3)"
$PY -m dal.evaluate --config $CFG --descriptor "$DESC" --runs "$ROOT_RUNS" --split train \
  --episodes "$($PY -c "from dal.config import load_config;print(load_config('$CFG')['eval']['train_episodes'])")" \
  --methods ppo greedy lawnmower straight random --save-traj 0 --out results/episodes_train.csv || log "train evaluation FAILED"
$PY -m dal.report --config $CFG --results results --runs "$ROOT_RUNS" --out REPORT.md || log "report FAILED"
log "all done"
