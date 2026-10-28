#!/usr/bin/env bash
# Phase 4-5 smoke test: short PPO training (S0, 20k steps, 2 envs) -> evaluation -> report.
set -euo pipefail
source "$(dirname "$0")/env.sh"
START=$(date +%s)
DESC=$($PY scripts/best_descriptor.py || echo cnn)
echo "[smoke] descriptor: $DESC"
rm -rf runs/smoke results/smoke
mkdir -p results/smoke/figs
$PY -m dal.train --config configs/smoke.yaml --descriptor "$DESC" --out runs/smoke | tail -n 5
T_TRAIN=$(( $(date +%s) - START ))
ls runs/smoke/S0_${DESC}_s0/checkpoints/ | head -n 3
test -f runs/smoke/S0_${DESC}_s0/progress.csv
$PY -m dal.evaluate --config configs/smoke.yaml --descriptor "$DESC" --runs runs/smoke --scenarios S0 \
  --split test --episodes 50 --out results/smoke/episodes_test.csv
$PY -m dal.evaluate --config configs/smoke.yaml --descriptor "$DESC" --runs runs/smoke --scenarios S0 \
  --methods ppo greedy lawnmower straight random --split train --episodes 30 --save-traj 0 \
  --out results/smoke/episodes_train.csv
for f in data_stats.json retrieval.csv gate.json gate.csv; do [ -f "results/$f" ] && cp "results/$f" results/smoke/; done
[ -f results/figs/data_overview.png ] && cp results/figs/data_overview.png results/smoke/figs/
$PY -m dal.report --config configs/smoke.yaml --results results/smoke --runs runs/smoke --out REPORT.md \
  --note " **SMOKE TEST (20k timesteps, 50 episodes, S0 only): the numbers are NOT meaningful.**"
END=$(date +%s)
printf "train_seconds=%d\ntotal_seconds=%d\n" "$T_TRAIN" "$((END - START))" | tee results/smoke/smoke_time.txt
$PY -c "import json;m=json.load(open('runs/smoke/S0_${DESC}_s0/train_meta.json'));print('[smoke] PPO fps', round(m['fps']))"
