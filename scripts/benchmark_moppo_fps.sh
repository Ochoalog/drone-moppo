#!/usr/bin/env bash
# P2: mede o throughput do treino do MOPPO antes (tag v0-pre-refactor) e depois (código atual).
# Roda 12 rollouts (24.576 passos) de cada; o fps vem do progress.csv, sem o rollout do gate (25%).
# Rodar com a CPU livre. Uso: bash scripts/benchmark_moppo_fps.sh [pasta_de_saida]
set -euo pipefail
unset PYTHONPATH
cd "$(dirname "$0")/.."
REPO="$PWD"
OUT="${1:-$REPO/results_mo_v2/benchmark_fps}"
WORKTREE="$OUT/worktree_v0"
STEPS=24576
mkdir -p "$OUT"

if [ ! -d "$WORKTREE" ]; then
  git worktree add -q --detach "$WORKTREE" v0-pre-refactor
fi

echo "[fps] carga antes: $(cut -d' ' -f1-3 /proc/loadavg)"
echo "[fps] v0 (antes) ..."
PYTHONPATH="$WORKTREE/src" DAL_DATA_DIR="$REPO/data" DAL_RUNS_DIR="$OUT/runs_v0" \
  .venv/bin/python -m dal.mo_ppo --config "$WORKTREE/configs_mo/default.yaml" \
  --timesteps "$STEPS" --out "$OUT/run_v0" > "$OUT/run_v0.log" 2>&1
echo "[fps] atual, config v1 (NEXT_STEP) ..."
.venv/bin/python -m dal.mo_ppo --config configs_mo/default.yaml \
  --timesteps "$STEPS" --out "$OUT/run_now_v1cfg" > "$OUT/run_now_v1cfg.log" 2>&1
echo "[fps] atual, config v2 (SAME_STEP) ..."
.venv/bin/python -m dal.mo_ppo --config configs_mo_v2/default.yaml \
  --timesteps "$STEPS" --out "$OUT/run_now_v2cfg" > "$OUT/run_now_v2cfg.log" 2>&1
echo "[fps] carga depois: $(cut -d' ' -f1-3 /proc/loadavg)"

.venv/bin/python - "$OUT" <<'EOF'
import sys
from pathlib import Path
import pandas as pd

out = Path(sys.argv[1])
rows = []
for run in ("run_v0", "run_now_v1cfg", "run_now_v2cfg"):
    progress = pd.read_csv(out / run / "progress.csv")
    elapsed = progress["time_elapsed"].diff()
    steps = progress["timesteps"].diff()
    rate = (steps / elapsed).iloc[1:]
    # Descarta o rollout com o gate (o mais lento) e o primeiro (aquecimento).
    rate = rate.drop(rate.idxmin())
    rows.append({"run": run, "fps_median_per_rollout": round(float(rate.median()), 1),
                 "fps_cumulative_reported": float(progress["fps"].iloc[-1]), "rollouts": len(progress)})
table = pd.DataFrame(rows)
table.to_csv(out / "fps.csv", index=False)
print(table.to_string(index=False))
EOF
git worktree remove --force "$WORKTREE"
