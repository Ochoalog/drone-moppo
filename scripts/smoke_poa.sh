#!/usr/bin/env bash
# POA smoke test: 20 paired episodes per method, P0 and P1, both beta variants -> REPORT_POA.md.
set -euo pipefail
source "$(dirname "$0")/env.sh"
START=$(date +%s)
CFG=configs_poa/poa.yaml
S=results_poa/smoke
rm -rf "$S"; mkdir -p "$S/figs"
$PY -m dal.poa --config $CFG --step eval --scenarios P0 P1 --episodes 20 --out $S/episodes.csv
for f in data_gate.json sanity_gate.json windows.json retrieval.csv beta.csv; do cp results_poa/$f $S/; done
cp results_poa/figs/data_overview.png results_poa/figs/maskcheck_*.png $S/figs/
$PY -m dal.poa_report --config $CFG --results $S --out REPORT_POA.md \
  --note " **SMOKE TEST (20 episodes, P0/P1): the numbers are NOT meaningful.**"
echo "total_seconds=$(( $(date +%s) - START ))" | tee $S/smoke_time.txt
