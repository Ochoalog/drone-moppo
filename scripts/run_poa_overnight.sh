#!/usr/bin/env bash
# Full Porto Alegre battery (no training): paired zero-shot + recalibrated evaluation of the
# Taquari models and baselines -> results_poa/ -> REPORT_POA.md. Resumable: finished steps are skipped.
# Launch: nohup systemd-inhibit --what=sleep:idle bash scripts/run_poa_overnight.sh > runs_poa/poa_$(date +%F).log 2>&1 &
set -uo pipefail
source "$(dirname "$0")/env.sh"
CFG=configs_poa/poa.yaml
R=results_poa
mkdir -p runs_poa $R
log() { echo "[$(date '+%F %T')] $*"; }
step() { local name=$1; shift; log "start $name"; "$@" && log "done $name" || log "FAILED $name (exit $?) - continuing"; }
[ -f data_poa/masks.npz ] || step data $PY -m dal.poa_data --config $CFG --step all
[ -f data_poa/cache/hand/obs.npy ] || { step features $PY -m dal.poa --config $CFG --step features; step cache $PY -m dal.poa --config $CFG --step cache; }
[ -f $R/windows.json ] || step units $PY -m dal.poa --config $CFG --step units
[ -f $R/sanity_gate.json ] || step gate $PY -m dal.poa --config $CFG --step gate
[ -f data_poa/beta_recal.json ] || step calib $PY -m dal.poa --config $CFG --step calib
[ -f $R/beta.csv ] || step beta $PY -m dal.poa --config $CFG --step beta
if [ ! -f $R/episodes.csv ]; then
  step eval $PY -m dal.poa --config $CFG --step eval --out $R/episodes.tmp.csv && mv $R/episodes.tmp.csv $R/episodes.csv
fi
if sha256sum -c ../checkpoints/taquari_2026-10-28.sha256 --quiet; then CK=passed; else CK=FAILED; fi
log "Taquari checksum after the run: $CK"
grep -q "after the full run" $R/validation.md 2>/dev/null || \
  printf '\n- Taquari manifest check **after the full run** (`sha256sum -c ../checkpoints/taquari_2026-10-28.sha256 --quiet`): %s (%s).\n' "$CK" "$(date '+%F %T')" >> $R/validation.md
step report $PY -m dal.poa_report --config $CFG
log "all done"
