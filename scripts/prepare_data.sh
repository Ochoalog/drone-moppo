#!/usr/bin/env bash
# Phase 1-2: download AOI windows, masks, alignment, figures, descriptors, caches, retrieval + beta.
set -euo pipefail
source "$(dirname "$0")/env.sh"
$PY -m dal.data --step download
$PY -m dal.data --step masks
$PY -m dal.align
$PY -m dal.data --step stats
$PY -m dal.viz
if [ ! -f data/features/hand_post_obs.npy ] || [ ! -f data/features/cnn_post_obs.npy ]; then
  OMP_NUM_THREADS=1 $PY -m dal.features --descriptors hand > runs/features_hand.log 2>&1 &
  DAL_TORCH_THREADS=7 $PY -m dal.features --descriptors cnn > runs/features_cnn.log 2>&1
  wait
fi
$PY -m dal.calibrate --descriptors hand cnn
