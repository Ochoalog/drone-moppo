#!/usr/bin/env bash
set -euo pipefail

echo "=========================================================="
echo "Starting MO Full Overnight Training and Evaluation"
echo "Date: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Host: $(hostname)"
echo "Branch: $(git rev-parse --abbrev-ref HEAD)"
echo "Commit: $(git rev-parse HEAD)"
echo "=========================================================="

# Check disk space
echo "Disk space before overnight:"
df -h .

# Run full MO pipeline (resumable, handles gates, evaluation, and reporting)
.venv/bin/python -m src.dal.mo_pipeline

# Check manifests at the end
echo "Verifying checksums at the end of execution..."
sha256sum -c ../checkpoints/taquari_2026-10-28.sha256 --quiet
sha256sum -c ../checkpoints/poa_2026-10-28.sha256 --quiet

echo "Disk space after overnight:"
df -h .

echo "=========================================================="
echo "MO Full Overnight Execution Completed Successfully"
echo "Date: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================================="
