#!/usr/bin/env bash
set -euo pipefail

echo "=========================================================="
echo "Starting MO Smoke Pipeline (< 10 min)"
echo "Date: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================================="

# Check disk space
echo "Disk space before smoke:"
df -h .

# Run 50k step smoke pipeline
.venv/bin/python -m src.dal.mo_pipeline --smoke

# Check manifests
echo "Checking Taquari manifest..."
sha256sum -c ../checkpoints/taquari_2026-10-28.sha256 --quiet
echo "Checking Porto Alegre manifest..."
sha256sum -c ../checkpoints/poa_2026-10-28.sha256 --quiet

echo "=========================================================="
echo "MO Smoke Pipeline Finished Successfully"
echo "Date: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================================="
