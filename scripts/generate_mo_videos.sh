#!/usr/bin/env bash
set -euo pipefail

echo "=========================================================="
echo "Generating Active Localization Videos and Animations"
echo "Date: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================================="

.venv/bin/python -u -m src.dal.mo_video

echo "=========================================================="
echo "Video Generation Completed Successfully!"
echo "Videos stored in: results_mo/videos/"
echo "=========================================================="
