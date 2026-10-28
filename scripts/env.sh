# Sourced by all scripts: isolated interpreter, no ROS PYTHONPATH leakage, one BLAS thread per process.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
unset PYTHONPATH
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
PY="$ROOT/.venv/bin/python"
mkdir -p "$ROOT/runs" "$ROOT/results"
