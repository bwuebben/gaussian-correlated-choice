#!/usr/bin/env bash
set -euo pipefail
TASK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$TASK_DIR"
PYTHON_BIN="${PY:-$TASK_DIR/.venv/bin/python}"
MODE="${1:---saved}"
if [[ "$#" -gt 1 || ( "$MODE" != --saved && "$MODE" != --full ) ]]; then
  echo 'Usage: ./reproduce.sh [--saved|--full]' >&2
  exit 2
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo 'Create .venv and install requirements.txt, or set PY to an executable Python interpreter.' >&2
  exit 1
fi
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export MPLCONFIGDIR="$TASK_DIR/.cache/matplotlib"
mkdir -p output "$MPLCONFIGDIR"
"$PYTHON_BIN" src/validate_population.py
"$PYTHON_BIN" src/validate_estimation.py
if [[ "$MODE" == --full ]]; then
  "$PYTHON_BIN" src/run_sparse.py
  "$PYTHON_BIN" src/run_finite.py
  "$PYTHON_BIN" src/run_misspecification.py
  "$PYTHON_BIN" src/run_geometry.py
  "$PYTHON_BIN" src/run_scaling.py
  "$PYTHON_BIN" src/run_supplement.py
fi
"$PYTHON_BIN" src/build_results.py
"$PYTHON_BIN" companion/src/examples.py
"$PYTHON_BIN" companion/src/geometry.py
"$PYTHON_BIN" commercial/src/economics.py
./build.sh
