#!/usr/bin/env bash
set -euo pipefail

python tests/create_toy_data.py
rm -rf tests/smoke_results

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python run_cmss.py --config configs/smoke.yaml --stage all --device cpu

python - <<'PY'
from pathlib import Path

required = [
    Path("tests/smoke_results/feedback_bundle.zip"),
    Path("tests/smoke_results/metrics/condition_metrics.csv"),
    Path("tests/smoke_results/summary/ablation_summary.csv"),
    Path("tests/smoke_results/summary/clean_noharm_summary.csv"),
]

missing = [str(path) for path in required if not path.exists() or path.stat().st_size == 0]
if missing:
    raise SystemExit("Missing smoke-test outputs: " + ", ".join(missing))

print("CMSS smoke test passed.")
PY
