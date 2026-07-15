#!/usr/bin/env bash
set -euo pipefail

DEVICE="${1:-cuda:0}"
python run_cmss.py --config configs/paper.yaml --stage all --device "${DEVICE}"
