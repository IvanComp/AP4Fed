#!/usr/bin/env bash
set -euo pipefail
python run_batch.py Experiments/*/config.json \
  --compose Docker/docker-compose.dynamic.yml \
  --repeat 2 \
  --tag dyn
