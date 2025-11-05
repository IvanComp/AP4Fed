#!/usr/bin/env bash
set -euo pipefail
# Always use dynamic compose template and let run_batch.py expand N clients from config.json
python run_batch.py Experiments/*/config.json \
  --compose Docker/docker-compose.dynamic.yml \
  --repeat 10 \
  --tag dyn
