#!/usr/bin/env bash
set -euo pipefail

# Run every Experiments/*/config.json ten times using your existing compose.
# Adjust --compose path if your compose lives elsewhere.
python run_batch.py Experiments/*/config.json \
  --compose Docker/docker-compose.yml \
  --repeat 10 \
  --compose-project EXP_BATCH

# If you want reproducibility via seeds, uncomment and set the correct dot path:
# python run_batch.py Experiments/*/config.json \
#   --compose Docker/docker-compose.yml \
#   --repeat 10 \
#   --seed-field seed --seed-start 100 --seed-step 1 \
#   --compose-project EXP_BATCH
