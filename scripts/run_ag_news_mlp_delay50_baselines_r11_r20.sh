#!/bin/zsh
set -euo pipefail

ROOT="/Users/ivan/Desktop/AP4Fed"
DOCKER_BIN_DIR="/Applications/Docker.app/Contents/Resources/bin"
OUTPUT_DIR="$ROOT/paper_campaigns/ag_news__mlp_delay50_fixed_baselines/experiments"
STAGING_DIR="$ROOT/paper_campaigns/ag_news__mlp_delay50_fixed_baselines/staging"

cd "$ROOT"

export PATH="$DOCKER_BIN_DIR:$PATH"
export PYTHONUNBUFFERED=1

exec python3 build_paper_experiments.py \
  --dataset AG_NEWS \
  --model MLP \
  --rounds 10 \
  --repeat 20 \
  --approaches never,random,expert_driven \
  --fixed-delay-seconds 50 \
  --output-dir "$OUTPUT_DIR" \
  --staging-dir "$STAGING_DIR" \
  --quiet \
  --continue-on-error \
  "$@"
