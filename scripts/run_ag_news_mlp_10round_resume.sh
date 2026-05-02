#!/bin/zsh
set -euo pipefail

ROOT="/Users/ivan/Desktop/AP4Fed"
DOCKER_BIN_DIR="/Applications/Docker.app/Contents/Resources/bin"

cd "$ROOT"

export PATH="$DOCKER_BIN_DIR:$PATH"
export PYTHONUNBUFFERED=1

exec python3 build_paper_experiments.py \
  --dataset AG_NEWS \
  --model MLP \
  --repeat 10 \
  --rounds 10 \
  --quiet \
  --continue-on-error \
  "$@"
