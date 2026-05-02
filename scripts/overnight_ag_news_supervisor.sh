#!/bin/zsh
set -u

ROOT="/Users/ivan/Desktop/AP4Fed"
LOG="$ROOT/overnight_ag_news_supervisor.log"
PYTHON_BIN="/Library/Frameworks/Python.framework/Versions/3.12/Resources/Python.app/Contents/MacOS/Python"
DOCKER_BIN="/Applications/Docker.app/Contents/Resources/bin/docker"

cd "$ROOT" || exit 1

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

echo "[$(timestamp)] supervisor started" >> "$LOG"

while true; do
  missing="$($PYTHON_BIN scripts/get_missing_ag_news_runs.py 2>>"$LOG")"
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "[$(timestamp)] missing-run probe failed with code $rc; retrying in 120s" >> "$LOG"
    sleep 120
    continue
  fi

  if [ "$missing" = "0" ]; then
    echo "[$(timestamp)] all AG_NEWS + MLP runs are complete" >> "$LOG"
    exit 0
  fi

  if pgrep -f "build_paper_experiments.py --dataset AG_NEWS --model MLP --repeat 10" >/dev/null 2>&1; then
    echo "[$(timestamp)] existing experiment process detected; missing=$missing; sleeping 300s" >> "$LOG"
    sleep 300
    continue
  fi

  if ! pgrep -f "$DOCKER_BIN" >/dev/null 2>&1 && ! "$DOCKER_BIN" info >/dev/null 2>>"$LOG"; then
    echo "[$(timestamp)] docker unavailable; missing=$missing; sleeping 300s" >> "$LOG"
    sleep 300
    continue
  fi

  echo "[$(timestamp)] relaunching campaign; missing=$missing" >> "$LOG"
  "$PYTHON_BIN" build_paper_experiments.py \
    --dataset AG_NEWS \
    --model MLP \
    --repeat 10 \
    --quiet \
    --continue-on-error >> "$LOG" 2>&1

  echo "[$(timestamp)] campaign process exited with code $?" >> "$LOG"
  sleep 30
done
