#!/bin/zsh
set -u

ROOT="/Users/ivan/Desktop/AP4Fed"
LOG="$ROOT/overnight_ag_news_mixed_rounds.log"
PYTHON_BIN="/Library/Frameworks/Python.framework/Versions/3.12/Resources/Python.app/Contents/MacOS/Python"
DOCKER_BIN="/Applications/Docker.app/Contents/Resources/bin/docker"
RUN_PATTERN="build_paper_experiments.py --dataset AG_NEWS --model MLP --repeat 10"
CURRENT_RUN_DIR="$ROOT/paper_campaigns/ag_news__mlp/staging/paper__rq2__voting_based__r08"
CURRENT_COMPOSE="$ROOT/Docker/docker-compose.generated.paper__rq2__voting_based__r08.yml"

cd "$ROOT" || exit 1

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

echo "[$(timestamp)] mixed-round overnight orchestrator started" >> "$LOG"

wait_for_current_100_round_run() {
  if ! pgrep -f "$RUN_PATTERN --quiet" >/dev/null 2>&1; then
    echo "[$(timestamp)] no active AG_NEWS build process detected; skipping 100-round wait" >> "$LOG"
    return 0
  fi

  echo "[$(timestamp)] waiting for current 100-round run to archive" >> "$LOG"
  while pgrep -f "$RUN_PATTERN --quiet" >/dev/null 2>&1; do
    if [ -f "$CURRENT_RUN_DIR/performance/FLwithAP_performance_metrics.csv" ]; then
      echo "[$(timestamp)] current 100-round run archived" >> "$LOG"
      return 0
    fi
    sleep 60
  done

  if [ -f "$CURRENT_RUN_DIR/performance/FLwithAP_performance_metrics.csv" ]; then
    echo "[$(timestamp)] current 100-round run archived after process exit" >> "$LOG"
    return 0
  fi

  echo "[$(timestamp)] active process ended before the 100-round run archived" >> "$LOG"
  return 1
}

stop_current_campaign() {
  echo "[$(timestamp)] stopping current AG_NEWS campaign processes" >> "$LOG"
  pkill -f "$RUN_PATTERN --quiet" >/dev/null 2>&1 || true
  pkill -f "docker-compose.generated.paper__rq2__voting_based__r08.yml up --build" >/dev/null 2>&1 || true
  if [ -f "$CURRENT_COMPOSE" ]; then
    "$DOCKER_BIN" compose -f "$CURRENT_COMPOSE" down --remove-orphans >> "$LOG" 2>&1 || true
  fi
}

complete_missing_10_round_runs() {
  while true; do
    missing="$($PYTHON_BIN scripts/get_missing_runs.py --dataset AG_NEWS --model MLP --repeat 10 --rounds 10 2>>"$LOG")"
    rc=$?
    if [ $rc -ne 0 ]; then
      echo "[$(timestamp)] missing probe failed with rc=$rc; retrying in 120s" >> "$LOG"
      sleep 120
      continue
    fi

    if [ "$missing" = "0" ]; then
      echo "[$(timestamp)] all AG_NEWS 10-round runs are complete" >> "$LOG"
      return 0
    fi

    echo "[$(timestamp)] launching 10-round completion pass; missing=$missing" >> "$LOG"
    "$PYTHON_BIN" build_paper_experiments.py \
      --dataset AG_NEWS \
      --model MLP \
      --repeat 10 \
      --rounds 10 \
      --quiet \
      --continue-on-error >> "$LOG" 2>&1

    echo "[$(timestamp)] 10-round completion pass exited with rc=$?" >> "$LOG"
    sleep 30
  done
}

wait_for_current_100_round_run || true
stop_current_campaign
echo "[$(timestamp)] rehoming archived 100-round runs" >> "$LOG"
"$PYTHON_BIN" scripts/rehome_ag_news_100round_runs.py >> "$LOG" 2>&1
echo "[$(timestamp)] rehome script exited with rc=$?" >> "$LOG"
complete_missing_10_round_runs
