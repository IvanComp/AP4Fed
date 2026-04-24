#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="python3"
VENV_DIR="${ROOT_DIR}/.venv-experiments"
VERIFY_MODE="local"
OLLAMA_URL=""
SKIP_VERIFY=0

usage() {
  cat <<'EOF'
Usage: ./scripts/bootstrap_experiment_env.sh [options]

Create a headless Python environment for AP4FED experiment runners on a fresh machine.

Options:
  --python PATH         Python interpreter to use (default: python3)
  --venv PATH           Target virtualenv path (default: .venv-experiments in repo root)
  --mode MODE           Verification mode: local, docker, or both (default: local)
  --ollama-url URL      Optional Ollama-compatible endpoint to verify
  --skip-verify         Create/install the environment without running preflight checks
  -h, --help            Show this help message
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --venv)
      VENV_DIR="$2"
      shift 2
      ;;
    --mode)
      VERIFY_MODE="$2"
      shift 2
      ;;
    --ollama-url)
      OLLAMA_URL="$2"
      shift 2
      ;;
    --skip-verify)
      SKIP_VERIFY=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "${VERIFY_MODE}" in
  local|docker|both)
    ;;
  *)
    echo "Invalid --mode '${VERIFY_MODE}'. Use local, docker, or both." >&2
    exit 2
    ;;
esac

"${PYTHON_BIN}" -m venv "${VENV_DIR}"

VENV_PYTHON="${VENV_DIR}/bin/python"
VENV_PIP="${VENV_DIR}/bin/pip"

"${VENV_PYTHON}" -m pip install --upgrade pip setuptools wheel
"${VENV_PIP}" install -r "${ROOT_DIR}/requirements-experiments.txt"

if [[ "${SKIP_VERIFY}" -eq 0 ]]; then
  VERIFY_CMD=("${VENV_PYTHON}" "${ROOT_DIR}/scripts/verify_experiment_env.py" "--mode" "${VERIFY_MODE}")
  if [[ -n "${OLLAMA_URL}" ]]; then
    VERIFY_CMD+=("--ollama-url" "${OLLAMA_URL}")
  fi
  "${VERIFY_CMD[@]}"
fi

cat <<EOF

Environment ready.

Activate it with:
  source "${VENV_DIR}/bin/activate"

Typical experiment commands:
  python "${ROOT_DIR}/build_paper_experiments.py" --mode local --rounds 100 --repeat 10
  python "${ROOT_DIR}/build_paper_experiments.py" --mode docker --rounds 100 --repeat 10
  python "${ROOT_DIR}/build_paper_experiments_500clients_k5.py" --mode local --rounds 100 --repeat 10
  python "${ROOT_DIR}/build_paper_experiments_500clients_k5.py" --mode docker --rounds 100 --repeat 10

The supercomputer only needs to generate CSV outputs.
Notebook analysis can stay on your local PC.
EOF
