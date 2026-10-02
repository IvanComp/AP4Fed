#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python3 -m venv "${repo_dir}/.venv-leonardo"
"${repo_dir}/.venv-leonardo/bin/python" -m pip install --upgrade pip
"${repo_dir}/.venv-leonardo/bin/python" -m pip install -r "${repo_dir}/leonardo/requirements-runner.txt"
