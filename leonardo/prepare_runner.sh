#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

python_version="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
python_major="${python_version%%.*}"
python_minor="${python_version##*.}"
if (( python_major < 3 || (python_major == 3 && python_minor < 9) )); then
    echo "Python 3.9 or newer is required; found ${python_version}." >&2
    echo "On Leonardo run: module load python/3.11.7" >&2
    exit 1
fi

python3 -m venv "${repo_dir}/.venv-leonardo"
"${repo_dir}/.venv-leonardo/bin/python" -m pip install --upgrade pip
"${repo_dir}/.venv-leonardo/bin/python" -m pip install -r "${repo_dir}/leonardo/requirements-runner.txt"
