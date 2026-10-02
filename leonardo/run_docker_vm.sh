#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
available_cpus="$(getconf _NPROCESSORS_ONLN)"
if (( available_cpus < 32 )); then
    echo "The AP4Fed VM campaign requires at least 32 online CPU cores; found ${available_cpus}." >&2
    exit 2
fi

docker compose version >/dev/null

if [[ ! -x "${repo_dir}/.venv-leonardo/bin/python" ]]; then
    "${repo_dir}/leonardo/prepare_runner.sh"
fi

cd "${repo_dir}"
exec "${repo_dir}/.venv-leonardo/bin/python" run_adept_campaign.py \
    --container-runtime docker \
    --host-cpus 32 \
    --server-cpus 2 \
    --low-spec-cpus 2 \
    --high-spec-cpus 3 \
    --skip-dataset-prefetch \
    "$@"
