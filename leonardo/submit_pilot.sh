#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <CINECA-project-account>" >&2
    exit 2
fi

if [[ -z "${WORK:-}" ]]; then
    echo "WORK is not defined." >&2
    exit 2
fi

account="$1"
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
pilot_root="${WORK}/AP4Fed-pattern-stress-pilot"

mkdir -p "${repo_dir}/leonardo/logs"
cd "${repo_dir}"
sbatch \
    --account="${account}" \
    --array=1-1 \
    --export="ALL,AP4FED_ROUNDS=2,AP4FED_OUTPUT_DIR=${pilot_root}" \
    leonardo/ap4fed_wave_array.sbatch
