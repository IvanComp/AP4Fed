#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Usage: $0 <CINECA-project-account> [max-parallel-waves]" >&2
    exit 2
fi

account="$1"
max_parallel="${2:-10}"
if (( max_parallel < 1 || max_parallel > 10 )); then
    echo "max-parallel-waves must be between 1 and 10." >&2
    exit 2
fi

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "${repo_dir}/leonardo/logs"
cd "${repo_dir}"
sbatch \
    --account="${account}" \
    --array="1-10%${max_parallel}" \
    leonardo/ap4fed_wave_array_combinedAP.sbatch
