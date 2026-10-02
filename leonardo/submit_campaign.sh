#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <CINECA-project-account>" >&2
    exit 2
fi

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "${repo_dir}/leonardo/logs"
cd "${repo_dir}"
sbatch --account="$1" leonardo/ap4fed_campaign.sbatch
