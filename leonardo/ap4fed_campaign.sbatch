#!/usr/bin/env bash
#SBATCH --job-name=ap4fed-adept
#SBATCH --partition=dcgp_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=112
#SBATCH --hint=nomultithread
#SBATCH --exclusive
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --signal=B:TERM@300
#SBATCH --output=leonardo/logs/slurm-%j.out
#SBATCH --error=leonardo/logs/slurm-%j.err

set -euo pipefail

module load python/3.11.7

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
runner="${repo_dir}/.venv-leonardo/bin/python"
sif_path="${AP4FED_SIF_PATH:-${repo_dir}/leonardo/ap4fed.sif}"
output_dir="${AP4FED_OUTPUT_DIR:-${WORK}/AP4Fed-pattern-stress-results}"

if [[ ! -x "${runner}" ]]; then
    echo "Missing ${runner}; run leonardo/prepare_runner.sh first." >&2
    exit 2
fi

if [[ ! -f "${sif_path}" ]]; then
    echo "Missing Singularity image: ${sif_path}" >&2
    exit 2
fi

cd "${repo_dir}"
"${runner}" run_adept_campaign.py \
    --container-runtime singularity \
    --singularity-image "${sif_path}" \
    --campaign-machine-id leonardo \
    --host-cpus 112 \
    --server-cpus 2 \
    --low-spec-cpus 7 \
    --high-spec-cpus 12 \
    --output-dir "${output_dir}" &
campaign_pid=$!

stop_campaign() {
    kill -TERM "${campaign_pid}" 2>/dev/null || true
    wait "${campaign_pid}" || true
    exit 143
}
trap stop_campaign TERM INT
wait "${campaign_pid}"
