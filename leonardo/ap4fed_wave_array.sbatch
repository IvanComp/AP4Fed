#!/usr/bin/env bash
#SBATCH --job-name=ap4fed-wave
#SBATCH --partition=dcgp_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=112
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --signal=B:TERM@300
#SBATCH --output=leonardo/logs/wave-%A_%a.out
#SBATCH --error=leonardo/logs/wave-%A_%a.err

set -euo pipefail

source_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
runner="${source_repo}/.venv-leonardo/bin/python"
sif_path="${AP4FED_SIF_PATH:-${source_repo}/leonardo/ap4fed.sif}"
campaign_root="${AP4FED_OUTPUT_DIR:-${WORK}/AP4Fed-pattern-stress-parallel}"
wave="${SLURM_ARRAY_TASK_ID}"
task_repo="${TMPDIR}/AP4Fed-wave-${wave}"

if [[ ! -x "${runner}" || ! -f "${sif_path}" ]]; then
    echo "Prepare the runner and SIF image before submitting the array." >&2
    exit 2
fi

mkdir -p "${task_repo}"
cp "${source_repo}/run_adept_campaign.py" "${task_repo}/"
cp -a "${source_repo}/Docker" "${task_repo}/Docker"
cp -a "${source_repo}/.git" "${task_repo}/.git"

cd "${task_repo}"
"${runner}" run_adept_campaign.py \
    --container-runtime singularity \
    --singularity-image "${sif_path}" \
    --campaign-machine-id "leonardo-wave-${wave}" \
    --host-cpus 112 \
    --server-cpus 2 \
    --low-spec-cpus 7 \
    --high-spec-cpus 12 \
    --only-wave "${wave}" \
    --output-dir "${campaign_root}/wave_${wave}" &
campaign_pid=$!

stop_campaign() {
    kill -TERM "${campaign_pid}" 2>/dev/null || true
    wait "${campaign_pid}" || true
    exit 143
}
trap stop_campaign TERM INT
wait "${campaign_pid}"
