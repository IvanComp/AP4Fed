#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
runner="${repo_dir}/.venv-leonardo/bin/python"
sif_path="${AP4FED_SIF_PATH:-${repo_dir}/leonardo/ap4fed.sif}"

failed=0

check_command() {
    local command_name="$1"
    if command -v "${command_name}" >/dev/null 2>&1; then
        echo "OK: ${command_name} is available"
    else
        echo "MISSING: ${command_name}" >&2
        failed=1
    fi
}

echo "AP4Fed Leonardo pre-flight check"
echo "Repository: ${repo_dir}"

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "WARNING: this terminal is already inside a Slurm allocation (${SLURM_JOB_ID})."
    echo "Use the RCM SSH session for preparation and submission; batch jobs will use Slurm."
else
    echo "OK: this is not an interactive Slurm allocation"
fi

check_command sbatch
check_command squeue

if command -v singularity >/dev/null 2>&1; then
    container_command="singularity"
elif command -v apptainer >/dev/null 2>&1; then
    container_command="apptainer"
else
    echo "MISSING: Singularity or Apptainer" >&2
    failed=1
    container_command=""
fi

if [[ -x "${runner}" ]]; then
    echo "OK: campaign Python environment exists"
else
    echo "MISSING: ${runner}" >&2
    echo "Run: ./leonardo/prepare_runner.sh" >&2
    failed=1
fi

if [[ -f "${sif_path}" ]]; then
    echo "OK: SIF image exists at ${sif_path}"
else
    echo "MISSING: ${sif_path}" >&2
    echo "The SIF is intentionally not stored in Git; transfer it separately to this path." >&2
    failed=1
fi

if [[ -z "${WORK:-}" ]]; then
    echo "MISSING: WORK environment variable" >&2
    failed=1
else
    echo "OK: results will use ${WORK}/AP4Fed-pattern-stress-parallel"
fi

if (( failed != 0 )); then
    echo "PRE-FLIGHT FAILED: fix the MISSING items before submitting experiments." >&2
    exit 1
fi

"${container_command}" inspect "${sif_path}" >/dev/null
echo "OK: SIF image can be opened"

"${runner}" "${repo_dir}/run_adept_campaign.py" \
    --container-runtime singularity \
    --singularity-image "${sif_path}" \
    --host-cpus 112 \
    --server-cpus 2 \
    --low-spec-cpus 7 \
    --high-spec-cpus 12 \
    --dry-run

echo "PRE-FLIGHT PASSED: the 600-run campaign is ready for Slurm submission."
