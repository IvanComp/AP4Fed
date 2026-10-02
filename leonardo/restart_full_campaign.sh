#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Usage: $0 <CINECA-project-account> [max-parallel-waves]" >&2
    exit 2
fi

if [[ -z "${WORK:-}" ]]; then
    echo "WORK is not defined." >&2
    exit 2
fi

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
campaign_root="${AP4FED_OUTPUT_DIR:-${WORK}/AP4Fed-pattern-stress-parallel}"
timestamp="$(date +%Y%m%d-%H%M%S)"

active_jobs="$(squeue --noheader --user="${USER}" --name=ap4fed-wave --format='%A_%a' 2>/dev/null || true)"
if [[ -n "${active_jobs}" ]]; then
    echo "Active AP4Fed wave jobs detected:" >&2
    echo "${active_jobs}" >&2
    echo "Cancel or wait for them before restarting the full campaign." >&2
    exit 2
fi

if [[ -e "${campaign_root}" ]]; then
    backup_root="${campaign_root}.backup-${timestamp}"
    mv "${campaign_root}" "${backup_root}"
    echo "Previous campaign archived at: ${backup_root}"
fi

exec "${repo_dir}/leonardo/submit_wave_array.sh" "$@"
