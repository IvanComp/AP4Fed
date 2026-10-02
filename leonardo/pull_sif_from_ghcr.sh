#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
sif_path="${AP4FED_SIF_PATH:-${repo_dir}/leonardo/ap4fed.sif}"
image_uri="docker://ghcr.io/ivancomp/ap4fed-leonardo:latest"

if ! command -v singularity >/dev/null 2>&1; then
    echo "Singularity is not available on this machine." >&2
    exit 1
fi

mkdir -p "$(dirname "${sif_path}")"
echo "Downloading ${image_uri}"
singularity pull --force "${sif_path}" "${image_uri}"
singularity inspect "${sif_path}" >/dev/null
echo "Created ${sif_path}"
