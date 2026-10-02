#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
image_tag="ap4fed-leonardo:latest"
sif_path="${1:-${repo_dir}/leonardo/ap4fed.sif}"
archive_path="$(mktemp "${TMPDIR:-/tmp}/ap4fed-image.XXXXXX.tar")"

cleanup() {
    rm -f "${archive_path}"
}
trap cleanup EXIT

if command -v singularity >/dev/null 2>&1; then
    container_engine="singularity"
elif command -v apptainer >/dev/null 2>&1; then
    container_engine="apptainer"
else
    echo "Install Singularity or Apptainer before building the SIF image." >&2
    exit 1
fi

docker build -t "${image_tag}" -f "${repo_dir}/Docker/Dockerfile.server" "${repo_dir}/Docker"
docker save -o "${archive_path}" "${image_tag}"
"${container_engine}" build --force "${sif_path}" "docker-archive://${archive_path}"
echo "Created ${sif_path}"
