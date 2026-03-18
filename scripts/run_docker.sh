#!/bin/bash
# Launch the sol-execbench Docker container with the right mounts.
#
# Usage:
#   ./scripts/run_docker.sh [--build] [docker-run-args...] [-- command...]
#
# Examples:
#   ./scripts/run_docker.sh                             # interactive shell
#   ./scripts/run_docker.sh --build                     # build image, then shell
#   ./scripts/run_docker.sh -- sol-execbench tests/sol_execbench/samples/rmsnorm --solution tests/sol_execbench/samples/rmsnorm/solution_cuda.json
#   ./scripts/run_docker.sh --gpus '"device=1"' -- bash
#
# Environment variables:
#   IMAGE_NAME          Docker image name    (default: sol-execbench)
#   IMAGE_TAG           Docker image tag     (default: latest)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

IMAGE_NAME="${IMAGE_NAME:-sol-execbench}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
IMAGE="${IMAGE_NAME}:${IMAGE_TAG}"

# Container-side paths (must match Dockerfile / entrypoint expectations)
CONTAINER_PROJECT="/sol-execbench"

# Parse --build flag, then split remaining args on "--"
BUILD=false
DOCKER_ARGS=()
CMD=()
seen_separator=false
for arg in "$@"; do
    if [ "$arg" = "--build" ] && ! $seen_separator; then
        BUILD=true
        continue
    fi
    if [ "$arg" = "--" ]; then
        seen_separator=true
        continue
    fi
    if $seen_separator; then
        CMD+=("$arg")
    else
        DOCKER_ARGS+=("$arg")
    fi
done

# Build the image if requested
if $BUILD; then
    echo "+ docker build -t ${IMAGE} -f ${REPO_ROOT}/docker/Dockerfile ${REPO_ROOT}"
    docker build \
        -t "${IMAGE}" \
        --build-arg HOST_UID="$(id -u)" \
        --build-arg HOST_GID="$(id -g)" \
        --build-arg HOST_USER="$(whoami)" \
        -f "${REPO_ROOT}/docker/Dockerfile" \
        "${REPO_ROOT}"
fi

LOCAL_FLASHINFER_TRACE_DIR="${REPO_ROOT}/data/flashinfer-trace"
if [ ! -d "${LOCAL_FLASHINFER_TRACE_DIR}" ]; then
    echo "WARNING: ${LOCAL_FLASHINFER_TRACE_DIR} does not exist"
    echo "       Run ./scripts/download_data.sh to download the flashinfer-trace dataset to run those problems."
fi
FLASHINFER_TRACE_DIR="/sol-execbench/data/flashinfer-trace"

# Default to interactive shell if no command given
if [ ${#CMD[@]} -eq 0 ]; then
    CMD=("bash")
fi

DOCKER_CMD=(
    docker run --rm -it
    --gpus all
    --ipc=host
    --privileged
    --ulimit memlock=-1
    --ulimit stack=67108864
    -v "${REPO_ROOT}:${CONTAINER_PROJECT}"
    -e "FLASHINFER_TRACE_DIR=${FLASHINFER_TRACE_DIR}"
    -e "SOL_EXECBENCH_GPU_CLK_MHZ=${SOL_EXECBENCH_GPU_CLK_MHZ:-}"
    -e "SOL_EXECBENCH_DRAM_CLK_MHZ=${SOL_EXECBENCH_DRAM_CLK_MHZ:-}"
    "${DOCKER_ARGS[@]}"
    "${IMAGE}"
    "${CMD[@]}"
)

echo "+ ${DOCKER_CMD[*]}"
exec "${DOCKER_CMD[@]}"
