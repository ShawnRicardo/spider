#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6}"

DATA_ID="${1:-${DATA_ID:-0}}"
if ! [[ "${DATA_ID}" =~ ^[0-9]+$ ]]; then
  echo "DATA_ID must be a non-negative integer, got: ${DATA_ID}" >&2
  exit 2
fi
export DATA_ID

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export WARP_CACHE_PATH="${WARP_CACHE_PATH:-/tmp/spider_warp_cache}"
mkdir -p "${WARP_CACHE_PATH}"

bash examples/asm_ourdata_watermelon/run_ik_ourdata_watermelon_asm_URDFCollision.sh "${DATA_ID}"

MJWP_DEVICE="${MJWP_DEVICE:-cuda:0}"
echo "Using MJWP device: ${MJWP_DEVICE}"

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
env -u LD_LIBRARY_PATH python examples/run_mjwp_watermelon.py \
  +override=ourdata_asm_PickSpoonBowlParams \
  task=watermelon_server \
  data_id="${DATA_ID}" \
  device="${MJWP_DEVICE}"
