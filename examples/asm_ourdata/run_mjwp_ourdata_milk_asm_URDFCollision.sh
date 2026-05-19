#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

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

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"

bash examples/asm_ourdata/run_ik_ourdata_milk_asm_URDFCollision.sh "${DATA_ID}"

if [[ -z "${MJWP_DEVICE:-}" ]]; then
  if env -u LD_LIBRARY_PATH python -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)"; then
    MJWP_DEVICE=cuda:0
  else
    MJWP_DEVICE=cpu
  fi
fi
echo "Using MJWP device: ${MJWP_DEVICE}"

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
env -u LD_LIBRARY_PATH python examples/run_mjwp.py \
  +override=ourdata_asm_PickSpoonBowlParams \
  data_id="${DATA_ID}" \
  device="${MJWP_DEVICE}"
