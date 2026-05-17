#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export WARP_CACHE_PATH="${WARP_CACHE_PATH:-/tmp/spider_warp_cache}"
mkdir -p "${WARP_CACHE_PATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"

bash examples/asm_ourdata/run_ik_ourdata_milk_asm.sh

env -u LD_LIBRARY_PATH python examples/run_mjwp.py \
  +override=ourdata_asm \
  viewer=none \
  show_viewer=false \
  save_video=true \
  save_info=true
