#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export WARP_CACHE_PATH="${WARP_CACHE_PATH:-/tmp/spider_warp_cache}"
mkdir -p "${WARP_CACHE_PATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"

HOST="${1:-0.0.0.0}"
PORT="${2:-8080}"

bash examples/asm_ourdata/run_ik_ourdata_milk_asm.sh

env -u LD_LIBRARY_PATH python examples/run_mjwp.py \
  +override=ourdata_asm \
  viewer=viser \
  show_viewer=true \
  viser_host="${HOST}" \
  viser_port="${PORT}" \
  wait_on_finish=true \
  save_video=false \
  save_info=true
