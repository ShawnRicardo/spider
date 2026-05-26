#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

export CUDA_VISIBLE_DEVICES=6

echo "Running two-object MJWP: task=pick_place data_id=0"

bash examples/asm_ourdata_2objs/run_ik_ourdata_2objs_asm_URDFCollision.sh

mkdir -p /tmp/spider_warp_cache

env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES=6 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl WARP_CACHE_PATH=/tmp/spider_warp_cache python examples/run_mjwp_2objs.py \
  +override=ourdata_asm_2objs \
  dataset_name=ourdata \
  task=pick_place \
  data_id=0 \
  device=cuda:0
