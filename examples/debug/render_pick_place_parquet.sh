#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="7"
# 可以使用 egl 了
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

python examples/debug/render_pick_place_parquet_asm_videos.py \
  --render-kinematic-front true \
  --render-kinematic-head true \
  --render-dynamic-front true \
  --render-dynamic-head true \
  --collision-mode urdf_mesh_scaled \
  --urdf-collision-mesh-scale 0.7 \
  --exclude-initial-penetrations false \
  "$@"
