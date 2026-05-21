#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="7"
# 可以使用 egl 了
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

# 选择新版碰撞 URDF 时，在命令后追加：--asm-variant asm_2 / asm_3 / asm_4
python examples/debug/render_pick_place_parquet_asm_videos.py \
  --render-kinematic-front true \
  --render-kinematic-head true \
  --render-dynamic-front true \
  --render-dynamic-head true \
  --collision-mode urdf_mesh_scaled \
  --urdf-collision-mesh-scale 1.0 \
  --exclude-initial-penetrations false \
  --asm-variant asm_5 \
  "$@"
