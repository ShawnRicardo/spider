#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

echo "Running two-object IK: task=pick_place data_id=0 left=apple:obj_0 right=banana:obj_1"

bash examples/asm_ourdata_2objs/generate_scene_ourdata_2objs_asm_URDFCollision.sh

# env -u LD_LIBRARY_PATH MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python examples/debug/render_ourdata_2objs_ref_keypoints_mujoco.py \
#   --dataset-dir example_datasets \
#   --dataset-name ourdata \
#   --robot-type asm \
#   --embodiment-type bimanual \
#   --task pick_place \
#   --data-id 0 \
#   --output-path example_datasets/processed/ourdata/asm/bimanual/pick_place/0/visualization_ref_keypoints_mujoco.mp4

env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES=6 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python spider/preprocess/ik_2objs.py \
  --dataset-dir example_datasets \
  --dataset-name ourdata \
  --robot-type asm \
  --embodiment-type bimanual \
  --task pick_place \
  --data-id 0 \
  --open-hand \
  --save-video \
  --visualize-hand-keypoints \
  --visualize-object-bbox \
  --visualize-object-axes \
  --visualize-robot-base-axes \
  --visualize-head-camera-axes \
  --no-show-viewer \
  --enable-collision
