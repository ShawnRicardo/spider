#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa

bash examples/asm_ourdata/generate_scene_ourdata_milk_asm.sh

env -u LD_LIBRARY_PATH python spider/preprocess/ik.py \
  --dataset-dir example_datasets \
  --dataset-name ourdata \
  --robot-type asm \
  --embodiment-type bimanual \
  --task milk \
  --data-id 0 \
  --enable-collision \
  --open-hand \
  --save-video \
  --visualize-hand-keypoints \
  --visualize-object-bbox \
  --visualize-object-axes \
  --visualize-world-axes \
  --visualize-robot-base-axes \
  --visualize-head-camera-axes \
  --no-show-viewer
