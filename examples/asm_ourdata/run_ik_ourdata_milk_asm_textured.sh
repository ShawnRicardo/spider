#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"

DATA_ID="${1:-${DATA_ID:-0}}"
TRAJECTORY_INTERPOLATION_FACTOR="${TRAJECTORY_INTERPOLATION_FACTOR:-1}"
SCENE_OFFSET_X="${SCENE_OFFSET_X:-0.00}"
SCENE_OFFSET_Y="${SCENE_OFFSET_Y:-0.00}"
SCENE_OFFSET_Z="${SCENE_OFFSET_Z:-0.00}"
ASM_COLLISION_MESH_SCALE="${ASM_COLLISION_MESH_SCALE:-1.0}"
if ! [[ "${DATA_ID}" =~ ^[0-9]+$ ]]; then
  echo "DATA_ID must be a non-negative integer, got: ${DATA_ID}" >&2
  exit 2
fi
if ! [[ "${TRAJECTORY_INTERPOLATION_FACTOR}" =~ ^[1-9][0-9]*$ ]]; then
  echo "TRAJECTORY_INTERPOLATION_FACTOR must be a positive integer, got: ${TRAJECTORY_INTERPOLATION_FACTOR}" >&2
  exit 2
fi
if ! [[ "${ASM_COLLISION_MESH_SCALE}" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]]; then
  echo "ASM_COLLISION_MESH_SCALE must be a positive number, got: ${ASM_COLLISION_MESH_SCALE}" >&2
  exit 2
fi

export DATA_ID
export TRAJECTORY_INTERPOLATION_FACTOR
export SCENE_OFFSET_X
export SCENE_OFFSET_Y
export SCENE_OFFSET_Z
export ASM_COLLISION_MESH_SCALE
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

echo "Running milk IK with textured object visual and URDFCollision-style parameters: DATA_ID=${DATA_ID}, TRAJECTORY_INTERPOLATION_FACTOR=${TRAJECTORY_INTERPOLATION_FACTOR}, SCENE_OFFSET_XYZ=${SCENE_OFFSET_X} ${SCENE_OFFSET_Y} ${SCENE_OFFSET_Z}, ASM_COLLISION_MESH_SCALE=${ASM_COLLISION_MESH_SCALE}"

bash examples/asm_ourdata/generate_scene_ourdata_milk_asm_textured.sh

# ik
env -u LD_LIBRARY_PATH python spider/preprocess/ik.py \
  --dataset-dir example_datasets \
  --dataset-name ourdata \
  --robot-type asm \
  --embodiment-type bimanual \
  --task milk \
  --data-id "${DATA_ID}" \
  --open-hand \
  --save-video \
  --no-show-viewer \
  --enable-collision
  # --visualize-hand-keypoints \
  # --visualize-object-bbox \
  # --visualize-object-axes \
  # --visualize-world-axes \
  # --visualize-robot-base-axes \
  # --visualize-head-camera-axes \
