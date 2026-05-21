#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-${MUJOCO_GL}}"

DATA_ID="${DATA_ID:-0}"
VIDEO_DIR="examples/debug/videos"
VIDEO_SUFFIX="hand_keypoints_object_bbox_object_axes_world_axes_robot_base_axes_head_camera_axes"
PROCESSED_DIR="example_datasets/processed/ourdata/asm/bimanual/milk/${DATA_ID}"
FRONT_VIDEO="${PROCESSED_DIR}/visualization_ik_${VIDEO_SUFFIX}.mp4"
DEBUG_VIDEO="${VIDEO_DIR}/visualization_ik_ourdata_milk_asm_debug_alignment_NewCollision_front.mp4"

mkdir -p "${VIDEO_DIR}"

bash examples/asm_ourdata/generate_scene_ourdata_milk_asm_NewCollision.sh

env -u LD_LIBRARY_PATH python spider/preprocess/ik.py \
    --dataset-dir example_datasets \
    --dataset-name ourdata \
    --robot-type asm \
    --embodiment-type bimanual \
    --task milk \
    --data-id "${DATA_ID}" \
    --enable-collision \
    --open-hand \
    --visualize-hand-keypoints \
    --visualize-object-bbox \
    --visualize-object-axes \
    --visualize-world-axes \
    --visualize-robot-base-axes \
    --visualize-head-camera-axes \
    --save-video \
    --no-show-viewer

if [[ ! -f "${FRONT_VIDEO}" ]]; then
  echo "Expected front IK video was not generated: ${FRONT_VIDEO}" >&2
  exit 1
fi

cp -f "${FRONT_VIDEO}" "${DEBUG_VIDEO}"
echo "Saved debug front video to ${DEBUG_VIDEO}"
