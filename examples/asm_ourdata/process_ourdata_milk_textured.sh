#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

TRAJECTORY_INTERPOLATION_FACTOR="${1:-${TRAJECTORY_INTERPOLATION_FACTOR:-1}}"
DATA_ID="${DATA_ID:-0}"
SCENE_OFFSET_X="${SCENE_OFFSET_X:-0.00}"
SCENE_OFFSET_Y="${SCENE_OFFSET_Y:-0.00}"
SCENE_OFFSET_Z="${SCENE_OFFSET_Z:-0.00}"
if ! [[ "${TRAJECTORY_INTERPOLATION_FACTOR}" =~ ^[1-9][0-9]*$ ]]; then
  echo "trajectory interpolation factor must be a positive integer, got: ${TRAJECTORY_INTERPOLATION_FACTOR}" >&2
  exit 2
fi
if ! [[ "${DATA_ID}" =~ ^[0-9]+$ ]]; then
  echo "DATA_ID must be a non-negative integer, got: ${DATA_ID}" >&2
  exit 2
fi

# Outputs:
#   Processed keypoints:
#     example_datasets/processed/ourdata/mano/bimanual/milk/${DATA_ID}/
#       - trajectory_keypoints.npz
#       - conversion_debug.npz
#   Task metadata:
#     example_datasets/processed/ourdata/mano/bimanual/milk/task_info.json
#   Object assets:
#     example_datasets/processed/ourdata/assets/objects/milk/
#       - visual.obj
#       - visual_mesh_textured.obj
#       - visual_mesh_texture.png
#       - convex/*.obj
#
# Prerequisite:
#   example_datasets/processed/ourdata/assets/robots/asm/bimanual.xml
#   must already exist so `ourdata.py` can align the local camera world to the
#   ASM simulation world before IK via the ASM neck D435 optical frame.

env -u LD_LIBRARY_PATH python spider/process_datasets/ourdata_textured.py \
  --workspace preprocessed/milk \
  --dataset-dir example_datasets \
  --dataset-name ourdata \
  --task milk \
  --data-id "${DATA_ID}" \
  --object-name milk \
  --object-id obj_0 \
  --embodiment-type bimanual \
  --ref-dt 0.03333333333333333 \
  --trajectory-interpolation-factor "${TRAJECTORY_INTERPOLATION_FACTOR}" \
  --scene-offset-xyz "${SCENE_OFFSET_X}" "${SCENE_OFFSET_Y}" "${SCENE_OFFSET_Z}" \
  --orientation-policy upright_preserve_heading \
  --world-to-sim-alignment d435_optical \
  --alignment-robot-xml example_datasets/processed/ourdata/assets/robots/asm/bimanual.xml

env -u LD_LIBRARY_PATH python spider/preprocess/decompose_fast.py \
  --dataset-dir example_datasets \
  --dataset-name ourdata \
  --robot-type asm \
  --embodiment-type bimanual \
  --task milk \
  --data-id "${DATA_ID}"
