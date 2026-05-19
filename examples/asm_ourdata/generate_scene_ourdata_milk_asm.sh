#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

DATA_ID="${DATA_ID:-0}"
ASM_COLLISION_MESH_SCALE="${ASM_COLLISION_MESH_SCALE:-0.7}"
if ! [[ "${DATA_ID}" =~ ^[0-9]+$ ]]; then
  echo "DATA_ID must be a non-negative integer, got: ${DATA_ID}" >&2
  exit 2
fi
if ! [[ "${ASM_COLLISION_MESH_SCALE}" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]]; then
  echo "ASM_COLLISION_MESH_SCALE must be a positive number, got: ${ASM_COLLISION_MESH_SCALE}" >&2
  exit 2
fi
export DATA_ID

# Outputs:
#   Robot assets:
#     example_datasets/processed/ourdata/assets/robots/asm/
#       - bimanual.xml
#       - right.xml
#       - left.xml
#   Task scene:
#     example_datasets/processed/ourdata/asm/bimanual/milk/
#       - scene.xml
#       - scene_eq.xml
#       - scene_act.xml
#       - task_info.json

# osmesa
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

env -u LD_LIBRARY_PATH python spider/preprocess/prepare_asm_mjcf.py \
  --dataset-dir example_datasets \
  --dataset-name ourdata \
  --source-urdf spider/assets/robots/asm_description/urdf/asm.urdf \
  --robot-type asm \
  --arm-kp 800 \
  --hand-kp 140 \
  --arm-damping 3.0 \
  --hand-damping 1.2 \
  --arm-armature 0.05 \
  --hand-armature 0.02 \
  --arm-frictionloss 0.0 \
  --hand-frictionloss 0.02 \
  --arm-force-scale 8.0 \
  --hand-force-scale 8.0 \
  --collision-mesh-scale "${ASM_COLLISION_MESH_SCALE}" \
  --variants bimanual right left

bash examples/asm_ourdata/process_ourdata_milk.sh

env -u LD_LIBRARY_PATH python spider/preprocess/generate_xml.py \
  --dataset-dir example_datasets \
  --dataset-name ourdata \
  --robot-type asm \
  --embodiment-type bimanual \
  --task milk \
  --data-id "${DATA_ID}" \
  --object-density 15000 \
  --object-armature 0.02 \
  --object-frictionloss 0.03 \
  --friction-scale 1.0 \
  --object-bbox-collision \
  --object-bbox-collision-margin 0.001 \
  --support-table-from-bbox \
  --support-table-collision-mode object_only \
  --support-table-height-mode first_frame_min \
  --support-table-z-offset=0.0 \
  --no-show-viewer

env -u LD_LIBRARY_PATH python spider/preprocess/generate_xml.py \
  --dataset-dir example_datasets \
  --dataset-name ourdata \
  --robot-type asm \
  --embodiment-type bimanual \
  --task milk \
  --data-id "${DATA_ID}" \
  --object-density 15000 \
  --object-armature 0.02 \
  --object-frictionloss 0.03 \
  --friction-scale 1.0 \
  --object-bbox-collision \
  --object-bbox-collision-margin 0.001 \
  --support-table-from-bbox \
  --support-table-collision-mode object_only \
  --support-table-height-mode first_frame_min \
  --support-table-z-offset=0.0 \
  --act-scene \
  --no-show-viewer
