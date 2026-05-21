#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

DATA_ID=0
ASM_COLLISION_MESH_SCALE=1.0
if ! [[ "${DATA_ID}" =~ ^[0-9]+$ ]]; then
  echo "DATA_ID must be a non-negative integer, got: ${DATA_ID}" >&2
  exit 2
fi
if ! [[ "${ASM_COLLISION_MESH_SCALE}" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]]; then
  echo "ASM_COLLISION_MESH_SCALE must be a positive number, got: ${ASM_COLLISION_MESH_SCALE}" >&2
  exit 2
fi
export DATA_ID

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

# Use the active URDF collision meshes directly instead of generated
# capsule/box primitive proxies. ASM_COLLISION_MESH_SCALE=1.0 means original
# URDF collision mesh size; lower values shrink them around their mesh centers.
env -u LD_LIBRARY_PATH python spider/preprocess/prepare_asm_mjcf.py \
  --dataset-dir example_datasets \
  --dataset-name ourdata \
  --source-urdf spider/assets/robots/asm_description/urdf/asm_7.urdf \
  --robot-type asm \
  --arm-kp 300 \
  --hand-kp 180 \
  --arm-damping 2.0 \
  --hand-damping 0.5 \
  --arm-armature 0.05 \
  --hand-armature 0.02 \
  --arm-frictionloss 0.0 \
  --hand-frictionloss 0.01 \
  --hand-force-scale 2.0 \
  --collision-geometry-mode urdf_mesh \
  --collision-mesh-scale "${ASM_COLLISION_MESH_SCALE}" \
  --variants bimanual right left

bash examples/asm_ourdata/process_ourdata_milk_PickSpoonBowlParams.sh

env -u LD_LIBRARY_PATH python spider/preprocess/generate_xml.py \
  --dataset-dir example_datasets \
  --dataset-name ourdata \
  --robot-type asm \
  --embodiment-type bimanual \
  --task milk \
  --data-id "${DATA_ID}" \
  --object-bbox-collision \
  --object-bbox-collision-margin 0.001 \
  --support-table-from-bbox \
  --support-table-collision-mode object_and_manipulator \
  --support-table-height-mode trajectory_min \
  --support-table-z-offset=-0.05 \
  --no-robot-object-collision \
  --no-object-object-collision \
  --no-object-floor-collision \
  --no-show-viewer

# env -u LD_LIBRARY_PATH python spider/preprocess/generate_xml.py \
#   --dataset-dir example_datasets \
#   --dataset-name ourdata \
#   --robot-type asm \
#   --embodiment-type bimanual \
#   --task milk \
#   --data-id "${DATA_ID}" \
#   --object-bbox-collision \
#   --object-bbox-collision-margin 0.001 \
#   --support-table-from-bbox \
#   --support-table-collision-mode object_and_hand \
#   --support-table-height-mode first_frame_min \
#   --support-table-z-offset=-0.07 \
#   --act-scene \
#   --no-show-viewer
