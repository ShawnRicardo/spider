#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

DATA_ID="${DATA_ID:-0}"
DATASET_NAME="${DATASET_NAME:-ourdata}"
TASK="${TASK:-robot}"
TRAJECTORY_INTERPOLATION_FACTOR="${TRAJECTORY_INTERPOLATION_FACTOR:-1}"
ASM_SOURCE_URDF="${ASM_SOURCE_URDF:-spider/assets/robots/asm_description/urdf/asm_7.urdf}"
ASM_COLLISION_MESH_SCALE="${ASM_COLLISION_MESH_SCALE:-1.0}"
SUPPORT_TABLE_COLLISION_MODE="${SUPPORT_TABLE_COLLISION_MODE:-object_and_manipulator}"
SUPPORT_TABLE_HEIGHT_MODE="${SUPPORT_TABLE_HEIGHT_MODE:-first_frame_min}"
SUPPORT_TABLE_Z_OFFSET="${SUPPORT_TABLE_Z_OFFSET:-0}"
OBJECT_BBOX_COLLISION_MARGIN="${OBJECT_BBOX_COLLISION_MARGIN:-0.001}"
ROBOT_OBJECT_COLLISION="${ROBOT_OBJECT_COLLISION:-true}"

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
export DATASET_NAME
export TASK
export TRAJECTORY_INTERPOLATION_FACTOR
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

env -u LD_LIBRARY_PATH python spider/preprocess/prepare_asm_mjcf.py \
  --dataset-dir example_datasets \
  --dataset-name "${DATASET_NAME}" \
  --source-urdf "${ASM_SOURCE_URDF}" \
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

bash examples/asm_ourdata_2objs/process_ourdata_2objs.sh "${TRAJECTORY_INTERPOLATION_FACTOR}"

generate_args=(
  spider/preprocess/generate_xml.py
  --dataset-dir example_datasets
  --dataset-name "${DATASET_NAME}"
  --robot-type asm
  --embodiment-type bimanual
  --task "${TASK}"
  --data-id "${DATA_ID}"
  --object-bbox-collision
  --object-bbox-collision-margin "${OBJECT_BBOX_COLLISION_MARGIN}"
  --support-table-from-bbox
  --support-table-collision-mode "${SUPPORT_TABLE_COLLISION_MODE}"
  --support-table-height-mode "${SUPPORT_TABLE_HEIGHT_MODE}"
  "--support-table-z-offset=${SUPPORT_TABLE_Z_OFFSET}"
  --no-show-viewer
)

if [[ "${ROBOT_OBJECT_COLLISION}" == "false" ]]; then
  generate_args+=(--no-robot-object-collision)
fi

env -u LD_LIBRARY_PATH python "${generate_args[@]}"
