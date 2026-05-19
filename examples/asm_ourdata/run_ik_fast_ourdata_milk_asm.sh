#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

DATA_ID="${1:-${DATA_ID:-0}}"
TRAJECTORY_INTERPOLATION_FACTOR="${TRAJECTORY_INTERPOLATION_FACTOR:-1}"
SCENE_OFFSET_X="${SCENE_OFFSET_X:-0.00}"
SCENE_OFFSET_Y="${SCENE_OFFSET_Y:-0.00}"
SCENE_OFFSET_Z="${SCENE_OFFSET_Z:-0.00}"
ASM_COLLISION_MESH_SCALE="${ASM_COLLISION_MESH_SCALE:-0.7}"
REF_DT="${REF_DT:-0.03333333333333333}"
SIM_DT="${SIM_DT:-0.005}"

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

echo "Running milk IK-fast pipeline with DATA_ID=${DATA_ID}, TRAJECTORY_INTERPOLATION_FACTOR=${TRAJECTORY_INTERPOLATION_FACTOR}, REF_DT=${REF_DT}, SIM_DT=${SIM_DT}, SCENE_OFFSET_XYZ=${SCENE_OFFSET_X} ${SCENE_OFFSET_Y} ${SCENE_OFFSET_Z}, ASM_COLLISION_MESH_SCALE=${ASM_COLLISION_MESH_SCALE}"

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

env -u LD_LIBRARY_PATH python spider/process_datasets/ourdata.py \
  --workspace preprocessed/milk \
  --dataset-dir example_datasets \
  --dataset-name ourdata \
  --task milk \
  --data-id "${DATA_ID}" \
  --object-name milk \
  --object-id obj_0 \
  --embodiment-type bimanual \
  --ref-dt "${REF_DT}" \
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

env -u LD_LIBRARY_PATH python spider/preprocess/ik_fast.py \
  --dataset-dir example_datasets \
  --dataset-name ourdata \
  --robot-type asm \
  --embodiment-type bimanual \
  --task milk \
  --data-id "${DATA_ID}" \
  --sim-dt "${SIM_DT}" \
  --ref-dt "${REF_DT}"
