#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

TRAJECTORY_INTERPOLATION_FACTOR=1
DATA_ID=0
DATASET_NAME=ourdata
WORKSPACE=preprocessed/robot
TASK=robot
RIGHT_OBJECT_ID=obj_1
LEFT_OBJECT_ID=obj_0
RIGHT_OBJECT_NAME=right_obj
LEFT_OBJECT_NAME=left_obj
REF_DT=0.03333333333333333
SCENE_OFFSET_X=0.00
SCENE_OFFSET_Y=0.00
SCENE_OFFSET_Z=0.00

if ! [[ "${TRAJECTORY_INTERPOLATION_FACTOR}" =~ ^[1-9][0-9]*$ ]]; then
  echo "TRAJECTORY_INTERPOLATION_FACTOR must be a positive integer, got: ${TRAJECTORY_INTERPOLATION_FACTOR}" >&2
  exit 2
fi
if ! [[ "${DATA_ID}" =~ ^[0-9]+$ ]]; then
  echo "DATA_ID must be a non-negative integer, got: ${DATA_ID}" >&2
  exit 2
fi
if [[ "${RIGHT_OBJECT_ID}" == "${LEFT_OBJECT_ID}" ]]; then
  echo "RIGHT_OBJECT_ID and LEFT_OBJECT_ID must be different, got: ${RIGHT_OBJECT_ID}" >&2
  exit 2
fi
if [[ "${RIGHT_OBJECT_NAME}" == "${LEFT_OBJECT_NAME}" ]]; then
  echo "RIGHT_OBJECT_NAME and LEFT_OBJECT_NAME must be different, got: ${RIGHT_OBJECT_NAME}" >&2
  exit 2
fi

echo "Processing two-object ourdata scene: WORKSPACE=${WORKSPACE}, TASK=${TASK}, DATA_ID=${DATA_ID}, right=${RIGHT_OBJECT_NAME}:${RIGHT_OBJECT_ID}, left=${LEFT_OBJECT_NAME}:${LEFT_OBJECT_ID}"

env -u LD_LIBRARY_PATH python spider/process_datasets/ourdata_textured_2objs.py \
  --workspace "${WORKSPACE}" \
  --dataset-dir example_datasets \
  --dataset-name "${DATASET_NAME}" \
  --task "${TASK}" \
  --data-id "${DATA_ID}" \
  --right-object-name "${RIGHT_OBJECT_NAME}" \
  --left-object-name "${LEFT_OBJECT_NAME}" \
  --right-object-id "${RIGHT_OBJECT_ID}" \
  --left-object-id "${LEFT_OBJECT_ID}" \
  --embodiment-type bimanual \
  --ref-dt "${REF_DT}" \
  --trajectory-interpolation-factor "${TRAJECTORY_INTERPOLATION_FACTOR}" \
  --scene-offset-xyz "${SCENE_OFFSET_X}" "${SCENE_OFFSET_Y}" "${SCENE_OFFSET_Z}" \
  --orientation-policy upright_preserve_heading \
  --world-to-sim-alignment d435_optical \
  --alignment-robot-xml "example_datasets/processed/${DATASET_NAME}/assets/robots/asm/bimanual.xml"

env -u LD_LIBRARY_PATH python spider/preprocess/decompose_fast.py \
  --dataset-dir example_datasets \
  --dataset-name "${DATASET_NAME}" \
  --robot-type asm \
  --embodiment-type bimanual \
  --task "${TASK}" \
  --data-id "${DATA_ID}"
