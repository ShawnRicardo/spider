#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

echo "Processing two-object ourdata scene: workspace=preprocessed/pick_place task=pick_place data_id=0 left=apple:obj_0 right=banana:obj_1 scene_offset_xyz=-0.10 0.00 0.00"

env -u LD_LIBRARY_PATH python spider/process_datasets/ourdata_textured_2objs.py \
  --workspace preprocessed/pick_place \
  --dataset-dir example_datasets \
  --dataset-name ourdata \
  --task pick_place \
  --data-id 0 \
  --right-object-name banana \
  --left-object-name apple \
  --right-object-id obj_1 \
  --left-object-id obj_0 \
  --embodiment-type bimanual \
  --ref-dt 0.03333333333333333 \
  --trajectory-interpolation-factor 1 \
  --scene-offset-xyz -0.15 0.00 0.10 \
  --orientation-policy upright_preserve_heading \
  --world-to-sim-alignment d435_optical \
  --alignment-robot-xml example_datasets/processed/ourdata/assets/robots/asm/bimanual.xml

env -u LD_LIBRARY_PATH python spider/preprocess/decompose_fast.py \
  --dataset-dir example_datasets \
  --dataset-name ourdata \
  --robot-type asm \
  --embodiment-type bimanual \
  --task pick_place \
  --data-id 0
