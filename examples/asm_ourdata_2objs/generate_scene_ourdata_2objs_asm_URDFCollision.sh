#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

echo "Generating two-object ASM scene: task=pick_place data_id=0 URDF=asm_7.urdf collision=urdf_mesh scale=1.0"

env -u LD_LIBRARY_PATH MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python spider/preprocess/prepare_asm_mjcf_2objs.py \
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
  --collision-mesh-scale 1.0 \
  --variants bimanual right left

bash examples/asm_ourdata_2objs/process_ourdata_2objs.sh

env -u LD_LIBRARY_PATH MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python spider/preprocess/generate_xml_2objs.py \
  --dataset-dir example_datasets \
  --dataset-name ourdata \
  --robot-type asm \
  --embodiment-type bimanual \
  --task pick_place \
  --data-id 0 \
  --object-bbox-collision \
  --object-bbox-collision-margin 0.001 \
  --support-table-from-bbox \
  --support-table-collision-mode object_and_manipulator \
  --support-table-height-mode first_frame_min \
  --support-table-z-offset=0 \
  --no-show-viewer
