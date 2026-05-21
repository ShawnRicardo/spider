#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

# Outputs:
#   Robot assets:
#     example_datasets/processed/oakink/assets/robots/asm/
#       - bimanual.xml
#       - right.xml
#       - left.xml
#   Task scene:
#     example_datasets/processed/oakink/asm/bimanual/pick_spoon_bowl/
#       - scene.xml
#       - scene_eq.xml
#       - task_info.json

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

env -u LD_LIBRARY_PATH python spider/preprocess/prepare_asm_mjcf_PickSpoonBowl.py \
  --dataset-dir example_datasets \
  --dataset-name oakink \
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
  --root-yaw-deg 0 \
  --variants bimanual right left

env -u LD_LIBRARY_PATH python spider/preprocess/generate_xml_PickSpoonBowl.py \
  --dataset-dir example_datasets \
  --dataset-name oakink \
  --robot-type asm \
  --embodiment-type bimanual \
  --task pick_spoon_bowl \
  --data-id 0 \
  --no-show-viewer

# env -u LD_LIBRARY_PATH python spider/preprocess/generate_xml_PickSpoonBowl.py \
#   --dataset-dir example_datasets \
#   --dataset-name oakink \
#   --robot-type asm \
#   --embodiment-type bimanual \
#   --task pick_spoon_bowl \
#   --data-id 0 \
#   --act-scene \
#   --no-show-viewer
