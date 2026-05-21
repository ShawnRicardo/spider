#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

export CUDA_VISIBLE_DEVICES=2

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

bash examples/asm/generate_scene_asm.sh

env -u LD_LIBRARY_PATH python spider/preprocess/ik_PickSpoonBowl.py \
  --dataset-dir example_datasets \
  --dataset-name oakink \
  --robot-type asm \
  --embodiment-type bimanual \
  --task pick_spoon_bowl \
  --data-id 0 \
  --open-hand \
  --enable-collision \
  --save-video \
  --no-show-viewer

# env -u LD_LIBRARY_PATH python spider/preprocess/ik_PickSpoonBowl.py \
#   --dataset-dir example_datasets \
#   --dataset-name oakink \
#   --robot-type asm \
#   --embodiment-type bimanual \
#   --task pick_spoon_bowl \
#   --data-id 0 \
#   --open-hand \
#   --enable-collision \
#   --act-scene \
#   --save-video \
#   --no-show-viewer
