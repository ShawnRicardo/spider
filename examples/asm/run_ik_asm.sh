#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa

bash examples/asm/generate_scene_asm.sh

env -u LD_LIBRARY_PATH python spider/preprocess/ik.py \
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

env -u LD_LIBRARY_PATH python spider/preprocess/ik.py \
  --dataset-dir example_datasets \
  --dataset-name oakink \
  --robot-type asm \
  --embodiment-type bimanual \
  --task pick_spoon_bowl \
  --data-id 0 \
  --open-hand \
  --enable-collision \
  --act-scene \
  --no-save-video \
  --no-show-viewer
