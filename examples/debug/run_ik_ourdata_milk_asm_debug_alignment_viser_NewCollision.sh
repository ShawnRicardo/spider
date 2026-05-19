#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

export CUDA_VISIBLE_DEVICES=7

# osmesa
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

HOST="${1:-127.0.0.1}"
PORT="${2:-8080}"

bash examples/asm_ourdata/generate_scene_ourdata_milk_asm_NewCollision.sh

env -u LD_LIBRARY_PATH python spider/preprocess/ik.py \
    --dataset-dir example_datasets \
    --dataset-name ourdata \
    --robot-type asm \
    --embodiment-type bimanual \
    --task milk \
    --data-id 0 \
    --enable-collision \
    --open-hand \
    --visualize-hand-keypoints \
    --visualize-object-bbox \
    --visualize-object-axes \
    --visualize-world-axes \
    --visualize-robot-base-axes \
    --visualize-head-camera-axes \
    --viewer viser \
    --show-viewer \
    --viser-host "${HOST}" \
    --viser-port "${PORT}" \
    --wait-on-finish
