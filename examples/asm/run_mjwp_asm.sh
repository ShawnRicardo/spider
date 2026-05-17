#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

export CUDA_VISIBLE_DEVICES=7

export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa

bash examples/asm/run_ik_asm.sh

env -u LD_LIBRARY_PATH python examples/run_mjwp.py \
  +override=oakink_asm \
  dataset_dir=example_datasets \
  dataset_name=oakink \
  task=pick_spoon_bowl \
  data_id=0 \
  robot_type=asm \
  embodiment_type=bimanual \
  viewer=none \
  show_viewer=false \
  save_video=true \
  save_info=true \
  contact_guidance=true \
  num_samples=32 \
  max_num_iterations=4 \
  max_sim_steps=-1
