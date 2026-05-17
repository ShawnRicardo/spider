#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

python examples/debug/run_milk_ik_to_act.py "$@"
