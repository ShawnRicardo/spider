#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/../.."

export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
if [ "$MUJOCO_GL" = "osmesa" ]; then
  export PYOPENGL_PLATFORM=osmesa
fi

env -u LD_LIBRARY_PATH python -u examples/run_mjwp.py \
  --config-name mjwp \
  viewer=none \
  show_viewer=false \
  save_video=true \
  save_info=true \
  "$@"
