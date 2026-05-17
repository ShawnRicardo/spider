## Examples Layout

- `examples/default/`: upstream or baseline shell entrypoints for original workflows.
- `examples/asm/`: ASM robot workflows on original OakInk data.
- `examples/asm_ourdata/`: ASM robot workflows on local `ourdata` inputs.
- `examples/debug/`: intermediate debugging scripts, including short IK checks and hand/object alignment visualization.
  - `run_ik_ourdata_milk_asm_debug_alignment_video.sh`: MuJoCo/IK debug video for `ourdata/milk`, with left/right hand targets, object bbox, object axes, world axes, robot-base axes, and head-camera axes overlaid in the simulator.
  - `run_ik_ourdata_milk_asm_debug_alignment_viser.sh`: the same IK-stage MuJoCo debug scene, but streamed through the browser via Viser for headless servers.
  - `run_mjwp_ourdata_milk_asm_debug_video.sh`: MJWP-stage side-by-side video for `ourdata/milk`; left is reference, right is simulation.
  - `run_mjwp_ourdata_milk_asm_debug_viser.sh`: MJWP-stage MuJoCo scene streamed through Viser after regenerating the `ourdata/milk` ASM IK reference.
  - `visualize_ourdata_milk_keypoints_viser.py`: browser-based 3D viewer for raw/processed `preprocessed/milk` hand/object keypoints.

Core Python entrypoints such as `run_mjwp.py`, `run_hdmi.py`, `run_dexmachina.py`, and `run_maniptrans.py` remain at the top level of `examples/` so existing Hydra and documentation references stay stable.
