# ASM Revised Integration Log

This file records every implementation change made for integrating the full ASM dual-arm dual-hand URDF into the SPIDER MJWP pipeline.

## Change 001 - Initialize revision log

- Goal: Create the required root-level implementation log before modifying code.
- Files changed:
  - `asm_revised.md`: lines 1-8 created.
- Details:
  - Added the log title and the first change entry.
  - Future entries will record goals, file paths, affected lines, validation commands, and validation results.
- Validation:
  - Command: `test -f asm_revised.md`
  - Result: pending; validated after file creation.

## Change 002 - Add ASM MJCF preparation script

- Goal: Convert the ROS-style full ASM URDF into SPIDER-compatible MuJoCo robot assets.
- Files changed:
  - `spider/preprocess/prepare_asm_mjcf.py`: lines 1-408 created.
- Details:
  - Added URDF mesh path normalization for `package://asm_description/...` assets.
  - Added flat mesh copying to `example_datasets/processed/<dataset>/assets/robots/asm/meshes/`.
  - Added MuJoCo URDF compilation via `mujoco.MjModel.from_xml_path` and `mujoco.mj_saveLastXML`.
  - Added SPIDER-specific palm/fingertip sites, track/trace sites, simplified `collision_hand_` geoms, groundplane materials, and position actuators.
  - Added `right`, `left`, and `bimanual` XML variant generation and validation.
- Validation:
  - Command: `python -m py_compile spider/preprocess/prepare_asm_mjcf.py`
  - Result: passed.

## Change 003 - Fix ASM site insertion after MuJoCo fixed-link collapsing

- Goal: Attach SPIDER palm and fingertip sites to the actual MJCF bodies produced by MuJoCo's URDF compiler.
- Files changed:
  - `spider/preprocess/prepare_asm_mjcf.py`: lines 205-294 updated.
- Details:
  - Added mesh-based body lookup helpers to find the body containing `right_palm_link`, `left_palm_link`, and each `*_tip_link` mesh.
  - Updated site/collision insertion to use the mesh geom local `pos` and `quat` because MuJoCo collapses fixed URDF links into geoms instead of preserving them as bodies.
- Validation:
  - Command: `python -m py_compile spider/preprocess/prepare_asm_mjcf.py`
  - Result: passed.

## Change 004 - Fix right/left XML side pruning roots

- Goal: Generate true single-side ASM XML variants even after MuJoCo collapses fixed `Base_R/Base_L` links.
- Files changed:
  - `spider/preprocess/prepare_asm_mjcf.py`: lines 347-358 updated.
- Details:
  - Updated single-side pruning to remove both `Base_*` and actual compiled roots `Link1_*`.
  - Kept actuator cleanup based on remaining joints.
- Validation:
  - Command: `python -m py_compile spider/preprocess/prepare_asm_mjcf.py`
  - Result: passed.

## Change 005 - Generate ASM processed robot assets

- Goal: Materialize SPIDER robot asset XML files for the full ASM model and single-side variants.
- Files changed:
  - `example_datasets/processed/oakink/assets/robots/asm/bimanual.xml`: generated from full ASM dual-arm dual-hand model.
  - `example_datasets/processed/oakink/assets/robots/asm/right.xml`: generated right arm + right hand variant.
  - `example_datasets/processed/oakink/assets/robots/asm/left.xml`: generated left arm + left hand variant.
  - `example_datasets/processed/oakink/assets/robots/asm/meshes/`: copied supported ASM STL meshes.
- Details:
  - `bimanual.xml` validated as `nq=54 nv=54 nu=54 nsite=32 ngeom=158 collision_hand=12`.
  - `right.xml` validated as `nq=27 nv=27 nu=27 nsite=16 ngeom=84 collision_hand=6`.
  - `left.xml` validated as `nq=27 nv=27 nu=27 nsite=16 ngeom=84 collision_hand=6`.
- Validation:
  - Command: `python spider/preprocess/prepare_asm_mjcf.py --dataset-dir example_datasets --dataset-name oakink --source-urdf spider/assets/robots/asm_description/urdf/asm.urdf --robot-type asm --variants bimanual right left`
  - Result: passed.

## Change 006 - Add ASM optimizer dimension handling

- Goal: Treat ASM as a fixed-base 54-DoF robot instead of a floating-base hand during MJWP sampling and reward weighting.
- Files changed:
  - `spider/config.py`: lines 345-353 updated.
  - `spider/simulators/mjwp.py`: lines 174-196 updated.
- Details:
  - Added `robot_type == "asm"` noise scaling branch so all ASM robot controls use `joint_noise_scale`.
  - Added `robot_type == "asm"` qpos reward weighting branch so robot DoFs use `joint_rew_scale` and object DoFs keep position/orientation weights.
- Validation:
  - Command: ASM MJWP smoke test in Change 011.
  - Result: passed; `trajectory_mjwp.npz` was generated with finite `qpos=(1, 20, 68)`, `qvel=(1, 20, 66)`, and `ctrl=(1, 20, 54)`.

## Change 007 - Add ASM MJWP override and runner

- Goal: Provide a conservative Hydra override and a repeatable shell entrypoint for ASM bimanual pick_spoon_bowl.
- Files changed:
  - `examples/config/override/oakink_asm.yaml`: lines 1-38 created.
  - `examples/run_mjwp_asm.sh`: lines 1-54 created.
- Details:
  - Added default `robot_type=asm`, `embodiment_type=bimanual`, `task=pick_spoon_bowl`, and headless viewer settings.
  - Added conservative sampling/noise/reward settings for smoke tests.
  - Added a shell pipeline for prepare ASM assets, generate scene, IK, and MJWP.
- Validation:
  - Command: `chmod +x examples/run_mjwp_asm.sh`
  - Result: passed.

## Change 008 - Make generated scene ground materials explicit

- Goal: Prevent scene generation from depending on robot XML carrying unused `right_groundplane` or `left_groundplane` materials.
- Files changed:
  - `spider/preprocess/generate_xml.py`: lines 231-243 updated.
- Details:
  - Replaced the single `groundplane` material insertion with explicit creation of `groundplane`, `right_groundplane`, and `left_groundplane` when missing.
  - This fixes ASM scene generation because `generate_xml.py` later assigns the floor material to `right_groundplane` for bimanual/right scenes.
- Validation:
  - Command: scene generation in Change 009.
  - Result: passed; ASM `scene.xml` loaded with `nq=68 nv=66 nu=54 npair=425`.

## Change 009 - Generate and validate ASM pick_spoon_bowl scene

- Goal: Build the task-level MuJoCo scene using the ASM robot asset and OakInk object assets.
- Files changed:
  - `example_datasets/processed/oakink/asm/bimanual/pick_spoon_bowl/scene.xml`: generated.
  - `example_datasets/processed/oakink/asm/bimanual/pick_spoon_bowl/scene_eq.xml`: generated.
  - `example_datasets/processed/oakink/asm/bimanual/pick_spoon_bowl/task_info.json`: updated by `generate_xml.py`.
- Details:
  - Used existing `trajectory_keypoints.npz` and object convex mesh directories from OakInk task info.
  - Generated 326 hand/object/floor contact pairs during scene assembly.
  - Validated final scene as `nq=68 nv=66 nu=54 npair=425 nsite=66 ngeom=181`.
- Validation:
  - Command: `python spider/preprocess/generate_xml.py --task=pick_spoon_bowl --dataset-name=oakink --data-id=0 --embodiment-type=bimanual --robot-type=asm --no-show-viewer`
  - Result: passed.
  - Command: `mujoco.MjModel.from_xml_path("example_datasets/processed/oakink/asm/bimanual/pick_spoon_bowl/scene.xml")`
  - Result: passed.

## Change 010 - Stabilize ASM IK initialization and rollout validation

- Goal: Make short ASM IK produce a numerically valid `trajectory_kinematic.npz` and avoid xhand-specific rollout instability.
- Files changed:
  - `spider/preprocess/ik.py`: lines 163-275 added helper functions for object qpos assignment, ASM neutral initial guesses, joint/control clipping, and reference-qpos control filling.
  - `spider/preprocess/ik.py`: lines 562-570 updated initial object qpos setup to use the shared object-qpos helper.
  - `spider/preprocess/ik.py`: lines 627-685 updated initial-guess search to use ASM neutral joint midpoints plus small clipped perturbations, preserve valid object free-joint poses, clip controls, and include mocap/site residuals in initial-guess scoring.
  - `spider/preprocess/ik.py`: lines 894-944 updated IK rollout validation to use `sim_dt` substeps, disable the old xhand-specific random rollout noise for ASM, clip controls, and log both `trajectory_kinematic.npz` and `trajectory_ikrollout.npz` saves.
- Details:
  - Before this change, short ASM IK wrote a finite `qpos`, but `qpos_rollout` exploded to roughly `[-8.6e5, 8.3e5]` and MuJoCo emitted QACC instability warnings.
  - After this change, the same 30-frame IK smoke test completed with no QACC warnings and `best_qpos_diff_sum=17.483332453267344`.
- Validation:
  - Command: `python -m py_compile spider/preprocess/ik.py`
  - Result: passed.
  - Command: `MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa env -u LD_LIBRARY_PATH python spider/preprocess/ik.py --task=pick_spoon_bowl --dataset-name=oakink --data-id=0 --embodiment-type=bimanual --robot-type=asm --open-hand --end-idx=30 --no-save-video --no-show-viewer`
  - Result: passed; generated `trajectory_kinematic.npz` and `trajectory_ikrollout.npz` under `example_datasets/processed/oakink/asm/bimanual/pick_spoon_bowl/0/`.
  - Command: load `trajectory_kinematic.npz` and compare against `scene.xml` dimensions.
  - Result: passed; `qpos=(27, 68)`, `qpos_rollout=(27, 68)`, `qvel=(27, 66)`, `contact=(29, 20)`, `contact_pos=(29, 20, 3)`, all arrays finite, scene is `nq=68 nv=66 nu=54 npair=425`.

## Change 011 - Validate ASM MJWP smoke test and harden runner environment

- Goal: Confirm that ASM `trajectory_kinematic.npz` can drive a short MJWP optimization and make the ASM runner consistently use the headless OpenGL environment.
- Files changed:
  - `examples/run_mjwp_asm.sh`: lines 17-56 updated.
  - `example_datasets/processed/oakink/asm/bimanual/pick_spoon_bowl/0/config.yaml`: generated by `examples/run_mjwp.py` during smoke test.
  - `example_datasets/processed/oakink/asm/bimanual/pick_spoon_bowl/0/run_mjwp_20260422_011349.log`: generated console log for smoke test.
  - `example_datasets/processed/oakink/asm/bimanual/pick_spoon_bowl/0/trajectory_mjwp.npz`: generated smoke-test MJWP output.
- Details:
  - Added `PYTHON_HEADLESS=(env -u LD_LIBRARY_PATH python)` and reused it for prepare, scene generation, IK, and MJWP so OSMesa is not polluted by host/conda GL library paths.
  - Ran the requested short MJWP smoke test with `num_samples=64`, `max_num_iterations=2`, `max_sim_steps=20`, `save_video=false`.
  - Smoke test produced one control tick covering 20 sim steps and saved `trajectory_mjwp.npz`.
  - Final smoke-test object tracking error was `pos=0.0167`, `quat=0.3207`.
- Validation:
  - Command: `bash -n examples/run_mjwp_asm.sh`
  - Result: passed.
  - Command: `MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa env -u LD_LIBRARY_PATH python examples/run_mjwp.py +override=oakink_asm task=pick_spoon_bowl data_id=0 robot_type=asm embodiment_type=bimanual viewer=none show_viewer=false save_video=false save_info=true num_samples=64 max_num_iterations=2 max_sim_steps=20`
  - Result: passed; terminal printed `Saved info` and `Final object tracking error: pos=0.0167, quat=0.3207`.
  - Command: load `trajectory_mjwp.npz` and check shapes/finite values.
  - Result: passed; key arrays include `qpos=(1, 20, 68)`, `qvel=(1, 20, 66)`, `ctrl=(1, 20, 54)`, all finite.

## Change 012 - Validate short headless video outputs

- Goal: Confirm that OSMesa headless rendering can save both IK and MJWP MP4 outputs for the ASM pipeline.
- Files changed:
  - `example_datasets/processed/oakink/asm/bimanual/pick_spoon_bowl/0/visualization_ik.mp4`: generated short 30-frame IK smoke video.
  - `example_datasets/processed/oakink/asm/bimanual/pick_spoon_bowl/0/visualization_mjwp.mp4`: generated short 20-sim-step MJWP smoke video.
  - `example_datasets/processed/oakink/asm/bimanual/pick_spoon_bowl/0/run_mjwp_20260422_011716.log`: generated console log for video-enabled MJWP smoke test.
  - `example_datasets/processed/oakink/asm/bimanual/pick_spoon_bowl/0/trajectory_kinematic.npz`: regenerated by the 30-frame IK video smoke test.
  - `example_datasets/processed/oakink/asm/bimanual/pick_spoon_bowl/0/trajectory_mjwp.npz`: regenerated by the video-enabled MJWP smoke test.
- Details:
  - The generated MP4 files are smoke-test outputs, not a full-length 1740-frame run.
  - IK video smoke saved `visualization_ik.mp4`; MuJoCo still printed two late QACC warnings, but the saved `qpos`, `qpos_rollout`, and `qvel` arrays remained finite and bounded.
  - MJWP video smoke saved `visualization_mjwp.mp4` and ended with object tracking error `pos=0.0162`, `quat=0.3069`.
- Validation:
  - Command: `MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa env -u LD_LIBRARY_PATH python spider/preprocess/ik.py --task=pick_spoon_bowl --dataset-name=oakink --data-id=0 --embodiment-type=bimanual --robot-type=asm --open-hand --end-idx=30 --save-video --no-show-viewer`
  - Result: passed; `visualization_ik.mp4` created and `trajectory_kinematic.npz` remained finite.
  - Command: `MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa env -u LD_LIBRARY_PATH python examples/run_mjwp.py +override=oakink_asm task=pick_spoon_bowl data_id=0 robot_type=asm embodiment_type=bimanual viewer=none show_viewer=false save_video=true save_info=true num_samples=64 max_num_iterations=2 max_sim_steps=20`
  - Result: passed; `visualization_mjwp.mp4` and `trajectory_mjwp.npz` created.
  - Command: `ls -lh example_datasets/processed/oakink/asm/bimanual/pick_spoon_bowl/0/trajectory_kinematic.npz example_datasets/processed/oakink/asm/bimanual/pick_spoon_bowl/0/trajectory_mjwp.npz example_datasets/processed/oakink/asm/bimanual/pick_spoon_bowl/0/visualization_ik.mp4 example_datasets/processed/oakink/asm/bimanual/pick_spoon_bowl/0/visualization_mjwp.mp4`
  - Result: passed; current sizes are `trajectory_kinematic.npz=63K`, `trajectory_mjwp.npz=65K`, `visualization_ik.mp4=57K`, `visualization_mjwp.mp4=41K`.

## Change 013 - Fix ASM video camera and regenerate longer validation videos

- Goal: Address the two visual issues found after the first smoke run: videos were too short to show time progression, and the `front` camera did not show the full ASM arm/hand system.
- Files changed:
  - `spider/preprocess/generate_xml.py`: lines 26-62 added `_camera_xyaxes()` and `_add_front_camera()`.
  - `spider/preprocess/generate_xml.py`: lines 834-835 changed camera creation to call `_add_front_camera(mj_spec, robot_type)`.
  - `example_datasets/processed/oakink/asm/bimanual/pick_spoon_bowl/scene.xml`: regenerated with ASM `front` camera at `pos=[2.8, -2.8, 1.7]`, `mode=fixed`.
  - `example_datasets/processed/oakink/asm/bimanual/pick_spoon_bowl/scene_eq.xml`: regenerated after scene update.
  - `example_datasets/processed/oakink/asm/bimanual/pick_spoon_bowl/0/visualization_ik.mp4`: regenerated from a 150-frame IK smoke run.
  - `example_datasets/processed/oakink/asm/bimanual/pick_spoon_bowl/0/visualization_mjwp.mp4`: regenerated from a 100-sim-step MJWP smoke run.
  - `example_datasets/processed/oakink/asm/bimanual/pick_spoon_bowl/0/trajectory_kinematic.npz`: regenerated from the 150-frame IK run.
  - `example_datasets/processed/oakink/asm/bimanual/pick_spoon_bowl/0/trajectory_mjwp.npz`: regenerated from the 100-sim-step MJWP run.
- Details:
  - Root cause for the unreadable video was that ASM reused the hand-only xhand `front` camera (`z=0.844`, `trackcom`) while ASM bodies sit around `z~=1.2`, so the camera was too low/close for the full-arm model.
  - Added an ASM-only fixed 3/4 camera centered on the arm/hand workspace; non-ASM robots still use the previous xhand camera.
  - Root cause for the `0:00` video length was the previous `max_sim_steps=20` smoke run: at `render_dt=0.02` and `fps=50`, that produces only about `0.2s` of video.
  - Regenerated longer validation videos: IK is now `3.0s`; MJWP is now `1.0s`.
- Validation:
  - Command: `python -m py_compile spider/preprocess/generate_xml.py`
  - Result: passed.
  - Command: `python spider/preprocess/generate_xml.py --task=pick_spoon_bowl --dataset-name=oakink --data-id=0 --embodiment-type=bimanual --robot-type=asm --no-show-viewer`
  - Result: passed; `scene.xml` loaded as `nq=68 nv=66 nu=54 npair=425 ncam=1`, camera `front` has `pos=[2.8, -2.8, 1.7]`, `mode=0` fixed.
  - Command: `MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa env -u LD_LIBRARY_PATH python spider/preprocess/ik.py --task=pick_spoon_bowl --dataset-name=oakink --data-id=0 --embodiment-type=bimanual --robot-type=asm --open-hand --end-idx=150 --save-video --no-show-viewer`
  - Result: passed; `visualization_ik.mp4` duration is `3.0s`; `trajectory_kinematic.npz` arrays are finite with `qpos=(147, 68)`, `qpos_rollout=(147, 68)`, `qvel=(147, 66)`.
  - Command: `MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa env -u LD_LIBRARY_PATH python examples/run_mjwp.py +override=oakink_asm task=pick_spoon_bowl data_id=0 robot_type=asm embodiment_type=bimanual viewer=none show_viewer=false save_video=true save_info=true num_samples=32 max_num_iterations=1 max_sim_steps=100`
  - Result: passed; `visualization_mjwp.mp4` duration is `1.0s`; `trajectory_mjwp.npz` arrays are finite with `qpos=(5, 20, 68)`, `qvel=(5, 20, 66)`, `ctrl=(5, 20, 54)`.
  - Command: extracted middle frames from the regenerated IK and MJWP videos.
  - Result: passed; frames now show the ASM arm/hand scene instead of the previous close-up texture-like view.

## Change 014 - Make ASM full-run progress visible and reduce default MJWP cost

- Goal: Fix the user-facing problem that `bash examples/run_mjwp_asm.sh` looked frozen after `best_qpos_diff_sum`, and restore a checker-textured floor/background closer to the xhand scene.
- Files changed:
  - `spider/preprocess/generate_xml.py`: lines 100-119 added `_bind_groundplane_material_textures()`.
  - `spider/preprocess/generate_xml.py`: lines 882 and 924 updated scene/equality-scene XML export to post-process groundplane materials with explicit `texture=` bindings.
  - `spider/preprocess/ik.py`: line 18 added `import time`.
  - `spider/preprocess/ik.py`: lines 619-620 added IK elapsed-time/progress state.
  - `spider/preprocess/ik.py`: lines 829-843 added periodic `IK progress: ...` logs every 100 frames and at completion.
  - `spider/preprocess/ik.py`: lines 858-864 added an explicit `Encoding IK video...` log before MP4 writing.
  - `examples/run_mjwp_asm.sh`: lines 9-13 changed default runner settings to `NUM_SAMPLES=32`, `MAX_NUM_ITERATIONS=1`, added `IK_END_IDX`, and kept full-length `MAX_SIM_STEPS=-1`.
  - `examples/run_mjwp_asm.sh`: lines 20, 28, 37, and 49-50 added stage markers so terminal output clearly shows which phase is running.
- Details:
  - The script was not truly hanging in IK; it had entered the long per-frame IK rollout/video stage without any intermediate progress output.
  - For ASM, the previous MJWP defaults (`128 samples`, `8 iterations`) were several orders more expensive than the tested preview settings and were impractical for a first full-length run, so the runner now defaults to a tractable preview/full-video configuration.
  - The ASM scene floor looked flat because `scene.xml` material entries lacked `texture=` bindings even though the checker textures existed; XML post-processing now restores `groundplane`, `right_groundplane`, and `left_groundplane` texture references explicitly.
- Validation:
  - Command: `python -m py_compile spider/preprocess/generate_xml.py spider/preprocess/ik.py`
  - Result: passed.
  - Command: `bash -n examples/run_mjwp_asm.sh`
  - Result: passed.
  - Command: `python spider/preprocess/generate_xml.py --task=pick_spoon_bowl --dataset-name=oakink --data-id=0 --embodiment-type=bimanual --robot-type=asm --no-show-viewer`
  - Result: passed; `scene.xml` now contains `material texture=` bindings: `groundplane->groundplane`, `right_groundplane->right_groundplane`, `left_groundplane->left_groundplane`.
  - Command: render a single frame from the regenerated ASM scene.
  - Result: passed; floor now shows a checker pattern instead of a flat gray plane.
  - Command: `MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa env -u LD_LIBRARY_PATH python spider/preprocess/ik.py --task=pick_spoon_bowl --dataset-name=oakink --data-id=0 --embodiment-type=bimanual --robot-type=asm --open-hand --end-idx=30 --no-save-video --no-show-viewer`
  - Result: passed; terminal now prints `IK progress: 1/30 ...` and `IK progress: 30/30 ...` before saving outputs.

## Change 015 - Tighten ASM front camera and bias it toward the task area

- Goal: Reduce the large white sky/horizon band in ASM videos and make the robot plus spoon/bowl region appear larger and more centered in frame.
- Files changed:
  - `spider/preprocess/generate_xml.py`: lines 43-54 updated the ASM-specific `front` camera placement.
- Details:
  - The previous ASM camera (`pos=[2.8, -2.8, 1.7]`, `target=[0.0, 0.0, 1.0]`) was too far away and looked too high above the task, which made the top white region large and the manipulation area too small.
  - The ASM `front` camera now uses a closer, lower 3/4 view aimed toward the actual interaction zone: `pos=[2.2, -2.2, 1.25]`, `target=[0.16, -0.10, 0.50]`.
  - This keeps the full dual-arm robot visible while enlarging the robot in frame and biasing the viewpoint toward the spoon/bowl workspace.
- Validation:
  - Command: `python spider/preprocess/generate_xml.py --task=pick_spoon_bowl --dataset-name=oakink --data-id=0 --embodiment-type=bimanual --robot-type=asm --no-show-viewer`
  - Result: passed; regenerated `scene.xml` and `scene_eq.xml` with the updated camera.
  - Command: render a middle frame from `example_datasets/processed/oakink/asm/bimanual/pick_spoon_bowl/0/trajectory_kinematic.npz` using the regenerated `scene.xml`.
  - Result: passed; camera reported `pos=[2.2, -2.2, 1.25]`, and the verification frame showed a tighter composition with less top white area and a larger robot/task region.

## Change 016 - Add VSCode URDF Visualizer package mapping for ASM

- Goal: Make `spider/assets/robots/asm_description/urdf/asm.urdf` open correctly in the installed VSCode URDF viewer by resolving `package://asm_description/...` mesh paths.
- Files changed:
  - `.vscode/settings.json`: lines 34-38 added `urdf-visualizer.packages` mapping for `asm_description`.
- Details:
  - The installed viewer extension is `morningfrog.urdf-visualizer`, whose local README documents that `package://<package_name>` is resolved through the workspace setting `urdf-visualizer.packages`.
  - The ASM URDF already had a valid ROS package root (`package.xml`, `CMakeLists.txt`) and all 75 unique mesh references under `package://asm_description/...` existed on disk, but the viewer had no package-name-to-directory mapping in workspace settings.
  - Added:
    - `"urdf-visualizer.packages": { "asm_description": "spider/assets/robots/asm_description" }`
- Validation:
  - Command: verify all `package://asm_description/...` references in `spider/assets/robots/asm_description/urdf/asm.urdf` exist on disk.
  - Result: passed; `75` unique package-relative asset paths, `0` missing.
  - Command: `python - <<'PY' ... json.load(open('.vscode/settings.json')) ... PY`
  - Result: passed; workspace settings JSON is valid.

## Change 017 - Add automatic ASM workspace support and fixed support table

- Goal: Lift the OakInk hand/object workspace to a fixed ASM tabletop height derived from robot geometry, while keeping the robot base unchanged and leaving a clear extension point for future hand/arm-table collision.
- Files changed:
  - `spider/preprocess/workspace_support.py`: lines 1-279 created.
  - `spider/preprocess/generate_xml.py`: lines 20-24 added workspace-support imports.
  - `spider/preprocess/generate_xml.py`: lines 175-201 added `_add_support_table_pairs()` with a collision-mode branch.
  - `spider/preprocess/generate_xml.py`: lines 309-356 added automatic ASM support-table / workspace-offset computation and logging.
  - `spider/preprocess/generate_xml.py`: lines 402-412 added the `support_table` box geom to `scene.xml`.
  - `spider/preprocess/generate_xml.py`: lines 830-839 added `support_table <-> object` explicit contact pairs.
- Details:
  - Added a shared helper module that computes:
    - robot total height from the compiled ASM robot XML,
    - first-frame object world-space bounds from `visual.obj`,
    - `table_surface_z = robot_height / 2`,
    - `workspace_z_offset = table_surface_z + 0.002 - object_first_frame_min_z`,
    - automatic support-table center and half extents from the object XY bounds plus a `0.10m` margin.
  - The helper stores `support_table_collision_mode`, currently `object_only`, so later enabling hand/arm-table collision only requires extending the collision-mode branch instead of restructuring `generate_xml.py`.
  - While implementing the helper, fixed an intermediate bug where mesh asset transforms (`mesh_pos` / `mesh_quat`) were being applied twice; after the fix, ASM robot height returned to the previously validated `1.39155m`.
  - `generate_xml.py` now writes these derived values into `example_datasets/processed/oakink/asm/bimanual/pick_spoon_bowl/task_info.json`, inserts a dedicated `support_table` geom into the worldbody, and creates explicit contact pairs only between the table and object collision geoms.
- Validation:
  - Command: `python -m py_compile spider/preprocess/workspace_support.py spider/preprocess/generate_xml.py spider/preprocess/ik.py`
  - Result: passed.
  - Command: `source /home/guolijun/anaconda3/etc/profile.d/conda.sh && conda activate spider && python spider/preprocess/generate_xml.py --task=pick_spoon_bowl --dataset-name=oakink --data-id=0 --embodiment-type=bimanual --robot-type=asm --no-show-viewer`
  - Result: passed; `scene.xml` regenerated with `support_table`, and logs reported `robot_height=1.3916`, `table_surface_z=0.6958`, `workspace_z_offset=0.6985`.
  - Command: inspect `example_datasets/processed/oakink/asm/bimanual/pick_spoon_bowl/task_info.json` and `scene.xml`.
  - Result: passed; `task_info.json` now contains `robot_height`, `table_surface_z`, `workspace_z_offset`, `object_first_frame_min_z`, `support_table_center`, `support_table_size`, and `support_table_collision_mode=object_only`, while `scene.xml` contains `support_table` plus `20` `support_table_<object>` pairs and no hand-table pairs.
  - Command: compute shifted first-frame object bounds with `compute_object_bounds(...)`.
  - Result: passed; shifted object minimum z is `0.697775...`, table surface z is `0.695775...`, so the clearance is exactly `0.002m`.

## Change 018 - Apply workspace lift consistently in IK

- Goal: Replace the old ASM-unfriendly `qpos_ref[:, :, 2] += z_offset` behavior with a consistent workspace lift that moves wrists, fingertip targets, and real object poses by the same amount before IK and rollout validation.
- Files changed:
  - `spider/preprocess/ik.py`: lines 17-19 added `json` import for reading derived support parameters.
  - `spider/preprocess/ik.py`: lines 34-37 added workspace-support imports.
  - `spider/preprocess/ik.py`: lines 343-420 replaced raw keypoint loading and the old one-line `z_offset` application with:
    - auto-loading of ASM support parameters from `task_info.json`,
    - fallback recomputation if the robot-scene `task_info.json` is missing,
    - shared `workspace_z_offset` application to wrist, finger, and object arrays,
    - logging of pre/post lift z ranges.
- Details:
  - Before this change, `ik.py` only lifted `qpos_ref`, so the mocap target sites moved up while the true object free joint stayed near the floor, which made the table-support idea internally inconsistent.
  - `ik.py` now copies the raw MANO arrays, computes or loads the same `workspace_z_offset` that `generate_xml.py` wrote, and applies that offset to:
    - `qpos_wrist_right / qpos_wrist_left`
    - `qpos_finger_right / qpos_finger_left`
    - `qpos_obj_right / qpos_obj_left`
  - The previous `z_offset` CLI parameter is still accepted as an additive debug offset, but ASM now defaults to the automatic support-table lift even when the user does not pass any extra height.
- Validation:
  - Command: `source /home/guolijun/anaconda3/etc/profile.d/conda.sh && conda activate spider && MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa env -u LD_LIBRARY_PATH python spider/preprocess/ik.py --task=pick_spoon_bowl --dataset-name=oakink --data-id=0 --embodiment-type=bimanual --robot-type=asm --open-hand --end-idx=10 --no-save-video --no-show-viewer`
  - Result: passed; IK logged `workspace_z_offset=0.6985`, `wrist_z 0.0545->0.7530`, `object_z 0.0377->0.7362`, saved `trajectory_kinematic.npz`, and saved `trajectory_ikrollout.npz`.
  - Command: `source /home/guolijun/anaconda3/etc/profile.d/conda.sh && conda activate spider && MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa env -u LD_LIBRARY_PATH python examples/run_mjwp.py +override=oakink_asm task=pick_spoon_bowl data_id=0 robot_type=asm embodiment_type=bimanual viewer=none show_viewer=false save_video=false save_info=true num_samples=8 max_num_iterations=1 max_sim_steps=10`
  - Result: passed; short MJWP smoke test completed with the lifted workspace, saved `trajectory_mjwp.npz`, and did not introduce new NaN / crash failures.

## Change 019 - Split ASM pipeline into minimal one-click stage scripts

- Goal: Replace the parameter-heavy shell workflow with three minimal stage scripts under `examples/` for scene generation, IK, and MJWP, while keeping the commands explicit and free of shell macro variables.
- Files changed:
  - `examples/generate_scene_asm.sh`: lines 1-21 created.
  - `examples/run_ik_asm.sh`: lines 1-18 created.
  - `examples/run_mjwp_asm.sh`: lines 1-23 rewritten as a clean MJWP entrypoint.
- Details:
  - Added `examples/generate_scene_asm.sh` to run:
    - `prepare_asm_mjcf.py`
    - `generate_xml.py`
    with all parameters written directly as CLI arguments.
  - Added `examples/run_ik_asm.sh` to call `generate_scene_asm.sh` first and then run full-length ASM IK with `--save-video`.
  - Reworked `examples/run_mjwp_asm.sh` into a minimal MJWP script that calls `run_ik_asm.sh` first and then launches `examples/run_mjwp.py` with direct Hydra overrides, `save_video=true`, `num_samples=32`, `max_num_iterations=1`, and `max_sim_steps=-1`.
  - Removed the old shell variables and progress `echo` lines from `run_mjwp_asm.sh` so the scripts stay focused on the essential commands only.
- Validation:
  - Command: `chmod +x examples/generate_scene_asm.sh examples/run_ik_asm.sh examples/run_mjwp_asm.sh`
  - Result: passed.
  - Command: `bash -n examples/generate_scene_asm.sh && bash -n examples/run_ik_asm.sh && bash -n examples/run_mjwp_asm.sh`
  - Result: passed.

## Change 020 - Document ASM scene output paths in the scene-generation script

- Goal: Make it easy to find the generated robot assets and scene files immediately after running `examples/generate_scene_asm.sh`.
- Files changed:
  - `examples/generate_scene_asm.sh`: lines 4-14 added output-path comments.
- Details:
  - Added a short header comment block that lists:
    - robot asset output directory: `example_datasets/processed/oakink/assets/robots/asm/`
    - task scene output directory: `example_datasets/processed/oakink/asm/bimanual/pick_spoon_bowl/`
  - The comment explicitly names `bimanual.xml`, `right.xml`, `left.xml`, `scene.xml`, `scene_eq.xml`, and `task_info.json`.
- Validation:
  - Command: `nl -ba examples/generate_scene_asm.sh`
  - Result: passed; the output-path comment block is present at the top of the script.

## Change 021 - Show the support table in videos and swap spoon/bowl hand ownership

- Goal: Fix two visible issues in the ASM `pick_spoon_bowl` pipeline: the support table was missing from rendered videos, and the generated retargeting still had the spoon associated with the wrong robot hand.
- Files changed:
  - `spider/preprocess/workspace_support.py`: lines 282-306 added task-specific object-slot swap helpers for ASM OakInk `pick_spoon_bowl`.
  - `spider/preprocess/generate_xml.py`: lines 20-25 added imports for the new swap helpers.
  - `spider/preprocess/generate_xml.py`: lines 315-344 added ASM `pick_spoon_bowl` object-slot swapping for `task_info`, first-frame object poses, and per-object contact sites.
  - `spider/preprocess/generate_xml.py`: lines 403-412 updated `support_table` to use the default visible geom group instead of hidden collision-only group 3.
  - `spider/preprocess/ik.py`: lines 34-38 added imports for the new swap helpers.
  - `spider/preprocess/ik.py`: lines 359-381 added synchronized swapping of wrist trajectories, fingertip trajectories, object trajectories, and per-hand contact arrays so the left hand now follows the original spoon-hand motion and the right hand follows the original bowl-hand motion.
- Details:
  - Root cause of the missing table was simple: `support_table` had been assigned MuJoCo geom `group=3`, but the default renderer options only show groups `0/1/2`, so the table existed physically in the scene but was filtered out from video rendering.
  - Root cause of the spoon-hand issue was subtler: swapping only the object slots changed labels but did not change which physical hand trajectory actually approached the spoon. The fix therefore swaps both:
    - the scene-side object slots (`right_object <= bowl`, `left_object <= spoon`)
    - the IK-side hand trajectories (`right hand <= original bowl hand`, `left hand <= original spoon hand`)
  - This keeps the default right-object/right-hand and left-object/left-hand constraint structure intact while making the **left hand** the hand that actually approaches the spoon in the retargeted motion.
- Validation:
  - Command: `python -m py_compile spider/preprocess/workspace_support.py spider/preprocess/generate_xml.py spider/preprocess/ik.py`
  - Result: passed.
  - Command: `source /home/guolijun/anaconda3/etc/profile.d/conda.sh && conda activate spider && python spider/preprocess/generate_xml.py --task=pick_spoon_bowl --dataset-name=oakink --data-id=0 --embodiment-type=bimanual --robot-type=asm --no-show-viewer`
  - Result: passed; logs reported `Swapping ASM pick_spoon_bowl object slots`, and regenerated `scene.xml` now contains `right_visual=file="objects/C12001/visual.obj"` (bowl), `left_visual=file="objects/O02@0030@00002/visual.obj"` (spoon), and a visible `support_table` geom with no hidden group tag.
  - Command: render a single offscreen image from the regenerated scene and first kinematic frame.
  - Result: passed; `/tmp/asm_ik_table_check.png` shows the support table in the camera view.
  - Command: `source /home/guolijun/anaconda3/etc/profile.d/conda.sh && conda activate spider && MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa env -u LD_LIBRARY_PATH python spider/preprocess/ik.py --task=pick_spoon_bowl --dataset-name=oakink --data-id=0 --embodiment-type=bimanual --robot-type=asm --open-hand --end-idx=10 --no-save-video --no-show-viewer`
  - Result: passed; logs reported `Swapping ASM pick_spoon_bowl hand-object mapping so left hand targets the spoon and right hand targets the bowl.`
  - Command: load the regenerated `trajectory_kinematic.npz` and compare palm-site distances to the swapped object slots.
  - Result: passed; `right_object` (bowl) is closer to `right_palm` (`0.1538`) than `left_palm` (`0.3402`), while `left_object` (spoon) is closer to `left_palm` (`0.1014`) than `right_palm` (`0.4974`).

## Change 022 - Rotate the ASM robot toward the task workspace and restyle the ASM table/camera

- Goal: Make the ASM `pick_spoon_bowl` videos read more naturally by turning the robot so its front notch faces the tabletop task area, changing the support table to a visible wood-brown color, and tightening the fixed camera framing so the robot occupies more of the frame.
- Files changed:
  - `spider/preprocess/prepare_asm_mjcf.py`: lines 45-46 added ASM root-rotation constants, lines 77-79 added a yaw-to-quaternion helper, lines 214-232 added `wrap_worldbody_in_root()`, and line 453 now wraps the compiled ASM robot under a rotated `asm_root` body before sites/actuators are added.
  - `spider/preprocess/generate_xml.py`: lines 50-81 rewrote ASM camera placement to aim at the support-table task zone with a narrower `fovy=38`, lines 454-464 changed `support_table` from groundplane material to an explicit brown `rgba`, and line 1001 now passes `support_table_spec` into the camera builder.
- Details:
  - I measured the ASM base footprint from `base_link1.STL` and confirmed that the current task workspace was aligned with one of the base legs rather than one of the front notches. Wrapping the robot inside a single `asm_root` body with `45 deg` yaw rotates the whole arm-hand assembly consistently without touching the downstream site names or actuator layout.
  - This change is intentionally done at the robot-asset layer so the scene, IK, and MJWP stages all inherit the same notion of "robot front" automatically the next time we regenerate `bimanual.xml -> scene.xml`.
  - The support table now uses `rgba="0.56 0.49 0.39 1"`, which makes it visually separate from the dark checker floor while keeping the geometry simple.
  - The ASM fixed camera now derives its pose from `support_table_spec`, pulls closer to the task zone, and uses a narrower field of view so the robot/table occupy more of the rendered frame.
  - Important limitation: the arm-column interpenetration you noticed is still mostly an IK/self-collision modeling limitation, not just a camera artifact. The robot visuals can still pass through the central support because the current IK stage does not yet include dedicated arm/base collision proxies. Rotating the robot toward the task helps the posture look less unnatural, but it does not fully solve arm self-collision by itself.
- Validation:
  - Command: `python -m py_compile spider/preprocess/prepare_asm_mjcf.py spider/preprocess/generate_xml.py`
  - Result: passed.
  - Command: `conda run -n spider env -u LD_LIBRARY_PATH python spider/preprocess/prepare_asm_mjcf.py --dataset-dir example_datasets --dataset-name oakink --source-urdf spider/assets/robots/asm_description/urdf/asm.urdf --robot-type asm --variants bimanual right left`
  - Result: passed; regenerated ASM robot assets, and `bimanual.xml` now has `worldbody/body[@name='asm_root']` with `quat="0.92388 0 0 0.382683"`.
  - Command: `conda run -n spider env -u LD_LIBRARY_PATH python spider/preprocess/generate_xml.py --dataset-dir example_datasets --dataset-name oakink --robot-type asm --embodiment-type bimanual --task pick_spoon_bowl --data-id 0 --no-show-viewer`
  - Result: passed; regenerated `scene.xml`, which now contains `support_table rgba="0.56 0.49 0.39 1"` and a closer ASM front camera at `pos="1.2552 -0.617051 1.04"`.
  - Command: `conda run -n spider env -u LD_LIBRARY_PATH MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa python spider/preprocess/ik.py --dataset-dir example_datasets --dataset-name oakink --robot-type asm --embodiment-type bimanual --task pick_spoon_bowl --data-id 0 --open-hand --end-idx 30 --save-video --no-show-viewer`
  - Result: passed; short IK regenerated `visualization_ik.mp4`, `trajectory_kinematic.npz`, and `trajectory_ikrollout.npz` with the updated robot orientation and scene camera.
  - Command: offscreen render of frame 15 from the regenerated `trajectory_kinematic.npz` using `scene.xml` and the fixed `front` camera.
  - Result: passed; `/tmp/asm_scene_mid_v3.png` shows the rotated ASM robot, the brown support table clearly separated from the checker floor, and a tighter crop than the previous camera setup.

## Change 023 - Add a one-frame ASM IK smoke-test script

- Goal: Provide a minimal one-click command under `examples/` for quickly generating a 1-frame ASM IK video, so camera/scene/IK changes can be checked without running the full sequence.
- Files changed:
  - `examples/run_ik_asm_test1frame.sh`: lines 1-19 created.
- Details:
  - Added a new script that mirrors `examples/run_ik_asm.sh` but fixes `--end-idx 1`.
  - The script still regenerates the ASM scene first via `bash examples/generate_scene_asm.sh`, then runs `ik.py` in headless OSMesa mode with `--save-video` and `--no-show-viewer`.
  - This gives a very fast smoke test for a single-frame IK render while keeping the same output directory and environment setup as the full ASM IK script.
- Validation:
  - Command: `chmod +x examples/run_ik_asm_test1frame.sh && bash -n examples/run_ik_asm_test1frame.sh`
  - Result: passed.
  - Command: `nl -ba examples/run_ik_asm_test1frame.sh`
  - Result: passed; the new script contains the expected 1-frame IK command.

## Change 024 - Restore the earlier ASM camera distance and remove the extra zoom

- Goal: Keep the ASM front camera at the previously preferred `target/pos` values while removing the extra zoom effect introduced by the temporary `fovy=38` override.
- Files changed:
  - `spider/preprocess/generate_xml.py`: line 79 removed the ASM-specific `fovy=38.0` override.
- Details:
  - The ASM `target` and `pos` values were already back at:
    - `target = [table_center_xy[0] * 0.55, table_center_xy[1] * 0.35, max(table_surface_z + 0.18, 0.86)]`
    - `pos = [table_center_xy[0] + 1.28, table_center_xy[1] - 0.92, max(table_surface_z + 0.40, 1.08)]`
  - The remaining reason the one-frame test still looked too close was the narrower field-of-view override. Removing `fovy=38.0` returns the ASM fixed camera to MuJoCo's default field of view while preserving the desired camera position.
- Validation:
  - Command: `python -m py_compile spider/preprocess/generate_xml.py`
  - Result: passed.
  - Command: `conda run -n spider env -u LD_LIBRARY_PATH python spider/preprocess/generate_xml.py --dataset-dir example_datasets --dataset-name oakink --robot-type asm --embodiment-type bimanual --task pick_spoon_bowl --data-id 0 --no-show-viewer`
  - Result: passed; regenerated `scene.xml` and `scene_eq.xml` with the updated camera.
  - Command: inspect `example_datasets/processed/oakink/asm/bimanual/pick_spoon_bowl/scene.xml` camera attributes.
  - Result: passed; the `front` camera keeps the expected `pos` but no longer contains a `fovy` attribute.

## Change 025 - Revert ASM root rotation and restore the earlier fixed camera while keeping the new table color

- Goal: Undo the recent ASM workspace-facing rotation and return the camera to the earlier fixed position, because the user preferred the previous robot/object arrangement; keep the newer brown table color unchanged.
- Files changed:
  - `spider/preprocess/prepare_asm_mjcf.py`: line 46 changed `ASM_ROOT_YAW_DEG` from `45.0` back to `0.0`.
  - `spider/preprocess/generate_xml.py`: lines 56-67 simplified the ASM camera branch back to the earlier fixed camera pose `pos=[2.2, -2.2, 1.25]`, `target=[0.16, -0.10, 0.50]` while keeping the table color logic untouched.
- Details:
  - The perceived "object placement shift" came from rotating the whole ASM robot under `asm_root`, not from changing the actual object coordinates. Setting `ASM_ROOT_YAW_DEG=0.0` returns the robot/object relative arrangement to the previous state.
  - The support table stays brown (`rgba="0.56 0.49 0.39 1"`); only the robot orientation and the ASM fixed camera pose were reverted.
- Validation:
  - Command: `python -m py_compile spider/preprocess/prepare_asm_mjcf.py spider/preprocess/generate_xml.py`
  - Result: passed.
  - Command: `conda run -n spider env -u LD_LIBRARY_PATH python spider/preprocess/prepare_asm_mjcf.py --dataset-dir example_datasets --dataset-name oakink --source-urdf spider/assets/robots/asm_description/urdf/asm.urdf --robot-type asm --variants bimanual right left`
  - Result: passed; regenerated ASM assets, and `bimanual.xml` now has `asm_root quat="1 0 0 0"` (reported as identity quaternion).
  - Command: `conda run -n spider env -u LD_LIBRARY_PATH python spider/preprocess/generate_xml.py --dataset-dir example_datasets --dataset-name oakink --robot-type asm --embodiment-type bimanual --task pick_spoon_bowl --data-id 0 --no-show-viewer`
  - Result: passed; regenerated `scene.xml` and `scene_eq.xml`.
  - Command: inspect `example_datasets/processed/oakink/asm/bimanual/pick_spoon_bowl/scene.xml` and `example_datasets/processed/oakink/assets/robots/asm/bimanual.xml`.
  - Result: passed; `front` camera is back at the earlier fixed pose and `support_table` still uses `rgba="0.56 0.49 0.39 1"`.

## Change 026 - Restore ASM pick_spoon_bowl hand-object semantics to match xhand

- Goal: Remove the temporary ASM-only left/right swap so OakInk `pick_spoon_bowl` goes back to the original xhand semantics: **right hand tracks the spoon, left hand tracks the bowl**.
- Files changed:
  - `spider/preprocess/workspace_support.py`: lines 296-305 removed the ASM `pick_spoon_bowl` object-slot swap helper.
  - `spider/preprocess/generate_xml.py`: lines 20-24 removed the swap-helper imports; lines 311-316 now keep MANO `task_info` and first-frame object slots in their original order.
  - `spider/preprocess/ik.py`: lines 34-37 removed the swap-helper imports; lines 343-355 now keep the raw wrist/fingertip/object trajectories in their original left/right assignment.
- Details:
  - The earlier ASM fork intentionally swapped spoon/bowl ownership so the left hand targeted the spoon. This change removes that override completely instead of trying to special-case around it.
  - After this change, the processed ASM `task_info.json` again preserves the MANO/xhand convention:
    - `right_object_mesh_dir = O02@0030@00002` (spoon)
    - `left_object_mesh_dir = C12001` (bowl)
  - This also means IK once again feeds the right palm/fingertips the original right-hand target cloud, rather than reusing the original left-hand motion.
- Validation:
  - Command: `python -m py_compile spider/preprocess/workspace_support.py spider/preprocess/generate_xml.py spider/preprocess/ik.py`
  - Result: passed.
  - Command: `source /home/guolijun/anaconda3/etc/profile.d/conda.sh && conda activate spider && bash examples/generate_scene_asm.sh`
  - Result: passed; regenerated `task_info.json` now reports `right_object_mesh_dir=processed/oakink/assets/objects/O02@0030@00002` and `left_object_mesh_dir=processed/oakink/assets/objects/C12001`.
  - Command: `source /home/guolijun/anaconda3/etc/profile.d/conda.sh && conda activate spider && MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa env -u LD_LIBRARY_PATH python spider/preprocess/ik.py --dataset-dir example_datasets --dataset-name oakink --robot-type asm --embodiment-type bimanual --task pick_spoon_bowl --data-id 0 --open-hand --end-idx 5 --save-video --no-show-viewer`
  - Result: passed; regenerated `trajectory_kinematic.npz` with the restored semantics.
  - Command: load the regenerated `trajectory_kinematic.npz` and compare palm-site distances to the two object bodies.
  - Result: passed; at frame 1 `right_object -> right_palm = 0.1150 < right_object -> left_palm = 0.4285`, and `left_object -> left_palm = 0.1393 < left_object -> right_palm = 0.3549`.

## Change 027 - Move the ASM tabletop workspace to the robot midline and push it forward

- Goal: Keep the automatic table-height lift, but move the whole spoon/bowl workspace farther away from the center column so the task sits on the robot midline and farther forward.
- Files changed:
  - `spider/preprocess/workspace_support.py`: lines 15-38 added `ASM_PICK_SPOON_BOWL_FORWARD_OFFSET_Y` and `workspace_xy_offset` to `WorkspaceSupportSpec`; lines 195-262 now compute a task-specific XY workspace shift; lines 265-293 load `workspace_xy_offset` back from `task_info.json`.
  - `spider/preprocess/generate_xml.py`: lines 328-350 now pass `dataset_name/robot_type/task` into `compute_workspace_support_spec()` and log the new XY offset into the saved task info.
  - `spider/preprocess/ik.py`: lines 357-441 now read `workspace_xy_offset` from the support spec and apply the same XY shift to wrist, fingertip, and object targets before building `qpos_ref`.
- Details:
  - For `oakink + asm + bimanual + pick_spoon_bowl`, the support-table center is now adjusted by:
    - `x_offset = -original_table_center_x` so the table sits on the robot midline
    - `y_offset = -0.18` so the table and targets move farther forward
  - The shift is saved into `task_info.json` as `workspace_xy_offset` and reused by IK, so the scene center and the mocap target cloud stay consistent.
  - The automatic height logic remains unchanged: `table_surface_z` and `workspace_z_offset` still come from robot height and object first-frame min-z.
- Validation:
  - Command: `source /home/guolijun/anaconda3/etc/profile.d/conda.sh && conda activate spider && bash examples/generate_scene_asm.sh`
  - Result: passed; logs reported `workspace_xy_offset=[-0.15520219244320363, -0.18]` and `support_table center=[0.0, -0.2970513695470197, 0.6757752299065931]`.
  - Command: inspect `example_datasets/processed/oakink/asm/bimanual/pick_spoon_bowl/task_info.json`.
  - Result: passed; `workspace_xy_offset` is saved and `support_table_center[0] == 0.0`.
  - Command: `source /home/guolijun/anaconda3/etc/profile.d/conda.sh && conda activate spider && MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa env -u LD_LIBRARY_PATH python spider/preprocess/ik.py --dataset-dir example_datasets --dataset-name oakink --robot-type asm --embodiment-type bimanual --task pick_spoon_bowl --data-id 0 --open-hand --end-idx 5 --save-video --no-show-viewer`
  - Result: passed; logs reported `Applied workspace_xy_offset=[-0.15520219244320363, -0.18]` before the usual Z lift.

## Change 028 - Switch the ASM front camera to a straight-on frontal view

- Goal: Replace the previous right-front 45-degree ASM view with a frontal camera that looks straight at the robot from the front.
- Files changed:
  - `spider/preprocess/generate_xml.py`: lines 48-62 changed the ASM `front` camera to `pos=[0.0, -2.45, 1.25]`, `target=[0.0, -0.12, 0.50]`.
- Details:
  - The robot root orientation remains unchanged (`ASM_ROOT_YAW_DEG = 0.0`); only the rendered viewpoint changes.
  - The brown table color from the previous revision is intentionally kept as-is.
  - The ASM camera still uses MuJoCo's default field of view; this change only repositions the fixed camera.
- Validation:
  - Command: `python -m py_compile spider/preprocess/generate_xml.py`
  - Result: passed.
  - Command: `source /home/guolijun/anaconda3/etc/profile.d/conda.sh && conda activate spider && bash examples/generate_scene_asm.sh`
  - Result: passed; regenerated `scene.xml`.
  - Command: inspect the `front` camera in `example_datasets/processed/oakink/asm/bimanual/pick_spoon_bowl/scene.xml`.
  - Result: passed; the camera is now at `pos=[0.0, -2.45, 1.25]` and uses a frontal `xyaxes`.
  - Command: `source /home/guolijun/anaconda3/etc/profile.d/conda.sh && conda activate spider && MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa env -u LD_LIBRARY_PATH python spider/preprocess/ik.py --dataset-dir example_datasets --dataset-name oakink --robot-type asm --embodiment-type bimanual --task pick_spoon_bowl --data-id 0 --open-hand --end-idx 5 --save-video --no-show-viewer`
  - Result: passed; regenerated `visualization_ik.mp4` from the new frontal view.

## Change 029 - Mirror the ASM tabletop workspace to the robot front and flip the frontal camera to +Y

- Goal: Fix the remaining direction error in the ASM `pick_spoon_bowl` view so the robot front truly faces the observer, the table sits in front of the robot rather than behind it, and the spoon/bowl appear left/right correctly from the viewer's perspective.
- Files changed:
  - `spider/preprocess/workspace_support.py`: lines 13-15 renamed the task-specific Y offset constant to a reference back-side offset; lines 227-240 now mirror the previously generated back-side tabletop depth to the robot's front side instead of pushing the workspace farther along `-Y`.
  - `spider/preprocess/generate_xml.py`: lines 53-60 moved the ASM `front` camera from `-Y` to `+Y` while keeping the same height and centered frontal composition.
- Details:
  - The previous "frontal" version still looked wrong because both the camera and the tabletop were on the `-Y` side of the robot. That made the red-box front structure appear turned away and also inverted the expected viewer-left/viewer-right arrangement for spoon vs bowl.
  - The new workspace rule treats the old table placement as a reference back-side depth and mirrors it across the robot origin:
    - old table center `y ~= -0.297`
    - new table center `y ~= +0.297`
  - With the camera also moved to `+Y`, the robot is now viewed from the true front. Since the right-hand spoon remains at positive world `x`, it correctly appears on the viewer's **left** in the rendered frame, while the left-hand bowl appears on the viewer's **right**.
- Validation:
  - Command: `python -m py_compile spider/preprocess/workspace_support.py spider/preprocess/generate_xml.py`
  - Result: passed.
  - Command: `source /home/guolijun/anaconda3/etc/profile.d/conda.sh && conda activate spider && bash examples/generate_scene_asm.sh`
  - Result: passed; logs reported `workspace_xy_offset=[-0.15520219244320363, 0.4141027390940394]` and `support_table center=[0.0, 0.2970513695470197, 0.6757752299065931]`.
  - Command: inspect `example_datasets/processed/oakink/asm/bimanual/pick_spoon_bowl/task_info.json` and `scene.xml`.
  - Result: passed; `support_table_center[1]` is now positive and the `front` camera is at `pos=[0.0, 2.45, 1.25]`.
  - Command: `source /home/guolijun/anaconda3/etc/profile.d/conda.sh && conda activate spider && MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa env -u LD_LIBRARY_PATH python spider/preprocess/ik.py --dataset-dir example_datasets --dataset-name oakink --robot-type asm --embodiment-type bimanual --task pick_spoon_bowl --data-id 0 --open-hand --end-idx 5 --save-video --no-show-viewer`
  - Result: passed; regenerated `visualization_ik.mp4`, `trajectory_kinematic.npz`, and `trajectory_ikrollout.npz` with the front-side workspace.
  - Command: project the regenerated frame-1 object and palm positions into the `front` camera frame.
  - Result: passed; `right_object camera_x = -0.1848`, `left_object camera_x = +0.1765`, confirming spoon-on-viewer-left and bowl-on-viewer-right.
  - Command: `source /home/guolijun/anaconda3/etc/profile.d/conda.sh && conda activate spider && MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa env -u LD_LIBRARY_PATH python examples/run_mjwp.py +override=oakink_asm task=pick_spoon_bowl data_id=0 robot_type=asm embodiment_type=bimanual viewer=none show_viewer=false save_video=false save_info=true num_samples=8 max_num_iterations=1 max_sim_steps=5`
  - Result: passed; regenerated `trajectory_mjwp.npz` without introducing new scene/reference mismatches.

## 里程碑！IK 数据生成成功

IK 数据生成成功，机器人面向我，有桌子，左手拿碗右手拿勺子，机器人本体不穿模。

关键修改1：`prepare_asm_mjcf.py (line 46)`，俯视图，顺时针 -90，逆时针+90，机器人的朝向只需要改这一行。

关键修改 2：`workspace_support.py (line 212)` ,桌子高度 = 比例 * 机器人高度；桌子厚度 `workspace_support.py (line 12)`

关键修改 3：`workspace_support.py (line 15)`，控制桌子离机器人远近
```
ASM_PICK_SPOON_BOWL_REFERENCE_BACK_OFFSET_Y = -0.18
```
更负，比如 -0.30，桌子会更远；改得更接近 0，比如 -0.08，桌子会更近

关键修改 4：左右平移桌子，`workspace_support.py (line 238)`，
```
[-xy_center[0] + delta_x, ...]
```

但是现在的问题是机器人和桌子会穿模

## Change 030 - Enable ASM contact guidance and wire up the `_act` scene/data pipeline

- Goal: Turn on the MJWP "glue force" path (`contact_guidance`) for ASM, slightly increase the guidance gains, and make sure the shell pipeline generates the required actuator scene and actuator IK data automatically.
- Files changed:
  - `examples/config/override/oakink_asm.yaml`: lines 17-32 now enable `contact_guidance`, force `improvement_threshold=0.0`, and set mild guidance gains to `init_pos_actuator_gain/bias=12.0` and `init_rot_actuator_gain/bias=0.12`.
  - `examples/generate_scene_asm.sh`: lines 35-43 now add a second `generate_xml.py --act-scene` call so the ASM pipeline exports `scene_act.xml` in addition to `scene.xml/scene_eq.xml`.
  - `examples/run_ik_asm.sh`: lines 20-30 now add a second IK pass with `--act-scene --no-save-video` so the pipeline also exports `trajectory_kinematic_act.npz` and `trajectory_ikrollout_act.npz`.
  - `examples/run_mjwp_asm.sh`: lines 21-23 now enable `contact_guidance=true` and raise `max_num_iterations` from `1` to `2`, because with only one iteration the guidance gains are zeroed on the sole optimization pass and effectively do nothing.
- Details:
  - The "glue force" in this repository corresponds to object xyz/rpy actuators in `scene_act.xml` whose effective `kp/kd` values are scheduled by `run_mjwp.py` during optimization.
  - The previous script setup could not use guidance even if enabled, because:
    - `scene_act.xml` and `trajectory_kinematic_act.npz` were not being generated by the standard ASM shell path.
    - `max_num_iterations=1` causes the last-iteration zeroing logic in `run_mjwp.py` to disable the guidance on the only iteration.
  - This revision fixes both issues while keeping the gain increase intentionally small relative to the global defaults.
- Validation:
  - Command: `bash -n examples/generate_scene_asm.sh && bash -n examples/run_ik_asm.sh && bash -n examples/run_mjwp_asm.sh`
  - Result: passed.
  - Command: `python -m py_compile spider/config.py examples/run_mjwp.py spider/preprocess/generate_xml.py spider/preprocess/ik.py`
  - Result: passed.
  - Command: `source /home/guolijun/anaconda3/etc/profile.d/conda.sh && conda activate spider && bash examples/generate_scene_asm.sh`
  - Result: passed; regenerated `scene.xml`, `scene_eq.xml`, and `scene_act.xml`.
  - Command: `source /home/guolijun/anaconda3/etc/profile.d/conda.sh && conda activate spider && MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa env -u LD_LIBRARY_PATH python spider/preprocess/ik.py --dataset-dir example_datasets --dataset-name oakink --robot-type asm --embodiment-type bimanual --task pick_spoon_bowl --data-id 0 --open-hand --act-scene --end-idx 5 --no-save-video --no-show-viewer`
  - Result: passed; generated `trajectory_kinematic_act.npz` and `trajectory_ikrollout_act.npz`.
  - Command: `source /home/guolijun/anaconda3/etc/profile.d/conda.sh && conda activate spider && MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa env -u LD_LIBRARY_PATH python examples/run_mjwp.py +override=oakink_asm dataset_dir=example_datasets dataset_name=oakink task=pick_spoon_bowl data_id=0 robot_type=asm embodiment_type=bimanual viewer=none show_viewer=false save_video=false save_info=true contact_guidance=true num_samples=8 max_num_iterations=2 max_sim_steps=5`
  - Result: completed and saved `trajectory_mjwp_act.npz` plus `config_act.yaml`, confirming the guidance pipeline is now wired through the `_act` assets/data path. This short smoke test still reported `NaNs or infs in rews: 8/8` and `Final object tracking error: pos=nan, quat=nan`, so contact guidance is now active but still needs further tuning for stable ASM behavior.

## Change 031 - Strengthen ASM contact guidance persistence and mildly raise actuator gains

- Goal: Make the ASM "glue force" stronger and longer-lasting during MJWP optimization, specifically to address the observed failure mode where the spoon starts in-hand on frame 1 but then drops during rollout.
- Files changed:
  - `examples/config/override/oakink_asm.yaml`: lines 29-33 now raise the object guidance gains from `12.0/0.12` to `14.0/0.14` and add `guidance_decay_ratio: 0.8` so the guidance decays more slowly across optimization iterations.
- Details:
  - The previous guidance settings only provided a mild pull on the object back toward the reference trajectory.
  - This revision strengthens both:
    - the magnitude of the object xyz/rpy actuator guidance
    - the persistence of that guidance across optimization iterations
  - The intent is to better support "keep holding after the first frame", not just "snap close on frame 1".
  - I intentionally did **not** overwrite the current `examples/run_mjwp_asm.sh` iteration/sample settings because the local script already uses a stronger manual override (`num_samples=32`, `max_num_iterations=4`) than the earlier baseline, and I wanted to preserve that user-side adjustment.
- Validation:
  - Command: `bash -n examples/run_mjwp_asm.sh`
  - Result: passed.
  - Command: `source /home/guolijun/anaconda3/etc/profile.d/conda.sh && conda activate spider && MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa env -u LD_LIBRARY_PATH python examples/run_mjwp.py +override=oakink_asm dataset_dir=example_datasets dataset_name=oakink task=pick_spoon_bowl data_id=0 robot_type=asm embodiment_type=bimanual viewer=none show_viewer=false save_video=false save_info=true contact_guidance=true num_samples=8 max_num_iterations=4 max_sim_steps=5`
  - Result: completed and saved `trajectory_mjwp_act.npz` with the stronger guidance settings, confirming the new parameters are active. The short smoke test still reported `NaNs or infs in rews: 8/8` on several optimization passes and ended with `Final object tracking error: pos=nan, quat=nan`, so the stronger glue force is active but not yet sufficient on its own to stabilize ASM grasp retention.

## Change 032 - Tune ASM arm/hand actuator dynamics and increase hand torque limits

- Goal: Reduce arm-side jitter, increase hand-side tracking strength, and raise the maximum force available to hand joints so the fingers can apply a stronger sustained squeeze during MJWP.
- Files changed:
  - `spider/preprocess/prepare_asm_mjcf.py`: lines 102-110 add explicit CLI parameters for `arm_kp`, `hand_kp`, arm/hand damping, armature, frictionloss, and `hand_force_scale`; lines 303-453 add joint-type detection and per-joint tuning logic; lines 573-583 apply that tuning before actuators are added.
  - `examples/generate_scene_asm.sh`: lines 19-33 now pass the tuned ASM generation parameters explicitly (`arm_kp=300`, `hand_kp=180`, `arm_damping=2.0`, `hand_damping=0.5`, `arm_armature=0.05`, `hand_armature=0.02`, `hand_frictionloss=0.01`, `hand_force_scale=2.0`).
- Details:
  - Arm joints are now treated separately from hand joints instead of inheriting the old global `damping=0 / armature=0.01 / frictionloss=0` defaults.
  - The new generation path writes:
    - arm joints: lower `kp`, larger damping/armature, no extra frictionloss
    - hand joints: higher `kp`, moderate damping/armature, small frictionloss
  - Hand-joint `actuatorfrcrange` is scaled by `2.0x`, which is the direct implementation of "increase the hand motor maximum" from the review discussion.
- Validation:
  - Command: `python -m py_compile spider/preprocess/prepare_asm_mjcf.py`
  - Result: passed.
  - Command: `source /home/guolijun/anaconda3/etc/profile.d/conda.sh && conda activate spider && bash examples/generate_scene_asm.sh`
  - Result: passed; regenerated ASM robot assets and scenes with `collision_hand=52` for `bimanual.xml`, confirming the new asset-generation path executes.

## Change 033 - Expand ASM hand collision proxies beyond palm and fingertips

- Goal: Give the ASM hand a more realistic grasp envelope by adding collision proxies on each finger link, rather than relying only on one palm box plus five fingertip spheres.
- Files changed:
  - `spider/preprocess/prepare_asm_mjcf.py`: lines 48-80 define per-finger proxy box sizes; lines 323-417 now add `collision_hand_{side}_{finger}_{link_idx}` box geoms for `link1` through `link4` on every finger, while retaining the palm box and fingertip spheres.
- Details:
  - The previous ASM hand had only:
    - `collision_hand_{side}_palm_0`
    - `collision_hand_{side}_{finger}_0` for fingertip spheres
  - The new version adds continuous finger-body contact proxies for all five fingers on both hands.
  - This change is intentionally limited to the hand; arm/base collision modeling is still out of scope for this revision.
- Validation:
  - Command: `source /home/guolijun/anaconda3/etc/profile.d/conda.sh && conda activate spider && bash examples/generate_scene_asm.sh`
  - Result: passed; validation output increased from sparse hand contact geometry to `collision_hand=52` in `bimanual.xml`, `collision_hand=26` in each single-arm asset.
  - Command: inspect `example_datasets/processed/oakink/assets/robots/asm/bimanual.xml`.
  - Result: passed; each finger now contains `collision_hand_<side>_<finger>_<link_idx>` geoms in addition to the original palm/tip proxies.

## Change 034 - Add MJWP grasp-closing bias, symmetric contact guidance indices, and finer ASM control timing

- Goal: Make MJWP hold a grasp after contact by biasing finger actuators closed once the palm is near the object, while also fixing the previous thumb-only / asymmetric contact guidance setup and increasing control update frequency.
- Files changed:
  - `spider/config.py`: lines 166-171 add grasp-bias and palm-site config fields; lines 335-373 resolve per-hand finger actuator ids/biases and palm site ids; lines 527-547 now use `thumb + index + middle` for both right and left contact guidance indices.
  - `examples/run_mjwp.py`: lines 237-276 add helper functions for object position extraction and control bias application; lines 358-373 read actuator control ranges; lines 535-584 apply sticky per-hand grasp-closing bias once palm-object distance drops below `0.09m`.
  - `examples/config/override/oakink_asm.yaml`: lines 19-33 set `max_num_iterations=4`, `ctrl_dt=0.1`, `knot_dt=0.1`, `nconmax_per_env=512`, and `njmax_per_env=1024` while keeping `contact_guidance=true` and the stronger guidance gains.
- Details:
  - Grasp bias follows the agreed rule:
    - thumb joints `1/3/4`: `+0.10 rad`
    - other fingers joints `1/3/4`: `+0.08 rad`
    - joint `2` is intentionally excluded
  - The bias is clipped against each actuator's `ctrlrange`, so it cannot exceed actuator limits.
  - The YAML file briefly had a duplicate `nconmax_per_env` key during this edit; that was removed before runtime validation.
- Validation:
  - Command: `python -m py_compile spider/config.py examples/run_mjwp.py`
  - Result: passed.
  - Command: `source /home/guolijun/anaconda3/etc/profile.d/conda.sh && conda activate spider && MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa env -u LD_LIBRARY_PATH python spider/preprocess/ik.py --dataset-dir example_datasets --dataset-name oakink --robot-type asm --embodiment-type bimanual --task pick_spoon_bowl --data-id 0 --open-hand --act-scene --end-idx 5 --no-save-video --no-show-viewer`
  - Result: passed; regenerated `trajectory_kinematic_act.npz` and `trajectory_ikrollout_act.npz`.
  - Command: `source /home/guolijun/anaconda3/etc/profile.d/conda.sh && conda activate spider && MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa env -u LD_LIBRARY_PATH python examples/run_mjwp.py +override=oakink_asm dataset_dir=example_datasets dataset_name=oakink task=pick_spoon_bowl data_id=0 robot_type=asm embodiment_type=bimanual viewer=none show_viewer=false save_video=false save_info=true contact_guidance=true num_samples=8 max_num_iterations=4 max_sim_steps=5`
  - Result: executed successfully on GPU and saved `trajectory_mjwp_act.npz` plus `config_act.yaml`. The new grasp-bias/contact-guidance path is live, but the short smoke test still reported `NaNs or infs in rews: 8/8` and `Final object tracking error: pos=nan, quat=nan`, so the implementation is complete but the ASM grasp remains numerically unstable.

## Change 035 - Switch ASM asset generation to `new_asm_description` and validate the new pipeline outputs

- Goal: Replace the old ASM source package with `spider/assets/robots/new_asm_description` as the default robot description for all ASM generation scripts, regenerate the processed `asm` assets/results tree, and verify that `scene -> IK -> MJWP` still runs end-to-end on the new robot.
- Files changed:
  - `spider/preprocess/prepare_asm_mjcf.py`: lines 92-95 now default `--source-urdf` to `spider/assets/robots/new_asm_description/urdf/asm.urdf`.
  - `examples/generate_scene_asm.sh`: lines 19-23 now explicitly pass `--source-urdf spider/assets/robots/new_asm_description/urdf/asm.urdf`.
- Details:
  - The new URDF differs only slightly from the previous ASM package, but it changes wrist/arm-to-hand fixed-joint transforms, so switching the source package at the generator entry points is the correct low-risk migration path.
  - The processed asset/output location remains unchanged:
    - robot assets: `example_datasets/processed/oakink/assets/robots/asm/`
    - experiment outputs: `example_datasets/processed/oakink/asm/bimanual/pick_spoon_bowl/`
  - The previously renamed `old_asm` folders are left untouched and no longer referenced by the active scripts.
- Validation:
  - Command: `bash -n examples/generate_scene_asm.sh && python -m py_compile spider/preprocess/prepare_asm_mjcf.py`
  - Result: passed.
  - Command: `source /home/guolijun/anaconda3/etc/profile.d/conda.sh && conda activate spider && bash examples/generate_scene_asm.sh`
  - Result: passed; regenerated `example_datasets/processed/oakink/assets/robots/asm/{bimanual,right,left}.xml`, plus `scene.xml`, `scene_eq.xml`, and `scene_act.xml` under `example_datasets/processed/oakink/asm/bimanual/pick_spoon_bowl/`.
  - Command: `source /home/guolijun/anaconda3/etc/profile.d/conda.sh && conda activate spider && MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa env -u LD_LIBRARY_PATH python spider/preprocess/ik.py --dataset-dir example_datasets --dataset-name oakink --robot-type asm --embodiment-type bimanual --task pick_spoon_bowl --data-id 0 --open-hand --end-idx 5 --save-video --no-show-viewer && MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa env -u LD_LIBRARY_PATH python spider/preprocess/ik.py --dataset-dir example_datasets --dataset-name oakink --robot-type asm --embodiment-type bimanual --task pick_spoon_bowl --data-id 0 --open-hand --act-scene --end-idx 5 --no-save-video --no-show-viewer`
  - Result: passed; saved `visualization_ik.mp4`, `trajectory_kinematic.npz`, `trajectory_ikrollout.npz`, `trajectory_kinematic_act.npz`, and `trajectory_ikrollout_act.npz` into `example_datasets/processed/oakink/asm/bimanual/pick_spoon_bowl/0/`.
  - Command: `source /home/guolijun/anaconda3/etc/profile.d/conda.sh && conda activate spider && MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa env -u LD_LIBRARY_PATH python examples/run_mjwp.py +override=oakink_asm dataset_dir=example_datasets dataset_name=oakink task=pick_spoon_bowl data_id=0 robot_type=asm embodiment_type=bimanual viewer=none show_viewer=false save_video=false save_info=true contact_guidance=true num_samples=8 max_num_iterations=4 max_sim_steps=5`
  - Result: passed as a short MJWP smoke test on GPU; saved `trajectory_mjwp_act.npz`, `config_act.yaml`, and `run_mjwp_act_20260424_194337.log` into `example_datasets/processed/oakink/asm/bimanual/pick_spoon_bowl/0/`. The pipeline itself is working with the new robot package, although the existing ASM MJWP instability remains (`NaNs or infs in rews` and `Final object tracking error: pos=nan, quat=nan`).

## Change 036 - Save two MJWP videos: helper-visualized and clean

- Goal: Keep the current MJWP debug video with helper geoms/contact visualization, while also writing a second clean video that hides auxiliary collision/debug geometry so the robot-object motion is easier to inspect.
- Files changed:
  - `spider/viewers/__init__.py`: lines 262-324 add `_make_render_options(include_helpers)` and extend `render_image(...)` so it can render either:
    - helper/debug view: contact points + contact forces on
    - clean view: helper geom groups and debug site groups hidden, contact overlays off
  - `examples/run_mjwp.py`: lines 670-692 now render two frame streams per saved frame (`images` and `images_clean`); lines 764-773 continue saving the original `visualization_mjwp*.mp4` and additionally save `visualization_mjwp_clean*.mp4`.
- Details:
  - The clean view hides:
    - `geomgroup[3]` and `geomgroup[4]`
    - `sitegroup[3]` and `sitegroup[4]`
    - `mjVIS_CONTACTPOINT`
    - `mjVIS_CONTACTFORCE`
  - This targets the exact helper content currently cluttering ASM MJWP videos:
    - `collision_hand_*` finger/palm proxies
    - ref/trace helper sites
    - contact-point/contact-force overlays
  - The original video naming is preserved, and the new additional file is:
    - `visualization_mjwp_clean.mp4`
    - or `visualization_mjwp_clean_act.mp4` when `contact_guidance=true`
- Validation:
  - Command: `python -m py_compile spider/viewers/__init__.py examples/run_mjwp.py`
  - Result: passed.
  - Command: inspect the updated render/save logic in `spider/viewers/__init__.py` and `examples/run_mjwp.py`.
  - Result: passed; the code now writes both the original helper-inclusive video and a second clean video path without changing the simulation/integration path.

## Change 037 - Add `ourdata/bottle` dataset conversion pipeline from RoboSimGS++ workspace outputs

- Goal: Adapt `preprocessed/bottle` into a first SPIDER-compatible custom dataset named `ourdata`, with task name `bottle`, right object slot active, left object slot empty, and outputs compatible with `generate_xml.py`, `ik.py`, and `run_mjwp.py`.
- Files changed:
  - `spider/process_datasets/ourdata.py`: lines 37-47 define fingertip vertex ids and object-tracking thresholds; lines 96-164 recover wrist/fingertip trajectories from `hawor/world_mocap.npz`; lines 224-293 estimate per-frame object translation from `masks + depth + cam_c2w`; lines 296-310 convert `sam3d/obj_3d_final.ply` into a usable `visual.obj` mesh via convex hull fallback when the source is only a point cloud; lines 313-464 write `trajectory_keypoints.npz`, `task_info.json`, `visual.obj`, and `conversion_debug.npz` under `example_datasets/processed/ourdata/...`.
  - `examples/process_ourdata_bottle.sh`: lines 1-32 run the new dataset processor and then `decompose_fast.py` to populate `convex/*.obj` and update `task_info.json`.
  - `examples/generate_scene_ourdata_bottle_asm.sh`: lines 1-55 regenerate ASM robot assets under `processed/ourdata/assets/robots/asm/` and then build `scene.xml`, `scene_eq.xml`, and `scene_act.xml` for `ourdata/bottle`.
  - `examples/run_ik_ourdata_bottle_asm.sh`: lines 1-30 run full-scene generation and then produce both standard and `_act` IK outputs for `ourdata/bottle`.
  - `examples/run_mjwp_ourdata_bottle_asm.sh`: lines 1-24 run MJWP on the new dataset using `contact_guidance=false`, because the first `ourdata` conversion currently does not provide valid contact labels.
- Details:
  - The first conversion intentionally keeps the hand and object pipelines simple and traceable:
    - fingertips use standard MANO vertex ids `744/320/443/554/671`
    - wrist pose is estimated from hand mesh geometry via PCA and fingertip directions
    - object rotation is fixed from `result/box_for_spider.npz`
    - object translation is tracked frame-by-frame from the best mask/depth component near the projected previous center
  - `trajectory_keypoints.npz` writes `contact_left/right` as zero masks but omits `contact_pos_*` so that current `generate_xml.py` cleanly falls back to zero contact-site offsets instead of misinterpreting a `(T, 5, 3)` tensor as a single-frame `(5, 3)` array.
  - `sam3d/obj_3d_final.ply` in `preprocessed/bottle` is a point cloud rather than a triangle mesh, so `ourdata.py` now converts it to `visual.obj` by taking the convex hull. This is good enough for a first integrated pipeline, but it is still a geometric approximation rather than a faithful surface reconstruction.
  - The resulting `ourdata` dataset is structurally valid and fully loadable by SPIDER. However, the converted hand/object trajectories are still far apart in world coordinates (for example, mean right-palm-to-object distance is about `0.56m`), so the semantic alignment of this first conversion will likely need another pass before the custom data is useful for grasp-quality experiments.
- Validation:
  - Command: `python -m py_compile spider/process_datasets/ourdata.py && bash -n examples/process_ourdata_bottle.sh examples/generate_scene_ourdata_bottle_asm.sh examples/run_ik_ourdata_bottle_asm.sh examples/run_mjwp_ourdata_bottle_asm.sh`
  - Result: passed.
  - Command: `source /home/guolijun/anaconda3/etc/profile.d/conda.sh && conda activate spider && bash examples/process_ourdata_bottle.sh`
  - Result: passed; generated:
    - `example_datasets/processed/ourdata/mano/bimanual/bottle/0/trajectory_keypoints.npz`
    - `example_datasets/processed/ourdata/mano/bimanual/bottle/0/conversion_debug.npz`
    - `example_datasets/processed/ourdata/mano/bimanual/bottle/task_info.json`
    - `example_datasets/processed/ourdata/assets/objects/bottle/visual.obj`
    - `example_datasets/processed/ourdata/assets/objects/bottle/convex/*.obj`
  - Command: inspect `example_datasets/processed/ourdata/mano/bimanual/bottle/0/trajectory_keypoints.npz`
  - Result: passed; shapes are:
    - `qpos_wrist_right/left: (185, 7)`
    - `qpos_finger_right/left: (185, 5, 7)`
    - `qpos_obj_right: (185, 7)`
    - `qpos_obj_left: (185, 7)` with the left slot empty (`xyz=0`, identity quaternion)
    - `contact_right/left: (185, 5)`
  - Command: `source /home/guolijun/anaconda3/etc/profile.d/conda.sh && conda activate spider && bash examples/generate_scene_ourdata_bottle_asm.sh`
  - Result: passed; generated:
    - `example_datasets/processed/ourdata/assets/robots/asm/{bimanual,right,left}.xml`
    - `example_datasets/processed/ourdata/asm/bimanual/bottle/scene.xml`
    - `example_datasets/processed/ourdata/asm/bimanual/bottle/scene_eq.xml`
    - `example_datasets/processed/ourdata/asm/bimanual/bottle/scene_act.xml`
  - Command: `source /home/guolijun/anaconda3/etc/profile.d/conda.sh && conda activate spider && MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa env -u LD_LIBRARY_PATH python spider/preprocess/ik.py --dataset-dir example_datasets --dataset-name ourdata --robot-type asm --embodiment-type bimanual --task bottle --data-id 0 --open-hand --end-idx 5 --no-save-video --no-show-viewer && MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa env -u LD_LIBRARY_PATH python spider/preprocess/ik.py --dataset-dir example_datasets --dataset-name ourdata --robot-type asm --embodiment-type bimanual --task bottle --data-id 0 --open-hand --act-scene --end-idx 5 --no-save-video --no-show-viewer`
  - Result: passed; generated short-horizon `trajectory_kinematic.npz`, `trajectory_ikrollout.npz`, `trajectory_kinematic_act.npz`, and `trajectory_ikrollout_act.npz` for `ourdata/bottle`.
  - Command: `source /home/guolijun/anaconda3/etc/profile.d/conda.sh && conda activate spider && MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa env -u LD_LIBRARY_PATH python examples/run_mjwp.py +override=oakink_asm dataset_dir=example_datasets dataset_name=ourdata task=bottle data_id=0 robot_type=asm embodiment_type=bimanual viewer=none show_viewer=false save_video=false save_info=true contact_guidance=false num_samples=8 max_num_iterations=4 max_sim_steps=5`
  - Result: passed on GPU as a short MJWP smoke test; generated `example_datasets/processed/ourdata/asm/bimanual/bottle/0/trajectory_mjwp.npz`, `config.yaml`, and `run_mjwp_20260425_232142.log`. Final short-horizon object tracking error was finite (`pos=0.0039`, `quat=0.0122`).

## Change 038 - Rename `ourdata` task from `bottom` to `bottle`

- Goal: Fix the custom dataset task name so every `ourdata` reference uses `bottle` instead of the mistaken `bottom`.
- Files changed:
  - `spider/process_datasets/ourdata.py`: line 317 changes the default task name from `bottom` to `bottle`.
  - `examples/process_ourdata_bottle.sh`: lines 4-32 update output comments and both `--task` arguments to `bottle`.
  - `examples/generate_scene_ourdata_bottle_asm.sh`: lines 4-55 update output comments, script chaining, and both `--task` arguments to `bottle`.
  - `examples/run_ik_ourdata_bottle_asm.sh`: lines 7-29 update the upstream script name and both `--task` arguments to `bottle`.
  - `examples/run_mjwp_ourdata_bottle_asm.sh`: lines 7-24 update the upstream script name and `task=bottle` override.
  - `asm_revised.md`: updated prior `ourdata` change notes to use `bottle` consistently.
- Details:
  - Renamed script files:
    - `examples/process_ourdata_bottom.sh` -> `examples/process_ourdata_bottle.sh`
    - `examples/generate_scene_ourdata_bottom_asm.sh` -> `examples/generate_scene_ourdata_bottle_asm.sh`
    - `examples/run_ik_ourdata_bottom_asm.sh` -> `examples/run_ik_ourdata_bottle_asm.sh`
    - `examples/run_mjwp_ourdata_bottom_asm.sh` -> `examples/run_mjwp_ourdata_bottle_asm.sh`
  - Moved generated dataset directories:
    - `example_datasets/processed/ourdata/mano/bimanual/bottom` -> `.../bottle`
    - `example_datasets/processed/ourdata/asm/bimanual/bottom` -> `.../bottle`
  - Rewrote generated metadata so `task_info.json`, `config.yaml`, and existing text logs under `processed/ourdata` now consistently use `bottle`.
- Validation:
  - Command: `bash -n examples/process_ourdata_bottle.sh examples/generate_scene_ourdata_bottle_asm.sh examples/run_ik_ourdata_bottle_asm.sh examples/run_mjwp_ourdata_bottle_asm.sh`
  - Result: passed.
  - Command: `source /home/guolijun/anaconda3/etc/profile.d/conda.sh && conda activate spider && bash examples/process_ourdata_bottle.sh`
  - Result: passed; regenerated `trajectory_keypoints.npz`, `conversion_debug.npz`, and `task_info.json` under `example_datasets/processed/ourdata/mano/bimanual/bottle/0/`.
  - Command: `source /home/guolijun/anaconda3/etc/profile.d/conda.sh && conda activate spider && bash examples/generate_scene_ourdata_bottle_asm.sh`
  - Result: passed; regenerated `scene.xml`, `scene_eq.xml`, and `scene_act.xml` under `example_datasets/processed/ourdata/asm/bimanual/bottle/`.
  - Command: `rg -n "bottom" examples spider/process_datasets/ourdata.py asm_revised.md example_datasets/processed/ourdata`
  - Result: no matches; all `ourdata`-related references are now `bottle`.
