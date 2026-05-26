#!/usr/bin/env python3
"""Render two-object ourdata hand reference keypoints in the MuJoCo scene.

This script is intentionally independent from IK. It renders the processed
input hand/object references with the ASM robot loaded, so the reference video
is still available when collision-enabled IK fails before it can save a video.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio
import mujoco
import numpy as np

from spider.io import get_processed_data_dir


RIGHT_RGBA = [1.0, 0.0, 0.0, 0.95]
LEFT_RGBA = [0.1, 0.35, 1.0, 0.95]
PALM_SIZE = [0.018, 0.028, 0.038]
TIP_SIZE = [0.014, 0.0, 0.0]
DEBUG_GROUP = 5
FINGER_NAMES = ["thumb", "index", "middle", "ring", "pinky"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="example_datasets")
    parser.add_argument("--dataset-name", default="ourdata")
    parser.add_argument("--robot-type", default="asm")
    parser.add_argument("--embodiment-type", default="bimanual")
    parser.add_argument("--task", default="pick_place")
    parser.add_argument("--data-id", type=int, default=0)
    parser.add_argument("--camera", default="front")
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=0.0)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=-1)
    parser.add_argument(
        "--output-path",
        default="",
        help="Default: processed asm task dir / visualization_ref_keypoints_mujoco.mp4",
    )
    return parser.parse_args()


def _load_ref_dt(dataset_dir: str, dataset_name: str, embodiment_type: str, task: str) -> float:
    task_info_path = (
        Path(dataset_dir)
        / "processed"
        / dataset_name
        / "mano"
        / embodiment_type
        / task
        / "task_info.json"
    )
    if not task_info_path.exists():
        return 1.0 / 30.0
    with open(task_info_path, "r", encoding="utf-8") as f:
        task_info = json.load(f)
    return float(task_info.get("ref_dt", 1.0 / 30.0))


def _add_marker_body(spec: mujoco.MjSpec, name: str, rgba: list[float], is_palm: bool) -> None:
    body = spec.worldbody.add_body(name=name, mocap=True)
    if is_palm:
        body.add_geom(
            name=f"{name}_geom",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=PALM_SIZE,
            rgba=rgba,
            group=DEBUG_GROUP,
            contype=0,
            conaffinity=0,
            density=0,
        )
    else:
        body.add_geom(
            name=f"{name}_geom",
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=TIP_SIZE,
            rgba=rgba,
            group=DEBUG_GROUP,
            contype=0,
            conaffinity=0,
            density=0,
        )


def _add_ref_markers(spec: mujoco.MjSpec) -> list[str]:
    names: list[str] = []
    for side, rgba in [("right", RIGHT_RGBA), ("left", LEFT_RGBA)]:
        palm_name = f"ref_{side}_palm"
        _add_marker_body(spec, palm_name, rgba, is_palm=True)
        names.append(palm_name)
        for finger_name in FINGER_NAMES:
            tip_name = f"ref_{side}_{finger_name}"
            _add_marker_body(spec, tip_name, rgba, is_palm=False)
            names.append(tip_name)
    return names


def _mocap_ids(model: mujoco.MjModel, names: list[str]) -> dict[str, int]:
    ids: dict[str, int] = {}
    for name in names:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id < 0:
            raise RuntimeError(f"Missing mocap body: {name}")
        mocap_id = int(model.body_mocapid[body_id])
        if mocap_id < 0:
            raise RuntimeError(f"Body is not mocap: {name}")
        ids[name] = mocap_id
    return ids


def _set_free_joint_qpos(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint_name: str,
    qpos: np.ndarray,
) -> None:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        return
    adr = int(model.jnt_qposadr[joint_id])
    data.qpos[adr : adr + 7] = qpos


def _set_ref_markers(
    data: mujoco.MjData,
    ids: dict[str, int],
    qpos_wrist_right: np.ndarray,
    qpos_finger_right: np.ndarray,
    qpos_wrist_left: np.ndarray,
    qpos_finger_left: np.ndarray,
    frame_idx: int,
) -> None:
    for side, qpos_wrist, qpos_finger in [
        ("right", qpos_wrist_right, qpos_finger_right),
        ("left", qpos_wrist_left, qpos_finger_left),
    ]:
        palm_id = ids[f"ref_{side}_palm"]
        data.mocap_pos[palm_id] = qpos_wrist[frame_idx, :3]
        data.mocap_quat[palm_id] = qpos_wrist[frame_idx, 3:7]
        for finger_idx, finger_name in enumerate(FINGER_NAMES):
            mocap_id = ids[f"ref_{side}_{finger_name}"]
            data.mocap_pos[mocap_id] = qpos_finger[frame_idx, finger_idx, :3]
            data.mocap_quat[mocap_id] = qpos_finger[frame_idx, finger_idx, 3:7]


def _render_options() -> mujoco.MjvOption:
    options = mujoco.MjvOption()
    mujoco.mjv_defaultOption(options)
    options.geomgroup[DEBUG_GROUP] = True
    options.sitegroup[DEBUG_GROUP] = True
    return options


def main() -> None:
    args = _parse_args()
    if args.frame_stride < 1:
        raise ValueError("--frame-stride must be >= 1")

    robot_dir = Path(
        get_processed_data_dir(
            args.dataset_dir,
            args.dataset_name,
            args.robot_type,
            args.embodiment_type,
            args.task,
            args.data_id,
        )
    )
    mano_dir = Path(
        get_processed_data_dir(
            args.dataset_dir,
            args.dataset_name,
            "mano",
            args.embodiment_type,
            args.task,
            args.data_id,
        )
    )
    scene_path = robot_dir.parent / "scene.xml"
    traj_path = mano_dir / "trajectory_keypoints.npz"
    output_path = (
        Path(args.output_path)
        if args.output_path
        else robot_dir / "visualization_ref_keypoints_mujoco.mp4"
    )
    if not scene_path.exists():
        raise FileNotFoundError(f"Scene XML not found: {scene_path}")
    if not traj_path.exists():
        raise FileNotFoundError(f"Trajectory keypoints not found: {traj_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading scene: {scene_path}")
    print(f"Loading keypoints: {traj_path}")
    spec = mujoco.MjSpec.from_file(str(scene_path))
    marker_names = _add_ref_markers(spec)
    model = spec.compile()
    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), int(args.width))
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), int(args.height))
    data = mujoco.MjData(model)

    # This debug render shows reference geometry only; disable contacts so it
    # cannot fail before recording the target hand/object positions.
    model.geom_contype[:] = 0
    model.geom_conaffinity[:] = 0

    ref = np.load(traj_path)
    qpos_wrist_right = ref["qpos_wrist_right"].astype(np.float64)
    qpos_finger_right = ref["qpos_finger_right"].astype(np.float64)
    qpos_obj_right = ref["qpos_obj_right"].astype(np.float64)
    qpos_wrist_left = ref["qpos_wrist_left"].astype(np.float64)
    qpos_finger_left = ref["qpos_finger_left"].astype(np.float64)
    qpos_obj_left = ref["qpos_obj_left"].astype(np.float64)

    num_frames = min(
        len(qpos_wrist_right),
        len(qpos_finger_right),
        len(qpos_obj_right),
        len(qpos_wrist_left),
        len(qpos_finger_left),
        len(qpos_obj_left),
    )
    frame_indices = np.arange(0, num_frames, args.frame_stride, dtype=np.int64)
    if args.max_frames > 0:
        frame_indices = frame_indices[: args.max_frames]
    fps = args.fps if args.fps > 0 else 1.0 / _load_ref_dt(
        args.dataset_dir, args.dataset_name, args.embodiment_type, args.task
    )

    marker_ids = _mocap_ids(model, marker_names)
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    options = _render_options()

    print(
        f"Rendering {len(frame_indices)} frames at {args.width}x{args.height}, "
        f"fps={fps:.3f}, camera={args.camera}, output={output_path}"
    )
    with imageio.get_writer(output_path, fps=fps) as writer:
        for out_idx, frame_idx in enumerate(frame_indices):
            data.qpos[:] = model.qpos0
            _set_free_joint_qpos(model, data, "right_object_joint", qpos_obj_right[frame_idx])
            _set_free_joint_qpos(model, data, "left_object_joint", qpos_obj_left[frame_idx])
            _set_ref_markers(
                data,
                marker_ids,
                qpos_wrist_right,
                qpos_finger_right,
                qpos_wrist_left,
                qpos_finger_left,
                int(frame_idx),
            )
            mujoco.mj_forward(model, data)
            try:
                renderer.update_scene(data, camera=args.camera, scene_option=options)
            except Exception:
                renderer.update_scene(data, camera=0, scene_option=options)
            writer.append_data(renderer.render())
            if out_idx == 0 or (out_idx + 1) % 100 == 0 or out_idx == len(frame_indices) - 1:
                print(f"Rendered {out_idx + 1}/{len(frame_indices)} frames")
    renderer.close()
    print(f"Saved reference keypoint video: {output_path}")


if __name__ == "__main__":
    main()
