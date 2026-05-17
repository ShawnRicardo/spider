# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Run IK for the given hand type and mode.

Input data format: npz file which contains qpos for key frames.

TODO: for enable collision, first use non collision as initial guess

Author: Chaoyi Pan
Date: 2025-07-06
"""

import json
import os
import time
from collections import Counter
from pathlib import Path

import loguru
import mujoco
import mujoco.viewer
import numpy as np
import tyro
from loop_rate_limiters import RateLimiter
from mujoco import MjSpec
from omegaconf import DictConfig, OmegaConf
from scipy import signal

from spider import ROOT
from spider.io import get_processed_data_dir
from spider.mujoco_utils import get_viewer
from spider.preprocess.workspace_support import (
    compute_workspace_support_spec,
    workspace_support_spec_from_task_info,
)
from spider.viewers import viser_viewer

DEBUG_GEOM_GROUP = 5
HEAD_CAMERA_FRAME_SITE_NAME = "head_camera_frame"
D435_OPTICAL_FRAME_SITE_NAME = "d435_optical_frame"
RIGHT_HAND_TARGET_RGBA = [1.0, 0.0, 0.0, 0.95]
LEFT_HAND_TARGET_RGBA = [0.15, 0.45, 1.0, 0.95]
OBJECT_TARGET_RGBA = [0.0, 1.0, 0.0, 0.6]
OBJECT_BBOX_RGBA = [1.0, 0.9, 0.0, 0.22]
OBJECT_CENTER_RGBA = [0.0, 1.0, 0.2, 0.95]
AXIS_X_RGBA = [1.0, 0.0, 0.0, 0.95]
AXIS_Y_RGBA = [0.0, 1.0, 0.0, 0.95]
AXIS_Z_RGBA = [0.0, 0.4, 1.0, 0.95]
AXIS_ORIGIN_RGBA = [1.0, 1.0, 1.0, 1.0]
LEGACY_CAMERA_MARKER_RGBA = [1.0, 0.0, 1.0, 0.55]
OPTICAL_CAMERA_MARKER_RGBA = [1.0, 0.45, 0.0, 0.95]


def _normalize_quat(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    norm = np.linalg.norm(quat)
    if norm < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return quat / norm


def _site_rgba_from_name(site_name: str, is_object_target: bool) -> list[float]:
    if is_object_target:
        return OBJECT_TARGET_RGBA
    if "left_" in site_name:
        return LEFT_HAND_TARGET_RGBA
    return RIGHT_HAND_TARGET_RGBA


def _add_axis_geoms(
    body_handle,
    prefix: str,
    axis_length: float,
    axis_thickness: float,
    group: int,
    origin_rgba: list[float] | None = None,
) -> None:
    half_length = 0.5 * axis_length
    thickness = axis_thickness
    body_handle.add_geom(
        name=f"{prefix}_origin",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        pos=[0.0, 0.0, 0.0],
        size=[max(axis_thickness * 3.0, 0.008)],
        rgba=AXIS_ORIGIN_RGBA if origin_rgba is None else origin_rgba,
        group=group,
        contype=0,
        conaffinity=0,
        density=0,
    )
    body_handle.add_geom(
        name=f"{prefix}_x",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[half_length, 0.0, 0.0],
        size=[half_length, thickness, thickness],
        rgba=AXIS_X_RGBA,
        group=group,
        contype=0,
        conaffinity=0,
        density=0,
    )
    body_handle.add_geom(
        name=f"{prefix}_y",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[0.0, half_length, 0.0],
        size=[thickness, half_length, thickness],
        rgba=AXIS_Y_RGBA,
        group=group,
        contype=0,
        conaffinity=0,
        density=0,
    )
    body_handle.add_geom(
        name=f"{prefix}_z",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[0.0, 0.0, half_length],
        size=[thickness, thickness, half_length],
        rgba=AXIS_Z_RGBA,
        group=group,
        contype=0,
        conaffinity=0,
        density=0,
    )


def _add_static_axes_body(
    mjspec: MjSpec,
    body_name: str,
    axis_length: float,
    axis_thickness: float,
    pos: list[float] | None = None,
    quat: list[float] | None = None,
) -> None:
    body_handle = mjspec.worldbody.add_body(
        name=body_name,
        pos=[0.0, 0.0, 0.0] if pos is None else pos,
        quat=[1.0, 0.0, 0.0, 0.0] if quat is None else quat,
        mocap=False,
    )
    _add_axis_geoms(
        body_handle,
        prefix=body_name,
        axis_length=axis_length,
        axis_thickness=axis_thickness,
        group=DEBUG_GEOM_GROUP,
    )


def _add_debug_object_body(
    mjspec: MjSpec,
    body_name: str,
    half_extents: np.ndarray,
    add_bbox: bool,
    add_axes: bool,
) -> None:
    body_handle = mjspec.worldbody.add_body(name=body_name, mocap=True)
    if add_bbox:
        body_handle.add_geom(
            name=f"{body_name}_bbox",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[0.0, 0.0, 0.0],
            size=half_extents.tolist(),
            rgba=OBJECT_BBOX_RGBA,
            group=DEBUG_GEOM_GROUP,
            contype=0,
            conaffinity=0,
            density=0,
        )
    body_handle.add_geom(
        name=f"{body_name}_center",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        pos=[0.0, 0.0, 0.0],
        size=[0.012],
        rgba=OBJECT_CENTER_RGBA,
        group=DEBUG_GEOM_GROUP,
        contype=0,
        conaffinity=0,
        density=0,
    )
    if add_axes:
        axis_length = max(float(np.max(half_extents)) * 1.5, 0.08)
        axis_thickness = max(axis_length * 0.06, 0.003)
        _add_axis_geoms(
            body_handle,
            prefix=f"{body_name}_axes",
            axis_length=axis_length,
            axis_thickness=axis_thickness,
            group=DEBUG_GEOM_GROUP,
        )


def _add_debug_camera_body(
    mjspec: MjSpec,
    body_name: str,
    marker_rgba: list[float],
    marker_size: list[float],
    axis_length: float = 0.14,
    axis_thickness: float = 0.0035,
    show_axes: bool = True,
) -> None:
    body_handle = mjspec.worldbody.add_body(name=body_name, mocap=True)
    body_handle.add_geom(
        name=f"{body_name}_center",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[max(marker_size)],
        rgba=marker_rgba,
        group=DEBUG_GEOM_GROUP,
        contype=0,
        conaffinity=0,
        density=0,
    )
    if not show_axes:
        return
    _add_axis_geoms(
        body_handle,
        prefix=f"{body_name}_axes",
        axis_length=axis_length,
        axis_thickness=axis_thickness,
        group=DEBUG_GEOM_GROUP,
        origin_rgba=marker_rgba,
    )


def _body_mocap_id(model: mujoco.MjModel, body_name: str) -> int:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id == -1:
        return -1
    return int(model.body_mocapid[body_id])


def _find_robot_base_body_id(model: mujoco.MjModel) -> int:
    for body_id in range(1, model.nbody):
        if int(model.body_parentid[body_id]) != 0:
            continue
        if int(model.body_mocapid[body_id]) != -1:
            continue
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if body_name is None:
            continue
        if (
            body_name.startswith("target_mocap_body_")
            or body_name.startswith("ref_")
            or body_name.startswith("debug_")
            or "object" in body_name
        ):
            continue
        return body_id
    return 1


def _load_object_debug_specs(
    dataset_name: str,
    task_info_path: str,
) -> dict[str, np.ndarray]:
    with open(task_info_path) as f:
        task_info = json.load(f)

    specs: dict[str, np.ndarray] = {}
    for side in ["right", "left"]:
        size_key = f"{side}_object_bbox_size_xyz_m"
        size_value = task_info.get(size_key)
        if size_value is not None:
            specs[side] = np.asarray(size_value, dtype=np.float64)

    if len(specs) > 0:
        return specs

    if dataset_name != "ourdata":
        return specs

    source_workspace = task_info.get("source_workspace")
    if source_workspace is None:
        return specs
    box_npz_path = Path(source_workspace) / "result" / "box_for_spider.npz"
    if not box_npz_path.exists():
        return specs
    box_data = np.load(box_npz_path)
    if "box_real_size_xyz_m" in box_data:
        specs["right"] = np.asarray(box_data["box_real_size_xyz_m"], dtype=np.float64)
    return specs


def _uses_camera_world_alignment(task_info: dict) -> bool:
    return task_info.get("world_to_sim_alignment_mode") == "d435_optical"


def _set_debug_body_pose(
    data: mujoco.MjData,
    mocap_id: int,
    pos: np.ndarray,
    quat_wxyz: np.ndarray,
) -> None:
    if mocap_id == -1:
        return
    data.mocap_pos[mocap_id] = np.asarray(pos, dtype=np.float64)
    data.mocap_quat[mocap_id] = _normalize_quat(quat_wxyz)


def _quat_from_xmat(xmat: np.ndarray) -> np.ndarray:
    quat = np.zeros(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quat, np.asarray(xmat, dtype=np.float64).reshape(9))
    return _normalize_quat(quat)


def _quat_angle_error(q1: np.ndarray, q2: np.ndarray) -> float:
    q1 = _normalize_quat(q1)
    q2 = _normalize_quat(q2)
    dot = float(np.clip(abs(np.dot(q1, q2)), 0.0, 1.0))
    return float(2.0 * np.arccos(dot))


def _geom_name(model: mujoco.MjModel, geom_id: int) -> str:
    if geom_id < 0:
        return "none"
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(geom_id))
    return name or f"geom_{int(geom_id)}"


def _contact_category(geom_name: str) -> str:
    if geom_name == "floor":
        return "floor"
    if geom_name == "support_table" or "table" in geom_name:
        return "table"
    if "object" in geom_name or geom_name.startswith("milk"):
        return "object"
    if geom_name.startswith("collision_"):
        return "robot"
    return "other"


def _contact_pair_category(geom1: str, geom2: str) -> str:
    categories = sorted([_contact_category(geom1), _contact_category(geom2)])
    return f"{categories[0]}--{categories[1]}"


def _summarize_contacts(model: mujoco.MjModel, data: mujoco.MjData) -> dict:
    ncon = int(data.ncon)
    category_counts: Counter[str] = Counter()
    negative_count = 0
    min_dist = None
    deepest_pair = None
    deepest_position = None
    for contact_idx in range(ncon):
        contact = data.contact[contact_idx]
        geom1 = _geom_name(model, contact.geom1)
        geom2 = _geom_name(model, contact.geom2)
        category_counts[_contact_pair_category(geom1, geom2)] += 1
        dist = float(contact.dist)
        if dist < 0.0:
            negative_count += 1
        if min_dist is None or dist < min_dist:
            min_dist = dist
            deepest_pair = [geom1, geom2]
            deepest_position = [float(v) for v in contact.pos]
    return {
        "ncon": ncon,
        "negative_contact_count": negative_count,
        "min_contact_dist": min_dist,
        "deepest_pair": deepest_pair,
        "deepest_position": deepest_position,
        "contact_category_counts": dict(category_counts),
    }


def _target_kind(site_name: str) -> str:
    if "object" in site_name:
        return "object"
    if "palm" in site_name:
        return "palm"
    return "fingertip"


def _stats(values: list[float]) -> dict[str, float | None]:
    if len(values) == 0:
        return {"mean": None, "max": None}
    values_np = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values_np)),
        "max": float(np.max(values_np)),
    }


def _compute_ik_collision_diagnostic(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    frame_idx: int,
    diagnostic_specs: list[tuple[str, int, int]],
    qpos_ref: np.ndarray,
) -> dict:
    contact_summary = _summarize_contacts(model, data)
    pos_errors_by_kind: dict[str, list[float]] = {
        "palm": [],
        "fingertip": [],
        "object": [],
    }
    rot_errors_by_kind: dict[str, list[float]] = {
        "palm": [],
        "object": [],
    }
    site_errors = {}

    for site_name, site_id, qpos_idx in diagnostic_specs:
        if site_id < 0:
            continue
        kind = _target_kind(site_name)
        target_pose = qpos_ref[frame_idx, qpos_idx]
        pos_err = float(np.linalg.norm(data.site_xpos[site_id] - target_pose[:3]))
        pos_errors_by_kind[kind].append(pos_err)
        site_entry = {"pos_err": pos_err}
        if kind in rot_errors_by_kind:
            site_quat = _quat_from_xmat(data.site_xmat[site_id])
            rot_err = _quat_angle_error(site_quat, target_pose[3:])
            rot_errors_by_kind[kind].append(rot_err)
            site_entry["rot_err"] = rot_err
        site_errors[site_name] = site_entry

    return {
        "frame": int(frame_idx),
        "sim_time": float(data.time),
        **contact_summary,
        "palm_pos_err": _stats(pos_errors_by_kind["palm"]),
        "fingertip_pos_err": _stats(pos_errors_by_kind["fingertip"]),
        "object_pos_err": _stats(pos_errors_by_kind["object"]),
        "palm_rot_err": _stats(rot_errors_by_kind["palm"]),
        "object_rot_err": _stats(rot_errors_by_kind["object"]),
        "site_errors": site_errors,
    }


def _format_optional_float(value: float | None) -> str:
    if value is None:
        return "none"
    return f"{value:.6f}"


def _write_ik_collision_diagnostics(
    output_dir: str,
    diagnostics: list[dict],
    *,
    failure: dict | None = None,
    suffix: str = "",
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    basename = f"ik_collision_diagnostics{suffix}"
    json_path = os.path.join(output_dir, f"{basename}.json")
    txt_path = os.path.join(output_dir, f"{basename}.txt")
    payload = {
        "failure": failure,
        "frames": diagnostics,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    lines = ["IK collision diagnostics", "=" * 80]
    if failure is not None:
        lines.append(f"failure: {failure}")
    lines.append(f"num_frames_recorded: {len(diagnostics)}")
    if diagnostics:
        min_dists = [
            d["min_contact_dist"]
            for d in diagnostics
            if d.get("min_contact_dist") is not None
        ]
        lines.append(
            "max_ncon: {}".format(max(int(d.get("ncon", 0)) for d in diagnostics))
        )
        if min_dists:
            lines.append(f"deepest_penetration: {min(min_dists):.6f}")
        lines.append("")
        lines.append(
            "frame sim_time ncon min_contact_dist deepest_pair "
            "palm_pos_max fingertip_pos_max object_pos_max palm_rot_max object_rot_max"
        )
        for item in diagnostics:
            pair = item.get("deepest_pair") or []
            pair_text = "<->".join(pair) if pair else "none"
            palm_pos = item.get("palm_pos_err", {}).get("max")
            fingertip_pos = item.get("fingertip_pos_err", {}).get("max")
            object_pos = item.get("object_pos_err", {}).get("max")
            palm_rot = item.get("palm_rot_err", {}).get("max")
            object_rot = item.get("object_rot_err", {}).get("max")
            lines.append(
                "{frame} {time:.4f} {ncon} {dist} {pair} {palm_pos} "
                "{finger_pos} {object_pos} {palm_rot} {object_rot}".format(
                    frame=item.get("frame", -1),
                    time=float(item.get("sim_time", 0.0)),
                    ncon=int(item.get("ncon", 0)),
                    dist=_format_optional_float(item.get("min_contact_dist")),
                    pair=pair_text,
                    palm_pos=_format_optional_float(palm_pos),
                    finger_pos=_format_optional_float(fingertip_pos),
                    object_pos=_format_optional_float(object_pos),
                    palm_rot=_format_optional_float(palm_rot),
                    object_rot=_format_optional_float(object_rot),
                )
            )
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    loguru.logger.info("Saved IK collision diagnostics to {} and {}", json_path, txt_path)


def _sync_debug_visualization_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    qpos_source: mujoco.MjData,
    frame_idx: int,
    qpos_obj_right: np.ndarray,
    qpos_obj_left: np.ndarray,
    debug_object_mocap_ids: dict[str, int],
    debug_robot_base_mocap_id: int,
    robot_base_body_id: int,
    debug_head_camera_mocap_id: int = -1,
    head_camera_site_id: int = -1,
    debug_optical_camera_mocap_id: int = -1,
    optical_camera_site_id: int = -1,
) -> None:
    data.qpos[:] = qpos_source.qpos.copy()
    data.qvel[:] = qpos_source.qvel.copy()
    if "right" in debug_object_mocap_ids:
        _set_debug_body_pose(
            data,
            debug_object_mocap_ids["right"],
            qpos_obj_right[frame_idx, :3],
            qpos_obj_right[frame_idx, 3:],
        )
    if "left" in debug_object_mocap_ids:
        _set_debug_body_pose(
            data,
            debug_object_mocap_ids["left"],
            qpos_obj_left[frame_idx, :3],
            qpos_obj_left[frame_idx, 3:],
        )
    mujoco.mj_forward(model, data)
    needs_second_forward = False
    if debug_robot_base_mocap_id != -1 and robot_base_body_id != -1:
        _set_debug_body_pose(
            data,
            debug_robot_base_mocap_id,
            data.xpos[robot_base_body_id],
            data.xquat[robot_base_body_id],
        )
        needs_second_forward = True
    if debug_head_camera_mocap_id != -1 and head_camera_site_id != -1:
        _set_debug_body_pose(
            data,
            debug_head_camera_mocap_id,
            data.site_xpos[head_camera_site_id],
            _quat_from_xmat(data.site_xmat[head_camera_site_id]),
        )
        needs_second_forward = True
    if debug_optical_camera_mocap_id != -1 and optical_camera_site_id != -1:
        _set_debug_body_pose(
            data,
            debug_optical_camera_mocap_id,
            data.site_xpos[optical_camera_site_id],
            _quat_from_xmat(data.site_xmat[optical_camera_site_id]),
        )
        needs_second_forward = True
    if needs_second_forward:
        mujoco.mj_forward(model, data)

# 给每个 target mocap body 添加一个 mocap body，并添加约束让它跟对应的 site 运动一致
def add_mocap_bodies(
    mjspec: MjSpec,
    sites_for_mimic: list[str],
    mocap_bodies: list[str],
    robot_conf: DictConfig = None,
    add_equality_constraint: bool = True,
    visualize_hand_keypoints: bool = False,
    visualize_target_geoms: bool = False,
):
    """Add mocap bodies to the model specification.
    Source: https://github.com/robfiras/loco-mujoco

    Args:
        mjspec (MjSpec): The model specification.
        sites_for_mimic (List[str]): The sites to mimic.
        mocap_bodies (List[str]): The names of the mocap bodies to be added to the model specification.
        mocap_bodies_init_pos: The initial positions of the mocap bodies.
        add_equality_constraint (bool): Whether to add equality constraints between the sites and the mocap bodies.

    """
    if robot_conf is not None and robot_conf.optimization_params.disable_joint_limits:
        for j in mjspec.joints:
            j.limited = False

    for j in mjspec.joints:
        j.actfrclimited = 0

    if robot_conf is not None and robot_conf.optimization_params.disable_collisions:
        for g in mjspec.geoms:
            g.contype = 0
            g.conaffinity = 0

    for mb_name in mocap_bodies:
        b_handle = mjspec.worldbody.add_body(name=mb_name, mocap=True)
        is_object_target = "object" in mb_name
        is_hand_target = not is_object_target
        site_group = 4 if visualize_hand_keypoints else 1
        rgba = _site_rgba_from_name(mb_name, is_object_target=is_object_target)
        show_target_geom = visualize_target_geoms or (
            visualize_hand_keypoints and is_hand_target
        )
        if show_target_geom:
            if "palm" in mb_name or is_object_target:
                b_handle.add_geom(
                    name=f"{mb_name}_geom",
                    type=mujoco.mjtGeom.mjGEOM_BOX,
                    size=[0.01, 0.02, 0.03],
                    rgba=rgba,
                    group=DEBUG_GEOM_GROUP,
                    contype=0,
                    conaffinity=0,
                    density=0,
                )
            else:
                b_handle.add_geom(
                    name=f"{mb_name}_geom",
                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    size=[0.012, 0.0, 0.0],
                    rgba=rgba,
                    group=DEBUG_GEOM_GROUP,
                    contype=0,
                    conaffinity=0,
                    density=0,
                )
        if visualize_hand_keypoints and is_hand_target:
            b_handle.add_site(
                name=mb_name,
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[0.012, 0.012, 0.012],
                rgba=rgba,
                group=site_group,
            )
        elif "palm" in mb_name or is_object_target:
            b_handle.add_site(
                name=mb_name,
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[0.01, 0.02, 0.03],
                rgba=rgba,
                group=site_group,
            )
        else:
            b_handle.add_site(
                name=mb_name,
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[0.01, 0.01, 0.01],
                rgba=rgba,
                group=site_group,
            )

    if add_equality_constraint:
        for b1, b2 in zip(sites_for_mimic, mocap_bodies, strict=False):
            if robot_conf is not None:
                eq_type = getattr(
                    mujoco.mjtEq,
                    robot_conf.site_joint_matches[b1].equality_constraint_type,
                )
                torque_scale = robot_conf.site_joint_matches[b1].torque_scale
            else:
                eq_type = mujoco.mjtEq.mjEQ_CONNECT
                torque_scale = 1.0

            constraint_data = np.zeros(11)
            constraint_data[10] = torque_scale
            e = mjspec.add_equality(
                name=f"{b1}_{b2}_equality_constraint",
                type=eq_type,
                name1=b1,
                name2=b2,
                objtype=mujoco.mjtObj.mjOBJ_SITE,
                data=constraint_data,
            )

            if robot_conf is not None:
                if hasattr(robot_conf.site_joint_matches[b1], "solref"):
                    test = len(robot_conf.site_joint_matches[b1].solref)
                    assert len(robot_conf.site_joint_matches[b1].solref) == 2, (
                        "solref must be a list of length 2"
                    )
                    e.solref = robot_conf.site_joint_matches[b1].solref
                if hasattr(robot_conf.site_joint_matches[b1], "solimp"):
                    assert len(robot_conf.site_joint_matches[b1].solimp) == 5, (
                        "solimp must be a list of length 5"
                    )
                    e.solimp = robot_conf.site_joint_matches[b1].solimp

    return mjspec

# 这个函数定义机器人侧语义顺序，不同 robot_type 有不同的 site 定义
def get_robot_sites(robot_type: str, embodiment_type: str):
    if robot_type in ["allegro", "metahand"]:   # 这两个机器人只有四个指头
        sites_in_robot = [
            "right_palm",
            "right_index_tip",
            "right_middle_tip",
            "right_ring_tip",
            "right_thumb_tip",
            "left_palm",
            "left_ring_tip",
            "left_middle_tip",
            "left_index_tip",
            "left_thumb_tip",
            "right_object",
            "left_object",
        ]
    else:
        sites_in_robot = [
            "right_palm",
            "right_thumb_tip",
            "right_index_tip",
            "right_middle_tip",
            "right_ring_tip",
            "right_pinky_tip",
            "left_palm",
            "left_thumb_tip",
            "left_index_tip",
            "left_middle_tip",
            "left_ring_tip",
            "left_pinky_tip",
            "right_object",
            "left_object",
        ]
    # 如果只有单手就加载单手
    if embodiment_type == "right":
        sites_in_robot = [s for s in sites_in_robot if "right" in s]
    elif embodiment_type == "left":
        sites_in_robot = [s for s in sites_in_robot if "left" in s]
    return sites_in_robot


def get_nq_obj(embodiment_type: str, act_scene: bool = False) -> int:
    if act_scene:
        return 0
    return 14 if embodiment_type == "bimanual" else 7


def set_object_qpos(
    qpos: np.ndarray,
    embodiment_type: str,
    qpos_obj_right_step: np.ndarray,
    qpos_obj_left_step: np.ndarray,
    act_scene: bool = False,
) -> None:
    if act_scene:
        return
    if embodiment_type == "bimanual":
        qpos[-14:-7] = qpos_obj_right_step
        qpos[-7:] = qpos_obj_left_step
    elif embodiment_type == "right":
        qpos[-7:] = qpos_obj_right_step
    elif embodiment_type == "left":
        qpos[-7:] = qpos_obj_left_step


def clip_robot_qpos_to_joint_ranges(
    model: mujoco.MjModel, qpos: np.ndarray, robot_qpos_dim: int
) -> None:
    scalar_joint_types = {
        int(mujoco.mjtJoint.mjJNT_SLIDE),
        int(mujoco.mjtJoint.mjJNT_HINGE),
    }
    for joint_id in range(model.njnt):
        qpos_adr = int(model.jnt_qposadr[joint_id])
        if qpos_adr >= robot_qpos_dim:
            continue
        if int(model.jnt_type[joint_id]) not in scalar_joint_types:
            continue
        if not bool(model.jnt_limited[joint_id]):
            continue
        low, high = model.jnt_range[joint_id]
        qpos[qpos_adr] = np.clip(qpos[qpos_adr], low, high)


def neutral_robot_qpos(model: mujoco.MjModel, robot_qpos_dim: int) -> np.ndarray:
    qpos = model.qpos0[:robot_qpos_dim].copy()
    scalar_joint_types = {
        int(mujoco.mjtJoint.mjJNT_SLIDE),
        int(mujoco.mjtJoint.mjJNT_HINGE),
    }
    for joint_id in range(model.njnt):
        qpos_adr = int(model.jnt_qposadr[joint_id])
        if qpos_adr >= robot_qpos_dim:
            continue
        if int(model.jnt_type[joint_id]) not in scalar_joint_types:
            continue
        if bool(model.jnt_limited[joint_id]):
            low, high = model.jnt_range[joint_id]
            qpos[qpos_adr] = 0.5 * (low + high)
    return qpos


def make_initial_guess_qpos(
    model: mujoco.MjModel,
    robot_type: str,
    embodiment_type: str,
    qpos_obj_right_step: np.ndarray,
    qpos_obj_left_step: np.ndarray,
    nq_obj: int,
    guess_idx: int,
    act_scene: bool = False,
) -> np.ndarray:
    if robot_type == "asm":
        qpos = model.qpos0.copy()
        robot_qpos_dim = model.nq - nq_obj
        qpos[:robot_qpos_dim] = neutral_robot_qpos(model, robot_qpos_dim)
        if guess_idx > 0:
            qpos[:robot_qpos_dim] += np.random.randn(robot_qpos_dim) * 0.03
        clip_robot_qpos_to_joint_ranges(model, qpos, robot_qpos_dim)
    else:
        qpos = np.random.rand(model.nq)

    set_object_qpos(
        qpos,
        embodiment_type,
        qpos_obj_right_step,
        qpos_obj_left_step,
        act_scene=act_scene,
    )
    return qpos


def clip_ctrl_to_range(model: mujoco.MjModel, ctrl: np.ndarray) -> None:
    for actuator_id in range(model.nu):
        if not bool(model.actuator_ctrllimited[actuator_id]):
            continue
        low, high = model.actuator_ctrlrange[actuator_id]
        ctrl[actuator_id] = np.clip(ctrl[actuator_id], low, high)


def clip_robot_qpos_trajectory_to_joint_ranges(
    model: mujoco.MjModel,
    qpos_traj: np.ndarray,
    robot_qpos_dim: int,
) -> np.ndarray:
    clipped = np.asarray(qpos_traj, dtype=np.float64).copy()
    scalar_joint_types = {
        int(mujoco.mjtJoint.mjJNT_SLIDE),
        int(mujoco.mjtJoint.mjJNT_HINGE),
    }
    clipped_stats = []
    for joint_id in range(model.njnt):
        qpos_adr = int(model.jnt_qposadr[joint_id])
        if qpos_adr >= robot_qpos_dim:
            continue
        if int(model.jnt_type[joint_id]) not in scalar_joint_types:
            continue
        if not bool(model.jnt_limited[joint_id]):
            continue
        low, high = model.jnt_range[joint_id]
        values = clipped[:, qpos_adr]
        below = values < low
        above = values > high
        if not np.any(below | above):
            continue
        below_violation = low - values[below] if np.any(below) else np.array([0.0])
        above_violation = values[above] - high if np.any(above) else np.array([0.0])
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        clipped_stats.append(
            {
                "name": joint_name or f"joint_{joint_id}",
                "count": int(np.count_nonzero(below | above)),
                "max_violation": float(
                    max(float(np.max(below_violation)), float(np.max(above_violation)))
                ),
                "range": (float(low), float(high)),
                "value_range": (float(np.min(values)), float(np.max(values))),
            }
        )
        clipped[:, qpos_adr] = np.clip(values, low, high)

    if clipped_stats:
        clipped_stats.sort(key=lambda item: item["max_violation"], reverse=True)
        total_values = sum(item["count"] for item in clipped_stats)
        worst = [
            (
                item["name"],
                item["count"],
                round(item["max_violation"], 4),
                tuple(round(v, 4) for v in item["range"]),
                tuple(round(v, 4) for v in item["value_range"]),
            )
            for item in clipped_stats[:12]
        ]
        loguru.logger.warning(
            "Clipped IK robot qpos to joint ranges: joints={}, values={}, max_violation={:.4f}, worst={}",
            len(clipped_stats),
            total_values,
            clipped_stats[0]["max_violation"],
            worst,
        )
    else:
        loguru.logger.info("IK robot qpos already within joint ranges.")
    return clipped


def set_ctrl_from_qpos(model: mujoco.MjModel, data: mujoco.MjData, nq_obj: int) -> None:
    data.ctrl[:] = 0.0
    ctrl_dim = min(model.nu, model.nq - nq_obj)
    data.ctrl[:ctrl_dim] = data.qpos[:ctrl_dim]
    clip_ctrl_to_range(model, data.ctrl)


def set_ctrl_from_reference_qpos(
    model: mujoco.MjModel, data: mujoco.MjData, qpos: np.ndarray, nq_obj: int
) -> None:
    data.ctrl[:] = 0.0
    ctrl_dim = min(model.nu, model.nq - nq_obj)
    data.ctrl[:ctrl_dim] = qpos[:ctrl_dim]
    clip_ctrl_to_range(model, data.ctrl)


# parameters
def main(
    dataset_dir: str = f"{ROOT}/../example_datasets",
    dataset_name: str = "oakink",
    robot_type: str = "allegro",
    embodiment_type: str = "bimanual",
    task: str = "pick_spoon_bowl",
    show_viewer: bool = True,
    save_video: bool = False,
    enable_collision: bool = False,
    start_idx: int = 0,
    end_idx: int = -1,
    sim_dt: float = 0.01,
    ref_dt: float = 0.02,
    data_id: int = 0,
    open_hand: bool = False,
    contact_detection_step_threshold: int = 3,
    finger_solimp_width: float = 0.01,
    finger_solimp_max_width: float = 0.0,
    wrist_solimp_width: float = 10.0,
    wrist_torque_scale: float = 10.0,
    object_solimp_width: float = 0.001,
    max_num_initial_guess: int = 8,
    average_frame_size: int = 3,
    aggregate_contact: bool = True,
    z_offset: float = 0.0,
    act_scene: bool = False,
    visualize_hand_keypoints: bool = False,
    visualize_object_bbox: bool = False,
    visualize_object_axes: bool = False,
    visualize_world_axes: bool = False,
    visualize_robot_base_axes: bool = False,
    visualize_head_camera_axes: bool = False,
    viewer: str = "mujoco",
    viser_host: str = "0.0.0.0",
    viser_port: int = 8080,
    wait_on_finish: bool = True,
):
    viewer = (viewer or "mujoco").strip().lower()
    if viewer not in {"mujoco", "viser", "none"}:
        raise ValueError(f"Unsupported viewer '{viewer}'. Expected mujoco, viser, or none.")
    use_mujoco_viewer = bool(show_viewer and viewer == "mujoco")
    use_viser_viewer = bool(show_viewer and viewer == "viser")

    # 处理路径
    # resolved processed directories
    dataset_dir = os.path.abspath(dataset_dir)
    processed_dir_robot = get_processed_data_dir(
        dataset_dir=dataset_dir,
        dataset_name=dataset_name,
        robot_type=robot_type,
        embodiment_type=embodiment_type,
        task=task,
        data_id=data_id,
    )
    processed_dir_mano = get_processed_data_dir(
        dataset_dir=dataset_dir,
        dataset_name=dataset_name,
        robot_type="mano",
        embodiment_type=embodiment_type,
        task=task,
        data_id=data_id,
    )
    os.makedirs(processed_dir_robot, exist_ok=True)
    # load model from processed scene
    if act_scene:
        model_path = f"{processed_dir_robot}/../scene_act.xml"
    else:
        model_path = f"{processed_dir_robot}/../scene.xml"
    # NOTE: sites in robot should follow the order of the xml file
    # 加载机器人的手部关键点数据，不同机器人的数据不同，而且手指数量甚至都不同，以及选择加载左手还是右手，或者是双手
    sites_in_robot = get_robot_sites(robot_type, embodiment_type)

    # 读取 oakink 迁移过来的 traj_kp.npz 文件，包含了手指、手腕、物体的位姿，以及接触状态（如果有的话）
    file_path = f"{processed_dir_mano}/trajectory_keypoints.npz"
    loaded_data = np.load(file_path)
    qpos_finger_right_raw = loaded_data["qpos_finger_right"][start_idx:end_idx]
    qpos_finger_left_raw = loaded_data["qpos_finger_left"][start_idx:end_idx]
    qpos_wrist_right_raw = loaded_data["qpos_wrist_right"][start_idx:end_idx]
    qpos_wrist_left_raw = loaded_data["qpos_wrist_left"][start_idx:end_idx]
    qpos_obj_right_raw = loaded_data["qpos_obj_right"][start_idx:end_idx]
    qpos_obj_left_raw = loaded_data["qpos_obj_left"][start_idx:end_idx]
    try:
        contact_left = loaded_data["contact_left"][start_idx:end_idx]
        contact_right = loaded_data["contact_right"][start_idx:end_idx]
    except:
        loguru.logger.warning("No contact data found, using all one")
        contact_left = np.ones((qpos_finger_right_raw.shape[0], 5))
        contact_right = np.ones((qpos_finger_left_raw.shape[0], 5))

    workspace_z_offset = float(z_offset)
    workspace_xy_offset = np.zeros(2, dtype=np.float64)
    support_task_info_path = f"{processed_dir_robot}/../task_info.json"
    support_spec = None
    mano_task_info_path = f"{processed_dir_mano}/../task_info.json"
    mano_task_info = {}
    if os.path.exists(mano_task_info_path):
        with open(mano_task_info_path) as f:
            mano_task_info = json.load(f)
    skip_workspace_support = (
        robot_type == "asm"
        and embodiment_type == "bimanual"
        and _uses_camera_world_alignment(mano_task_info)
    )
    if skip_workspace_support:
        loguru.logger.info(
            "Skipping ASM workspace support because keypoints are already aligned by camera."
        )
    if robot_type == "asm" and embodiment_type == "bimanual" and not skip_workspace_support:
        if os.path.exists(support_task_info_path):
            with open(support_task_info_path) as f:
                support_spec = workspace_support_spec_from_task_info(json.load(f))
        if support_spec is None:
            robot_assets_dir = (
                f"{dataset_dir}/processed/{dataset_name}/assets/robots/{robot_type}"
            )
            robot_xml_name = (
                "bimanual.xml"
                if embodiment_type == "bimanual"
                else f"{embodiment_type}.xml"
            )
            support_spec, _ = compute_workspace_support_spec(
                dataset_dir=dataset_dir,
                robot_xml_path=f"{robot_assets_dir}/{robot_xml_name}",
                task_info=mano_task_info,
                dataset_name=dataset_name,
                robot_type=robot_type,
                embodiment_type=embodiment_type,
                task=task,
                qpos_obj_right_first=qpos_obj_right_raw[0],
                qpos_obj_left_first=qpos_obj_left_raw[0],
            )
        workspace_z_offset = support_spec.workspace_z_offset
        workspace_xy_offset = support_spec.workspace_xy_offset.copy()
        loguru.logger.info(
            "ASM workspace support: robot_height={:.4f}, table_surface_z={:.4f}, object_first_frame_min_z={:.4f}, workspace_z_offset={:.4f}, workspace_xy_offset={}",
            support_spec.robot_height,
            support_spec.table_surface_z,
            support_spec.object_first_frame_min_z,
            support_spec.workspace_z_offset,
            support_spec.workspace_xy_offset.tolist(),
        )
    if abs(z_offset) > 1e-9 and support_spec is not None:
        workspace_z_offset += float(z_offset)
        loguru.logger.warning(
            "Applying additional manual z_offset={:.4f} on top of ASM auto workspace_z_offset; total workspace_z_offset={:.4f}",
            z_offset,
            workspace_z_offset,
        )

    qpos_finger_right = qpos_finger_right_raw.copy()
    qpos_finger_left = qpos_finger_left_raw.copy()
    qpos_wrist_right = qpos_wrist_right_raw.copy()
    qpos_wrist_left = qpos_wrist_left_raw.copy()
    qpos_obj_right = qpos_obj_right_raw.copy()
    qpos_obj_left = qpos_obj_left_raw.copy()
    if np.linalg.norm(workspace_xy_offset) > 1e-9:
        qpos_finger_right[:, :, :2] += workspace_xy_offset
        qpos_finger_left[:, :, :2] += workspace_xy_offset
        qpos_wrist_right[:, :2] += workspace_xy_offset
        qpos_wrist_left[:, :2] += workspace_xy_offset
        qpos_obj_right[:, :2] += workspace_xy_offset
        qpos_obj_left[:, :2] += workspace_xy_offset
        loguru.logger.info(
            "Applied workspace_xy_offset={}; wrist_xy [{:.4f}, {:.4f}] -> [{:.4f}, {:.4f}]",
            workspace_xy_offset.tolist(),
            float(qpos_wrist_right_raw[:, 0].mean()),
            float(qpos_wrist_right_raw[:, 1].mean()),
            float(qpos_wrist_right[:, 0].mean()),
            float(qpos_wrist_right[:, 1].mean()),
        )
    if abs(workspace_z_offset) > 1e-9:
        qpos_finger_right[:, :, 2] += workspace_z_offset
        qpos_finger_left[:, :, 2] += workspace_z_offset
        qpos_wrist_right[:, 2] += workspace_z_offset
        qpos_wrist_left[:, 2] += workspace_z_offset
        qpos_obj_right[:, 2] += workspace_z_offset
        qpos_obj_left[:, 2] += workspace_z_offset
        loguru.logger.info(
            "Applied workspace_z_offset={:.4f}; wrist_z {:.4f}->{:.4f}, object_z {:.4f}->{:.4f}",
            workspace_z_offset,
            float(qpos_wrist_right_raw[:, 2].min()),
            float(qpos_wrist_right[:, 2].min()),
            float(np.minimum(qpos_obj_right_raw[:, 2], qpos_obj_left_raw[:, 2]).min()),
            float(np.minimum(qpos_obj_right[:, 2], qpos_obj_left[:, 2]).min()),
        )
    contact_ref = np.concatenate([contact_right, contact_left], axis=1)
    if aggregate_contact:
        contact_aggregated = np.any(contact_ref, axis=-1)
        for i in range(contact_ref.shape[1]):
            contact_ref[:, i] = contact_aggregated
    # get the first contact frame where contact_left turns to 1 (two 1s consecutive)
    first_contact_frame_left = np.zeros(5) + qpos_finger_right.shape[0]
    first_contact_frame_right = np.zeros(5) + qpos_finger_left.shape[0]
    for j in range(5):
        for i in range(contact_detection_step_threshold, len(contact_left)):
            if contact_left[i - contact_detection_step_threshold : i, j].all():
                first_contact_frame_left[j] = i
                break
        for i in range(contact_detection_step_threshold, len(contact_right)):
            if contact_right[i - contact_detection_step_threshold : i, j].all():
                first_contact_frame_right[j] = i
                break
    # 读完之后，再拼接成 qpos_ref，shape 是 (T, 14, 7)，每个维度分别是时间步、关键点数量（手腕+5指尖+物体），以及每个关键点的位姿（位置+四元数）
    qpos_ref = np.concatenate(
        [
            qpos_wrist_right[:, None],
            qpos_finger_right,
            qpos_wrist_left[:, None],
            qpos_finger_left,
            qpos_obj_right[:, None],
            qpos_obj_left[:, None],
        ],
        axis=1,
    )
    use_debug_render = any(
        [
            visualize_hand_keypoints,
            visualize_object_bbox,
            visualize_object_axes,
            visualize_world_axes,
            visualize_robot_base_axes,
            visualize_head_camera_axes,
        ]
    )
    object_debug_specs: dict[str, np.ndarray] = {}
    if visualize_object_bbox or visualize_object_axes:
        try:
            object_debug_specs = _load_object_debug_specs(
                dataset_name=dataset_name,
                task_info_path=f"{processed_dir_mano}/../task_info.json",
            )
        except FileNotFoundError:
            loguru.logger.warning(
                "Object debug metadata not found; object bbox/axes visualization will be skipped"
            )
        if len(object_debug_specs) == 0:
            loguru.logger.warning(
                "No object bbox metadata available for dataset={} task={}; object bbox/axes visualization will be skipped",
                dataset_name,
                task,
            )

    # 读 scene.xml，拿到模型和数据
    # load model
    mj_model = mujoco.MjModel.from_xml_path(model_path)
    mj_model.opt.timestep = sim_dt
    mj_data = mujoco.MjData(mj_model)

    # NOTE: sites for mimic should follow the order of data
    # index_map 给每个语义点分配 qpos_idx
    index_map = {}
    cnt = 0
    for sides in ["right", "left"]:
        for body_name in [
            "palm",
            "thumb_tip",
            "index_tip",
            "middle_tip",
            "ring_tip",
            "pinky_tip",
        ]:
            index_map[f"{sides}_{body_name}"] = {
                "qpos_idx": cnt,
                "mocap_idx": -1,
                "eq_constraint_idx": -1,
            }
            cnt += 1
    # add objects
    index_map["right_object"] = {
        "qpos_idx": cnt,
        "mocap_idx": -1,
        "eq_constraint_idx": -1,
    }
    cnt += 1
    index_map["left_object"] = {
        "qpos_idx": cnt,
        "mocap_idx": -1,
        "eq_constraint_idx": -1,
    }
    cnt += 1

    # 定义要被追踪的 site；随后通过名字匹配得到 mano2mimic_site_idx
    # 这里的 sites 其实就是物体手上关键的接触点，被收集起来后作为作为 ik 要追踪的目标 site
    sites_for_mimic = [
        "right_palm",
        "right_thumb_tip",
        "right_index_tip",
        "right_middle_tip",
        "right_ring_tip",
        "right_pinky_tip",
        "left_palm",
        "left_thumb_tip",
        "left_index_tip",
        "left_middle_tip",
        "left_ring_tip",
        "left_pinky_tip",
        "right_object",
        "left_object",
    ]

    # special case: allegro hand
    if robot_type in ["allegro", "metahand"]:
        sites_for_mimic.remove("right_pinky_tip")
        sites_for_mimic.remove("left_pninky_tip")

    if embodiment_type == "right":
        sites_for_mimic = [s for s in sites_for_mimic if "right" in s]
    elif embodiment_type == "left":
        sites_for_mimic = [s for s in sites_for_mimic if "left" in s]

    site_ids = [
        mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, s)
        for s in sites_for_mimic
    ]
    # make sure all site_ids are valid, i.e. no -1
    assert all(site_id != -1 for site_id in site_ids), f"site_ids: {site_ids}"
    mano2mimic_site_idx = []
    for s in sites_for_mimic:
        # find robot site name
        for site_name in sites_in_robot:
            if site_name == s:
                mano2mimic_site_idx.append(sites_in_robot.index(site_name))

    # create mocap sites for retargeting
    site_joint_matches = {}
    for key in sites_for_mimic:
        if "palm" in key:  # palm: strong rotation constraint, weak position constraint
            constraint_type = "mjEQ_WELD"
            solimp = [0.0, 0.95, wrist_solimp_width, 0.5, 2.0]
            torque_scale = wrist_torque_scale
            solref = [0.02, 1.0]
        elif "object" in key:  # object: strong position and rotation constraint
            constraint_type = "mjEQ_WELD"
            torque_scale = 10.0
            solimp = [0.9, 0.95, object_solimp_width, 0.5, 2.0]
            solref = [0.002, 1.0]
        else:  # finger: weak rotation constraint, strong position constraint (but weaker than object)
            constraint_type = "mjEQ_CONNECT"
            if "thumb" in key or "index" in key or "middle" in key:
                width_scale = 1.0
            else:
                width_scale = 3.0
            finger_width = finger_solimp_width * width_scale
            if finger_solimp_max_width > 0.0:
                finger_width = min(finger_width, finger_solimp_max_width)
            solimp = [0.0, 0.95, finger_width, 0.5, 2.0]
            solref = [0.01, 1.0]
            torque_scale = 1.0
        site_joint_matches[key] = {
            "equality_constraint_type": constraint_type,
            "torque_scale": torque_scale,
            "solref": solref,
            "solimp": solimp,
        }

    robot_conf = OmegaConf.create(
        {
            "optimization_params": {
                "disable_joint_limits": False,
                "disable_collisions": not enable_collision,
            },
            "site_joint_matches": site_joint_matches,
        }
    )
    target_mocap_bodies = ["target_mocap_body_" + s for s in sites_for_mimic]
    mj_spec = mujoco.MjSpec.from_file(model_path)

    # ================================
    # add constraints to the free body
    # ================================
    mjspec = add_mocap_bodies(
        mj_spec,
        sites_for_mimic,
        target_mocap_bodies,
        robot_conf,
        add_equality_constraint=True,
        visualize_hand_keypoints=visualize_hand_keypoints,
        visualize_target_geoms=use_viser_viewer,
    )

    if visualize_world_axes:
        _add_static_axes_body(
            mjspec,
            body_name="debug_world_axes",
            axis_length=0.25,
            axis_thickness=0.004,
        )
    if visualize_robot_base_axes:
        robot_base_debug_handle = mjspec.worldbody.add_body(
            name="debug_robot_base_axes",
            mocap=True,
        )
        _add_axis_geoms(
            robot_base_debug_handle,
            prefix="debug_robot_base_axes",
            axis_length=0.18,
            axis_thickness=0.004,
            group=DEBUG_GEOM_GROUP,
        )
    if visualize_head_camera_axes:
        _add_debug_camera_body(
            mjspec,
            body_name="debug_head_camera_axes",
            marker_rgba=LEGACY_CAMERA_MARKER_RGBA,
            marker_size=[0.012, 0.008, 0.006],
            show_axes=False,
        )
        _add_debug_camera_body(
            mjspec,
            body_name="debug_d435_optical_axes",
            marker_rgba=OPTICAL_CAMERA_MARKER_RGBA,
            marker_size=[0.008, 0.005, 0.0035],
            axis_length=0.11,
            axis_thickness=0.003,
        )
    if visualize_object_bbox or visualize_object_axes:
        for side, size_xyz in object_debug_specs.items():
            if size_xyz.shape != (3,):
                loguru.logger.warning(
                    "Ignoring invalid {} object bbox size with shape {}",
                    side,
                    size_xyz.shape,
                )
                continue
            _add_debug_object_body(
                mjspec,
                body_name=f"debug_{side}_object_frame",
                half_extents=0.5 * size_xyz,
                add_bbox=visualize_object_bbox,
                add_axes=visualize_object_axes,
            )

    # ================================
    # add constraints to relative bodies, i.e. stick the object to the finger
    # ================================
    finger_names = [
        "thumb_tip",
        "index_tip",
        "middle_tip",
        "ring_tip",
        "pinky_tip",
    ]
    if robot_type in ["allegro", "metahand"]:
        finger_names = finger_names[:4]

    sides = {
        "right": ["right"],
        "left": ["left"],
        "bimanual": ["right", "left"],
    }[embodiment_type]

    # add position sensor to sites_for_mimic
    for i in range(len(sites_for_mimic)):
        site_name = sites_for_mimic[i]
        mjspec.add_sensor(
            name=f"pos_{site_name}",
            type=mujoco.mjtSensor.mjSENS_FRAMEPOS,
            objtype=mujoco.mjtObj.mjOBJ_SITE,
            objname=site_name,
        )

    mj_model_ik = mj_spec.compile()
    mj_model_ik.opt.timestep = sim_dt
    mj_model_ik.opt.iterations = 20
    mj_model_ik.opt.ls_iterations = 50
    if not enable_collision:
        mj_model_ik.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    mj_model_ik.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_ACTUATION
    mj_data_ik = mujoco.MjData(mj_model_ik)

    # update index_map
    for target_mocap_body in target_mocap_bodies:
        body_name = target_mocap_body[18:]
        body_id = mujoco.mj_name2id(
            mj_model_ik, mujoco.mjtObj.mjOBJ_BODY, target_mocap_body
        )
        mocap_id = mj_model_ik.body_mocapid[body_id]
        index_map[body_name]["mocap_idx"] = mocap_id
        # print(body_name, body_id, mocap_id)

    debug_object_mocap_ids = {
        side: _body_mocap_id(mj_model_ik, f"debug_{side}_object_frame")
        for side in object_debug_specs
    }
    debug_robot_base_mocap_id = (
        _body_mocap_id(mj_model_ik, "debug_robot_base_axes")
        if visualize_robot_base_axes
        else -1
    )
    debug_head_camera_mocap_id = (
        _body_mocap_id(mj_model_ik, "debug_head_camera_axes")
        if visualize_head_camera_axes
        else -1
    )
    debug_optical_camera_mocap_id = (
        _body_mocap_id(mj_model_ik, "debug_d435_optical_axes")
        if visualize_head_camera_axes
        else -1
    )
    robot_base_body_id = (
        _find_robot_base_body_id(mj_model_ik) if visualize_robot_base_axes else -1
    )
    head_camera_site_id = (
        mujoco.mj_name2id(
            mj_model_ik, mujoco.mjtObj.mjOBJ_SITE, HEAD_CAMERA_FRAME_SITE_NAME
        )
        if visualize_head_camera_axes
        else -1
    )
    optical_camera_site_id = (
        mujoco.mj_name2id(
            mj_model_ik, mujoco.mjtObj.mjOBJ_SITE, D435_OPTICAL_FRAME_SITE_NAME
        )
        if visualize_head_camera_axes
        else -1
    )
    if visualize_head_camera_axes and head_camera_site_id == -1:
        loguru.logger.warning(
            "Requested head camera axes visualization, but site '{}' was not found in the IK model.",
            HEAD_CAMERA_FRAME_SITE_NAME,
        )
        debug_head_camera_mocap_id = -1
    if visualize_head_camera_axes and optical_camera_site_id == -1:
        loguru.logger.warning(
            "Requested D435 optical camera axes visualization, but site '{}' was not found in the IK model.",
            D435_OPTICAL_FRAME_SITE_NAME,
        )
        debug_optical_camera_mocap_id = -1

    # set object position
    nq_obj = get_nq_obj(embodiment_type, act_scene=act_scene)
    set_object_qpos(
        mj_data_ik.qpos,
        embodiment_type,
        qpos_obj_right[0],
        qpos_obj_left[0],
        act_scene=act_scene,
    )

    # set the mocap sites to the tip positions
    # for i, site_id in enumerate(site_ids):
    for i in range(len(sites_for_mimic)):
        site_id = site_ids[i]
        site_name = sites_for_mimic[i]
        mano_id = mano2mimic_site_idx[i]
        mocap_id = i
        mj_data_ik.mocap_pos[mocap_id] = qpos_ref[0, mano_id, :3]
        mj_data_ik.mocap_quat[mocap_id] = qpos_ref[0, mano_id, 3:]
    _sync_debug_visualization_state(
        model=mj_model_ik,
        data=mj_data_ik,
        qpos_source=mj_data_ik,
        frame_idx=0,
        qpos_obj_right=qpos_obj_right,
        qpos_obj_left=qpos_obj_left,
        debug_object_mocap_ids=debug_object_mocap_ids,
        debug_robot_base_mocap_id=debug_robot_base_mocap_id,
        robot_base_body_id=robot_base_body_id,
        debug_head_camera_mocap_id=debug_head_camera_mocap_id,
        head_camera_site_id=head_camera_site_id,
        debug_optical_camera_mocap_id=debug_optical_camera_mocap_id,
        optical_camera_site_id=optical_camera_site_id,
    )
    viewer_body_entity_and_ids: list[tuple[object, int]] = []
    if use_viser_viewer:
        viser_viewer.init_viser(
            app_name="spider-ik",
            host=viser_host,
            port=viser_port,
        )
        viewer_body_entity_and_ids = viser_viewer.build_and_log_scene_from_spec(
            spec=mj_spec,
            model=mj_model_ik,
            entity_root="ik",
            build_ref=False,
            include_debug_groups=True,
        )
        loguru.logger.info(
            "IK Viser viewer is available at http://{}:{}/ . Use SSH port forwarding if running remotely.",
            viser_host,
            viser_port,
        )

    # rollout mujoco
    # reference dt inferred from MANO keypoint data spacing if available; default to 0.02
    rate_limiter = RateLimiter(1 / ref_dt)
    H = qpos_finger_right.shape[0]
    cnt = 0
    if save_video:
        import imageio

        mj_model_ik.vis.global_.offwidth = 720
        mj_model_ik.vis.global_.offheight = 480
        renderer = mujoco.Renderer(mj_model_ik, height=480, width=720)
    # TODO: move it to mujoco_utils
    run_viewer = get_viewer(use_mujoco_viewer, mj_model_ik, mj_data_ik)

    # random initial guess to find a stable initial pose

    ref_mocap_ids = []
    ref_site_ids = []
    track_site_ids = []
    # get track site ids
    for sid in range(mj_model.nsite):
        name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_SITE, sid)
        if name is not None and name.startswith("track"):
            track_site_ids.append(sid)
    for sid in track_site_ids:
        track_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_SITE, sid)
        ref_name = track_name.replace("track", "ref")
        # get mocap id of ref site
        mocap_body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, ref_name)
        mocap_id = mj_model.body_mocapid[mocap_body_id]
        ref_mocap_ids.append(mocap_id)
        # get site id of ref site
        ref_site_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, ref_name)
        ref_site_ids.append(ref_site_id)

    diagnostic_specs = []
    for site_name in sites_for_mimic:
        site_id = mujoco.mj_name2id(mj_model_ik, mujoco.mjtObj.mjOBJ_SITE, site_name)
        diagnostic_specs.append((site_name, site_id, index_map[site_name]["qpos_idx"]))
    ik_collision_diagnostics: list[dict] = []
    diagnostic_suffix = "_act" if act_scene else ""

    def record_ik_diagnostic(frame_idx: int, phase: str, **extra: object) -> dict:
        try:
            diagnostic = _compute_ik_collision_diagnostic(
                mj_model_ik,
                mj_data_ik,
                frame_idx,
                diagnostic_specs,
                qpos_ref,
            )
        except Exception as diagnostic_error:
            diagnostic = {
                "frame": int(frame_idx),
                "sim_time": float(mj_data_ik.time),
                "diagnostic_error": repr(diagnostic_error),
            }
        diagnostic["phase"] = phase
        diagnostic.update(extra)
        ik_collision_diagnostics.append(diagnostic)
        return diagnostic

    def step_ik_model(
        frame_idx: int,
        phase: str,
        *,
        substep: int,
        candidate_idx: int | None = None,
    ) -> None:
        try:
            mujoco.mj_step(mj_model_ik, mj_data_ik)
        except mujoco.FatalError as exc:
            diagnostic = record_ik_diagnostic(
                frame_idx,
                phase,
                substep=substep,
                candidate_idx=candidate_idx,
                mujoco_error=str(exc),
            )
            failure = {
                "frame": int(frame_idx),
                "phase": phase,
                "substep": int(substep),
                "candidate_idx": candidate_idx,
                "mujoco_error": str(exc),
                "last_diagnostic": diagnostic,
            }
            _write_ik_collision_diagnostics(
                processed_dir_robot,
                ik_collision_diagnostics,
                failure=failure,
                suffix=diagnostic_suffix,
            )
            raise

    with run_viewer() as gui:
        if use_mujoco_viewer and hasattr(gui, "opt"):
            if visualize_hand_keypoints:
                gui.opt.sitegroup[4] = True
            if (
                visualize_object_bbox
                or visualize_object_axes
                or visualize_world_axes
                or visualize_robot_base_axes
                or visualize_head_camera_axes
            ):
                gui.opt.geomgroup[DEBUG_GEOM_GROUP] = True
        ik_start_time = time.perf_counter()
        last_progress_log = -1
        cnt = 0
        while cnt < H:
            if cnt == 0:
                # reset distance cost
                cost_sum = 0.0
                # reset data buffer
                best_qpos_init = np.zeros(mj_model_ik.nq)
                best_qpos_diff_sum = np.inf
                for guess_idx in range(max_num_initial_guess):
                    mj_data_ik.qpos[:] = make_initial_guess_qpos(
                        mj_model_ik,
                        robot_type,
                        embodiment_type,
                        qpos_obj_right[cnt],
                        qpos_obj_left[cnt],
                        nq_obj,
                        guess_idx,
                        act_scene=act_scene,
                    )
                    mj_data_ik.qvel[:] = np.zeros(mj_model_ik.nv)
                    set_ctrl_from_qpos(mj_model_ik, mj_data_ik, nq_obj)
                    mocap_id_list = []
                    qpos_id_list = []
                    for k, v in index_map.items():
                        if v["mocap_idx"] != -1:
                            mocap_idx = v["mocap_idx"]
                            qpos_idx = v["qpos_idx"]
                            mocap_id_list.append(mocap_idx)
                            qpos_id_list.append(qpos_idx)
                            mj_data_ik.mocap_pos[mocap_idx] = qpos_ref[
                                cnt, qpos_idx, :3
                            ]
                            if "tip" in k:
                                mj_data_ik.mocap_pos[mocap_idx] += (
                                    np.random.randn(3) * 0.002
                                )
                            mj_data_ik.mocap_quat[mocap_idx] = qpos_ref[
                                cnt, qpos_idx, 3:
                            ]
                    qpos_diff_sum = 0.0
                    for substep in range(30):
                        set_ctrl_from_qpos(mj_model_ik, mj_data_ik, nq_obj)
                        step_ik_model(
                            cnt,
                            "initial_guess",
                            substep=substep,
                            candidate_idx=guess_idx,
                        )
                    # compute mocap diff
                    for mocap_id, qpos_id in zip(
                        mocap_id_list, qpos_id_list, strict=False
                    ):
                        mocap_pos = mj_data_ik.mocap_pos[mocap_id]
                        mocap_quat = mj_data_ik.mocap_quat[mocap_id]
                        qpos_pos = qpos_ref[cnt, qpos_id, :3]
                        qpos_quat = qpos_ref[cnt, qpos_id, 3:]
                        mocap_diff = np.linalg.norm(mocap_pos - qpos_pos)
                        qpos_diff = np.linalg.norm(mocap_quat - qpos_quat)
                        qpos_diff_sum += mocap_diff + qpos_diff
                    mj_data.qpos[:] = mj_data_ik.qpos.copy()
                    mj_data.qvel[:] = mj_data_ik.qvel.copy() * 0.0
                    set_ctrl_from_qpos(mj_model, mj_data, nq_obj)
                    mujoco.mj_forward(mj_model, mj_data)
                    for i in range(30):
                        mujoco.mj_step(mj_model, mj_data)
                        qpos_diff_sum += np.linalg.norm(mj_data.qpos - mj_data_ik.qpos)
                    if qpos_diff_sum < best_qpos_diff_sum:
                        best_qpos_init = mj_data_ik.qpos.copy()
                        best_qpos_diff_sum = qpos_diff_sum
                loguru.logger.info(f"best_qpos_diff_sum: {best_qpos_diff_sum}")
                mj_data_ik.qpos[:] = best_qpos_init
                mj_data_ik.qvel[:] = 0.0
                set_ctrl_from_qpos(mj_model_ik, mj_data_ik, nq_obj)
                step_ik_model(cnt, "post_initial_guess", substep=0)
                qpos_list = []
                contact_pos_list = []
                contact_list = []
                images = []

            for k, v in index_map.items():
                if v["mocap_idx"] != -1:
                    mj_data_ik.mocap_pos[v["mocap_idx"]] = qpos_ref[
                        cnt, v["qpos_idx"], :3
                    ]
                    mj_data_ik.mocap_quat[v["mocap_idx"]] = qpos_ref[
                        cnt, v["qpos_idx"], 3:
                    ]

            for substep in range(max(1, int(ref_dt / sim_dt))):
                step_ik_model(cnt, "frame_solve", substep=substep)
            record_ik_diagnostic(cnt, "frame_solve")

            # set site position and set it to ref mocap position (use original mj_model and mj_data)
            mj_data.qpos[:] = mj_data_ik.qpos.copy()
            mj_data.qvel[:] = 0.0
            set_ctrl_from_qpos(mj_model, mj_data, nq_obj)

            # override joint position according to contact state
            if open_hand:
                for side in ["right", "left"]:
                    for finger in ["thumb", "index", "middle", "ring", "pinky"]:
                        # get joint index
                        joint_ids = []
                        for jid in range(mj_model.njnt):
                            joint_name = mujoco.mj_id2name(
                                mj_model, mujoco.mjtObj.mjOBJ_JOINT, jid
                            )
                            if side in joint_name and finger in joint_name:
                                joint_ids.append(jid)
                        if len(joint_ids) > 0:
                            for joint_idx in joint_ids:
                                current_joint_pos = mj_data.qpos[joint_idx]
                                zero_joint_pos = 0.0
                                # Map sides and fingers to their respective indices
                                side_map = {
                                    "right": first_contact_frame_right,
                                    "left": first_contact_frame_left,
                                }
                                finger_map = {
                                    "thumb": 0,
                                    "index": 1,
                                    "middle": 2,
                                    "ring": 3,
                                    "pinky": 4,
                                }

                                contact_frame_list = side_map[side]
                                finger_idx = finger_map[finger]
                                contact_frame = contact_frame_list[finger_idx]

                                # Use smooth transition with clipping
                                ratio = np.clip(cnt / max(contact_frame, 1), 0.0, 1.0)
                                ratio = 1.0 - np.cos(ratio * np.pi * 0.5)
                                joint_pos = (
                                    ratio * current_joint_pos
                                    + (1 - ratio) * zero_joint_pos
                                )
                                mj_data.qpos[joint_idx] = joint_pos

            mujoco.mj_kinematics(mj_model, mj_data)
            for i in range(len(ref_mocap_ids)):
                mocap_id = ref_mocap_ids[i]
                track_site_id = track_site_ids[i]
                track_site_name = mujoco.mj_id2name(
                    mj_model, mujoco.mjtObj.mjOBJ_SITE, track_site_id
                )
                mj_data.mocap_pos[mocap_id] = mj_data.site_xpos[track_site_id].copy()

            contact = np.zeros(len(track_site_ids))
            contact_map = {
                "right_thumb": 0,
                "right_index": 1,
                "right_middle": 2,
                "right_ring": 3,
                "right_pinky": 4,
                "left_thumb": 5,
                "left_index": 6,
                "left_middle": 7,
                "left_ring": 8,
                "left_pinky": 9,
            }
            for i in range(len(track_site_ids)):
                track_site_name = mujoco.mj_id2name(
                    mj_model, mujoco.mjtObj.mjOBJ_SITE, track_site_ids[i]
                )
                for k, v in contact_map.items():
                    if k in track_site_name and "object" in track_site_name:
                        contact[i] = contact_ref[cnt, v]
                        break

            mujoco.mj_forward(mj_model, mj_data)
            contact_pos_list.append(mj_data.mocap_pos.copy())
            # get contact list
            # logic: for each track_site, check its corresponding object site (e.g. track site named "track_hand_right_index_tip" should correspond to "track_object_right_index_tip")
            # similarly, "track_object_right_index_tip" should correspond to "track_hand_right_index_tip"
            # after find its corresponding object site, check if the distance between the two sites is less than 0.01, if so, set contact to 1, otherwise set contact to 0
            # contact order should follow track site definition order
            # contact size is equal to check sites number
            # for i in range(len(track_site_ids)):
            #     track_site_id = track_site_ids[i]
            #     track_site_pos = mj_data.site_xpos[track_site_id].copy()
            #     track_site_name = mujoco.mj_id2name(
            #         mj_model, mujoco.mjtObj.mjOBJ_SITE, track_site_id
            #     )
            #     if "hand" in track_site_name:
            #         match_site_name = track_site_name.replace("hand", "object")
            #     elif "object" in track_site_name:
            #         match_site_name = track_site_name.replace("object", "hand")
            #     else:
            #         raise ValueError(f"Invalid track site name: {track_site_name}")
            #     match_site_id = mujoco.mj_name2id(
            #         mj_model, mujoco.mjtObj.mjOBJ_SITE, match_site_name
            #     )
            #     match_site_pos = mj_data.site_xpos[match_site_id].copy()
            #     if np.linalg.norm(track_site_pos - match_site_pos) < 0.01:
            #         contact[i] = 1
            #     else:
            #         contact[i] = 0
            contact_list.append(contact)

            # get contact point distance
            for i in range(len(sites_for_mimic)):
                site_name = sites_for_mimic[i]

            qpos_list.append(mj_data.qpos.copy())
            if save_video:
                opt = mujoco.MjvOption()
                mujoco.mjv_defaultOption(opt)
                if visualize_hand_keypoints:
                    opt.sitegroup[4] = True
                if (
                    visualize_object_bbox
                    or visualize_object_axes
                    or visualize_world_axes
                    or visualize_robot_base_axes
                    or visualize_head_camera_axes
                ):
                    opt.geomgroup[DEBUG_GEOM_GROUP] = True
                if use_debug_render:
                    _sync_debug_visualization_state(
                        model=mj_model_ik,
                        data=mj_data_ik,
                        qpos_source=mj_data,
                        frame_idx=cnt,
                        qpos_obj_right=qpos_obj_right,
                        qpos_obj_left=qpos_obj_left,
                        debug_object_mocap_ids=debug_object_mocap_ids,
                        debug_robot_base_mocap_id=debug_robot_base_mocap_id,
                        robot_base_body_id=robot_base_body_id,
                        debug_head_camera_mocap_id=debug_head_camera_mocap_id,
                        head_camera_site_id=head_camera_site_id,
                        debug_optical_camera_mocap_id=debug_optical_camera_mocap_id,
                        optical_camera_site_id=optical_camera_site_id,
                    )
                    renderer.update_scene(data=mj_data_ik, camera="front", scene_option=opt)
                else:
                    renderer.update_scene(data=mj_data, camera="front", scene_option=opt)
                images.append(renderer.render())
            if use_mujoco_viewer or use_viser_viewer:
                _sync_debug_visualization_state(
                    model=mj_model_ik,
                    data=mj_data_ik,
                    qpos_source=mj_data,
                    frame_idx=cnt,
                    qpos_obj_right=qpos_obj_right,
                    qpos_obj_left=qpos_obj_left,
                    debug_object_mocap_ids=debug_object_mocap_ids,
                    debug_robot_base_mocap_id=debug_robot_base_mocap_id,
                    robot_base_body_id=robot_base_body_id,
                    debug_head_camera_mocap_id=debug_head_camera_mocap_id,
                    head_camera_site_id=head_camera_site_id,
                    debug_optical_camera_mocap_id=debug_optical_camera_mocap_id,
                    optical_camera_site_id=optical_camera_site_id,
                )
            if use_viser_viewer:
                viser_viewer.log_frame(
                    data=mj_data_ik,
                    sim_time=float(mj_data.time),
                    viewer_body_entity_and_ids=viewer_body_entity_and_ids,
                    playback_fps=float(1.0 / ref_dt),
                )
            if use_mujoco_viewer:
                gui.sync()
                rate_limiter.sleep()
            cnt += 1
            progress_bucket = cnt // 100
            if (
                progress_bucket > last_progress_log
                or cnt == H
                or cnt == 1
            ):
                elapsed = time.perf_counter() - ik_start_time
                loguru.logger.info(
                    "IK progress: {}/{} frames ({:.1f}%), elapsed {:.1f}s",
                    cnt,
                    H,
                    100.0 * cnt / H,
                    elapsed,
                )
                last_progress_log = progress_bucket
            if cnt == H:
                cost_mean = cost_sum / H
                if use_mujoco_viewer:
                    # check if the rollout is good, if so, break
                    user_input = input("Is the rollout good? (y/n): ")
                    if user_input.lower() == "y":
                        break
                    else:
                        cnt = 0
                else:
                    break

        file_dir = processed_dir_robot
        os.makedirs(file_dir, exist_ok=True)
        _write_ik_collision_diagnostics(
            file_dir,
            ik_collision_diagnostics,
            suffix=diagnostic_suffix,
        )
        video_debug_parts = []
        if visualize_hand_keypoints:
            video_debug_parts.append("hand_keypoints")
        if visualize_object_bbox:
            video_debug_parts.append("object_bbox")
        if visualize_object_axes:
            video_debug_parts.append("object_axes")
        if visualize_world_axes:
            video_debug_parts.append("world_axes")
        if visualize_robot_base_axes:
            video_debug_parts.append("robot_base_axes")
        if visualize_head_camera_axes:
            video_debug_parts.append("head_camera_axes")
        video_debug_suffix = (
            "_" + "_".join(video_debug_parts) if len(video_debug_parts) > 0 else ""
        )
        if save_video:
            loguru.logger.info(
                "Encoding IK video with {} frames to {}/visualization_ik{}{}.mp4",
                len(images),
                file_dir,
                video_debug_suffix,
                "_act" if act_scene else "",
            )
            imageio.mimsave(
                f"{file_dir}/visualization_ik{video_debug_suffix}{'_act' if act_scene else ''}.mp4",
                images,
                fps=int(1 / ref_dt),
            )
            loguru.logger.info(
                f"Saved visualization video to {file_dir}/visualization_ik{video_debug_suffix}{'_act' if act_scene else ''}.mp4"
            )

        qpos_list = np.array(qpos_list)

        # average filter
        def moving_average_filter(signal_data, window_size=5):
            return np.convolve(
                signal_data, np.ones(window_size) / window_size, mode="valid"
            )

        # Apply moving average filter
        filtered_qpos_list = np.zeros(
            (qpos_list.shape[0] - average_frame_size + 1, qpos_list.shape[1])
        )
        for i in range(qpos_list.shape[1]):
            filtered_qpos_list[:, i] = moving_average_filter(
                qpos_list[:, i], average_frame_size
            )
        qpos_list = filtered_qpos_list

        def low_pass_filter(signal_data, cutoff_frequency=10, order=4):
            nyquist = 0.5 * (1 / ref_dt)
            normal_cutoff = cutoff_frequency / nyquist
            b, a = signal.butter(order, normal_cutoff, btype="low", analog=False)
            return signal.filtfilt(b, a, signal_data)

        # Apply low pass filter
        # for i in range(qpos_list.shape[1]):
        #     qpos_list[:, i] = low_pass_filter(qpos_list[:, i])

        qpos_list = np.asarray(qpos_list, dtype=np.float64)
        robot_qpos_dim = mj_model.nq - nq_obj
        qpos_list = clip_robot_qpos_trajectory_to_joint_ranges(
            mj_model,
            qpos_list,
            robot_qpos_dim,
        )

        H = qpos_list.shape[0]
        # get qvel
        qvel_list = np.zeros((H - 1, mj_model_ik.nv))
        for i in range(1, H):
            mujoco.mj_differentiatePos(
                mj_model_ik,
                qvel_list[i - 1, :],
                ref_dt,
                qpos_list[i - 1, :],
                qpos_list[i, :],
            )
        qpos_list = qpos_list[1:]
        contact_pos_list = np.array(contact_pos_list)[1:]
        contact_list = np.array(contact_list)[1:]
        assert qpos_list.shape[0] == qvel_list.shape[0]

        # directly rollout ctrl to get qpos_rollout
        # Use sim_dt substeps here; full-arm ASM is too stiff for a single ref_dt
        # dynamics step when the IK reference is only sampled at 50 Hz.
        mj_model.opt.timestep = sim_dt
        rollout_substeps = max(1, int(ref_dt / sim_dt))
        mj_data.qpos[:] = qpos_list[0]
        mj_data.qvel[:] = qvel_list[0]
        set_ctrl_from_qpos(mj_model, mj_data, nq_obj)
        for _ in range(rollout_substeps):
            mujoco.mj_step(mj_model, mj_data)
        H = qpos_list.shape[0]
        qpos_rollout = np.zeros((H, mj_model.nq))
        qvel_rollout = np.zeros((H, mj_model.nv))
        qpos_rollout[0] = qpos_list[0]
        for i in range(1, H):
            set_ctrl_from_reference_qpos(mj_model, mj_data, qpos_list[i], nq_obj)
            noise = np.random.randn(mj_model.nu) * 0.2
            if robot_type == "asm":
                noise *= 0.0
            else:
                noise[:6] *= 0.0
                noise[22:28] *= 0.0
            mj_data.ctrl[:] += noise
            clip_ctrl_to_range(mj_model, mj_data.ctrl)
            for _ in range(rollout_substeps):
                mujoco.mj_step(mj_model, mj_data)
            qpos_rollout[i] = mj_data.qpos.copy()

        if act_scene:
            out_npz = f"{file_dir}/trajectory_kinematic_act.npz"
        else:
            out_npz = f"{file_dir}/trajectory_kinematic.npz"
        np.savez(
            out_npz,
            qpos=qpos_list,
            qpos_rollout=qpos_rollout,
            qvel=qvel_list,
            contact=contact_list,
            contact_pos=contact_pos_list,
            frequency=1 / ref_dt,
        )
        loguru.logger.info(f"Saved {out_npz}")
        if act_scene:
            out_npz = f"{file_dir}/trajectory_ikrollout_act.npz"
        else:
            out_npz = f"{file_dir}/trajectory_ikrollout.npz"
        np.savez(
            out_npz,
            qpos=qpos_rollout,
        )
        loguru.logger.info(f"Saved {out_npz}")

    if use_viser_viewer and wait_on_finish:
        loguru.logger.info(
            "IK complete. Keeping Viser server alive on http://{}:{}/ . Press Ctrl+C to exit.",
            viser_host,
            viser_port,
        )
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    tyro.cli(main)
