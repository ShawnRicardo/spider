# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import xml.etree.ElementTree as ET

import loguru
import mujoco
import mujoco.viewer
import numpy as np
import tyro
from loop_rate_limiters import RateLimiter

from spider import ROOT
from spider.io import get_processed_data_dir
from spider.preprocess.workspace_support import (
    SUPPORT_TABLE_HALF_THICKNESS,
    SUPPORT_TABLE_MARGIN,
    SUPPORT_TABLE_COLLISION_MODE_OBJECT_AND_HAND,
    SUPPORT_TABLE_COLLISION_MODE_OBJECT_AND_MANIPULATOR,
    SUPPORT_TABLE_COLLISION_MODE_OBJECT_ONLY,
    WorkspaceSupportSpec,
    compute_workspace_support_spec,
    get_support_table_collision_targets,
    quat_wxyz_to_rotmat,
)

SUPPORT_TABLE_HEIGHT_MODE_TRAJECTORY_MIN = "trajectory_min"
SUPPORT_TABLE_HEIGHT_MODE_FIRST_FRAME_MIN = "first_frame_min"
D435_RENDER_CAMERA_NAME = "d435_optical_render"
D435_OPTICAL_FRAME_SITE_NAME = "d435_optical_frame"


def _format_float(value: float) -> str:
    return f"{value:.6g}"


def _camera_xyaxes(
    pos: list[float], target: list[float], up: list[float] | None = None
) -> list[float]:
    """Return MuJoCo camera xyaxes for a fixed camera looking at target."""
    pos_arr = np.asarray(pos, dtype=float)
    target_arr = np.asarray(target, dtype=float)
    up_arr = np.asarray([0.0, 0.0, 1.0] if up is None else up, dtype=float)

    z_axis = pos_arr - target_arr
    z_axis = z_axis / np.linalg.norm(z_axis)
    x_axis = np.cross(up_arr, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    return [*x_axis.tolist(), *y_axis.tolist()]


def _add_front_camera(
    mj_spec: mujoco.MjSpec,
    robot_type: str,
    support_table_spec=None,
) -> None:
    if robot_type == "asm":
        table_center = (
            np.asarray(support_table_spec.table_center, dtype=np.float64)
            if support_table_spec is not None
            else np.zeros(3, dtype=np.float64)
        )
        if table_center[1] > 0.0 and abs(table_center[1]) >= abs(table_center[0]):
            # OakInk bowl/spoon places the workspace on +Y and the ASM root is
            # rotated to face +Y, so the front render camera should also sit on
            # +Y looking back toward the robot.
            pos = [0.0, 3.15, 1.45]
            target = [0.0, 0.12, 0.72]
        else:
            # Camera-aligned scenes such as milk keep ASM front on +X.
            pos = [3.15, 0.0, 1.45]
            target = [0.12, 0.0, 0.72]
        mj_spec.worldbody.add_camera(
            name="front",
            pos=pos,
            xyaxes=_camera_xyaxes(pos, target),
            mode=mujoco.mjtCamLight.mjCAMLIGHT_FIXED,
        )
        return

    mj_spec.worldbody.add_camera(
        name="front",
        pos=[0.031, 0.941, 0.844],
        xyaxes=[-0.999, 0.033, -0.000, -0.022, -0.667, 0.745],
        mode=mujoco.mjtCamLight.mjCAMLIGHT_TRACKCOM,
    )


def _find_body_with_site(body, site_name: str):
    for site in body.sites:
        if site.name == site_name:
            return body, site
    for child in body.bodies:
        found_body, found_site = _find_body_with_site(child, site_name)
        if found_site is not None:
            return found_body, found_site
    return None, None


def _quat_to_rotmat(quat_wxyz) -> np.ndarray:
    mat = np.zeros(9, dtype=np.float64)
    mujoco.mju_quat2Mat(mat, np.asarray(quat_wxyz, dtype=np.float64))
    return mat.reshape(3, 3)


def _add_d435_optical_camera(
    mj_spec: mujoco.MjSpec,
    robot_type: str,
    fovy: float = 50.0,
) -> None:
    if robot_type != "asm":
        return
    body, site = _find_body_with_site(mj_spec.worldbody, D435_OPTICAL_FRAME_SITE_NAME)
    if site is None or body is None:
        loguru.logger.warning(
            "Skipping D435 render camera because site '{}' was not found.",
            D435_OPTICAL_FRAME_SITE_NAME,
        )
        return

    site_rot = _quat_to_rotmat(site.quat)
    optical_x = site_rot[:, 0]
    optical_y = site_rot[:, 1]
    # MuJoCo camera convention uses +X to the right, +Y up, and looks along -Z.
    # D435 optical uses +X right, +Y down, +Z forward.
    body.add_camera(
        name=D435_RENDER_CAMERA_NAME,
        pos=site.pos,
        xyaxes=[*optical_x.tolist(), *(-optical_y).tolist()],
        fovy=float(fovy),
    )


def _uses_camera_world_alignment(task_info: dict) -> bool:
    return task_info.get("world_to_sim_alignment_mode") == "d435_optical"


def _object_bbox_half_extents(
    task_info: dict,
    side: str,
    margin: float,
) -> list[float] | None:
    size_value = task_info.get(f"{side}_object_bbox_size_xyz_m")
    if size_value is None:
        return None
    size_xyz = np.asarray(size_value, dtype=np.float64)
    if size_xyz.shape != (3,) or not np.isfinite(size_xyz).all():
        loguru.logger.warning(
            "Ignoring invalid {} object bbox size for collision: {}",
            side,
            size_value,
        )
        return None
    if np.any(size_xyz <= 0):
        loguru.logger.warning(
            "Ignoring non-positive {} object bbox size for collision: {}",
            side,
            size_xyz.tolist(),
        )
        return None
    half_extents = 0.5 * size_xyz + max(float(margin), 0.0)
    return half_extents.tolist()


def _object_bbox_world_corners(qpos: np.ndarray, size_xyz: np.ndarray) -> np.ndarray:
    half_extents = 0.5 * np.asarray(size_xyz, dtype=np.float64)
    local_corners = np.array(
        [
            [sx * half_extents[0], sy * half_extents[1], sz * half_extents[2]]
            for sx in (-1.0, 1.0)
            for sy in (-1.0, 1.0)
            for sz in (-1.0, 1.0)
        ],
        dtype=np.float64,
    )
    qpos = np.asarray(qpos, dtype=np.float64)
    rotation = quat_wxyz_to_rotmat(qpos[3:])
    return local_corners @ rotation.T + qpos[:3]


def _active_bbox_trajectory_points(
    task_info: dict,
    embodiment_type: str,
    qpos_obj_right: np.ndarray,
    qpos_obj_left: np.ndarray,
) -> tuple[list[np.ndarray], dict[str, float], dict[str, float]]:
    object_entries = []
    if embodiment_type in ["right", "bimanual"]:
        right_size = task_info.get("right_object_bbox_size_xyz_m")
        if right_size is not None:
            object_entries.append(
                ("right", np.asarray(right_size, dtype=np.float64), qpos_obj_right)
            )
    if embodiment_type in ["left", "bimanual"]:
        left_size = task_info.get("left_object_bbox_size_xyz_m")
        if left_size is not None:
            object_entries.append(
                ("left", np.asarray(left_size, dtype=np.float64), qpos_obj_left)
            )

    trajectory_points = []
    first_frame_min_z = {}
    trajectory_min_z = {}
    for side, size_xyz, qpos_traj in object_entries:
        if (
            size_xyz.shape != (3,)
            or not np.isfinite(size_xyz).all()
            or np.any(size_xyz <= 0)
        ):
            loguru.logger.warning(
                "Skipping invalid {} object bbox size for support table: {}",
                side,
                size_xyz.tolist(),
            )
            continue
        qpos_traj = np.asarray(qpos_traj, dtype=np.float64)
        if qpos_traj.ndim == 1:
            qpos_traj = qpos_traj[None, :]
        if qpos_traj.shape[1] != 7 or len(qpos_traj) == 0:
            loguru.logger.warning(
                "Skipping invalid {} object qpos trajectory for support table: shape={}",
                side,
                qpos_traj.shape,
            )
            continue

        first_corners = _object_bbox_world_corners(qpos_traj[0], size_xyz)
        first_frame_min_z[side] = float(first_corners[:, 2].min())
        side_min_z = np.inf
        for qpos in qpos_traj:
            corners = _object_bbox_world_corners(qpos, size_xyz)
            side_min_z = min(side_min_z, float(corners[:, 2].min()))
            trajectory_points.append(corners)
        trajectory_min_z[side] = float(side_min_z)

    if len(trajectory_points) == 0:
        raise ValueError("No valid object bbox trajectories found for support table.")
    return trajectory_points, first_frame_min_z, trajectory_min_z


def _compute_bbox_support_table_spec(
    task_info: dict,
    embodiment_type: str,
    qpos_obj_right: np.ndarray,
    qpos_obj_left: np.ndarray,
    margin: float,
    half_thickness: float,
    z_offset: float,
    height_mode: str = SUPPORT_TABLE_HEIGHT_MODE_TRAJECTORY_MIN,
    collision_mode: str = SUPPORT_TABLE_COLLISION_MODE_OBJECT_ONLY,
) -> tuple[WorkspaceSupportSpec, dict[str, float]]:
    trajectory_points, first_frame_min_z, trajectory_min_z = (
        _active_bbox_trajectory_points(
            task_info=task_info,
            embodiment_type=embodiment_type,
            qpos_obj_right=qpos_obj_right,
            qpos_obj_left=qpos_obj_left,
        )
    )
    all_points = np.concatenate(trajectory_points, axis=0)
    lower = all_points.min(axis=0)
    upper = all_points.max(axis=0)
    object_first_frame_min_z = float(min(first_frame_min_z.values()))
    if height_mode == SUPPORT_TABLE_HEIGHT_MODE_TRAJECTORY_MIN:
        table_anchor_z = float(lower[2])
    elif height_mode == SUPPORT_TABLE_HEIGHT_MODE_FIRST_FRAME_MIN:
        table_anchor_z = object_first_frame_min_z
    else:
        raise ValueError(
            "support_table_height_mode must be one of "
            f"{[SUPPORT_TABLE_HEIGHT_MODE_TRAJECTORY_MIN, SUPPORT_TABLE_HEIGHT_MODE_FIRST_FRAME_MIN]}, "
            f"got {height_mode!r}"
        )
    table_surface_z = float(table_anchor_z + z_offset)

    xy_center = 0.5 * (lower[:2] + upper[:2])
    xy_half_extent = 0.5 * (upper[:2] - lower[:2]) + max(float(margin), 0.0)
    table_half_thickness = max(float(half_thickness), 1e-4)
    table_center = np.array(
        [xy_center[0], xy_center[1], table_surface_z - table_half_thickness],
        dtype=np.float64,
    )
    table_size = np.array(
        [xy_half_extent[0], xy_half_extent[1], table_half_thickness],
        dtype=np.float64,
    )
    return (
        WorkspaceSupportSpec(
            robot_height=0.0,
            table_surface_z=table_surface_z,
            workspace_z_offset=0.0,
            workspace_xy_offset=np.zeros(2, dtype=np.float64),
            object_first_frame_min_z=object_first_frame_min_z,
            table_center=table_center,
            table_size=table_size,
            collision_mode=collision_mode,
        ),
        trajectory_min_z,
    )


def _add_object_xyzrpy_actuators(
    xml_text: str,
    object_armature: float,
    object_frictionloss: float,
    object_pos_kp: float,
    object_pos_kd: float,
    object_rot_kp: float,
    object_rot_kd: float,
) -> str:
    root = ET.fromstring(xml_text)
    worldbody = root.find("worldbody")
    if worldbody is None:
        return xml_text

    actuator = root.find("actuator")
    if actuator is None:
        actuator = ET.SubElement(root, "actuator")

    existing_actuators = {
        elem.get("name")
        for elem in actuator.findall("*")
        if elem.get("name") is not None
    }

    joint_defs = [
        ("pos_x", "slide", "1 0 0", "pos"),
        ("pos_y", "slide", "0 1 0", "pos"),
        ("pos_z", "slide", "0 0 1", "pos"),
        ("rot_x", "hinge", "1 0 0", "rot"),
        ("rot_y", "hinge", "0 1 0", "rot"),
        ("rot_z", "hinge", "0 0 1", "rot"),
    ]

    for side in ("right", "left"):
        body = worldbody.find(f".//body[@name='{side}_object']")
        if body is None:
            continue
        free_joint_name = f"{side}_object_joint"
        free_joint = None
        for joint in body.findall("joint"):
            if joint.get("name") == free_joint_name:
                free_joint = joint
                break
        if free_joint is None:
            continue

        body_children = list(body)
        insert_index = body_children.index(free_joint)
        body.remove(free_joint)

        for offset, (suffix, joint_type, axis, group) in enumerate(joint_defs):
            joint_name = f"{side}_object_{suffix}"
            joint_attrs = {
                "name": joint_name,
                "type": joint_type,
                "axis": axis,
                "armature": _format_float(object_armature),
                "frictionloss": _format_float(object_frictionloss),
            }
            body.insert(insert_index + offset, ET.Element("joint", joint_attrs))

            actuator_name = joint_name
            if actuator_name not in existing_actuators:
                kp = object_pos_kp if group == "pos" else object_rot_kp
                kd = object_pos_kd if group == "pos" else object_rot_kd
                actuator_attrs = {
                    "name": actuator_name,
                    "joint": joint_name,
                    "kp": _format_float(kp),
                    "kv": _format_float(kd),
                }
                actuator.append(ET.Element("position", actuator_attrs))
                existing_actuators.add(actuator_name)

    try:
        ET.indent(root, space="  ")
    except AttributeError:
        pass
    return ET.tostring(root, encoding="unicode")


def _bind_groundplane_material_textures(xml_text: str) -> str:
    root = ET.fromstring(xml_text)
    asset = root.find("asset")
    if asset is None:
        return xml_text

    desired = {
        "groundplane": "groundplane",
        "right_groundplane": "right_groundplane",
        "left_groundplane": "left_groundplane",
        "right_object_visual_material": "right_object_visual_texture",
        "left_object_visual_material": "left_object_visual_texture",
    }
    for material in asset.findall("material"):
        name = material.get("name")
        texture = desired.get(name)
        if texture is not None:
            material.set("texture", texture)

    try:
        ET.indent(root, space="  ")
    except AttributeError:
        pass
    return ET.tostring(root, encoding="unicode")

# 选择物体 obj，将物体加入到 xml
def _select_object_visual_file(
    mesh_dir: str | None,
    use_visual_mesh_as_collision: bool,
) -> tuple[str | None, str | None, bool]:
    if not mesh_dir:
        return None, None, False
    plain_visual_file = f"{mesh_dir}/visual.obj"
    mesh_textured_visual_file = f"{mesh_dir}/visual_mesh_textured.obj"
    mesh_texture_file = f"{mesh_dir}/visual_mesh_texture.png"
    textured_visual_file = f"{mesh_dir}/visual_textured.obj"
    texture_file = f"{mesh_dir}/visual_texture.png"
    if (
        not use_visual_mesh_as_collision
        and os.path.exists(mesh_textured_visual_file)
        and os.path.exists(mesh_texture_file)
    ):
        return mesh_textured_visual_file, mesh_texture_file, True
    if (
        not use_visual_mesh_as_collision
        and os.path.exists(textured_visual_file)
        and os.path.exists(texture_file)
    ):
        return textured_visual_file, texture_file, True
    return plain_visual_file, None, False


def _add_support_table_pairs(
    mj_spec: mujoco.MjSpec,
    support_table_name: str,
    object_collision_names: list[str],
    hand_collision_names: list[str],
    collision_mode: str,
    solref: list[float],
    friction: list[float],
) -> int:
    pair_count = 0
    target_geom_names = get_support_table_collision_targets(
        object_collision_names,
        hand_collision_names,
        collision_mode=collision_mode,
    )
    for geom_name in target_geom_names:
        condim = 4 if ("thumb" in geom_name or "index" in geom_name) else 3
        mj_spec.add_pair(
            name=f"{support_table_name}_{geom_name}",
            geomname1=support_table_name,
            geomname2=geom_name,
            solref=solref,
            friction=friction,
            condim=condim,
        )
        pair_count += 1
    return pair_count


def main(
    dataset_dir: str = f"{ROOT}/../example_datasets",
    dataset_name: str = "oakink",
    robot_type: str = "allegro",
    embodiment_type: str = "bimanual",
    task: str = "pick_spoon_bowl",
    data_id: int = 0,
    hand_floor_collision: bool = False,
    object_floor_collision: bool = True,
    object_object_collision: bool = True,
    object_density: float = 1000,
    use_visual_mesh_as_collision: bool = False,
    object_bbox_collision: bool = False,
    object_bbox_collision_margin: float = 0.003,
    support_table_from_bbox: bool = False,
    support_table_collision_mode: str = SUPPORT_TABLE_COLLISION_MODE_OBJECT_ONLY,
    support_table_margin: float = SUPPORT_TABLE_MARGIN,
    support_table_half_thickness: float = SUPPORT_TABLE_HALF_THICKNESS,
    support_table_z_offset: float = 0.0,
    support_table_height_mode: str = SUPPORT_TABLE_HEIGHT_MODE_TRAJECTORY_MIN,
    robot_object_collision: bool = True,
    object_armature: float = 0.0001,
    object_frictionloss: float = 0.0001,
    friction_scale: float = 1.0,
    show_viewer: bool = True,
    act_scene: bool = False,
):
    dataset_dir = os.path.abspath(dataset_dir)
    if support_table_collision_mode not in {
        SUPPORT_TABLE_COLLISION_MODE_OBJECT_ONLY,
        SUPPORT_TABLE_COLLISION_MODE_OBJECT_AND_HAND,
        SUPPORT_TABLE_COLLISION_MODE_OBJECT_AND_MANIPULATOR,
    }:
        raise ValueError(
            "support_table_collision_mode must be one of "
            f"{[SUPPORT_TABLE_COLLISION_MODE_OBJECT_ONLY, SUPPORT_TABLE_COLLISION_MODE_OBJECT_AND_HAND, SUPPORT_TABLE_COLLISION_MODE_OBJECT_AND_MANIPULATOR]}, "
            f"got {support_table_collision_mode!r}"
        )
    if support_table_height_mode not in {
        SUPPORT_TABLE_HEIGHT_MODE_TRAJECTORY_MIN,
        SUPPORT_TABLE_HEIGHT_MODE_FIRST_FRAME_MIN,
    }:
        raise ValueError(
            "support_table_height_mode must be one of "
            f"{[SUPPORT_TABLE_HEIGHT_MODE_TRAJECTORY_MIN, SUPPORT_TABLE_HEIGHT_MODE_FIRST_FRAME_MIN]}, "
            f"got {support_table_height_mode!r}"
        )
    processed_dir = get_processed_data_dir(
        dataset_dir=dataset_dir,
        dataset_name=dataset_name,
        robot_type=robot_type,
        embodiment_type=embodiment_type,
        task=task,
        data_id=data_id,
    )
    os.makedirs(processed_dir, exist_ok=True)

    # choose robot XML based on embodiment_type
    robots_assets_dir = (
        f"{dataset_dir}/processed/{dataset_name}/assets/robots/{robot_type}"
    )
    robot_xml_name = (
        "bimanual.xml" if embodiment_type == "bimanual" else f"{embodiment_type}.xml"
    )
    robot_xml_path = f"{robots_assets_dir}/{robot_xml_name}"
    if not os.path.exists(robot_xml_path):
        raise FileNotFoundError(f"Robot XML not found: {robot_xml_path}")

    # load robot xml as base scene
    mj_spec = mujoco.MjSpec.from_file(robot_xml_path)
    if robot_type == "asm":
        mj_spec.option.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_FILTERPARENT)
    use_default_collision_contacts = robot_type == "asm" and robot_object_collision

    def _default_contact_attrs(
        enabled: bool,
        condim: int = 3,
        *,
        contype: int = 1,
        conaffinity: int = 1,
    ) -> dict:
        attrs = {
            "contype": contype if enabled else 0,
            "conaffinity": conaffinity if enabled else 0,
        }
        if enabled:
            attrs["condim"] = condim
        return attrs

    def _object_contact_attrs(enabled: bool, condim: int = 3) -> dict:
        if (
            enabled
            and support_table_spec is not None
            and support_table_spec.collision_mode
            == SUPPORT_TABLE_COLLISION_MODE_OBJECT_ONLY
        ):
            # Keep object contacts with the robot on bit 1, and allow the table
            # to collide only with objects through bit 2.
            return _default_contact_attrs(
                True,
                condim=condim,
                contype=0b11,
                conaffinity=0b11,
            )
        return _default_contact_attrs(enabled, condim=condim)

    def _support_table_contact_attrs(enabled: bool, condim: int = 3) -> dict:
        if (
            enabled
            and support_table_spec is not None
            and support_table_spec.collision_mode
            == SUPPORT_TABLE_COLLISION_MODE_OBJECT_ONLY
        ):
            return _default_contact_attrs(
                True,
                condim=condim,
                contype=0b10,
                conaffinity=0b10,
            )
        return _default_contact_attrs(enabled, condim=condim)

    # Configure compiler for robust relative mesh loading
    assets_root_dir = f"{dataset_dir}/processed/{dataset_name}/assets"
    # get relative dir of assets_root_dir and scene.xml
    assets_root_dir_rel = os.path.relpath(assets_root_dir, f"{processed_dir}/..")
    # Set meshdir relative to where scene.xml will be saved
    original_meshdir = mj_spec.meshdir
    mj_spec.meshdir = assets_root_dir_rel
    mj_spec.texturedir = assets_root_dir_rel

    # Rewrite robot mesh file paths to be relative to the assets directory
    robots_dir_abs = f"{assets_root_dir}/robots/{robot_type}"
    for mesh in getattr(mj_spec, "meshes", []):
        original = mesh.file
        # Determine absolute location of the original reference
        if os.path.isabs(original):
            candidate_abs = original
        else:
            # Prefer resolving relative to the robot's assets directory
            candidate_abs = os.path.normpath(
                os.path.join(robots_dir_abs, original_meshdir, original)
            )
        # Compute path relative to assets root
        try:
            file_rel_to_assets = os.path.relpath(candidate_abs, assets_root_dir)
        except ValueError:
            # Fallback: keep original if relpath fails (e.g., different drives)
            file_rel_to_assets = original
        mesh.file = file_rel_to_assets

    # load contact info for placing contact sites (optional)
    keypoint_data_dir = get_processed_data_dir(
        dataset_dir=dataset_dir,
        dataset_name=dataset_name,
        robot_type="mano",
        embodiment_type=embodiment_type,
        task=task,
        data_id=data_id,
    )
    contact_npz_path = f"{keypoint_data_dir}/trajectory_keypoints.npz"
    loaded_data = np.load(contact_npz_path)
    try:
        contact_left = loaded_data["contact_left"]
        contact_pos_left = loaded_data["contact_pos_left"]
        contact_right = loaded_data["contact_right"]
        contact_pos_right = loaded_data["contact_pos_right"]
    except KeyError:
        loguru.logger.warning(
            f"No contact data found at {contact_npz_path}; falling back to zeros"
        )
        contact_left = np.zeros((10, 5))
        contact_pos_left = np.zeros((5, 3))
        contact_right = np.zeros((10, 5))
        contact_pos_right = np.zeros((5, 3))
    finger_names = [
        "thumb_tip",
        "index_tip",
        "middle_tip",
        "ring_tip",
        "pinky_tip",
    ]
    if robot_type in ["allegro", "metahand"]:
        finger_names = finger_names[:4]

    task_info_path = f"{keypoint_data_dir}/../task_info.json"
    task_info = {}
    with open(task_info_path) as f:
        task_info = json.load(f)
    qpos_obj_right_first = loaded_data["qpos_obj_right"][0].copy()
    qpos_obj_left_first = loaded_data["qpos_obj_left"][0].copy()
    right_convex_dir = task_info.get("right_object_convex_dir")
    right_convex_dir = f"{dataset_dir}/{right_convex_dir}"
    left_convex_dir = task_info.get("left_object_convex_dir")
    left_convex_dir = f"{dataset_dir}/{left_convex_dir}"
    right_mesh_dir = task_info.get("right_object_mesh_dir")
    right_mesh_dir = f"{dataset_dir}/{right_mesh_dir}"
    left_mesh_dir = task_info.get("left_object_mesh_dir")
    left_mesh_dir = f"{dataset_dir}/{left_mesh_dir}"

    support_table_spec = None
    support_table_per_object_min_z = {}
    disable_workspace_support = (
        robot_type == "asm"
        and embodiment_type == "bimanual"
        and _uses_camera_world_alignment(task_info)
    )
    if disable_workspace_support:
        task_info.pop("workspace_z_offset", None)
        task_info.pop("workspace_xy_offset", None)
        loguru.logger.info(
            "Skipping ASM workspace support because task data is already aligned by camera."
        )
        if support_table_from_bbox:
            support_table_spec, support_table_per_object_min_z = (
                _compute_bbox_support_table_spec(
                    task_info=task_info,
                    embodiment_type=embodiment_type,
                    qpos_obj_right=loaded_data["qpos_obj_right"],
                    qpos_obj_left=loaded_data["qpos_obj_left"],
                    margin=support_table_margin,
                    half_thickness=support_table_half_thickness,
                    z_offset=support_table_z_offset,
                    height_mode=support_table_height_mode,
                    collision_mode=support_table_collision_mode,
                )
            )
            task_info.update(support_table_spec.to_json_dict())
            task_info["support_table_height_mode"] = support_table_height_mode
            task_info["support_table_source"] = (
                "object_bbox_first_frame_bottom"
                if support_table_height_mode == SUPPORT_TABLE_HEIGHT_MODE_FIRST_FRAME_MIN
                else "object_bbox_trajectory_bottom"
            )
            task_info["object_bbox_first_frame_min_z"] = float(
                support_table_spec.object_first_frame_min_z
            )
            task_info["object_bbox_trajectory_min_z"] = float(
                min(support_table_per_object_min_z.values())
            )
            task_info["object_bbox_table_anchor_z"] = float(
                support_table_spec.table_surface_z - support_table_z_offset
            )
            task_info["support_table_z_offset"] = float(support_table_z_offset)
            task_info["robot_object_collision"] = bool(robot_object_collision)
            loguru.logger.info(
                "ASM bbox support_table: surface_z={:.4f}, z_offset={:.4f}, height_mode={}, center={}, size={}, trajectory_min_z={}, collision_mode={}, robot_object_collision={}",
                support_table_spec.table_surface_z,
                support_table_z_offset,
                support_table_height_mode,
                support_table_spec.table_center.tolist(),
                support_table_spec.table_size.tolist(),
                support_table_per_object_min_z,
                support_table_spec.collision_mode,
                robot_object_collision,
            )
    if robot_type == "asm" and embodiment_type == "bimanual" and not disable_workspace_support:
        support_table_spec, support_table_per_object_min_z = (
            compute_workspace_support_spec(
                dataset_dir=dataset_dir,
                robot_xml_path=robot_xml_path,
                task_info=task_info,
                dataset_name=dataset_name,
                robot_type=robot_type,
                embodiment_type=embodiment_type,
                task=task,
                qpos_obj_right_first=qpos_obj_right_first,
                qpos_obj_left_first=qpos_obj_left_first,
                collision_mode=SUPPORT_TABLE_COLLISION_MODE_OBJECT_ONLY,
            )
        )
        task_info.update(support_table_spec.to_json_dict())
        loguru.logger.info(
            "ASM workspace support: robot_height={:.4f}, table_surface_z={:.4f}, object_first_frame_min_z={:.4f}, workspace_z_offset={:.4f}, workspace_xy_offset={}",
            support_table_spec.robot_height,
            support_table_spec.table_surface_z,
            support_table_spec.object_first_frame_min_z,
            support_table_spec.workspace_z_offset,
            support_table_spec.workspace_xy_offset.tolist(),
        )
        loguru.logger.info(
            "ASM support_table: center={}, size={}, collision_mode={}",
            support_table_spec.table_center.tolist(),
            support_table_spec.table_size.tolist(),
            support_table_spec.collision_mode,
        )
        if len(support_table_per_object_min_z) > 1:
            per_object_values = list(support_table_per_object_min_z.values())
            if max(per_object_values) - min(per_object_values) > 1e-4:
                loguru.logger.warning(
                    "ASM support_table aligned to global lowest object min-z; per-object min-z={}",
                    support_table_per_object_min_z,
                )

    # add assets
    mj_spec.add_texture(
        name="skybox",
        builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
        rgb1=[0.3, 0.5, 0.7],
        rgb2=[0, 0, 0],
        width=512,
        height=3072,
    )
    mj_spec.add_texture(
        name="groundplane",
        builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
        mark=mujoco.mjtMark.mjMARK_EDGE,
        rgb1=[0.2, 0.3, 0.4],
        rgb2=[0.1, 0.2, 0.3],
        markrgb=[0.8, 0.8, 0.8],
        width=300,
        height=300,
    )
    existing_material_names = {m.name for m in getattr(mj_spec, "materials", [])}
    for ground_material_name in ["groundplane", "right_groundplane", "left_groundplane"]:
        if ground_material_name in existing_material_names:
            continue
        mj_spec.add_material(
            name=ground_material_name,
            textures=["groundplane"],
            texuniform=True,
            texrepeat=[5, 5],
            reflectance=0.2,
        )
        existing_material_names.add(ground_material_name)

    # add floor
    if embodiment_type in ["right", "bimanual"]:
        material_name = "right_groundplane"
    else:
        material_name = "left_groundplane"
    mj_spec.worldbody.add_geom(
        name="floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[0, 0, 0.05],
        pos=[0, 0, 0.0],
        material=material_name,
        **_default_contact_attrs(use_default_collision_contacts),
    )
    if support_table_spec is not None:
        mj_spec.worldbody.add_geom(
            name="support_table",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=support_table_spec.table_size.tolist(),
            pos=support_table_spec.table_center.tolist(),
            group=0,
            rgba=[0.56, 0.49, 0.39, 1.0],
            **_support_table_contact_attrs(use_default_collision_contacts),
        )

    # Visual meshes (non-colliding)
    right_visual_file, right_visual_texture_file, right_visual_is_textured = _select_object_visual_file(
        right_mesh_dir,
        use_visual_mesh_as_collision,
    )
    left_visual_file, left_visual_texture_file, left_visual_is_textured = _select_object_visual_file(
        left_mesh_dir,
        use_visual_mesh_as_collision,
    )
    if (
        embodiment_type in ["right", "bimanual"]
        and right_visual_file
        and os.path.exists(right_visual_file)
    ):
        file_rel_to_meshdir = os.path.relpath(right_visual_file, assets_root_dir)
        right_mesh_kwargs = {"name": "right_visual", "file": file_rel_to_meshdir}
        if right_visual_is_textured:
            right_mesh_kwargs["inertia"] = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
        # 把物体 obj mesh 加入到 xml
        mj_spec.add_mesh(**right_mesh_kwargs)
        if right_visual_is_textured and right_visual_texture_file:
            texture_rel = os.path.relpath(
                right_visual_texture_file,
                assets_root_dir,
            )
            mj_spec.add_texture(
                name="right_object_visual_texture",
                type=mujoco.mjtTexture.mjTEXTURE_2D,
                file=texture_rel,
                colorspace=mujoco.mjtColorSpace.mjCOLORSPACE_SRGB,
            )
            mj_spec.add_material(
                name="right_object_visual_material",
                textures=["right_object_visual_texture"],
                rgba=[1.0, 1.0, 1.0, 1.0],
                specular=0.0,
                shininess=0.0,
            )
    if (
        embodiment_type in ["left", "bimanual"]
        and left_visual_file
        and os.path.exists(left_visual_file)
    ):
        file_rel_to_meshdir = os.path.relpath(left_visual_file, assets_root_dir)
        left_mesh_kwargs = {"name": "left_visual", "file": file_rel_to_meshdir}
        if left_visual_is_textured:
            left_mesh_kwargs["inertia"] = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
        mj_spec.add_mesh(**left_mesh_kwargs)
        if left_visual_is_textured and left_visual_texture_file:
            texture_rel = os.path.relpath(
                left_visual_texture_file,
                assets_root_dir,
            )
            mj_spec.add_texture(
                name="left_object_visual_texture",
                type=mujoco.mjtTexture.mjTEXTURE_2D,
                file=texture_rel,
                colorspace=mujoco.mjtColorSpace.mjCOLORSPACE_SRGB,
            )
            mj_spec.add_material(
                name="left_object_visual_material",
                textures=["left_object_visual_texture"],
                rgba=[1.0, 1.0, 1.0, 1.0],
                specular=0.0,
                shininess=0.0,
            )

    # Right object meshes
    right_object_files = []
    if embodiment_type in ["right", "bimanual"]:
        if use_visual_mesh_as_collision and right_visual_file:
            if os.path.exists(right_visual_file):
                # Reuse the visual mesh for collision (no extra collision mesh).
                right_object_files = ["visual"]
        elif right_convex_dir and os.path.isdir(right_convex_dir):
            right_object_files = sorted(
                [f for f in os.listdir(right_convex_dir) if f.endswith(".obj")]
            )
            for f in right_object_files:
                suffix = f.split(".")[0]
                file_abs = f"{right_convex_dir}/{f}"
                file_rel_to_meshdir = os.path.relpath(file_abs, assets_root_dir)
                mj_spec.add_mesh(name=f"right_{suffix}", file=file_rel_to_meshdir)

    # Left object meshes
    left_object_files = []
    if embodiment_type in ["left", "bimanual"]:
        if use_visual_mesh_as_collision and left_visual_file:
            if os.path.exists(left_visual_file):
                # Reuse the visual mesh for collision (no extra collision mesh).
                left_object_files = ["visual"]
        elif left_convex_dir and os.path.isdir(left_convex_dir):
            left_object_files = sorted(
                [f for f in os.listdir(left_convex_dir) if f.endswith(".obj")]
            )
            for f in left_object_files:
                suffix = f.split(".")[0]
                file_abs = f"{left_convex_dir}/{f}"
                file_rel_to_meshdir = os.path.relpath(file_abs, assets_root_dir)
                mj_spec.add_mesh(name=f"left_{suffix}", file=file_rel_to_meshdir)

    # add object to model
    # 将物体模型动态加到 scene，并挂 7-DoF 自由度
    # 右手物体和左手物体
    right_object_collision_names = []
    if embodiment_type in ["right", "bimanual"]:
        right_object_handle = mj_spec.worldbody.add_body(
            name="right_object",
            mocap=False,
        )
        right_object_handle.add_joint(
            name="right_object_joint",
            type=mujoco.mjtJoint.mjJNT_FREE,
            armature=object_armature,
            frictionloss=object_frictionloss,
        )
        # add geom to object
        for obj_file in right_object_files:
            suffix = obj_file.split(".")[0]
            is_visual_collision = use_visual_mesh_as_collision and suffix == "visual"
            geom_name = f"right_object_{suffix}"
            use_mesh_for_contact = (
                (suffix.isdigit() or is_visual_collision)
                and not object_bbox_collision
            )
            if use_mesh_for_contact:
                rgba = [0, 1, 0, 0]
                density = object_density
                if is_visual_collision:
                    geom_name = "right_object_collision_visual"
                right_object_collision_names.append(geom_name)
                group = 3
            elif suffix.isdigit() or is_visual_collision:
                # With bbox collision enabled, keep mesh geoms for mass/inertia
                # only. Otherwise mesh contacts plus bbox contacts double-count
                # impulses and can launch the object during MJWP rollouts.
                rgba = [0, 1, 0, 0]
                density = object_density
                group = 3
            else:
                rgba = [1, 1, 1, 1]
                density = 0
                group = 0
            right_object_handle.add_geom(
                name=geom_name,
                type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname=f"right_{suffix}",
                pos=[0, 0, 0],
                rgba=rgba,
                density=density,
                group=group,
                **_object_contact_attrs(
                    use_default_collision_contacts and use_mesh_for_contact
                ),
            )
        if object_bbox_collision:
            bbox_half_extents = _object_bbox_half_extents(
                task_info,
                side="right",
                margin=object_bbox_collision_margin,
            )
            if bbox_half_extents is not None:
                geom_name = "right_object_bbox_collision"
                right_object_collision_names.append(geom_name)
                right_object_handle.add_geom(
                    name=geom_name,
                    type=mujoco.mjtGeom.mjGEOM_BOX,
                    pos=[0, 0, 0],
                    size=bbox_half_extents,
                    rgba=[0.0, 0.8, 0.0, 0.0],
                    density=0,
                    group=3,
                    **_object_contact_attrs(use_default_collision_contacts),
                )
        # add site to object
        right_object_handle.add_site(
            name="right_object",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.01, 0.02, 0.03],
            pos=[0, 0, 0],
            rgba=[1, 0, 0, 1],
            group=3,
        )
        # add visual mesh (non-colliding)
        if "right_visual" in [m.name for m in mj_spec.meshes]:
            right_visual_geom_kwargs = {
                "name": "right_object_visual",
                "type": mujoco.mjtGeom.mjGEOM_MESH,
                "meshname": "right_visual",
                "pos": [0, 0, 0],
                "conaffinity": 0,
                "contype": 0,
                "rgba": [1, 1, 1, 1],
                "density": 0,
                "group": 0,
            }
            if right_visual_is_textured:
                right_visual_geom_kwargs["material"] = "right_object_visual_material"
            right_object_handle.add_geom(**right_visual_geom_kwargs)
        # add trace site to the object (for visualization)
        right_object_handle.add_site(
            name="trace_right_object",
            pos=[0, 0, 0],
            size=[0.01, 0.01, 0.01],
            rgba=[0, 1, 0, 1],
            group=4,
        )
        # add contact site to the object (for virtual constraint)
        # 加入 site 关键点，供 IK/跟踪使用
        for i, finger_name in enumerate(finger_names):
            right_object_handle.add_site(
                name=f"track_object_right_{finger_name}",
                pos=contact_pos_right[i],
                size=[0.01, 0.01, 0.01],
                rgba=[0, 1, 0, 1],
                group=4,
            )
            mocap_handle = mj_spec.worldbody.add_body(
                name=f"ref_object_right_{finger_name}",
                pos=[0, 0, 0],
                quat=[1, 0, 0, 0],
                mocap=True,
            )
            mocap_handle.add_site(
                name=f"ref_object_right_{finger_name}",
                pos=[0, 0, 0],
                size=[0.02, 0.02, 0.02],
                group=4,
                rgba=[0, 1, 0, 1],
            )
            mocap_handle = mj_spec.worldbody.add_body(
                name=f"ref_hand_right_{finger_name}",
                pos=[0, 0, 0],
                quat=[1, 0, 0, 0],
                mocap=True,
            )
            mocap_handle.add_site(
                name=f"ref_hand_right_{finger_name}",
                pos=[0, 0, 0],
                size=[0.02, 0.02, 0.02],
                group=4,
                rgba=[0, 1, 0, 1],
            )

    left_object_collision_names = []
    if embodiment_type in ["left", "bimanual"]:
        left_object_handle = mj_spec.worldbody.add_body(
            name="left_object",
            mocap=False,
            gravcomp=(
                1 if len(left_object_files) == 0 else 0
            ),  # if left object is not present, set gravcomp to 1 to avoid gravity
        )
        left_joint_handle = left_object_handle.add_joint(
            name="left_object_joint",
            type=mujoco.mjtJoint.mjJNT_FREE,
            armature=object_armature,
            frictionloss=object_frictionloss,
        )
        # add geom to object
        for obj_file in left_object_files:
            suffix = obj_file.split(".")[0]
            is_visual_collision = use_visual_mesh_as_collision and suffix == "visual"
            geom_name = f"left_object_{suffix}"
            use_mesh_for_contact = (
                (suffix.isdigit() or is_visual_collision)
                and not object_bbox_collision
            )
            if use_mesh_for_contact:
                rgba = [0, 1, 0, 0]
                density = object_density
                if is_visual_collision:
                    geom_name = "left_object_collision_visual"
                left_object_collision_names.append(geom_name)
                group = 3
            elif suffix.isdigit() or is_visual_collision:
                rgba = [0, 1, 0, 0]
                density = object_density
                group = 3
            else:
                rgba = [1, 1, 1, 1]
                density = 0
                group = 0
            left_object_handle.add_geom(
                name=geom_name,
                type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname=f"left_{suffix}",
                pos=[0, 0, 0],
                rgba=rgba,
                density=density,
                group=group,
                **_object_contact_attrs(
                    use_default_collision_contacts and use_mesh_for_contact
                ),
            )
        if object_bbox_collision:
            bbox_half_extents = _object_bbox_half_extents(
                task_info,
                side="left",
                margin=object_bbox_collision_margin,
            )
            if bbox_half_extents is not None:
                geom_name = "left_object_bbox_collision"
                left_object_collision_names.append(geom_name)
                left_object_handle.add_geom(
                    name=geom_name,
                    type=mujoco.mjtGeom.mjGEOM_BOX,
                    pos=[0, 0, 0],
                    size=bbox_half_extents,
                    rgba=[0.0, 0.8, 0.0, 0.0],
                    density=0,
                    group=3,
                    **_object_contact_attrs(use_default_collision_contacts),
                )
        # add mass to object if there is no left object
        if len(left_object_files) == 0:
            left_object_handle.add_geom(
                name="left_object_mass",
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                pos=[0.5, 0.5, 0.5],  # put it far away to avoid collision
                size=[0.1, 0.1, 0.1],
                density=10,
                rgba=[0.0, 0.0, 0.0, 0.0],
                group=3,
            )
            left_joint_handle.frictionloss = 1.0
            left_joint_handle.armature = 1.0
        # add site to object
        left_object_handle.add_site(
            name="left_object",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.01, 0.02, 0.03],
            pos=[0, 0, 0],
            rgba=[1, 0, 0, 1],
            group=3,
        )
        # add visual mesh (non-colliding)
        if "left_visual" in [m.name for m in mj_spec.meshes]:
            left_visual_geom_kwargs = {
                "name": "left_object_visual",
                "type": mujoco.mjtGeom.mjGEOM_MESH,
                "meshname": "left_visual",
                "pos": [0, 0, 0],
                "conaffinity": 0,
                "contype": 0,
                "rgba": [1, 1, 1, 1],
                "density": 0,
                "group": 0,
            }
            if left_visual_is_textured:
                left_visual_geom_kwargs["material"] = "left_object_visual_material"
            left_object_handle.add_geom(**left_visual_geom_kwargs)
        # add trace site to the object
        left_object_handle.add_site(
            name="trace_left_object",
            pos=[0, 0, 0],
            size=[0.01, 0.01, 0.01],
            rgba=[0, 1, 0, 1],
            group=4,
        )
        # add contact site to the object
        for i, finger_name in enumerate(finger_names):
            # if left object is not present, add contact site to the right object
            if len(left_object_files) == 0:
                handle = right_object_handle
            else:
                handle = left_object_handle
            handle.add_site(
                name=f"track_object_left_{finger_name}",
                pos=contact_pos_left[i],
                size=[0.01, 0.01, 0.01],
                rgba=[0, 1, 0, 1],
                group=4,
            )
            mocap_handle = mj_spec.worldbody.add_body(
                name=f"ref_object_left_{finger_name}",
                pos=[0, 0, 0],
                quat=[1, 0, 0, 0],
                mocap=True,
            )
            mocap_handle.add_site(
                name=f"ref_object_left_{finger_name}",
                pos=[0, 0, 0],
                size=[0.02, 0.02, 0.02],
                group=4,
                rgba=[0, 1, 0, 1],
            )
            mocap_handle = mj_spec.worldbody.add_body(
                name=f"ref_hand_left_{finger_name}",
                pos=[0, 0, 0],
                quat=[1, 0, 0, 0],
                mocap=True,
            )
            mocap_handle.add_site(
                name=f"ref_hand_left_{finger_name}",
                pos=[0, 0, 0],
                size=[0.02, 0.02, 0.02],
                group=4,
                rgba=[0, 1, 0, 1],
            )

    object_collision_names = right_object_collision_names + left_object_collision_names
    loguru.logger.info(f"Added {len(object_collision_names)} objects to model")

    def _collision_names_with_prefix(prefix: str) -> list[str]:
        names = []
        for geom_id in range(len(mj_spec.geoms)):
            geom = mj_spec.geoms[geom_id]
            name = geom.name or ""
            if name.startswith(prefix):
                names.append(name)
        return names

    hand_collision_names = _collision_names_with_prefix("collision_hand_")
    arm_collision_names = _collision_names_with_prefix("collision_arm_")
    body_collision_names = _collision_names_with_prefix("collision_body_")
    manipulator_collision_names = hand_collision_names + arm_collision_names
    robot_collision_names = manipulator_collision_names + body_collision_names

    if use_default_collision_contacts:
        loguru.logger.info(
            "Using {} ASM URDF collision mesh geoms with default geom collision masks; no explicit contact pairs added",
            len(robot_collision_names),
        )
    else:
        # add contact pairs
        default_solref = [0.02, 1]
        default_friction = [
            1.0 * friction_scale,
            1.0 * friction_scale,
            0.1 * friction_scale,
            0.0,
            0.0,
        ]
        small_friction = [
            0.01 * friction_scale,
            0.01 * friction_scale,
            0.0001 * friction_scale,
            0.0,
            0.0,
        ]

        def _collision_condim(name: str) -> int:
            if name.startswith("collision_hand_") and (
                "thumb" in name or "index" in name
            ):
                return 4
            return 3

        collision_pair_keys: set[tuple[str, str]] = set()

        def _add_collision_pair_once(
            geomname1: str,
            geomname2: str,
            *,
            friction: list[float],
            condim: int,
        ) -> bool:
            if geomname1 == geomname2:
                return False
            key = tuple(sorted((geomname1, geomname2)))
            if key in collision_pair_keys:
                return False
            collision_pair_keys.add(key)
            mj_spec.add_pair(
                name=f"{geomname1}_{geomname2}",
                geomname1=geomname1,
                geomname2=geomname2,
                solref=default_solref,
                friction=friction,
                condim=condim,
            )
            return True

        robot_collision_names_for_object = (
            robot_collision_names + ["floor"]
            if object_floor_collision
            else robot_collision_names
        )

        contact_cnt = 0

        # robot URDF collision mesh <-> object collision
        if robot_object_collision:
            for object_collision_name in object_collision_names:
                for robot_collision_name in robot_collision_names_for_object:
                    condim = _collision_condim(robot_collision_name)
                    friction = default_friction
                    if _add_collision_pair_once(
                        robot_collision_name,
                        object_collision_name,
                        friction=friction,
                        condim=condim,
                    ):
                        contact_cnt += 1
        else:
            loguru.logger.info(
                "Skipping robot-object contact pairs; robot_object_collision=false"
            )

        # object <-> object collision
        if (
            object_object_collision
            and embodiment_type == "bimanual"
            and len(right_object_collision_names) > 0
            and len(left_object_collision_names) > 0
        ):
            for right_object_collision_name in right_object_collision_names:
                for left_object_collision_name in left_object_collision_names:
                    mj_spec.add_pair(
                        name=f"{right_object_collision_name}_{left_object_collision_name}",
                        geomname1=right_object_collision_name,
                        geomname2=left_object_collision_name,
                        solref=default_solref,
                        friction=small_friction,
                        condim=3,
                    )
                    contact_cnt += 1
        if support_table_spec is not None:
            support_table_robot_collision_names = (
                manipulator_collision_names
                if support_table_spec.collision_mode
                == SUPPORT_TABLE_COLLISION_MODE_OBJECT_AND_MANIPULATOR
                else hand_collision_names
            )
            contact_cnt += _add_support_table_pairs(
                mj_spec=mj_spec,
                support_table_name="support_table",
                object_collision_names=object_collision_names,
                hand_collision_names=support_table_robot_collision_names,
                collision_mode=support_table_spec.collision_mode,
                solref=default_solref,
                friction=default_friction,
            )
        # hand <-> floor collision
        if hand_floor_collision:
            for hand_collision_name in hand_collision_names:
                if _add_collision_pair_once(
                    hand_collision_name,
                    "floor",
                    friction=default_friction,
                    condim=3,
                ):
                    contact_cnt += 1

        loguru.logger.info(f"Added {contact_cnt} contact pairs")

    # add camera
    _add_front_camera(mj_spec, robot_type, support_table_spec=support_table_spec)
    _add_d435_optical_camera(mj_spec, robot_type)

    # add reference sites for both hands and objects
    # for side in ["right", "left"]:
    #     for finger_name in finger_names:
    #         mocap_handle = mj_spec.worldbody.add_body(
    #             name=f"ref_{side}_{finger_name}",
    #             pos=[0, 0, 0],
    #             quat=[1, 0, 0, 0],
    #             mocap=True,
    #         )
    #         mocap_handle.add_site(
    #             name=f"ref_{side}_{finger_name}",
    #             pos=[0, 0, 0],
    #             size=[0.02, 0.02, 0.02],
    #             group=4,
    #             rgba=[0, 1, 0, 1],
    #         )

    mj_model = mj_spec.compile()
    mj_data = mujoco.MjData(mj_model)

    # save model in processed dir, use a stable name
    xml_file = _bind_groundplane_material_textures(mj_spec.to_xml())
    export_file_path = f"{processed_dir}/../scene.xml"
    if not act_scene:
        with open(export_file_path, "w") as f:
            f.write(xml_file)
        loguru.logger.info(f"Saved model to {export_file_path}")

    if act_scene:
        xml_file_act = _add_object_xyzrpy_actuators(
            xml_file,
            object_armature=object_armature,
            object_frictionloss=object_frictionloss,
            object_pos_kp=0,
            object_pos_kd=0,
            object_rot_kp=0,
            object_rot_kd=0,
        )
        export_file_path_act = f"{processed_dir}/../scene_act.xml"
        with open(export_file_path_act, "w") as f:
            f.write(xml_file_act)
        loguru.logger.info(
            f"Saved model with object actuators to {export_file_path_act}"
        )

    # save another model with has equality constraints between track site and ref site
    for sid in range(mj_model.nsite):
        site_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_SITE, sid)
        if "track" in site_name:
            # get ref site name by replacing "track" with "ref"
            ref_site_name = site_name.replace("track", "ref")
            e = mj_spec.add_equality(
                name=f"{site_name}_equality_constraint",
                type=mujoco.mjtEq.mjEQ_CONNECT,
                name1=site_name,
                name2=ref_site_name,
                objtype=mujoco.mjtObj.mjOBJ_SITE,
                data=np.zeros(11),
            )
            e.solref = [0.02, 1.0]
            # disable the constraint when the distance is large
            e.solimp = [0.0, 1.0, 100.0, 0.5, 2.0]
    mj_model_eq = mj_spec.compile()
    xml_file_eq = _bind_groundplane_material_textures(mj_spec.to_xml())
    export_file_path_eq = f"{processed_dir}/../scene_eq.xml"
    if not act_scene:
        with open(export_file_path_eq, "w") as f:
            f.write(xml_file_eq)
        loguru.logger.info(
            f"Saved model with equality constraints to {export_file_path_eq}"
        )

    # save task info
    task_info["robot_type"] = robot_type
    task_info["friction_scale"] = float(friction_scale)
    task_info["object_frictionloss"] = float(object_frictionloss)
    with open(f"{processed_dir}/../task_info.json", "w") as f:
        json.dump(task_info, f, indent=2)

    # visualize model
    if show_viewer:
        with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
            rate_limiter = RateLimiter(1 / mj_model.opt.timestep)
            while viewer.is_running():
                mujoco.mj_step(mj_model, mj_data)
                viewer.sync()
                rate_limiter.sleep()


if __name__ == "__main__":
    tyro.cli(main)
