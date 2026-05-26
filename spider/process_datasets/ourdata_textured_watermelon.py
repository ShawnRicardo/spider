# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Convert local reconstructed scene outputs into a SPIDER keypoint dataset.

This processor adapts a single-object local workspace into the SPIDER
`processed/<dataset>/mano/...` format expected by `generate_xml.py`, `ik.py`,
and `run_mjwp.py`.

Current assumptions:
- workspace contains a single manipulated object
- the object is assigned to the right object slot
- the left object slot is empty
- hand trajectories come from `hawor/world_mocap.npz`
- object 6D poses come from FoundationPose `center_pose/*.txt` when available
- object mesh scale comes from `result/box_for_spider.npz`
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from itertools import permutations
from pathlib import Path

import mujoco
import numpy as np
import trimesh
import tyro
from loguru import logger
from scipy.interpolate import PchipInterpolator
from scipy.spatial.transform import Rotation as R, Slerp

import spider
from spider.io import get_mesh_dir, get_processed_data_dir

FINGERTIP_VERTEX_IDS = {
    "thumb": 744,
    "index": 320,
    "middle": 443,
    "ring": 554,
    "pinky": 671,
}
FINGER_ORDER = ["thumb", "index", "middle", "ring", "pinky"]
IDENTITY_QUAT_WXYZ = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
OBJECT_MIN_AREA = 400
OBJECT_MIN_VALID_DEPTH = 20
WORLD_UP = np.array([0.0, 0.0, 1.0], dtype=np.float64)
SH_C0 = 0.28209479177387814
ALIGNMENT_MODE_NONE = "none"
ALIGNMENT_MODE_D435_OPTICAL = "d435_optical"
HEAD_CAMERA_SITE_NAME = "head_camera_frame"
D435_OPTICAL_FRAME_SITE_NAME = "d435_optical_frame"
SUPPORTED_ALIGNMENT_MODES = {
    ALIGNMENT_MODE_NONE,
    ALIGNMENT_MODE_D435_OPTICAL,
}
PLY_SCALAR_DTYPES = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "<i2",
    "int16": "<i2",
    "ushort": "<u2",
    "uint16": "<u2",
    "int": "<i4",
    "int32": "<i4",
    "uint": "<u4",
    "uint32": "<u4",
    "float": "<f4",
    "float32": "<f4",
    "double": "<f8",
    "float64": "<f8",
}
TEXTURED_BOX_FACES = [
    ("pos_x", 0, 1.0, (1, 2), (1.0, 1.0)),
    ("neg_x", 0, -1.0, (1, 2), (-1.0, 1.0)),
    ("pos_y", 1, 1.0, (0, 2), (-1.0, 1.0)),
    ("neg_y", 1, -1.0, (0, 2), (1.0, 1.0)),
    ("pos_z", 2, 1.0, (0, 1), (1.0, 1.0)),
    ("neg_z", 2, -1.0, (0, 1), (1.0, -1.0)),
]


@dataclass
class ObjectCandidate:
    label: int
    area: int
    centroid_uv: np.ndarray
    center_world: np.ndarray
    point_count: int


@dataclass
class RigidAlignment:
    mode: str
    rotation: np.ndarray
    translation: np.ndarray
    source_origin: np.ndarray
    target_origin: np.ndarray
    source_frame: np.ndarray
    target_frame: np.ndarray
    robot_xml: str | None
    num_frames: int
    reference_camera_frame_index: int
    source_camera_pose: np.ndarray
    target_camera_pose: np.ndarray
    target_camera_frame_name: str
    legacy_head_camera_pose: np.ndarray


@dataclass
class CameraBundle:
    path: Path
    images: np.ndarray | None
    depths: np.ndarray | None
    intrinsic: np.ndarray
    cam_c2w: np.ndarray


@dataclass
class WorkspaceInputs:
    world_mocap_path: Path
    camera_npz_path: Path
    box_npz_path: Path
    object_ply_path: Path
    masks_dir: Path | None
    object_pose_dir: Path | None


def _normalize(vec: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < 1e-8:
        if fallback is None:
            raise ValueError("cannot normalize near-zero vector")
        fallback_norm = float(np.linalg.norm(fallback))
        if fallback_norm < 1e-8:
            raise ValueError("fallback vector is also near zero")
        return fallback / fallback_norm
    return vec / norm


def _rotation_matrix_to_wxyz(rotation_matrix: np.ndarray) -> np.ndarray:
    u, _, vh = np.linalg.svd(rotation_matrix.astype(np.float64))
    rotation_matrix = u @ vh
    if np.linalg.det(rotation_matrix) < 0.0:
        u[:, -1] *= -1.0
        rotation_matrix = u @ vh
    quat_xyzw = R.from_matrix(rotation_matrix).as_quat()
    return np.array(
        [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]],
        dtype=np.float32,
    )


def _normalize_quat_wxyz(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    return quat / np.maximum(norm, 1e-12)


def _continuous_quat_wxyz(quats: np.ndarray) -> np.ndarray:
    quats = _normalize_quat_wxyz(quats).copy()
    for idx in range(1, quats.shape[0]):
        if float(np.dot(quats[idx - 1], quats[idx])) < 0.0:
            quats[idx] *= -1.0
    return quats


def _interpolation_times(num_frames: int, factor: int) -> tuple[np.ndarray, np.ndarray]:
    if factor < 1:
        raise ValueError(f"trajectory_interpolation_factor must be >= 1, got {factor}")
    if num_frames < 2 or factor == 1:
        times = np.arange(num_frames, dtype=np.float64)
        return times, times
    src_times = np.arange(num_frames, dtype=np.float64)
    dst_times = np.linspace(
        0.0,
        float(num_frames - 1),
        (num_frames - 1) * factor + 1,
        dtype=np.float64,
    )
    return src_times, dst_times


def _resample_positions_pchip(
    positions: np.ndarray, src_times: np.ndarray, dst_times: np.ndarray
) -> np.ndarray:
    positions = np.asarray(positions, dtype=np.float64)
    if len(src_times) == len(dst_times):
        return positions.copy()
    return PchipInterpolator(src_times, positions, axis=0)(dst_times)


def _interpolate_quaternions_wxyz(
    quats: np.ndarray, src_times: np.ndarray, dst_times: np.ndarray
) -> np.ndarray:
    quats = np.asarray(quats, dtype=np.float64)
    if len(src_times) == len(dst_times):
        return _normalize_quat_wxyz(quats)
    quats = _continuous_quat_wxyz(quats)
    rotations = R.from_quat(quats[:, [1, 2, 3, 0]])
    out_xyzw = Slerp(src_times, rotations)(dst_times).as_quat()
    return _normalize_quat_wxyz(out_xyzw[:, [3, 0, 1, 2]])


def _interpolate_pose_trajectory(qpos: np.ndarray, factor: int) -> np.ndarray:
    qpos = np.asarray(qpos)
    if qpos.shape[-1] != 7:
        raise ValueError(f"expected pose trajectory with last dim 7, got {qpos.shape}")
    src_times, dst_times = _interpolation_times(qpos.shape[0], factor)
    if len(src_times) == len(dst_times):
        return qpos.copy()

    out = np.empty((len(dst_times), *qpos.shape[1:]), dtype=np.float64)
    out[..., :3] = _resample_positions_pchip(qpos[..., :3], src_times, dst_times)
    leading_shape = qpos.shape[1:-1]
    if leading_shape:
        for index in np.ndindex(leading_shape):
            out[(slice(None), *index, slice(3, 7))] = _interpolate_quaternions_wxyz(
                qpos[(slice(None), *index, slice(3, 7))],
                src_times,
                dst_times,
            )
    else:
        out[:, 3:7] = _interpolate_quaternions_wxyz(
            qpos[:, 3:7], src_times, dst_times
        )
    return out.astype(qpos.dtype, copy=False)


def _interpolate_contact_trajectory(contact: np.ndarray, factor: int) -> np.ndarray:
    contact = np.asarray(contact)
    src_times, dst_times = _interpolation_times(contact.shape[0], factor)
    if len(src_times) == len(dst_times):
        return contact.copy()
    indices = np.clip(np.rint(dst_times).astype(np.int64), 0, contact.shape[0] - 1)
    return contact[indices].astype(contact.dtype, copy=False)


def _rigid_transform_points(
    points: np.ndarray, rotation: np.ndarray, translation: np.ndarray
) -> np.ndarray:
    return (points @ rotation.T + translation).astype(np.float32)


def _orthonormalize_rotation(rotation_matrix: np.ndarray) -> np.ndarray:
    u, _, vh = np.linalg.svd(rotation_matrix.astype(np.float64))
    rotation_matrix = u @ vh
    if np.linalg.det(rotation_matrix) < 0.0:
        u[:, -1] *= -1.0
        rotation_matrix = u @ vh
    return rotation_matrix


def _site_pose(model: mujoco.MjModel, data: mujoco.MjData, site_name: str) -> np.ndarray:
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if site_id == -1:
        raise ValueError(f"site {site_name} not found in {model}")
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = np.asarray(data.site_xmat[site_id], dtype=np.float64).reshape(3, 3)
    pose[:3, 3] = np.asarray(data.site_xpos[site_id], dtype=np.float64)
    return pose


def _compute_target_camera_pose(robot_xml: Path, site_name: str) -> np.ndarray:
    if not robot_xml.exists():
        raise FileNotFoundError(f"alignment robot xml not found: {robot_xml}")
    model = mujoco.MjModel.from_xml_path(str(robot_xml))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return _site_pose(model, data, site_name)


def _resolve_world_to_sim_alignment(
    alignment_mode: str,
    alignment_robot_xml: str | None,
    cam_c2w: np.ndarray | None = None,
) -> RigidAlignment:
    if alignment_mode not in SUPPORTED_ALIGNMENT_MODES:
        raise ValueError(
            f"world_to_sim_alignment must be one of {sorted(SUPPORTED_ALIGNMENT_MODES)}, "
            f"got {alignment_mode}"
        )

    identity_rotation = np.eye(3, dtype=np.float64)
    zero_translation = np.zeros(3, dtype=np.float64)
    identity_pose = np.eye(4, dtype=np.float64)
    if alignment_mode == ALIGNMENT_MODE_NONE:
        return RigidAlignment(
            mode=alignment_mode,
            rotation=identity_rotation,
            translation=zero_translation,
            source_origin=zero_translation.copy(),
            target_origin=zero_translation.copy(),
            source_frame=identity_rotation.copy(),
            target_frame=identity_rotation.copy(),
            robot_xml=None,
            num_frames=0,
            reference_camera_frame_index=-1,
            source_camera_pose=identity_pose.copy(),
            target_camera_pose=identity_pose.copy(),
            target_camera_frame_name="",
            legacy_head_camera_pose=identity_pose.copy(),
        )

    if alignment_robot_xml is None:
        raise ValueError("alignment_robot_xml is required when alignment is enabled")
    robot_xml_path = Path(alignment_robot_xml).resolve()

    if alignment_mode != ALIGNMENT_MODE_D435_OPTICAL:
        raise NotImplementedError(
            f"alignment mode {alignment_mode} is not implemented"
        )
    if cam_c2w is None or len(cam_c2w) == 0:
        raise ValueError("cam_c2w is required for d435_optical alignment")
    source_camera_pose = np.asarray(cam_c2w[0], dtype=np.float64)
    target_camera_pose = _compute_target_camera_pose(
        robot_xml_path, D435_OPTICAL_FRAME_SITE_NAME
    )
    legacy_head_camera_pose = _compute_target_camera_pose(
        robot_xml_path, HEAD_CAMERA_SITE_NAME
    )
    sim_from_world = target_camera_pose @ np.linalg.inv(source_camera_pose)
    rotation = _orthonormalize_rotation(sim_from_world[:3, :3])
    translation = sim_from_world[:3, 3]
    return RigidAlignment(
        mode=alignment_mode,
        rotation=rotation,
        translation=translation,
        source_origin=source_camera_pose[:3, 3].copy(),
        target_origin=target_camera_pose[:3, 3].copy(),
        source_frame=source_camera_pose[:3, :3].copy(),
        target_frame=target_camera_pose[:3, :3].copy(),
        robot_xml=str(robot_xml_path),
        num_frames=1,
        reference_camera_frame_index=0,
        source_camera_pose=source_camera_pose,
        target_camera_pose=target_camera_pose,
        target_camera_frame_name=D435_OPTICAL_FRAME_SITE_NAME,
        legacy_head_camera_pose=legacy_head_camera_pose,
    )


def _resolve_object_rotation(
    box_rotation: np.ndarray, orientation_policy: str
) -> np.ndarray:
    if orientation_policy == "preserve_box":
        return box_rotation.astype(np.float64)
    if orientation_policy != "upright_preserve_heading":
        raise ValueError(
            "orientation_policy must be one of "
            "{'preserve_box', 'upright_preserve_heading'}"
        )

    # `box_rotation_R` comes from PCA with the longest OBB axis in column 0.
    # For upright carton-like objects we keep the horizontal heading from the
    # remaining axes while forcing that principal axis to align with world up.
    rot = box_rotation.astype(np.float64)
    heading_candidates = [rot[:, 1], rot[:, 2]]
    heading_scores = [
        float(np.linalg.norm(candidate - np.dot(candidate, WORLD_UP) * WORLD_UP))
        for candidate in heading_candidates
    ]
    heading_idx = 1 + int(np.argmax(heading_scores))
    heading_axis = rot[:, heading_idx]
    heading_proj = heading_axis - np.dot(heading_axis, WORLD_UP) * WORLD_UP
    fallback_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if np.linalg.norm(heading_proj) < 1e-8:
        other_idx = 3 - heading_idx
        other_axis = rot[:, other_idx]
        heading_proj = other_axis - np.dot(other_axis, WORLD_UP) * WORLD_UP
        fallback_axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    heading_proj = _normalize(heading_proj, fallback_axis)

    upright_rot = np.zeros((3, 3), dtype=np.float64)
    upright_rot[:, 0] = WORLD_UP
    if heading_idx == 1:
        upright_rot[:, 1] = heading_proj
        upright_rot[:, 2] = _normalize(np.cross(upright_rot[:, 0], upright_rot[:, 1]))
    else:
        upright_rot[:, 2] = heading_proj
        upright_rot[:, 1] = _normalize(np.cross(upright_rot[:, 2], upright_rot[:, 0]))

    u, _, vh = np.linalg.svd(upright_rot)
    upright_rot = u @ vh
    if np.linalg.det(upright_rot) < 0.0:
        u[:, -1] *= -1.0
        upright_rot = u @ vh
    return upright_rot


def _project_world_to_uv(point_world: np.ndarray, intrinsic: np.ndarray, cam_c2w: np.ndarray) -> np.ndarray:
    point_h = np.concatenate([point_world.astype(np.float64), [1.0]])
    point_cam = np.linalg.inv(cam_c2w) @ point_h
    if point_cam[2] <= 1e-8:
        return np.array([np.nan, np.nan], dtype=np.float64)
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    u = fx * point_cam[0] / point_cam[2] + cx
    v = fy * point_cam[1] / point_cam[2] + cy
    return np.array([u, v], dtype=np.float64)


def _estimate_wrist_pose(verts: np.ndarray, side: str) -> tuple[np.ndarray, np.ndarray]:
    fingertips = {
        finger: verts[vertex_id].astype(np.float64)
        for finger, vertex_id in FINGERTIP_VERTEX_IDS.items()
    }
    tip_centroid = np.stack([fingertips[finger] for finger in FINGER_ORDER], axis=0).mean(axis=0)
    hand_centroid = verts.mean(axis=0)

    centered = verts - hand_centroid
    cov = centered.T @ centered / max(len(centered), 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order]

    longitudinal_target = tip_centroid - hand_centroid
    longitudinal_scores = np.abs(eigvecs.T @ _normalize(longitudinal_target, eigvecs[:, 0]))
    longitudinal_idx = int(np.argmax(longitudinal_scores))
    longitudinal_axis = eigvecs[:, longitudinal_idx]
    if np.dot(longitudinal_axis, longitudinal_target) < 0.0:
        longitudinal_axis = -longitudinal_axis

    remaining_indices = [idx for idx in range(3) if idx != longitudinal_idx]
    if side == "right":
        lateral_target = fingertips["index"] - fingertips["pinky"]
    else:
        lateral_target = fingertips["pinky"] - fingertips["index"]
    lateral_target = _normalize(lateral_target, eigvecs[:, remaining_indices[0]])
    lateral_candidates = eigvecs[:, remaining_indices]
    lateral_scores = np.abs(lateral_candidates.T @ lateral_target)
    lateral_axis = lateral_candidates[:, int(np.argmax(lateral_scores))]
    if np.dot(lateral_axis, lateral_target) < 0.0:
        lateral_axis = -lateral_axis

    normal_axis = np.cross(longitudinal_axis, lateral_axis)
    normal_axis = _normalize(normal_axis, eigvecs[:, remaining_indices[-1]])
    lateral_axis = np.cross(normal_axis, longitudinal_axis)
    lateral_axis = _normalize(lateral_axis, lateral_target)
    normal_axis = np.cross(longitudinal_axis, lateral_axis)
    normal_axis = _normalize(normal_axis, normal_axis)

    rotation_matrix = np.column_stack([longitudinal_axis, lateral_axis, normal_axis])
    if np.linalg.det(rotation_matrix) < 0.0:
        normal_axis = -normal_axis
        rotation_matrix = np.column_stack([longitudinal_axis, lateral_axis, normal_axis])

    longitudinal_projection = verts @ longitudinal_axis
    wrist_threshold = np.percentile(longitudinal_projection, 10.0)
    wrist_points = verts[longitudinal_projection <= wrist_threshold]
    if len(wrist_points) == 0:
        wrist_points = verts
    wrist_position = wrist_points.mean(axis=0).astype(np.float32)
    wrist_quat = _rotation_matrix_to_wxyz(rotation_matrix)
    return wrist_position, wrist_quat


def _estimate_hand_trajectory(verts_seq: np.ndarray, side: str) -> tuple[np.ndarray, np.ndarray]:
    num_frames = verts_seq.shape[0]
    qpos_wrist = np.zeros((num_frames, 7), dtype=np.float32)
    qpos_finger = np.zeros((num_frames, 5, 7), dtype=np.float32)

    fingertip_ids = [FINGERTIP_VERTEX_IDS[finger] for finger in FINGER_ORDER]
    qpos_finger[:, :, :3] = verts_seq[:, fingertip_ids, :].astype(np.float32)
    qpos_finger[:, :, 3:] = IDENTITY_QUAT_WXYZ

    for frame_idx in range(num_frames):
        wrist_pos, wrist_quat = _estimate_wrist_pose(verts_seq[frame_idx], side=side)
        qpos_wrist[frame_idx, :3] = wrist_pos
        qpos_wrist[frame_idx, 3:] = wrist_quat
    return qpos_wrist, qpos_finger


def _component_world_center(mask: np.ndarray, depth: np.ndarray, intrinsic: np.ndarray, cam_c2w: np.ndarray) -> tuple[np.ndarray, int]:
    ys, xs = np.where(mask)
    z = depth[ys, xs].astype(np.float64)
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    x = (xs.astype(np.float64) - cx) * z / fx
    y = (ys.astype(np.float64) - cy) * z / fy
    points_cam = np.stack([x, y, z], axis=1)
    points_world = (cam_c2w[:3, :3] @ points_cam.T).T + cam_c2w[:3, 3]
    return points_world.mean(axis=0), len(points_world)


def _extract_object_candidates(mask_image: np.ndarray, depth: np.ndarray, intrinsic: np.ndarray, cam_c2w: np.ndarray) -> list[ObjectCandidate]:
    import cv2

    depth_h, depth_w = depth.shape
    if mask_image.shape[:2] != (depth_h, depth_w):
        mask_image = cv2.resize(mask_image, (depth_w, depth_h), interpolation=cv2.INTER_NEAREST)

    binary = (mask_image > (0 if mask_image.max() <= 10 else 127)).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)

    candidates: list[ObjectCandidate] = []
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < OBJECT_MIN_AREA:
            continue
        component_mask = labels == label
        valid_mask = component_mask & np.isfinite(depth) & (depth > 1e-6)
        if int(valid_mask.sum()) < OBJECT_MIN_VALID_DEPTH:
            continue
        center_world, point_count = _component_world_center(valid_mask, depth, intrinsic, cam_c2w)
        centroid_uv = centroids[label].astype(np.float64)
        candidates.append(
            ObjectCandidate(
                label=label,
                area=area,
                centroid_uv=centroid_uv,
                center_world=center_world.astype(np.float32),
                point_count=point_count,
            )
        )
    return candidates


def _interpolate_positions(positions: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    if not valid_mask.any():
        raise RuntimeError("failed to estimate any valid object positions")
    if valid_mask.all():
        return positions.astype(np.float32)

    frame_ids = np.arange(len(positions), dtype=np.float64)
    valid_ids = frame_ids[valid_mask]
    interpolated = positions.copy().astype(np.float64)
    for axis in range(3):
        interpolated[:, axis] = np.interp(frame_ids, valid_ids, positions[valid_mask, axis])
    return interpolated.astype(np.float32)


def _estimate_object_trajectory(
    masks_dir: Path,
    depths: np.ndarray,
    intrinsic: np.ndarray,
    cam_c2w: np.ndarray,
    initial_center_world: np.ndarray,
    num_frames: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    mask_files = sorted(masks_dir.glob("*.png"))[:num_frames]
    if len(mask_files) < num_frames:
        raise RuntimeError(
            f"expected at least {num_frames} mask frames, found {len(mask_files)} in {masks_dir}"
        )

    centers_world = np.full((num_frames, 3), np.nan, dtype=np.float32)
    valid_mask = np.zeros(num_frames, dtype=bool)
    selected_labels = np.full(num_frames, -1, dtype=np.int32)
    selected_areas = np.zeros(num_frames, dtype=np.int32)
    selected_uv = np.full((num_frames, 2), np.nan, dtype=np.float32)
    reference_uv = np.full((num_frames, 2), np.nan, dtype=np.float32)

    previous_center = initial_center_world.astype(np.float64)
    for frame_idx in range(num_frames):
        import cv2

        mask_image = cv2.imread(str(mask_files[frame_idx]), cv2.IMREAD_GRAYSCALE)
        if mask_image is None:
            continue
        candidates = _extract_object_candidates(
            mask_image=mask_image,
            depth=depths[frame_idx],
            intrinsic=intrinsic,
            cam_c2w=cam_c2w[frame_idx],
        )
        if not candidates:
            continue

        projected_reference = _project_world_to_uv(previous_center, intrinsic, cam_c2w[frame_idx])
        reference_uv[frame_idx] = projected_reference.astype(np.float32)

        best_candidate = None
        best_score = math.inf
        for candidate in candidates:
            world_distance = float(np.linalg.norm(candidate.center_world - previous_center))
            if np.all(np.isfinite(projected_reference)):
                pixel_distance = float(np.linalg.norm(candidate.centroid_uv - projected_reference))
            else:
                pixel_distance = 0.0
            score = world_distance + 0.002 * pixel_distance
            if score < best_score:
                best_score = score
                best_candidate = candidate

        if best_candidate is None:
            continue

        centers_world[frame_idx] = best_candidate.center_world
        valid_mask[frame_idx] = True
        selected_labels[frame_idx] = best_candidate.label
        selected_areas[frame_idx] = best_candidate.area
        selected_uv[frame_idx] = best_candidate.centroid_uv.astype(np.float32)
        previous_center = best_candidate.center_world.astype(np.float64)

    centers_world = _interpolate_positions(centers_world, valid_mask)
    debug = {
        "valid_mask": valid_mask.astype(np.uint8),
        "selected_labels": selected_labels,
        "selected_areas": selected_areas,
        "selected_uv": selected_uv,
        "reference_uv": reference_uv,
    }
    return centers_world, debug


def _first_existing(paths: list[Path], description: str) -> Path:
    for path in paths:
        if path.exists():
            return path
    formatted = ", ".join(str(path) for path in paths)
    raise FileNotFoundError(f"required {description} not found; tried: {formatted}")


def _optional_first_existing_dir(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.is_dir() and len(list(path.glob("*.txt"))) > 0:
            return path
    return None


def _resolve_workspace_inputs(workspace_path: Path, object_id: str) -> WorkspaceInputs:
    world_mocap_path = workspace_path / "hawor" / "world_mocap.npz"
    camera_npz_path = _first_existing(
        [
            workspace_path / "da3.npz",
            workspace_path / "da3" / "da3.npz",
        ],
        "da3.npz",
    )
    box_npz_path = _first_existing(
        [
            workspace_path / "result" / object_id / "box_for_spider.npz",
            workspace_path / "result" / "box_for_spider.npz",
        ],
        "box_for_spider.npz",
    )
    object_ply_path = _first_existing(
        [
            workspace_path / "sam3d" / object_id / "obj_mesh_final.glb",
            workspace_path / "sam3d" / object_id / "obj_mesh_final.obj",
            workspace_path / "sam3d" / object_id / "obj_mesh_final.ply",
            workspace_path / "sam3d" / object_id / "obj_3d_final.ply",
            workspace_path / "sam3d" / "obj_mesh_final.glb",
            workspace_path / "sam3d" / "obj_mesh_final.obj",
            workspace_path / "sam3d" / "obj_mesh_final.ply",
            workspace_path / "sam3d" / "obj_3d_final.ply",
        ],
        "obj_mesh_final.glb, obj_mesh_final.obj, obj_mesh_final.ply, or obj_3d_final.ply",
    )
    masks_candidates = [
        workspace_path / "masks" / "objects_full",
        workspace_path / "masks",
    ]
    masks_dir = next(
        (path for path in masks_candidates if path.is_dir() and len(list(path.glob("*.png"))) > 0),
        None,
    )
    object_pose_dir = _optional_first_existing_dir(
        [
            workspace_path
            / "result"
            / object_id
            / "foundationpose_debug"
            / "center_pose",
            workspace_path / "result" / object_id / "poses",
        ]
    )
    for required_path in [world_mocap_path, camera_npz_path, box_npz_path, object_ply_path]:
        if not required_path.exists():
            raise FileNotFoundError(f"required input not found: {required_path}")
    if object_pose_dir is None and masks_dir is None:
        raise FileNotFoundError(
            "required object trajectory source not found: expected FoundationPose "
            f"poses under {workspace_path / 'result' / object_id} or masks under {workspace_path / 'masks'}"
        )
    return WorkspaceInputs(
        world_mocap_path=world_mocap_path,
        camera_npz_path=camera_npz_path,
        box_npz_path=box_npz_path,
        object_ply_path=object_ply_path,
        masks_dir=masks_dir,
        object_pose_dir=object_pose_dir,
    )


def _load_camera_bundle(camera_npz_path: Path) -> CameraBundle:
    data = np.load(camera_npz_path)
    if "cam_c2w" not in data.files:
        raise KeyError(f"cam_c2w not found in {camera_npz_path}")
    if "intrinsic" in data.files:
        intrinsic = data["intrinsic"].astype(np.float64)
    elif "intrinsics" in data.files:
        intrinsic = data["intrinsics"][0].astype(np.float64)
    else:
        raise KeyError(f"intrinsic/intrinsics not found in {camera_npz_path}")
    depths = data["depths"].astype(np.float32) if "depths" in data.files else None
    images = data["images"] if "images" in data.files else None
    return CameraBundle(
        path=camera_npz_path,
        images=images,
        depths=depths,
        intrinsic=intrinsic,
        cam_c2w=data["cam_c2w"].astype(np.float64),
    )


def _load_foundationpose_object_trajectory(
    pose_dir: Path,
    cam_c2w: np.ndarray,
    num_frames: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    pose_files = sorted(pose_dir.glob("*.txt"))
    if len(pose_files) < num_frames:
        raise RuntimeError(
            f"expected at least {num_frames} pose frames, found {len(pose_files)} in {pose_dir}"
        )

    object_poses_world = np.zeros((num_frames, 4, 4), dtype=np.float64)
    for frame_idx, pose_file in enumerate(pose_files[:num_frames]):
        pose_cam_obj = np.loadtxt(pose_file, dtype=np.float64)
        if pose_cam_obj.shape != (4, 4):
            raise ValueError(f"invalid object pose shape {pose_cam_obj.shape} in {pose_file}")
        object_poses_world[frame_idx] = cam_c2w[frame_idx] @ pose_cam_obj

    centers_world = object_poses_world[:, :3, 3].astype(np.float32)
    rotations_world = np.stack(
        [_orthonormalize_rotation(pose[:3, :3]) for pose in object_poses_world],
        axis=0,
    )
    debug = {
        "valid_mask": np.ones(num_frames, dtype=np.uint8),
        "selected_labels": np.zeros(num_frames, dtype=np.int32),
        "selected_areas": np.zeros(num_frames, dtype=np.int32),
        "selected_uv": np.full((num_frames, 2), np.nan, dtype=np.float32),
        "reference_uv": np.full((num_frames, 2), np.nan, dtype=np.float32),
        "pose_source_files": np.asarray([str(path) for path in pose_files[:num_frames]]),
        "object_poses_world": object_poses_world.astype(np.float32),
    }
    return centers_world, rotations_world.astype(np.float64), debug


def _load_object_mesh(source_ply: Path) -> trimesh.Trimesh:
    geometry = trimesh.load(source_ply, process=False, maintain_order=True)
    if isinstance(geometry, trimesh.Scene):
        geometry = trimesh.util.concatenate(tuple(geometry.geometry.values()))
    if isinstance(geometry, trimesh.points.PointCloud):
        if len(geometry.vertices) == 0:
            raise RuntimeError(f"point cloud at {source_ply} is empty")
        mesh = geometry.convex_hull
    elif isinstance(geometry, trimesh.Trimesh):
        mesh = geometry if len(geometry.faces) > 0 else geometry.convex_hull
    else:
        raise TypeError(f"unsupported geometry type from {source_ply}: {type(geometry)}")
    if mesh.is_empty:
        raise RuntimeError(f"failed to build a mesh from {source_ply}")
    return mesh


def _source_mesh_has_faces(source_mesh: Path) -> bool:
    if source_mesh.suffix.lower() == ".ply":
        return _ply_has_faces(source_mesh)
    geometry = trimesh.load(source_mesh, process=False, maintain_order=True)
    if isinstance(geometry, trimesh.Scene):
        return any(
            isinstance(part, trimesh.Trimesh) and len(part.faces) > 0
            for part in geometry.geometry.values()
        )
    return isinstance(geometry, trimesh.Trimesh) and len(geometry.faces) > 0


def _ply_has_faces(source_ply: Path) -> bool:
    with source_ply.open("rb") as file:
        first_line = file.readline()
        if first_line.strip() != b"ply":
            raise ValueError(f"{source_ply} is not a PLY file")
        while True:
            raw_line = file.readline()
            if raw_line == b"":
                raise ValueError(f"{source_ply} ended before PLY end_header")
            line = raw_line.decode("ascii", errors="strict").strip()
            if line == "end_header":
                return False
            parts = line.split()
            if len(parts) >= 3 and parts[0] == "element" and parts[1] == "face":
                return int(parts[2]) > 0


def _ply_scalar_dtype(property_type: str, file_format: str) -> str:
    if property_type not in PLY_SCALAR_DTYPES:
        raise ValueError(f"unsupported PLY scalar property type: {property_type}")
    dtype = PLY_SCALAR_DTYPES[property_type]
    if file_format == "binary_big_endian" and dtype.startswith("<"):
        dtype = ">" + dtype[1:]
    return dtype


def _read_ply_vertex_table(source_ply: Path) -> np.ndarray:
    """Read fixed-width PLY vertex properties into a structured array.

    The SAM3D milk object is a binary Gaussian PLY with custom fields such as
    f_dc_0/1/2. Trimesh can read the coordinates, but it does not preserve
    these Gaussian color fields, so we parse the vertex table directly.
    """
    with source_ply.open("rb") as file:
        first_line = file.readline()
        if first_line.strip() != b"ply":
            raise ValueError(f"{source_ply} is not a PLY file")

        file_format: str | None = None
        vertex_count: int | None = None
        vertex_properties: list[tuple[str, str]] = []
        current_element: str | None = None

        while True:
            raw_line = file.readline()
            if raw_line == b"":
                raise ValueError(f"{source_ply} ended before PLY end_header")
            line = raw_line.decode("ascii", errors="strict").strip()
            if line == "end_header":
                break
            if not line or line.startswith("comment"):
                continue

            parts = line.split()
            if parts[0] == "format":
                file_format = parts[1]
            elif parts[0] == "element":
                current_element = parts[1]
                if current_element == "vertex":
                    vertex_count = int(parts[2])
            elif parts[0] == "property" and current_element == "vertex":
                if parts[1] == "list":
                    raise ValueError(
                        f"{source_ply} has list vertex properties, which are not supported"
                    )
                vertex_properties.append((parts[2], parts[1]))

        if file_format not in {"binary_little_endian", "binary_big_endian"}:
            raise ValueError(
                f"{source_ply} uses unsupported PLY format {file_format!r}; "
                "expected binary_little_endian or binary_big_endian"
            )
        if vertex_count is None or vertex_count <= 0:
            raise ValueError(f"{source_ply} has invalid vertex count {vertex_count}")
        if not vertex_properties:
            raise ValueError(f"{source_ply} has no vertex properties")

        dtype = np.dtype(
            [
                (name, _ply_scalar_dtype(property_type, file_format))
                for name, property_type in vertex_properties
            ]
        )
        vertices = np.fromfile(file, dtype=dtype, count=vertex_count)
    if len(vertices) != vertex_count:
        raise ValueError(
            f"{source_ply} vertex table is truncated: expected {vertex_count}, got {len(vertices)}"
        )
    return vertices


def _load_gaussian_colored_points(source_ply: Path) -> tuple[np.ndarray, np.ndarray, str]:
    vertices = _read_ply_vertex_table(source_ply)
    names = set(vertices.dtype.names or ())
    required_xyz = {"x", "y", "z"}
    if not required_xyz.issubset(names):
        raise ValueError(f"{source_ply} is missing x/y/z vertex fields")

    points = np.stack(
        [vertices["x"], vertices["y"], vertices["z"]],
        axis=1,
    ).astype(np.float32)

    if {"f_dc_0", "f_dc_1", "f_dc_2"}.issubset(names):
        sh_dc = np.stack(
            [vertices["f_dc_0"], vertices["f_dc_1"], vertices["f_dc_2"]],
            axis=1,
        ).astype(np.float32)
        rgb_float = np.clip(0.5 + SH_C0 * sh_dc, 0.0, 1.0)
        colors = np.rint(rgb_float * 255.0).astype(np.uint8)
        color_source = "gaussian_sh_dc"
    elif {"red", "green", "blue"}.issubset(names):
        colors = np.stack(
            [vertices["red"], vertices["green"], vertices["blue"]],
            axis=1,
        ).astype(np.uint8)
        color_source = "ply_rgb"
    else:
        raise ValueError(
            f"{source_ply} has no f_dc_0/1/2 or red/green/blue color fields"
        )

    finite_mask = np.isfinite(points).all(axis=1)
    if not np.all(finite_mask):
        logger.warning(
            "Dropping {} non-finite colored object points from {}",
            int(np.count_nonzero(~finite_mask)),
            source_ply,
        )
        points = points[finite_mask]
        colors = colors[finite_mask]
    if len(points) == 0:
        raise RuntimeError(f"{source_ply} has no finite colored object points")
    return points, colors, color_source


def _prepare_object_points_with_mesh_alignment(
    points: np.ndarray,
    scale_factor: float,
    mesh_alignment_debug: dict[str, np.ndarray | float | bool | list[int]],
) -> np.ndarray:
    if not np.isfinite(scale_factor) or scale_factor <= 0.0:
        raise ValueError(f"invalid scale_factor {scale_factor}")
    prepared_points = np.asarray(points, dtype=np.float32) * np.float32(scale_factor)
    rotation = np.asarray(
        mesh_alignment_debug["mesh_bbox_alignment_rotation"],
        dtype=np.float32,
    )
    if bool(mesh_alignment_debug["mesh_bbox_alignment_applied"]):
        prepared_points = prepared_points @ rotation.T
    bounds_center = np.asarray(
        mesh_alignment_debug["mesh_bbox_alignment_bounds_center_offset"],
        dtype=np.float32,
    )
    prepared_points = prepared_points - bounds_center[None, :]
    return prepared_points.astype(np.float32)


def _write_colored_point_ply(
    output_path: Path,
    points: np.ndarray,
    colors: np.ndarray,
) -> None:
    points = np.asarray(points, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.uint8)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"colored PLY points must have shape (N, 3), got {points.shape}")
    if colors.shape != points.shape:
        raise ValueError(
            f"colored PLY colors must have shape {points.shape}, got {colors.shape}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ply_dtype = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ]
    )
    table = np.empty(len(points), dtype=ply_dtype)
    table["x"] = points[:, 0]
    table["y"] = points[:, 1]
    table["z"] = points[:, 2]
    table["red"] = colors[:, 0]
    table["green"] = colors[:, 1]
    table["blue"] = colors[:, 2]
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    with output_path.open("wb") as file:
        file.write(header.encode("ascii"))
        table.tofile(file)


def _face_uv_from_points(
    points: np.ndarray,
    bbox_size_xyz_m: np.ndarray,
    face_spec: tuple[str, int, float, tuple[int, int], tuple[float, float]],
) -> np.ndarray:
    _, _, _, uv_axes, uv_signs = face_spec
    half_extents = 0.5 * np.asarray(bbox_size_xyz_m, dtype=np.float32)
    u_axis, v_axis = uv_axes
    u_sign, v_sign = uv_signs
    uv = np.empty((len(points), 2), dtype=np.float32)
    uv[:, 0] = 0.5 + 0.5 * u_sign * points[:, u_axis] / max(float(half_extents[u_axis]), 1e-9)
    uv[:, 1] = 0.5 + 0.5 * v_sign * points[:, v_axis] / max(float(half_extents[v_axis]), 1e-9)
    return np.clip(uv, 0.0, 1.0)


def _fill_texture_holes(texture: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    if np.all(valid_mask):
        return texture
    if not np.any(valid_mask):
        return texture
    from scipy import ndimage

    _, nearest_indices = ndimage.distance_transform_edt(
        ~valid_mask,
        return_indices=True,
    )
    filled = texture.copy()
    empty_rows, empty_cols = np.nonzero(~valid_mask)
    filled[empty_rows, empty_cols] = texture[
        nearest_indices[0, empty_rows, empty_cols],
        nearest_indices[1, empty_rows, empty_cols],
    ]
    return filled


def _bake_face_texture(
    uv: np.ndarray,
    colors: np.ndarray,
    face_resolution: int,
    fallback_color: np.ndarray,
) -> np.ndarray:
    if len(uv) == 0:
        return np.tile(
            fallback_color[None, None, :],
            (face_resolution, face_resolution, 1),
        ).astype(np.uint8)

    cols = np.clip(
        np.rint(uv[:, 0] * (face_resolution - 1)).astype(np.int32),
        0,
        face_resolution - 1,
    )
    rows = np.clip(
        np.rint((1.0 - uv[:, 1]) * (face_resolution - 1)).astype(np.int32),
        0,
        face_resolution - 1,
    )
    accum = np.zeros((face_resolution, face_resolution, 3), dtype=np.float64)
    counts = np.zeros((face_resolution, face_resolution), dtype=np.int32)
    np.add.at(accum, (rows, cols), colors.astype(np.float64))
    np.add.at(counts, (rows, cols), 1)

    texture = np.tile(
        fallback_color[None, None, :],
        (face_resolution, face_resolution, 1),
    ).astype(np.float64)
    valid_mask = counts > 0
    texture[valid_mask] = accum[valid_mask] / counts[valid_mask, None]
    texture = _fill_texture_holes(
        np.clip(np.rint(texture), 0, 255).astype(np.uint8),
        valid_mask,
    )
    return texture


def _build_textured_box_vertices_and_uvs(
    bbox_size_xyz_m: np.ndarray,
    atlas_cols: int,
    atlas_rows: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    half = 0.5 * np.asarray(bbox_size_xyz_m, dtype=np.float32)
    vertices: list[list[float]] = []
    uvs: list[list[float]] = []
    faces: list[list[int]] = []
    face_names: list[str] = []
    for face_idx, face_spec in enumerate(TEXTURED_BOX_FACES):
        face_name, normal_axis, normal_sign, uv_axes, uv_signs = face_spec
        u_axis, v_axis = uv_axes
        u_sign, v_sign = uv_signs
        normal_coord = normal_sign * half[normal_axis]
        face_vertex_indices = []
        for u_raw, v_raw in [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]:
            point = np.zeros(3, dtype=np.float32)
            point[normal_axis] = normal_coord
            point[u_axis] = (2.0 * u_raw - 1.0) * half[u_axis] * u_sign
            point[v_axis] = (2.0 * v_raw - 1.0) * half[v_axis] * v_sign
            vertices.append(point.tolist())
            face_vertex_indices.append(len(vertices))

            cell_col = face_idx % atlas_cols
            cell_row = face_idx // atlas_cols
            atlas_u = (cell_col + u_raw) / atlas_cols
            atlas_v = 1.0 - (cell_row + (1.0 - v_raw)) / atlas_rows
            uvs.append([atlas_u, atlas_v])
        faces.append(face_vertex_indices)
        face_names.append(face_name)
    return (
        np.asarray(vertices, dtype=np.float32),
        np.asarray(uvs, dtype=np.float32),
        np.asarray(faces, dtype=np.int32),
        face_names,
    )


def _write_textured_box_obj(
    obj_path: Path,
    mtl_path: Path,
    texture_path: Path,
    bbox_size_xyz_m: np.ndarray,
) -> None:
    vertices, uvs, faces, face_names = _build_textured_box_vertices_and_uvs(
        bbox_size_xyz_m,
        atlas_cols=3,
        atlas_rows=2,
    )
    with mtl_path.open("w", encoding="utf-8") as file:
        file.write("newmtl milk_textured\n")
        file.write("Ka 1.000000 1.000000 1.000000\n")
        file.write("Kd 1.000000 1.000000 1.000000\n")
        file.write("Ks 0.000000 0.000000 0.000000\n")
        file.write("Ns 1.000000\n")
        file.write("d 1.000000\n")
        file.write(f"map_Kd {texture_path.name}\n")

    with obj_path.open("w", encoding="utf-8") as file:
        file.write(f"mtllib {mtl_path.name}\n")
        file.write("o milk_textured_box\n")
        for vertex in vertices:
            file.write(f"v {vertex[0]:.9f} {vertex[1]:.9f} {vertex[2]:.9f}\n")
        for uv in uvs:
            file.write(f"vt {uv[0]:.9f} {uv[1]:.9f}\n")
        file.write("usemtl milk_textured\n")
        for face_idx, (face_name, face) in enumerate(zip(face_names, faces, strict=True)):
            _, normal_axis, normal_sign, _, _ = TEXTURED_BOX_FACES[face_idx]
            desired_normal = np.zeros(3, dtype=np.float32)
            desired_normal[normal_axis] = normal_sign
            face_vertices = vertices[face - 1]
            actual_normal = np.cross(
                face_vertices[1] - face_vertices[0],
                face_vertices[2] - face_vertices[0],
            )
            if float(np.dot(actual_normal, desired_normal)) < 0.0:
                face = face[::-1]
            file.write(f"g {face_name}\n")
            refs = [f"{idx}/{idx}" for idx in face]
            file.write(f"f {refs[0]} {refs[1]} {refs[2]}\n")
            file.write(f"f {refs[0]} {refs[2]} {refs[3]}\n")


def _bake_textured_box_assets(
    object_mesh_dir: Path,
    points: np.ndarray,
    colors: np.ndarray,
    bbox_size_xyz_m: np.ndarray,
    face_resolution: int,
) -> tuple[Path, Path, Path, tuple[int, int]]:
    if face_resolution < 32:
        raise ValueError(f"object texture face resolution is too small: {face_resolution}")
    points = np.asarray(points, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.uint8)
    bbox_size = np.asarray(bbox_size_xyz_m, dtype=np.float32)
    if points.shape[0] != colors.shape[0]:
        raise ValueError(
            f"points/colors length mismatch for texture bake: {points.shape[0]} vs {colors.shape[0]}"
        )
    if bbox_size.shape != (3,) or np.any(bbox_size <= 0) or not np.isfinite(bbox_size).all():
        raise ValueError(f"invalid bbox size for texture bake: {bbox_size_xyz_m}")

    fallback_color = np.median(colors, axis=0).astype(np.uint8)
    half = 0.5 * bbox_size
    normalized = np.abs(points) / np.maximum(half[None, :], 1e-9)
    face_axis = np.argmax(normalized, axis=1)
    face_sign = np.where(points[np.arange(len(points)), face_axis] >= 0.0, 1.0, -1.0)

    atlas_cols = 3
    atlas_rows = 2
    atlas = np.zeros(
        (atlas_rows * face_resolution, atlas_cols * face_resolution, 3),
        dtype=np.uint8,
    )
    for face_idx, face_spec in enumerate(TEXTURED_BOX_FACES):
        _, normal_axis, normal_sign, _, _ = face_spec
        face_mask = (face_axis == normal_axis) & (face_sign == normal_sign)
        uv = _face_uv_from_points(points[face_mask], bbox_size, face_spec)
        face_texture = _bake_face_texture(
            uv,
            colors[face_mask],
            face_resolution=face_resolution,
            fallback_color=fallback_color,
        )
        cell_col = face_idx % atlas_cols
        cell_row = face_idx // atlas_cols
        row0 = cell_row * face_resolution
        col0 = cell_col * face_resolution
        atlas[row0 : row0 + face_resolution, col0 : col0 + face_resolution] = face_texture

    from PIL import Image

    texture_path = object_mesh_dir / "visual_texture.png"
    mtl_path = object_mesh_dir / "visual_textured.mtl"
    obj_path = object_mesh_dir / "visual_textured.obj"
    Image.fromarray(atlas, mode="RGB").save(texture_path)
    _write_textured_box_obj(
        obj_path=obj_path,
        mtl_path=mtl_path,
        texture_path=texture_path,
        bbox_size_xyz_m=bbox_size,
    )
    return obj_path, mtl_path, texture_path, (int(atlas.shape[1]), int(atlas.shape[0]))


def _extract_mesh_vertex_rgb(mesh: trimesh.Trimesh) -> np.ndarray | None:
    vertex_colors = getattr(mesh.visual, "vertex_colors", None)
    if vertex_colors is None or len(vertex_colors) != len(mesh.vertices):
        return None
    colors = np.asarray(vertex_colors)
    if colors.ndim != 2 or colors.shape[1] < 3:
        return None
    colors = colors[:, :3]
    if colors.dtype.kind == "f" and float(np.nanmax(colors)) <= 1.0:
        colors = colors * 255.0
    colors = np.clip(colors, 0, 255).astype(np.uint8)
    if not np.isfinite(colors.astype(np.float32)).all():
        return None
    return colors


def _simplify_colored_mesh(
    mesh: trimesh.Trimesh,
    max_faces: int,
) -> trimesh.Trimesh:
    if max_faces <= 0 or len(mesh.faces) <= max_faces:
        return mesh.copy()

    colors = _extract_mesh_vertex_rgb(mesh)
    if colors is None:
        return mesh.copy()

    try:
        import open3d as o3d

        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(
            np.asarray(mesh.vertices, dtype=np.float64)
        )
        o3d_mesh.triangles = o3d.utility.Vector3iVector(
            np.asarray(mesh.faces, dtype=np.int32)
        )
        o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(
            colors.astype(np.float64) / 255.0
        )
        simplified = o3d_mesh.simplify_quadric_decimation(int(max_faces))
        simplified.remove_unreferenced_vertices()
        simplified_vertices = np.asarray(simplified.vertices, dtype=np.float64)
        simplified_faces = np.asarray(simplified.triangles, dtype=np.int64)
        simplified_colors = np.asarray(simplified.vertex_colors, dtype=np.float64)
        if (
            len(simplified_vertices) == 0
            or len(simplified_faces) == 0
            or len(simplified_colors) != len(simplified_vertices)
        ):
            raise RuntimeError("Open3D returned an empty or uncolored simplified mesh")
        output = trimesh.Trimesh(
            vertices=simplified_vertices,
            faces=simplified_faces,
            process=False,
        )
        output.visual.vertex_colors = np.clip(
            simplified_colors * 255.0,
            0,
            255,
        ).astype(np.uint8)
        logger.info(
            "Simplified textured object mesh for MuJoCo visual: faces {} -> {}",
            len(mesh.faces),
            len(output.faces),
        )
        return output
    except Exception as exc:
        logger.warning(
            "Could not simplify colored object mesh to {} faces; using full mesh: {}",
            max_faces,
            exc,
        )
        return mesh.copy()


def _write_textured_mesh_obj(
    obj_path: Path,
    mtl_path: Path,
    texture_path: Path,
    mesh: trimesh.Trimesh,
    tile_size: int,
) -> tuple[int, int]:
    if tile_size < 2:
        raise ValueError(f"object_texture_tile_size is too small: {tile_size}")
    if len(mesh.faces) == 0:
        raise ValueError("cannot bake a textured OBJ from a mesh with no faces")

    colors = _extract_mesh_vertex_rgb(mesh)
    if colors is None:
        raise ValueError("source mesh does not contain per-vertex RGB colors")

    face_count = int(len(mesh.faces))
    atlas_cols = int(math.ceil(math.sqrt(face_count)))
    atlas_rows = int(math.ceil(face_count / atlas_cols))
    atlas_width = atlas_cols * tile_size
    atlas_height = atlas_rows * tile_size
    atlas = np.zeros((atlas_height, atlas_width, 3), dtype=np.uint8)

    uv_triplets: list[tuple[tuple[float, float], tuple[float, float], tuple[float, float]]] = []
    margin = 0.25
    for face_idx, face in enumerate(np.asarray(mesh.faces, dtype=np.int64)):
        row = face_idx // atlas_cols
        col = face_idx % atlas_cols
        x0 = col * tile_size
        y0 = row * tile_size
        face_color = np.mean(colors[face], axis=0)
        atlas[y0 : y0 + tile_size, x0 : x0 + tile_size] = np.clip(
            face_color,
            0,
            255,
        ).astype(np.uint8)
        left = (x0 + margin) / atlas_width
        right = (x0 + tile_size - margin) / atlas_width
        top = 1.0 - (y0 + margin) / atlas_height
        bottom = 1.0 - (y0 + tile_size - margin) / atlas_height
        uv_triplets.append(((left, bottom), (right, bottom), (left, top)))

    from PIL import Image

    Image.fromarray(atlas).save(texture_path)
    with mtl_path.open("w", encoding="utf-8") as file:
        file.write("newmtl milk_visual_texture\n")
        file.write("Ka 1.000000 1.000000 1.000000\n")
        file.write("Kd 1.000000 1.000000 1.000000\n")
        file.write("Ks 0.000000 0.000000 0.000000\n")
        file.write("Ns 1.000000\n")
        file.write("illum 2\n")
        file.write(f"map_Kd {texture_path.name}\n")

    with obj_path.open("w", encoding="utf-8") as file:
        file.write(f"mtllib {mtl_path.name}\n")
        file.write("usemtl milk_visual_texture\n")
        for vertex in np.asarray(mesh.vertices, dtype=np.float64):
            file.write(
                f"v {vertex[0]:.9g} {vertex[1]:.9g} {vertex[2]:.9g}\n"
            )
        for uv_triplet in uv_triplets:
            for uv in uv_triplet:
                file.write(f"vt {uv[0]:.9g} {uv[1]:.9g}\n")
        for face_idx, face in enumerate(np.asarray(mesh.faces, dtype=np.int64)):
            vertex_refs = face + 1
            vt_base = face_idx * 3 + 1
            file.write(
                f"f {vertex_refs[0]}/{vt_base} "
                f"{vertex_refs[1]}/{vt_base + 1} "
                f"{vertex_refs[2]}/{vt_base + 2}\n"
            )
    return atlas_width, atlas_height


def _bake_textured_mesh_assets(
    object_mesh_dir: Path,
    mesh: trimesh.Trimesh,
    max_faces: int,
    tile_size: int,
) -> tuple[Path, Path, Path, tuple[int, int], int]:
    textured_mesh = _simplify_colored_mesh(mesh, max_faces=max_faces)
    obj_path = object_mesh_dir / "visual_mesh_textured.obj"
    mtl_path = object_mesh_dir / "visual_mesh_textured.mtl"
    texture_path = object_mesh_dir / "visual_mesh_texture.png"
    atlas_size = _write_textured_mesh_obj(
        obj_path=obj_path,
        mtl_path=mtl_path,
        texture_path=texture_path,
        mesh=textured_mesh,
        tile_size=tile_size,
    )
    return obj_path, mtl_path, texture_path, atlas_size, int(len(textured_mesh.faces))


def _scale_object_mesh(mesh: trimesh.Trimesh, scale_factor: float) -> trimesh.Trimesh:
    if not np.isfinite(scale_factor) or scale_factor <= 0.0:
        raise ValueError(f"invalid scale_factor {scale_factor}")
    scaled_mesh = mesh.copy()
    scaled_mesh.apply_scale(scale_factor)
    return scaled_mesh


def _prepare_object_mesh_for_bbox_frame(
    mesh: trimesh.Trimesh,
    scale_factor: float,
    bbox_size_xyz_m: np.ndarray,
    align_axes_to_bbox: bool,
    max_axis_rel_error: float = 0.25,
) -> tuple[trimesh.Trimesh, dict[str, np.ndarray | float | bool | list[int]]]:
    """Scale, axis-align, and center the object mesh in the bbox local frame.

    FoundationPose poses and `box_for_spider.npz` describe the object body frame.
    SAM3D point clouds may use a different local axis order. We infer the axis
    permutation by matching scaled mesh extents to bbox extents.
    """
    prepared_mesh = _scale_object_mesh(mesh, scale_factor=scale_factor)
    bbox_size = np.asarray(bbox_size_xyz_m, dtype=np.float64)
    if bbox_size.shape != (3,) or not np.isfinite(bbox_size).all() or np.any(bbox_size <= 0):
        raise ValueError(f"invalid bbox_size_xyz_m for mesh alignment: {bbox_size_xyz_m}")

    source_extents = np.asarray(prepared_mesh.extents, dtype=np.float64)
    if source_extents.shape != (3,) or not np.isfinite(source_extents).all() or np.any(source_extents <= 0):
        raise ValueError(f"invalid mesh extents for mesh alignment: {source_extents}")

    best_perm = (0, 1, 2)
    best_rel_error = float("inf")
    best_score = float("inf")
    for perm in permutations(range(3)):
        permuted_extents = source_extents[list(perm)]
        rel_error = np.abs(permuted_extents - bbox_size) / np.maximum(bbox_size, 1e-9)
        score = float(np.linalg.norm(rel_error))
        if score < best_score:
            best_score = score
            best_rel_error = float(np.max(rel_error))
            best_perm = perm

    rotation = np.eye(3, dtype=np.float64)
    alignment_applied = False
    if align_axes_to_bbox and best_rel_error <= max_axis_rel_error:
        permutation_matrix = np.zeros((3, 3), dtype=np.float64)
        for target_axis, source_axis in enumerate(best_perm):
            permutation_matrix[target_axis, source_axis] = 1.0
        if np.linalg.det(permutation_matrix) < 0.0:
            permutation_matrix[int(np.argmin(bbox_size)), :] *= -1.0
        rotation = permutation_matrix
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = rotation
        prepared_mesh.apply_transform(transform)
        alignment_applied = True
    elif align_axes_to_bbox:
        logger.warning(
            "Skipping mesh->bbox axis alignment because best axis extent error is {:.3f}; "
            "source_extents={}, bbox_size={}, best_perm={}",
            best_rel_error,
            source_extents.tolist(),
            bbox_size.tolist(),
            list(best_perm),
        )

    bounds_center = np.asarray(prepared_mesh.bounds, dtype=np.float64).mean(axis=0)
    prepared_mesh.apply_translation(-bounds_center)
    target_extents = np.asarray(prepared_mesh.extents, dtype=np.float64)
    debug = {
        "mesh_bbox_alignment_applied": alignment_applied,
        "mesh_bbox_alignment_perm": list(best_perm),
        "mesh_bbox_alignment_rotation": rotation,
        "mesh_bbox_alignment_source_extents": source_extents,
        "mesh_bbox_alignment_target_extents": target_extents,
        "mesh_bbox_alignment_bbox_size": bbox_size,
        "mesh_bbox_alignment_max_rel_error": best_rel_error,
        "mesh_bbox_alignment_bounds_center_offset": bounds_center,
    }
    return prepared_mesh, debug


def main(
    workspace: str = "preprocessed/watermelon_server",
    dataset_dir: str = f"{spider.ROOT}/../example_datasets",
    dataset_name: str = "ourdata",
    task: str = "watermelon_server",
    data_id: int = 0,
    object_name: str = "watermelon",
    object_id: str = "obj_0",
    embodiment_type: str = "bimanual",
    ref_dt: float = 1.0 / 30.0,
    orientation_policy: str = "preserve_box",
    world_to_sim_alignment: str = ALIGNMENT_MODE_NONE,
    alignment_robot_xml: str | None = None,
    align_object_mesh_to_bbox: bool = True,
    trajectory_interpolation_factor: int = 1,
    scene_offset_xyz: tuple[float, float, float] = (0.10, 0.0, 0.0),
    object_texture_face_resolution: int = 512,
    object_textured_mesh_max_faces: int = 120000,
    object_texture_tile_size: int = 4,
) -> None:
    workspace_path = Path(workspace).resolve()
    dataset_dir_path = Path(dataset_dir).resolve()
    if embodiment_type != "bimanual":
        raise ValueError("first version of ourdata processor only supports bimanual")

    inputs = _resolve_workspace_inputs(workspace_path, object_id=object_id)

    output_dir = Path(
        get_processed_data_dir(
            dataset_dir=str(dataset_dir_path),
            dataset_name=dataset_name,
            robot_type="mano",
            embodiment_type=embodiment_type,
            task=task,
            data_id=data_id,
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    object_mesh_dir = Path(
        get_mesh_dir(
            dataset_dir=str(dataset_dir_path),
            dataset_name=dataset_name,
            object_name=object_name,
        )
    )
    object_mesh_dir.mkdir(parents=True, exist_ok=True)

    mocap_data = np.load(inputs.world_mocap_path)
    camera_bundle = _load_camera_bundle(inputs.camera_npz_path)
    box_data = np.load(inputs.box_npz_path)

    right_verts = mocap_data["right_verts"].astype(np.float32)
    left_verts = mocap_data["left_verts"].astype(np.float32)
    depths = camera_bundle.depths
    intrinsic = camera_bundle.intrinsic
    cam_c2w = camera_bundle.cam_c2w

    frame_counts = [
        right_verts.shape[0],
        left_verts.shape[0],
        cam_c2w.shape[0],
    ]
    if depths is not None:
        frame_counts.append(depths.shape[0])
    if inputs.object_pose_dir is not None:
        frame_counts.append(len(list(inputs.object_pose_dir.glob("*.txt"))))
    elif inputs.masks_dir is not None:
        frame_counts.append(len(list(inputs.masks_dir.glob("*.png"))))
    num_frames = min(frame_counts)
    if num_frames <= 0:
        raise RuntimeError("no overlapping frames found across workspace outputs")

    right_verts = right_verts[:num_frames]
    left_verts = left_verts[:num_frames]
    if depths is not None:
        depths = depths[:num_frames]
    cam_c2w = cam_c2w[:num_frames]

    logger.info("Converting workspace {} -> dataset={} task={} frames={}", workspace_path, dataset_name, task, num_frames)

    raw_right_verts = right_verts.copy()
    raw_left_verts = left_verts.copy()
    raw_qpos_wrist_right, raw_qpos_finger_right = _estimate_hand_trajectory(
        raw_right_verts, side="right"
    )
    raw_qpos_wrist_left, raw_qpos_finger_left = _estimate_hand_trajectory(
        raw_left_verts, side="left"
    )

    initial_center_world = box_data["box_center_world"].astype(np.float64)
    box_rotation = box_data["box_rotation_R"].astype(np.float64)
    if inputs.object_pose_dir is not None:
        object_centers_world, object_rotations_world, object_debug = (
            _load_foundationpose_object_trajectory(
                pose_dir=inputs.object_pose_dir,
                cam_c2w=cam_c2w,
                num_frames=num_frames,
            )
        )
        object_pose_source = "foundationpose_center_pose"
        object_pose_dir = str(inputs.object_pose_dir)
    else:
        if depths is None:
            raise RuntimeError(
                f"{camera_bundle.path} does not contain depths required for mask/depth object tracking"
            )
        if inputs.masks_dir is None:
            raise FileNotFoundError(
                "No FoundationPose center_pose directory was found and no object mask "
                "directory is available for mask/depth fallback."
            )
        object_centers_world, object_debug = _estimate_object_trajectory(
            masks_dir=inputs.masks_dir,
            depths=depths,
            intrinsic=intrinsic,
            cam_c2w=cam_c2w,
            initial_center_world=initial_center_world,
            num_frames=num_frames,
        )
        object_rotation_source = _resolve_object_rotation(
            box_rotation=box_rotation,
            orientation_policy=orientation_policy,
        )
        object_rotations_world = np.tile(object_rotation_source[None, :, :], (num_frames, 1, 1))
        object_pose_source = "mask_depth"
        object_pose_dir = None
    raw_object_centers_world = object_centers_world.copy()
    raw_object_rotations_world = object_rotations_world.copy()

    alignment = _resolve_world_to_sim_alignment(
        alignment_mode=world_to_sim_alignment,
        alignment_robot_xml=alignment_robot_xml,
        cam_c2w=cam_c2w,
    )
    if alignment.mode != ALIGNMENT_MODE_NONE:
        logger.info(
            "Applying world->sim alignment mode={} using {} frames; robot_xml={}",
            alignment.mode,
            alignment.num_frames,
            alignment.robot_xml,
        )
        logger.info(
            "Alignment source origin={} target origin={} translation={}",
            alignment.source_origin.tolist(),
            alignment.target_origin.tolist(),
            alignment.translation.tolist(),
        )

    right_verts = _rigid_transform_points(
        raw_right_verts,
        rotation=alignment.rotation,
        translation=alignment.translation,
    )
    left_verts = _rigid_transform_points(
        raw_left_verts,
        rotation=alignment.rotation,
        translation=alignment.translation,
    )
    object_centers_sim = _rigid_transform_points(
        raw_object_centers_world,
        rotation=alignment.rotation,
        translation=alignment.translation,
    )
    aligned_initial_center_world = _rigid_transform_points(
        initial_center_world[None],
        rotation=alignment.rotation,
        translation=alignment.translation,
    )[0]
    object_rotations_sim = np.stack(
        [
            _orthonormalize_rotation(alignment.rotation @ object_rotation)
            for object_rotation in raw_object_rotations_world
        ],
        axis=0,
    )
    object_quats_wxyz = np.stack(
        [_rotation_matrix_to_wxyz(object_rotation) for object_rotation in object_rotations_sim],
        axis=0,
    )

    qpos_wrist_right, qpos_finger_right = _estimate_hand_trajectory(
        right_verts, side="right"
    )
    qpos_wrist_left, qpos_finger_left = _estimate_hand_trajectory(
        left_verts, side="left"
    )
    qpos_obj_right = np.zeros((num_frames, 7), dtype=np.float32)
    qpos_obj_right[:, :3] = object_centers_sim
    qpos_obj_right[:, 3:] = object_quats_wxyz.astype(np.float32)

    qpos_obj_left = np.zeros((num_frames, 7), dtype=np.float32)
    qpos_obj_left[:, 3] = 1.0

    contact_right = np.zeros((num_frames, 5), dtype=np.float32)
    contact_left = np.zeros((num_frames, 5), dtype=np.float32)

    scene_offset_xyz_arr = np.asarray(scene_offset_xyz, dtype=np.float32)
    if scene_offset_xyz_arr.shape != (3,):
        raise ValueError(
            f"scene_offset_xyz must have exactly 3 values, got {scene_offset_xyz}"
        )
    if not np.all(np.isfinite(scene_offset_xyz_arr)):
        raise ValueError(f"scene_offset_xyz must be finite, got {scene_offset_xyz}")
    if np.linalg.norm(scene_offset_xyz_arr) > 1e-9:
        qpos_wrist_right[:, :3] += scene_offset_xyz_arr
        qpos_finger_right[:, :, :3] += scene_offset_xyz_arr
        qpos_obj_right[:, :3] += scene_offset_xyz_arr
        qpos_wrist_left[:, :3] += scene_offset_xyz_arr
        qpos_finger_left[:, :, :3] += scene_offset_xyz_arr

        left_object_pos_empty = np.all(
            np.linalg.norm(qpos_obj_left[:, :3], axis=1) < 1e-6
        )
        left_object_quat_empty = np.all(
            np.linalg.norm(qpos_obj_left[:, 3:] - IDENTITY_QUAT_WXYZ, axis=1)
            < 1e-6
        )
        if not (left_object_pos_empty and left_object_quat_empty):
            qpos_obj_left[:, :3] += scene_offset_xyz_arr
        logger.info(
            "Applied scene_offset_xyz={}; right_object_xyz [{:.4f}, {:.4f}, {:.4f}] -> [{:.4f}, {:.4f}, {:.4f}]",
            scene_offset_xyz_arr.tolist(),
            float(object_centers_sim[0, 0]),
            float(object_centers_sim[0, 1]),
            float(object_centers_sim[0, 2]),
            float(qpos_obj_right[0, 0]),
            float(qpos_obj_right[0, 1]),
            float(qpos_obj_right[0, 2]),
        )

    interpolation_factor = int(trajectory_interpolation_factor)
    source_num_frames = int(num_frames)
    if interpolation_factor < 1:
        raise ValueError(
            f"trajectory_interpolation_factor must be >= 1, got {interpolation_factor}"
        )
    if interpolation_factor > 1:
        qpos_wrist_right = _interpolate_pose_trajectory(
            qpos_wrist_right, interpolation_factor
        )
        qpos_finger_right = _interpolate_pose_trajectory(
            qpos_finger_right, interpolation_factor
        )
        qpos_obj_right = _interpolate_pose_trajectory(
            qpos_obj_right, interpolation_factor
        )
        qpos_wrist_left = _interpolate_pose_trajectory(
            qpos_wrist_left, interpolation_factor
        )
        qpos_finger_left = _interpolate_pose_trajectory(
            qpos_finger_left, interpolation_factor
        )
        qpos_obj_left = _interpolate_pose_trajectory(
            qpos_obj_left, interpolation_factor
        )
        contact_right = _interpolate_contact_trajectory(
            contact_right, interpolation_factor
        )
        contact_left = _interpolate_contact_trajectory(
            contact_left, interpolation_factor
        )
        num_frames = int(qpos_wrist_right.shape[0])
        logger.info(
            "Interpolated keypoint trajectory: {} -> {} frames (factor={})",
            source_num_frames,
            num_frames,
            interpolation_factor,
        )

    trajectory_path = output_dir / "trajectory_keypoints.npz"
    np.savez(
        trajectory_path,
        qpos_wrist_right=qpos_wrist_right,
        qpos_finger_right=qpos_finger_right,
        qpos_obj_right=qpos_obj_right,
        qpos_wrist_left=qpos_wrist_left,
        qpos_finger_left=qpos_finger_left,
        qpos_obj_left=qpos_obj_left,
        contact_right=contact_right,
        contact_left=contact_left,
    )

    object_scale_factor = float(box_data["scale_factor"])
    object_bbox_size_xyz_m = box_data["box_real_size_xyz_m"].astype(np.float32)
    source_object_has_faces = _source_mesh_has_faces(inputs.object_ply_path)
    visual_mesh, mesh_alignment_debug = _prepare_object_mesh_for_bbox_frame(
        _load_object_mesh(inputs.object_ply_path),
        scale_factor=object_scale_factor,
        bbox_size_xyz_m=object_bbox_size_xyz_m,
        align_axes_to_bbox=align_object_mesh_to_bbox,
    )
    visual_obj_path = object_mesh_dir / "visual.obj"
    visual_mesh.export(visual_obj_path)
    visual_mesh_ply_path = object_mesh_dir / "visual_mesh.ply"
    visual_mesh.export(visual_mesh_ply_path)
    colored_ply_path = object_mesh_dir / "visual_colored.ply"
    colored_ply_written = False
    colored_ply_num_points = 0
    colored_ply_color_source = ""
    textured_obj_path = object_mesh_dir / "visual_mesh_textured.obj"
    textured_mtl_path = object_mesh_dir / "visual_mesh_textured.mtl"
    texture_png_path = object_mesh_dir / "visual_mesh_texture.png"
    textured_assets_written = False
    texture_atlas_size = (0, 0)
    textured_mesh_face_count = 0
    try:
        mesh_vertex_rgb = _extract_mesh_vertex_rgb(visual_mesh)
        if source_object_has_faces and mesh_vertex_rgb is not None:
            _write_colored_point_ply(
                colored_ply_path,
                points=np.asarray(visual_mesh.vertices, dtype=np.float32),
                colors=mesh_vertex_rgb,
            )
            colored_ply_written = True
            colored_ply_num_points = int(len(visual_mesh.vertices))
            colored_ply_color_source = f"{inputs.object_ply_path.suffix.lower()}_vertex_rgb"
            (
                textured_obj_path,
                textured_mtl_path,
                texture_png_path,
                texture_atlas_size,
                textured_mesh_face_count,
            ) = _bake_textured_mesh_assets(
                object_mesh_dir=object_mesh_dir,
                mesh=visual_mesh,
                max_faces=int(object_textured_mesh_max_faces),
                tile_size=int(object_texture_tile_size),
            )
            textured_assets_written = True
            logger.info(
                "Saved textured object mesh assets to {}, {}, {} "
                "(faces={}, atlas={}x{}, source={})",
                textured_obj_path,
                textured_mtl_path,
                texture_png_path,
                textured_mesh_face_count,
                texture_atlas_size[0],
                texture_atlas_size[1],
                inputs.object_ply_path,
            )
        else:
            colored_points_raw, colored_points_rgb, colored_ply_color_source = (
                _load_gaussian_colored_points(inputs.object_ply_path)
            )
            colored_points = _prepare_object_points_with_mesh_alignment(
                colored_points_raw,
                scale_factor=object_scale_factor,
                mesh_alignment_debug=mesh_alignment_debug,
            )
            _write_colored_point_ply(
                colored_ply_path,
                points=colored_points,
                colors=colored_points_rgb,
            )
            colored_ply_written = True
            colored_ply_num_points = int(len(colored_points))
            logger.info(
                "Saved colored object PLY to {} (points={}, color_source={})",
                colored_ply_path,
                colored_ply_num_points,
                colored_ply_color_source,
            )
            _, _, _, texture_atlas_size = _bake_textured_box_assets(
                object_mesh_dir=object_mesh_dir,
                points=colored_points,
                colors=colored_points_rgb,
                bbox_size_xyz_m=object_bbox_size_xyz_m,
                face_resolution=int(object_texture_face_resolution),
            )
            textured_obj_path = object_mesh_dir / "visual_textured.obj"
            textured_mtl_path = object_mesh_dir / "visual_textured.mtl"
            texture_png_path = object_mesh_dir / "visual_texture.png"
            textured_assets_written = True
            logger.info(
                "Saved textured object box assets to {}, {}, {} (atlas={}x{})",
                textured_obj_path,
                textured_mtl_path,
                texture_png_path,
                texture_atlas_size[0],
                texture_atlas_size[1],
            )
    except Exception as exc:
        logger.warning(
            "Could not export colored/textured object assets from {}: {}",
            inputs.object_ply_path,
            exc,
        )

    task_info = {
        "task": task,
        "dataset_name": dataset_name,
        "robot_type": "mano",
        "embodiment_type": embodiment_type,
        "data_id": data_id,
        "ref_dt": ref_dt,
        "right_object_mesh_dir": str(object_mesh_dir.relative_to(dataset_dir_path)),
        "left_object_mesh_dir": None,
        "right_object_convex_dir": None,
        "left_object_convex_dir": None,
        "right_object_colored_ply": (
            str(colored_ply_path.relative_to(dataset_dir_path))
            if colored_ply_written
            else None
        ),
        "right_object_visual_mesh_ply": str(
            visual_mesh_ply_path.relative_to(dataset_dir_path)
        ),
        "right_object_visual_source_has_faces": bool(source_object_has_faces),
        "right_object_textured_visual": (
            str(textured_obj_path.relative_to(dataset_dir_path))
            if textured_assets_written
            else None
        ),
        "right_object_texture_png": (
            str(texture_png_path.relative_to(dataset_dir_path))
            if textured_assets_written
            else None
        ),
        "source_workspace": str(workspace_path),
        "object_id": object_id,
        "num_frames": int(num_frames),
        "source_num_frames": int(source_num_frames),
        "trajectory_interpolation_factor": int(interpolation_factor),
        "trajectory_interpolation_method": "pchip_xyz_slerp_quat_nearest_contact",
        "scene_offset_xyz": scene_offset_xyz_arr.tolist(),
        "camera_bundle_path": str(camera_bundle.path),
        "object_pose_source": object_pose_source,
        "object_pose_dir": object_pose_dir,
        "object_scale_factor": object_scale_factor,
        "object_colored_ply_num_points": colored_ply_num_points,
        "object_colored_ply_color_source": colored_ply_color_source,
        "object_visual_source_path": str(inputs.object_ply_path),
        "object_texture_face_resolution": int(object_texture_face_resolution),
        "object_textured_mesh_max_faces": int(object_textured_mesh_max_faces),
        "object_texture_tile_size": int(object_texture_tile_size),
        "object_textured_mesh_face_count": int(textured_mesh_face_count),
        "object_texture_atlas_size": list(texture_atlas_size),
        "object_orientation_policy": orientation_policy,
        "align_object_mesh_to_bbox": align_object_mesh_to_bbox,
        "object_mesh_bbox_alignment_applied": bool(
            mesh_alignment_debug["mesh_bbox_alignment_applied"]
        ),
        "object_mesh_bbox_alignment_perm": mesh_alignment_debug[
            "mesh_bbox_alignment_perm"
        ],
        "object_mesh_bbox_alignment_rotation": mesh_alignment_debug[
            "mesh_bbox_alignment_rotation"
        ].tolist(),
        "object_mesh_bbox_alignment_source_extents": mesh_alignment_debug[
            "mesh_bbox_alignment_source_extents"
        ].tolist(),
        "object_mesh_bbox_alignment_target_extents": mesh_alignment_debug[
            "mesh_bbox_alignment_target_extents"
        ].tolist(),
        "object_mesh_bbox_alignment_max_rel_error": float(
            mesh_alignment_debug["mesh_bbox_alignment_max_rel_error"]
        ),
        "right_object_bbox_size_xyz_m": object_bbox_size_xyz_m.tolist(),
        "left_object_bbox_size_xyz_m": None,
        "world_to_sim_alignment_mode": alignment.mode,
        "world_to_sim_rotation_matrix": alignment.rotation.tolist(),
        "world_to_sim_translation_xyz_m": alignment.translation.tolist(),
        "world_to_sim_source_origin_xyz_m": alignment.source_origin.tolist(),
        "world_to_sim_target_origin_xyz_m": alignment.target_origin.tolist(),
        "world_to_sim_source_frame": alignment.source_frame.tolist(),
        "world_to_sim_target_frame": alignment.target_frame.tolist(),
        "alignment_robot_xml": alignment.robot_xml,
        "reference_camera_frame_index": int(alignment.reference_camera_frame_index),
        "reference_world_camera_pose": alignment.source_camera_pose.tolist(),
        "reference_sim_camera_pose": alignment.target_camera_pose.tolist(),
        "reference_target_camera_frame_name": alignment.target_camera_frame_name,
        "reference_target_camera_pose": alignment.target_camera_pose.tolist(),
        "reference_legacy_head_camera_pose": alignment.legacy_head_camera_pose.tolist(),
    }
    task_info_path = output_dir.parent / "task_info.json"
    with task_info_path.open("w", encoding="utf-8") as file:
        json.dump(task_info, file, indent=2)

    debug_path = output_dir / "conversion_debug.npz"
    np.savez(
        debug_path,
        raw_right_wrist=qpos_wrist_right if alignment.mode == ALIGNMENT_MODE_NONE else raw_qpos_wrist_right,
        raw_left_wrist=qpos_wrist_left if alignment.mode == ALIGNMENT_MODE_NONE else raw_qpos_wrist_left,
        raw_right_tip_positions=qpos_finger_right[:, :, :3] if alignment.mode == ALIGNMENT_MODE_NONE else raw_qpos_finger_right[:, :, :3],
        raw_left_tip_positions=qpos_finger_left[:, :, :3] if alignment.mode == ALIGNMENT_MODE_NONE else raw_qpos_finger_left[:, :, :3],
        aligned_right_wrist=qpos_wrist_right,
        aligned_left_wrist=qpos_wrist_left,
        right_tip_positions=qpos_finger_right[:, :, :3],
        left_tip_positions=qpos_finger_left[:, :, :3],
        raw_object_centers_world=raw_object_centers_world,
        raw_object_rotations_world=raw_object_rotations_world.astype(np.float32),
        right_object_centers_world=object_centers_sim,
        right_object_centers_sim=qpos_obj_right[:, :3],
        right_object_centers_sim_before_scene_offset=object_centers_sim,
        right_object_rotations_sim=object_rotations_sim.astype(np.float32),
        object_tracking_valid_mask=object_debug["valid_mask"],
        object_tracking_selected_labels=object_debug["selected_labels"],
        object_tracking_selected_areas=object_debug["selected_areas"],
        object_tracking_selected_uv=object_debug["selected_uv"],
        object_tracking_reference_uv=object_debug["reference_uv"],
        object_pose_source=np.array(object_pose_source),
        object_pose_dir=np.array("" if object_pose_dir is None else object_pose_dir),
        object_pose_source_files=object_debug.get(
            "pose_source_files",
            np.asarray([], dtype=str),
        ),
        raw_object_poses_world=object_debug.get(
            "object_poses_world",
            np.zeros((0, 4, 4), dtype=np.float32),
        ),
        initial_box_center_world=initial_center_world.astype(np.float32),
        aligned_initial_box_center_world=aligned_initial_center_world.astype(np.float32),
        object_bbox_size_xyz_m=object_bbox_size_xyz_m,
        object_scale_factor=np.array(object_scale_factor, dtype=np.float32),
        object_colored_ply_path=np.array(
            str(colored_ply_path) if colored_ply_written else ""
        ),
        object_colored_ply_num_points=np.array(
            colored_ply_num_points,
            dtype=np.int32,
        ),
        object_colored_ply_color_source=np.array(colored_ply_color_source),
        object_textured_visual_path=np.array(
            str(textured_obj_path) if textured_assets_written else ""
        ),
        object_texture_png_path=np.array(
            str(texture_png_path) if textured_assets_written else ""
        ),
        object_texture_face_resolution=np.array(
            int(object_texture_face_resolution),
            dtype=np.int32,
        ),
        object_textured_mesh_max_faces=np.array(
            int(object_textured_mesh_max_faces),
            dtype=np.int32,
        ),
        object_texture_tile_size=np.array(
            int(object_texture_tile_size),
            dtype=np.int32,
        ),
        object_textured_mesh_face_count=np.array(
            int(textured_mesh_face_count),
            dtype=np.int32,
        ),
        object_texture_atlas_size=np.asarray(texture_atlas_size, dtype=np.int32),
        object_orientation_policy=np.array(orientation_policy),
        mesh_bbox_alignment_applied=np.array(
            mesh_alignment_debug["mesh_bbox_alignment_applied"]
        ),
        mesh_bbox_alignment_perm=np.asarray(
            mesh_alignment_debug["mesh_bbox_alignment_perm"],
            dtype=np.int32,
        ),
        mesh_bbox_alignment_rotation=mesh_alignment_debug[
            "mesh_bbox_alignment_rotation"
        ].astype(np.float32),
        mesh_bbox_alignment_source_extents=mesh_alignment_debug[
            "mesh_bbox_alignment_source_extents"
        ].astype(np.float32),
        mesh_bbox_alignment_target_extents=mesh_alignment_debug[
            "mesh_bbox_alignment_target_extents"
        ].astype(np.float32),
        mesh_bbox_alignment_bbox_size=mesh_alignment_debug[
            "mesh_bbox_alignment_bbox_size"
        ].astype(np.float32),
        mesh_bbox_alignment_max_rel_error=np.array(
            mesh_alignment_debug["mesh_bbox_alignment_max_rel_error"],
            dtype=np.float32,
        ),
        mesh_bbox_alignment_bounds_center_offset=mesh_alignment_debug[
            "mesh_bbox_alignment_bounds_center_offset"
        ].astype(np.float32),
        world_to_sim_alignment_mode=np.array(alignment.mode),
        world_to_sim_rotation_matrix=alignment.rotation.astype(np.float32),
        world_to_sim_translation_xyz_m=alignment.translation.astype(np.float32),
        world_to_sim_source_origin_xyz_m=alignment.source_origin.astype(np.float32),
        world_to_sim_target_origin_xyz_m=alignment.target_origin.astype(np.float32),
        world_to_sim_source_frame=alignment.source_frame.astype(np.float32),
        world_to_sim_target_frame=alignment.target_frame.astype(np.float32),
        reference_camera_frame_index=np.array(
            alignment.reference_camera_frame_index, dtype=np.int32
        ),
        reference_world_camera_pose=alignment.source_camera_pose.astype(np.float32),
        reference_sim_camera_pose=alignment.target_camera_pose.astype(np.float32),
        reference_target_camera_frame_name=np.array(
            alignment.target_camera_frame_name
        ),
        reference_target_camera_pose=alignment.target_camera_pose.astype(np.float32),
        reference_legacy_head_camera_pose=alignment.legacy_head_camera_pose.astype(
            np.float32
        ),
        object_quats_wxyz=object_quats_wxyz.astype(np.float32),
        source_num_frames=np.array(source_num_frames, dtype=np.int32),
        num_frames=np.array(num_frames, dtype=np.int32),
        trajectory_interpolation_factor=np.array(
            interpolation_factor, dtype=np.int32
        ),
        scene_offset_xyz=scene_offset_xyz_arr.astype(np.float32),
    )

    logger.info("Saved trajectory keypoints to {}", trajectory_path)
    logger.info("Saved task_info to {}", task_info_path)
    logger.info("Saved visual mesh to {}", visual_obj_path)
    if colored_ply_written:
        logger.info("Saved colored visual point cloud to {}", colored_ply_path)
    if textured_assets_written:
        logger.info("Saved textured visual mesh to {}", textured_obj_path)
    logger.info("Saved conversion debug data to {}", debug_path)


if __name__ == "__main__":
    tyro.cli(main)
