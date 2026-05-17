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
ALIGNMENT_MODE_NONE = "none"
ALIGNMENT_MODE_D435_OPTICAL = "d435_optical"
HEAD_CAMERA_SITE_NAME = "head_camera_frame"
D435_OPTICAL_FRAME_SITE_NAME = "d435_optical_frame"
SUPPORTED_ALIGNMENT_MODES = {
    ALIGNMENT_MODE_NONE,
    ALIGNMENT_MODE_D435_OPTICAL,
}


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
            workspace_path / "mega_sam.npz",
        ],
        "camera/depth npz",
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
            workspace_path / "sam3d" / object_id / "obj_3d_final.ply",
            workspace_path / "sam3d" / "obj_3d_final.ply",
        ],
        "obj_3d_final.ply",
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
    geometry = trimesh.load(source_ply, process=False)
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
    workspace: str = "preprocessed/milk",
    dataset_dir: str = f"{spider.ROOT}/../example_datasets",
    dataset_name: str = "ourdata",
    task: str = "milk",
    data_id: int = 0,
    object_name: str = "milk",
    object_id: str = "obj_0",
    embodiment_type: str = "bimanual",
    ref_dt: float = 1.0 / 30.0,
    orientation_policy: str = "preserve_box",
    world_to_sim_alignment: str = ALIGNMENT_MODE_NONE,
    alignment_robot_xml: str | None = None,
    align_object_mesh_to_bbox: bool = True,
    trajectory_interpolation_factor: int = 1,
    scene_offset_xyz: tuple[float, float, float] = (0.10, 0.0, 0.0),
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
    visual_mesh, mesh_alignment_debug = _prepare_object_mesh_for_bbox_frame(
        _load_object_mesh(inputs.object_ply_path),
        scale_factor=object_scale_factor,
        bbox_size_xyz_m=object_bbox_size_xyz_m,
        align_axes_to_bbox=align_object_mesh_to_bbox,
    )
    visual_obj_path = object_mesh_dir / "visual.obj"
    visual_mesh.export(visual_obj_path)

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
    logger.info("Saved conversion debug data to {}", debug_path)


if __name__ == "__main__":
    tyro.cli(main)
