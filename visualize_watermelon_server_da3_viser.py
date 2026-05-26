#!/usr/bin/env python3
"""Viser viewer for raw hand keypoints and object poses in a preprocessed scene.

This script intentionally stays in the DA3/world coordinate frame. It does not
apply the ASM d435_optical world-to-sim alignment used by IK/MJWP.
"""

from __future__ import annotations

import argparse
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import loguru
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as R


RIGHT_HAND_COLOR = np.array([255, 80, 80], dtype=np.uint8)
RIGHT_VERT_COLOR = np.array([255, 185, 185], dtype=np.uint8)
LEFT_HAND_COLOR = np.array([80, 130, 255], dtype=np.uint8)
LEFT_VERT_COLOR = np.array([180, 210, 255], dtype=np.uint8)
OBJECT_CENTER_COLOR = np.array([0, 220, 120], dtype=np.uint8)
OBJECT_BBOX_COLOR = np.array([255, 220, 0], dtype=np.uint8)
CAMERA_COLOR = np.array([255, 140, 0], dtype=np.uint8)
OBJECT_PALETTE = [
    np.array([255, 220, 0], dtype=np.uint8),
    np.array([0, 220, 180], dtype=np.uint8),
    np.array([220, 120, 255], dtype=np.uint8),
    np.array([255, 120, 80], dtype=np.uint8),
]

BBOX_EDGES = [
    (0, 1),
    (2, 3),
    (4, 5),
    (6, 7),
    (0, 2),
    (1, 3),
    (4, 6),
    (5, 7),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
]

MANO_HAND_EDGES = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
]


@dataclass
class ViewerConfig:
    scene: str = "pick_place"
    workspace: str | None = None
    object_id: str = "obj_0"
    object_ids: tuple[str, ...] | None = None
    pose_dir: str | None = None
    object_pose_mode: str = "camera_to_object"
    host: str = "0.0.0.0"
    port: int = 8080
    fps: int = 30
    show_hand_vertices: bool = True
    show_object_mesh: bool = True
    fit_object_mesh_to_bbox: bool = True
    point_stride: int = 4
    hand_vertex_stride: int = 4
    point_size_joints: float = 0.018
    point_size_vertices: float = 0.004
    point_size_object: float = 0.025
    object_axes_length: float = 0.12
    verbose: bool = True


@dataclass
class SceneData:
    num_frames: int
    cam_c2w: np.ndarray
    right_joints: np.ndarray
    left_joints: np.ndarray
    right_vertices: np.ndarray | None
    left_vertices: np.ndarray | None
    objects: list["ObjectData"]


@dataclass
class ObjectData:
    object_id: str
    pose_dir: Path
    centers: np.ndarray
    quats_wxyz: np.ndarray
    extents: np.ndarray
    mesh: trimesh.Trimesh | None


def _parse_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value!r}")


def parse_args() -> ViewerConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize raw hands and one or more objects in DA3 world "
            "coordinates with Viser."
        )
    )
    parser.add_argument(
        "--scene",
        default=ViewerConfig.scene,
        help="Predefined workspace under preprocessed/. Ignored only when --workspace is set.",
    )
    parser.add_argument(
        "--workspace",
        default=None,
        help="Raw workspace directory. Defaults to preprocessed/<scene>.",
    )
    parser.add_argument(
        "--object-id",
        default=ViewerConfig.object_id,
        help="Single object id to visualize when --object-ids is not provided.",
    )
    parser.add_argument(
        "--object-ids",
        nargs="+",
        default=None,
        help="One or more object ids to visualize, e.g. --object-ids obj_0 obj_1.",
    )
    parser.add_argument(
        "--pose-dir",
        default=None,
        help=(
            "Optional object pose directory override. By default the script tries "
            "result/<object-id>/poses and result/<object-id>/foundationpose_debug/center_pose."
        ),
    )
    parser.add_argument(
        "--object-pose-mode",
        default=ViewerConfig.object_pose_mode,
        choices=[
            "camera_to_object",
            "world_to_object",
            "object_to_camera",
            "object_to_world",
        ],
        help=(
            "How to interpret result/<object-id>/poses/*.txt. "
            "camera_to_object uses T_world_obj = da3.cam_c2w @ pose."
        ),
    )
    parser.add_argument("--host", default=ViewerConfig.host)
    parser.add_argument("--port", type=int, default=ViewerConfig.port)
    parser.add_argument("--fps", type=int, default=ViewerConfig.fps)
    parser.add_argument("--show-hand-vertices", type=_parse_bool, default=True)
    parser.add_argument("--show-object-mesh", type=_parse_bool, default=True)
    parser.add_argument(
        "--fit-object-mesh-to-bbox",
        type=_parse_bool,
        default=ViewerConfig.fit_object_mesh_to_bbox,
        help=(
            "Scale the loaded object mesh so its local bounds match "
            "box_for_spider.npz box_real_size_xyz_m. This keeps the mesh in real meters."
        ),
    )
    parser.add_argument("--point-stride", type=int, default=ViewerConfig.point_stride)
    parser.add_argument(
        "--hand-vertex-stride",
        type=int,
        default=ViewerConfig.hand_vertex_stride,
    )
    parser.add_argument("--point-size-joints", type=float, default=ViewerConfig.point_size_joints)
    parser.add_argument(
        "--point-size-vertices",
        type=float,
        default=ViewerConfig.point_size_vertices,
    )
    parser.add_argument("--point-size-object", type=float, default=ViewerConfig.point_size_object)
    parser.add_argument("--object-axes-length", type=float, default=ViewerConfig.object_axes_length)
    parser.add_argument("--verbose", type=_parse_bool, default=ViewerConfig.verbose)
    args = parser.parse_args()
    workspace = args.workspace or str(Path("preprocessed") / args.scene)
    return ViewerConfig(
        scene=args.scene,
        workspace=workspace,
        object_id=args.object_id,
        object_ids=tuple(args.object_ids) if args.object_ids else None,
        pose_dir=args.pose_dir,
        object_pose_mode=args.object_pose_mode,
        host=args.host,
        port=args.port,
        fps=max(1, int(args.fps)),
        show_hand_vertices=args.show_hand_vertices,
        show_object_mesh=args.show_object_mesh,
        fit_object_mesh_to_bbox=args.fit_object_mesh_to_bbox,
        point_stride=max(1, int(args.point_stride)),
        hand_vertex_stride=max(1, int(args.hand_vertex_stride)),
        point_size_joints=args.point_size_joints,
        point_size_vertices=args.point_size_vertices,
        point_size_object=args.point_size_object,
        object_axes_length=args.object_axes_length,
        verbose=args.verbose,
    )


def _rotmat_to_wxyz(rotmat: np.ndarray) -> np.ndarray:
    quat_xyzw = R.from_matrix(np.asarray(rotmat, dtype=np.float64)).as_quat()
    return np.array(
        [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]],
        dtype=np.float32,
    )


def _wxyz_to_rotmat(quat_wxyz: np.ndarray) -> np.ndarray:
    quat_wxyz = np.asarray(quat_wxyz, dtype=np.float64)
    quat_wxyz = quat_wxyz / max(float(np.linalg.norm(quat_wxyz)), 1e-12)
    quat_xyzw = np.array(
        [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]],
        dtype=np.float64,
    )
    return R.from_quat(quat_xyzw).as_matrix()


def _line_colors(num_segments: int, color: np.ndarray) -> np.ndarray:
    color = np.asarray(color, dtype=np.uint8)
    return np.tile(color[None, None, :], (num_segments, 2, 1))


def _hand_segments(joints: np.ndarray) -> np.ndarray:
    joints = np.asarray(joints, dtype=np.float32)
    segments = []
    for a, b in MANO_HAND_EDGES:
        if a < len(joints) and b < len(joints):
            segments.append([joints[a], joints[b]])
    return np.asarray(segments, dtype=np.float32)


def _obb_corners(center: np.ndarray, rotmat: np.ndarray, extents: np.ndarray) -> np.ndarray:
    center = np.asarray(center, dtype=np.float32)
    rotmat = np.asarray(rotmat, dtype=np.float32)
    half = 0.5 * np.asarray(extents, dtype=np.float32)
    corners = []
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                corners.append(
                    center
                    + sx * rotmat[:, 0] * half[0]
                    + sy * rotmat[:, 1] * half[1]
                    + sz * rotmat[:, 2] * half[2]
                )
    return np.asarray(corners, dtype=np.float32)


def _obb_segments(corners: np.ndarray) -> np.ndarray:
    return np.asarray([[corners[a], corners[b]] for a, b in BBOX_EDGES], dtype=np.float32)


def _camera_segments(cam_c2w: np.ndarray, scale: float = 0.08) -> np.ndarray:
    origin = cam_c2w[:3, 3].astype(np.float32)
    rot = cam_c2w[:3, :3].astype(np.float32)
    endpoints = [origin + rot[:, i] * scale for i in range(3)]
    return np.asarray([[origin, endpoint] for endpoint in endpoints], dtype=np.float32)


def _load_mesh(path: Path) -> trimesh.Trimesh:
    geometry = trimesh.load(path, process=False, maintain_order=True)
    if isinstance(geometry, trimesh.Scene):
        geometry = geometry.to_mesh()
    if isinstance(geometry, trimesh.points.PointCloud):
        geometry = geometry.convex_hull
    if not isinstance(geometry, trimesh.Trimesh):
        raise TypeError(f"unsupported mesh type from {path}: {type(geometry)}")
    if len(geometry.faces) == 0:
        geometry = geometry.convex_hull
    mesh = geometry.copy()
    if len(mesh.vertices) > 0:
        mesh.apply_translation(-mesh.bounds.mean(axis=0))
    return mesh


def _fit_mesh_to_bbox_extents(
    mesh: trimesh.Trimesh,
    bbox_extents: np.ndarray,
    source_path: Path,
) -> trimesh.Trimesh:
    bbox_extents = np.asarray(bbox_extents, dtype=np.float64)
    mesh_extents = np.asarray(mesh.extents, dtype=np.float64)
    if (
        bbox_extents.shape != (3,)
        or not np.all(np.isfinite(bbox_extents))
        or np.any(bbox_extents <= 0.0)
    ):
        raise ValueError(f"invalid bbox extents for object mesh fitting: {bbox_extents}")
    if not np.all(np.isfinite(mesh_extents)) or np.any(mesh_extents <= 1e-9):
        raise ValueError(
            f"invalid source mesh extents for {source_path}: {mesh_extents}"
        )

    fitted = mesh.copy()
    bounds_center = np.asarray(fitted.bounds, dtype=np.float64).mean(axis=0)
    scale_xyz = bbox_extents / mesh_extents
    fitted.vertices = (np.asarray(fitted.vertices, dtype=np.float64) - bounds_center) * scale_xyz
    fitted.apply_translation(-np.asarray(fitted.bounds, dtype=np.float64).mean(axis=0))
    loguru.logger.info(
        "Fitted object mesh to real bbox size: source={} raw_extents={} bbox_extents={} scale_xyz={} fitted_extents={}",
        source_path,
        np.round(mesh_extents, 6).tolist(),
        np.round(bbox_extents, 6).tolist(),
        np.round(scale_xyz, 6).tolist(),
        np.round(np.asarray(fitted.extents, dtype=np.float64), 6).tolist(),
    )
    return fitted


def _candidate_mesh_paths(workspace: Path, object_id: str) -> list[Path]:
    return [
        workspace / "result" / object_id / "scaled_mesh.glb",
        workspace / "result" / object_id / "scaled_mesh.ply",
        workspace / "sam3d" / object_id / "obj_mesh_final.glb",
        workspace / "sam3d" / object_id / "obj_mesh_final.obj",
        workspace / "sam3d" / object_id / "obj_mesh_final.ply",
        workspace / "sam3d" / object_id / "obj_3d_final.ply",
    ]


def _load_object_mesh(
    workspace: Path,
    object_id: str,
    bbox_extents: np.ndarray | None = None,
    fit_to_bbox: bool = True,
) -> trimesh.Trimesh | None:
    for path in _candidate_mesh_paths(workspace, object_id):
        if path.exists():
            try:
                mesh = _load_mesh(path)
                if fit_to_bbox and bbox_extents is not None:
                    mesh = _fit_mesh_to_bbox_extents(mesh, bbox_extents, path)
                loguru.logger.info(
                    "Loaded object mesh {} (verts={}, faces={})",
                    path,
                    len(mesh.vertices),
                    len(mesh.faces),
                )
                return mesh
            except Exception as exc:
                loguru.logger.warning("Failed to load object mesh {}: {}", path, exc)
    loguru.logger.warning("No object mesh found under {}", workspace)
    return None


def _load_poses(pose_dir: Path) -> np.ndarray:
    pose_files = sorted(pose_dir.glob("*.txt"))
    if not pose_files:
        raise FileNotFoundError(f"no pose txt files found under {pose_dir}")
    poses = []
    for path in pose_files:
        pose = np.loadtxt(path, dtype=np.float64)
        if pose.shape != (4, 4):
            raise ValueError(f"{path} has shape {pose.shape}, expected (4, 4)")
        poses.append(pose)
    return np.stack(poses, axis=0)


def _resolve_workspace(config: ViewerConfig) -> Path:
    workspace = config.workspace or str(Path("preprocessed") / config.scene)
    return Path(workspace).resolve()


def _resolve_da3_path(workspace: Path) -> Path:
    candidates = [
        workspace / "da3.npz",
        workspace / "da3" / "da3.npz",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "required da3.npz missing. Tried: "
        + ", ".join(str(path) for path in candidates)
    )


def _resolve_object_ids(config: ViewerConfig) -> tuple[str, ...]:
    if config.object_ids:
        return config.object_ids
    return (config.object_id,)


def _resolve_pose_dir(workspace: Path, object_id: str, pose_dir_override: str | None) -> Path:
    if pose_dir_override is not None:
        pose_dir = Path(pose_dir_override)
        if not pose_dir.is_absolute():
            pose_dir = workspace / pose_dir
        if not pose_dir.exists():
            raise FileNotFoundError(f"object pose directory override does not exist: {pose_dir}")
        return pose_dir

    candidates = [
        workspace / "result" / object_id / "foundationpose_debug" / "center_pose",
        workspace / "result" / object_id / "poses",
    ]
    for pose_dir in candidates:
        if pose_dir.exists() and any(pose_dir.glob("*.txt")):
            return pose_dir
    raise FileNotFoundError(
        "no object pose txt files found. Tried: "
        + ", ".join(str(path) for path in candidates)
    )


def _object_world_poses(
    poses: np.ndarray,
    cam_c2w: np.ndarray,
    mode: str,
) -> np.ndarray:
    if mode == "camera_to_object":
        return np.einsum("nij,njk->nik", cam_c2w, poses)
    if mode == "world_to_object":
        return poses.copy()
    if mode == "object_to_camera":
        return np.einsum("nij,njk->nik", cam_c2w, np.linalg.inv(poses))
    if mode == "object_to_world":
        return np.linalg.inv(poses)
    raise ValueError(f"unsupported object pose mode: {mode}")


def _print_range(name: str, points: np.ndarray) -> None:
    points = points.reshape(-1, points.shape[-1])[:, :3]
    loguru.logger.info(
        "{} xyz min={} max={} mean={}",
        name,
        np.round(points.min(axis=0), 4).tolist(),
        np.round(points.max(axis=0), 4).tolist(),
        np.round(points.mean(axis=0), 4).tolist(),
    )


def _print_distance_stats(data: SceneData) -> None:
    tips = np.concatenate(
        [
            data.right_joints[:, [4, 8, 12, 16, 20], :],
            data.left_joints[:, [4, 8, 12, 16, 20], :],
        ],
        axis=1,
    )
    for obj in data.objects:
        dist = np.linalg.norm(tips - obj.centers[:, None, :], axis=2)
        closest = dist.min(axis=1)
        closest_idx = dist.argmin(axis=1)
        closest_z_delta = tips[np.arange(len(tips)), closest_idx, 2] - obj.centers[:, 2]
        loguru.logger.info(
            "{} closest fingertip-object distance: mean={:.4f} min={:.4f} max={:.4f} first={:.4f} mid={:.4f} last={:.4f}",
            obj.object_id,
            float(closest.mean()),
            float(closest.min()),
            float(closest.max()),
            float(closest[0]),
            float(closest[len(closest) // 2]),
            float(closest[-1]),
        )
        loguru.logger.info(
            "{} closest fingertip z - object z: mean={:.4f} min={:.4f} max={:.4f} first={:.4f} mid={:.4f} last={:.4f}",
            obj.object_id,
            float(closest_z_delta.mean()),
            float(closest_z_delta.min()),
            float(closest_z_delta.max()),
            float(closest_z_delta[0]),
            float(closest_z_delta[len(closest_z_delta) // 2]),
            float(closest_z_delta[-1]),
        )


def load_scene(config: ViewerConfig) -> SceneData:
    workspace = _resolve_workspace(config)
    da3_path = _resolve_da3_path(workspace)
    mocap_path = workspace / "hawor" / "world_mocap.npz"

    for path in [da3_path, mocap_path]:
        if not path.exists():
            raise FileNotFoundError(f"required input missing: {path}")
    object_ids = _resolve_object_ids(config)

    da3 = np.load(da3_path, allow_pickle=True)
    mocap = np.load(mocap_path, allow_pickle=True)

    cam_c2w = da3["cam_c2w"].astype(np.float64)
    right_joints = mocap["right_joints"].astype(np.float32)
    left_joints = mocap["left_joints"].astype(np.float32)
    right_vertices = mocap["right_verts"].astype(np.float32) if "right_verts" in mocap else None
    left_vertices = mocap["left_verts"].astype(np.float32) if "left_verts" in mocap else None

    object_pose_items: list[tuple[str, Path, np.ndarray, np.lib.npyio.NpzFile]] = []
    for object_id in object_ids:
        box_path = workspace / "result" / object_id / "box_for_spider.npz"
        if not box_path.exists():
            raise FileNotFoundError(f"required input missing: {box_path}")
        pose_dir = _resolve_pose_dir(
            workspace,
            object_id,
            config.pose_dir if len(object_ids) == 1 else None,
        )
        box = np.load(box_path, allow_pickle=True)
        poses = _load_poses(pose_dir)
        object_pose_items.append((object_id, pose_dir, poses, box))

    num_frames = min(
        [len(cam_c2w), len(right_joints), len(left_joints)]
        + [len(item[2]) for item in object_pose_items]
    )
    if right_vertices is not None:
        num_frames = min(num_frames, len(right_vertices))
    if left_vertices is not None:
        num_frames = min(num_frames, len(left_vertices))
    if num_frames <= 0:
        raise RuntimeError("no overlapping frames found")

    cam_c2w = cam_c2w[:num_frames]
    right_joints = right_joints[:num_frames]
    left_joints = left_joints[:num_frames]
    if right_vertices is not None:
        right_vertices = right_vertices[:num_frames, :: config.hand_vertex_stride, :]
    if left_vertices is not None:
        left_vertices = left_vertices[:num_frames, :: config.hand_vertex_stride, :]

    objects: list[ObjectData] = []
    for object_id, pose_dir, poses, box in object_pose_items:
        poses = poses[:num_frames]
        object_poses = _object_world_poses(poses, cam_c2w, config.object_pose_mode)
        object_centers = object_poses[:, :3, 3].astype(np.float32)
        object_quats = np.stack(
            [_rotmat_to_wxyz(pose[:3, :3]) for pose in object_poses],
            axis=0,
        )
        object_extents = box["box_real_size_xyz_m"].astype(np.float32)
        object_mesh = (
            _load_object_mesh(
                workspace,
                object_id,
                bbox_extents=object_extents,
                fit_to_bbox=config.fit_object_mesh_to_bbox,
            )
            if config.show_object_mesh
            else None
        )
        objects.append(
            ObjectData(
                object_id=object_id,
                pose_dir=pose_dir,
                centers=object_centers,
                quats_wxyz=object_quats.astype(np.float32),
                extents=object_extents,
                mesh=object_mesh,
            )
        )

    data = SceneData(
        num_frames=num_frames,
        cam_c2w=cam_c2w.astype(np.float32),
        right_joints=right_joints,
        left_joints=left_joints,
        right_vertices=right_vertices,
        left_vertices=left_vertices,
        objects=objects,
    )

    loguru.logger.info("scene={} workspace={}", config.scene, workspace)
    loguru.logger.info("da3_path={}", da3_path)
    for obj in objects:
        loguru.logger.info("{} pose_dir={}", obj.object_id, obj.pose_dir)
    loguru.logger.info(
        "frames: da3={} right_joints={} left_joints={} object_pose_counts={} viewer={}",
        len(da3["cam_c2w"]),
        len(mocap["right_joints"]),
        len(mocap["left_joints"]),
        {item[0]: len(item[2]) for item in object_pose_items},
        num_frames,
    )
    loguru.logger.info("object_pose_mode={}", config.object_pose_mode)
    _print_range("right_joints", data.right_joints)
    _print_range("left_joints", data.left_joints)
    for obj in data.objects:
        _print_range(f"{obj.object_id}_centers", obj.centers)
    _print_distance_stats(data)
    return data


def main(config: ViewerConfig) -> None:
    import viser  # type: ignore

    data = load_scene(config)
    server = viser.ViserServer(
        host=config.host,
        port=config.port,
        label=f"{config.scene} DA3 raw viewer",
        verbose=config.verbose,
    )
    server.scene.set_up_direction("+z")
    server.scene.add_grid(
        "/world/grid",
        section_color=(0.75, 0.75, 0.75),
        cell_color=(0.88, 0.88, 0.88),
    )

    look_at = np.mean([obj.centers[0] for obj in data.objects], axis=0).astype(np.float64)
    camera_pos = look_at + np.array([0.7, -1.0, 0.45], dtype=np.float64)

    @server.on_client_connect
    def _(client) -> None:
        client.camera.position = camera_pos
        client.camera.look_at = look_at

    right_joint_handle = server.scene.add_point_cloud(
        "/raw/right_joints",
        data.right_joints[0],
        RIGHT_HAND_COLOR,
        point_size=config.point_size_joints,
        point_shape="sparkle",
    )
    left_joint_handle = server.scene.add_point_cloud(
        "/raw/left_joints",
        data.left_joints[0],
        LEFT_HAND_COLOR,
        point_size=config.point_size_joints,
        point_shape="sparkle",
    )
    right_line_handle = server.scene.add_line_segments(
        "/raw/right_hand_skeleton",
        _hand_segments(data.right_joints[0]),
        _line_colors(len(MANO_HAND_EDGES), RIGHT_HAND_COLOR),
        line_width=2.5,
    )
    left_line_handle = server.scene.add_line_segments(
        "/raw/left_hand_skeleton",
        _hand_segments(data.left_joints[0]),
        _line_colors(len(MANO_HAND_EDGES), LEFT_HAND_COLOR),
        line_width=2.5,
    )

    right_vertices_handle = None
    left_vertices_handle = None
    if data.right_vertices is not None:
        right_vertices_handle = server.scene.add_point_cloud(
            "/raw/right_hand_vertices",
            data.right_vertices[0],
            RIGHT_VERT_COLOR,
            point_size=config.point_size_vertices,
            point_shape="circle",
            visible=config.show_hand_vertices,
        )
    if data.left_vertices is not None:
        left_vertices_handle = server.scene.add_point_cloud(
            "/raw/left_hand_vertices",
            data.left_vertices[0],
            LEFT_VERT_COLOR,
            point_size=config.point_size_vertices,
            point_shape="circle",
            visible=config.show_hand_vertices,
        )

    object_handles = []
    for obj_idx, obj in enumerate(data.objects):
        bbox_color = OBJECT_PALETTE[obj_idx % len(OBJECT_PALETTE)]
        center_color = OBJECT_CENTER_COLOR if obj_idx == 0 else bbox_color
        prefix = f"/raw/{obj.object_id}"
        corners = _obb_corners(
            obj.centers[0],
            _wxyz_to_rotmat(obj.quats_wxyz[0]),
            obj.extents,
        )
        center_handle = server.scene.add_point_cloud(
            f"{prefix}/center",
            obj.centers[0][None, :],
            center_color,
            point_size=config.point_size_object,
            point_shape="rounded",
        )
        corners_handle = server.scene.add_point_cloud(
            f"{prefix}/bbox_corners",
            corners,
            bbox_color,
            point_size=config.point_size_joints,
            point_shape="diamond",
        )
        bbox_handle = server.scene.add_line_segments(
            f"{prefix}/bbox_edges",
            _obb_segments(corners),
            _line_colors(len(BBOX_EDGES), bbox_color),
            line_width=2.0,
        )
        frame_handle = server.scene.add_frame(
            f"{prefix}/frame",
            axes_length=config.object_axes_length,
            axes_radius=0.004,
            origin_radius=0.01,
            position=obj.centers[0],
            wxyz=obj.quats_wxyz[0],
        )
        mesh_handle = None
        if obj.mesh is not None:
            mesh_handle = server.scene.add_mesh_trimesh(
                f"{prefix}/mesh",
                obj.mesh,
                position=obj.centers[0],
                wxyz=obj.quats_wxyz[0],
                visible=config.show_object_mesh,
            )
        object_handles.append(
            {
                "object": obj,
                "center": center_handle,
                "corners": corners_handle,
                "bbox": bbox_handle,
                "frame": frame_handle,
                "mesh": mesh_handle,
            }
        )

    camera_frame_handle = server.scene.add_frame(
        "/da3/current_camera",
        axes_length=0.08,
        axes_radius=0.002,
        origin_radius=0.006,
        position=data.cam_c2w[0, :3, 3],
        wxyz=_rotmat_to_wxyz(data.cam_c2w[0, :3, :3]),
    )
    camera_lines_handle = server.scene.add_line_segments(
        "/da3/current_camera_axes_lines",
        _camera_segments(data.cam_c2w[0]),
        np.asarray(
            [
                [[255, 60, 60], [255, 60, 60]],
                [[60, 220, 60], [60, 220, 60]],
                [[60, 120, 255], [60, 120, 255]],
            ],
            dtype=np.uint8,
        ),
        line_width=2.0,
    )
    camera_path = data.cam_c2w[:, :3, 3]
    if len(camera_path) > 1:
        server.scene.add_line_segments(
            "/da3/camera_path",
            np.stack([camera_path[:-1], camera_path[1:]], axis=1),
            _line_colors(len(camera_path) - 1, CAMERA_COLOR),
            line_width=1.5,
        )

    display_folder = server.gui.add_folder("Display")
    with display_folder:
        show_joints = server.gui.add_checkbox("Hand Joints", initial_value=True)
        show_vertices = server.gui.add_checkbox(
            "Hand Vertices",
            initial_value=config.show_hand_vertices,
            disabled=right_vertices_handle is None and left_vertices_handle is None,
        )
        show_object = server.gui.add_checkbox("Object Center/BBox", initial_value=True)
        has_object_mesh = any(item["mesh"] is not None for item in object_handles)
        show_mesh = server.gui.add_checkbox(
            "Object Mesh",
            initial_value=config.show_object_mesh and has_object_mesh,
            disabled=not has_object_mesh,
        )
        show_camera = server.gui.add_checkbox("DA3 Camera", initial_value=True)

    timeline_folder = server.gui.add_folder("Timeline")
    with timeline_folder:
        frame_slider = server.gui.add_slider(
            "Frame",
            min=0,
            max=max(0, data.num_frames - 1),
            step=1,
            initial_value=0,
        )
        fps_slider = server.gui.add_slider(
            "FPS",
            min=1,
            max=60,
            step=1,
            initial_value=config.fps,
        )
        loop_checkbox = server.gui.add_checkbox("Loop", initial_value=True)
        btn_rev = server.gui.add_button("Play Backward")
        btn_pause = server.gui.add_button("Pause")
        btn_fwd = server.gui.add_button("Play Forward")

    playback_speed = {"value": 0}

    def apply_visibility() -> None:
        right_joint_handle.visible = show_joints.value
        left_joint_handle.visible = show_joints.value
        right_line_handle.visible = show_joints.value
        left_line_handle.visible = show_joints.value
        if right_vertices_handle is not None:
            right_vertices_handle.visible = show_vertices.value
        if left_vertices_handle is not None:
            left_vertices_handle.visible = show_vertices.value
        for item in object_handles:
            item["center"].visible = show_object.value
            item["corners"].visible = show_object.value
            item["bbox"].visible = show_object.value
            item["frame"].visible = show_object.value
            if item["mesh"] is not None:
                item["mesh"].visible = show_object.value and show_mesh.value
        camera_frame_handle.visible = show_camera.value
        camera_lines_handle.visible = show_camera.value

    def render_frame(frame_idx: int) -> None:
        idx = max(0, min(int(frame_idx), data.num_frames - 1))
        right_joint_handle.points = data.right_joints[idx]
        left_joint_handle.points = data.left_joints[idx]
        right_line_handle.points = _hand_segments(data.right_joints[idx])
        left_line_handle.points = _hand_segments(data.left_joints[idx])
        if right_vertices_handle is not None and data.right_vertices is not None:
            right_vertices_handle.points = data.right_vertices[idx]
        if left_vertices_handle is not None and data.left_vertices is not None:
            left_vertices_handle.points = data.left_vertices[idx]

        for item in object_handles:
            obj = item["object"]
            center = obj.centers[idx]
            quat = obj.quats_wxyz[idx]
            rot = _wxyz_to_rotmat(quat)
            corners_frame = _obb_corners(center, rot, obj.extents)
            item["center"].points = center[None, :]
            item["corners"].points = corners_frame
            item["bbox"].points = _obb_segments(corners_frame)
            item["frame"].position = tuple(center.tolist())
            item["frame"].wxyz = tuple(quat.tolist())
            if item["mesh"] is not None:
                item["mesh"].position = tuple(center.tolist())
                item["mesh"].wxyz = tuple(quat.tolist())

        camera_frame_handle.position = tuple(data.cam_c2w[idx, :3, 3].tolist())
        camera_frame_handle.wxyz = tuple(_rotmat_to_wxyz(data.cam_c2w[idx, :3, :3]).tolist())
        camera_lines_handle.points = _camera_segments(data.cam_c2w[idx])
        apply_visibility()
        try:
            server.flush()
        except Exception:
            pass

    for widget in [show_joints, show_vertices, show_object, show_mesh, show_camera]:
        @widget.on_update
        def _(_) -> None:
            apply_visibility()

    @frame_slider.on_update
    def _(_) -> None:
        render_frame(int(frame_slider.value))

    @btn_rev.on_click
    def _(_) -> None:
        playback_speed["value"] = -1

    @btn_pause.on_click
    def _(_) -> None:
        playback_speed["value"] = 0

    @btn_fwd.on_click
    def _(_) -> None:
        playback_speed["value"] = 1

    def playback_loop() -> None:
        while True:
            speed = playback_speed["value"]
            if speed != 0:
                next_frame = int(frame_slider.value) + speed
                if next_frame > int(frame_slider.max):
                    next_frame = 0 if loop_checkbox.value else int(frame_slider.max)
                elif next_frame < 0:
                    next_frame = int(frame_slider.max) if loop_checkbox.value else 0
                frame_slider.value = next_frame
            time.sleep(1.0 / max(1.0, float(fps_slider.value)))

    threading.Thread(target=playback_loop, daemon=True).start()
    render_frame(0)
    loguru.logger.info(
        "Viser viewer running at http://{}:{}/ (frames={}, DA3 world coordinates)",
        config.host,
        config.port,
        data.num_frames,
    )
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main(parse_args())
