#!/usr/bin/env python3
"""Browser-based 3D viewer for raw/processed ourdata milk keypoints.

This is a headless debug utility intended for server usage. It launches a
Viser server and visualizes:

- raw hand vertices from `preprocessed/milk/hawor/world_mocap.npz`
- raw hand keypoints derived with the same logic used by `ourdata.py`
- raw object pose trajectory derived from FoundationPose `center_pose`
- raw object OBB / object mesh from `box_for_spider.npz` + `obj_3d_final.ply`
- optional processed overlays from `example_datasets/processed/ourdata/...`

It is meant to answer a single question quickly:
are the hands and object in the same 3D coordinate frame before IK?
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import loguru
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as R

from spider.io import get_processed_data_dir
from spider.process_datasets import ourdata as ourdata_utils

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

RIGHT_HAND_COLOR = np.array([255, 80, 80], dtype=np.uint8)
RIGHT_HAND_VERT_COLOR = np.array([255, 180, 180], dtype=np.uint8)
LEFT_HAND_COLOR = np.array([90, 140, 255], dtype=np.uint8)
LEFT_HAND_VERT_COLOR = np.array([180, 210, 255], dtype=np.uint8)
RAW_OBJECT_CENTER_COLOR = np.array([0, 220, 120], dtype=np.uint8)
RAW_OBJECT_CORNER_COLOR = np.array([255, 220, 0], dtype=np.uint8)
PROC_RIGHT_HAND_COLOR = np.array([255, 0, 255], dtype=np.uint8)
PROC_LEFT_HAND_COLOR = np.array([0, 220, 220], dtype=np.uint8)
PROC_OBJECT_CENTER_COLOR = np.array([255, 160, 0], dtype=np.uint8)
PROC_OBJECT_CORNER_COLOR = np.array([255, 240, 160], dtype=np.uint8)


@dataclass
class ViewerConfig:
    # ---------------------------------------------------------------------
    # 输入数据相关参数
    # ---------------------------------------------------------------------
    # `workspace` 指向你自己的原始工作区，也就是 preprocessed/milk 这一层。
    # 这里包含 raw 的 hand/object/camera 数据，是本脚本最主要的输入。
    workspace: str = "preprocessed/milk"

    # `dataset_dir / dataset_name / task / data_id / embodiment_type`
    # 只在“叠加 processed 结果”时使用，用于定位
    # example_datasets/processed/ourdata/.../trajectory_keypoints.npz
    # 这套参数和 `ourdata.py` / `run_ik.py` 保持一致。
    dataset_dir: str = "example_datasets"
    dataset_name: str = "ourdata"
    task: str = "milk"
    data_id: int = 0
    embodiment_type: str = "bimanual"
    object_id: str = "obj_0"

    # raw 物体姿态显示策略。
    # - preserve_box: 直接使用 box_for_spider.npz 中的原始 box_rotation_R
    # - upright_preserve_heading: 强制物体直立，但尽量保留水平朝向
    orientation_policy: str = "preserve_box"

    # ---------------------------------------------------------------------
    # 是否叠加 processed 结果
    # ---------------------------------------------------------------------
    # True: 额外读取 example_datasets/processed/.../trajectory_keypoints.npz
    # False: 只显示 raw workspace 数据
    show_processed_overlay: bool = True

    # ---------------------------------------------------------------------
    # 可视化显示相关参数
    # ---------------------------------------------------------------------
    # 是否显示 raw 的整手顶点云。
    # 这个点很多，但对检查 hand mesh 的整体位置最直接。
    show_raw_hand_vertices: bool = True

    # 是否显示 raw 物体 mesh。
    # 如果只关心中心和 bbox，可以关掉它。
    show_raw_object_mesh: bool = True

    # 是否显示 processed 物体 mesh。
    # 默认关闭，避免 raw/processed 两层 mesh 叠在一起太乱。
    show_processed_object_mesh: bool = False

    # ---------------------------------------------------------------------
    # 服务器与显示尺度参数
    # ---------------------------------------------------------------------
    # Viser 服务监听地址与端口。服务器上一般用 0.0.0.0，然后本地做 SSH 端口转发。
    host: str = "0.0.0.0"
    port: int = 8080

    # 点云/关键点/物体中心的显示半径（不是物理尺寸，只是渲染大小）。
    point_size_vertices: float = 0.004
    point_size_keypoints: float = 0.02
    point_size_object: float = 0.025

    # 物体坐标轴显示长度，便于检查姿态方向。
    object_axes_length: float = 0.12

    # 是否输出 Viser 自身的启动日志。
    verbose: bool = True


@dataclass
class RawSceneData:
    num_frames: int
    right_vertices: np.ndarray
    left_vertices: np.ndarray
    right_keypoints: np.ndarray
    left_keypoints: np.ndarray
    object_centers: np.ndarray
    object_quats_wxyz: np.ndarray
    object_extents: np.ndarray
    object_mesh: trimesh.Trimesh


@dataclass
class ProcessedSceneData:
    num_frames: int
    right_keypoints: np.ndarray
    left_keypoints: np.ndarray
    object_centers: np.ndarray
    object_quats_wxyz: np.ndarray
    object_extents: np.ndarray
    object_mesh: trimesh.Trimesh | None
    task_info: dict


def _parse_bool(value: str) -> bool:
    """Parse a human-friendly bool string for argparse.

    Accepts: true/false, 1/0, yes/no, y/n, on/off.
    """
    lowered = value.strip().lower()
    if lowered in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(
        f"invalid boolean value: {value!r} (expected true/false)"
    )


def build_argparser() -> argparse.ArgumentParser:
    """Build a standard argparse parser.

    参数按“输入路径 / processed 叠加 / 可视化显示 / 服务器配置”分组，
    方便直接 `--help` 查看。
    """
    parser = argparse.ArgumentParser(
        description=(
            "启动一个基于 Viser 的 3D 调试页面，用于查看 "
            "preprocessed/milk 中 raw 手关键点、手顶点云、物体中心、bbox、mesh，"
            "并可选叠加 processed 结果进行对比。"
        )
    )

    # ------------------------------------------------------------------
    # 输入路径参数
    # ------------------------------------------------------------------
    io_group = parser.add_argument_group(
        "输入路径参数",
        "这些参数决定读取哪一份 raw workspace，以及 processed overlay 的定位方式。",
    )
    io_group.add_argument(
        "--workspace",
        default=ViewerConfig.workspace,
        help=(
            "原始工作区目录。默认是 preprocessed/milk。"
            "脚本会从这里读取 world_mocap.npz、da3.npz、"
            "box_for_spider.npz、obj_3d_final.ply、FoundationPose center_pose。"
        ),
    )
    io_group.add_argument(
        "--dataset-dir",
        default=ViewerConfig.dataset_dir,
        help=(
            "processed 数据根目录。只有在 show_processed_overlay=true 时才使用。"
        ),
    )
    io_group.add_argument(
        "--dataset-name",
        default=ViewerConfig.dataset_name,
        help="processed 数据集名称，默认 ourdata。",
    )
    io_group.add_argument(
        "--task",
        default=ViewerConfig.task,
        help="任务名，默认 milk。",
    )
    io_group.add_argument(
        "--data-id",
        type=int,
        default=ViewerConfig.data_id,
        help="样本编号，默认 0。",
    )
    io_group.add_argument(
        "--embodiment-type",
        default=ViewerConfig.embodiment_type,
        choices=["right", "left", "bimanual"],
        help="手型配置。当前 milk 默认 bimanual。",
    )
    io_group.add_argument(
        "--object-id",
        default=ViewerConfig.object_id,
        help="原始 workspace 中的物体目录名，默认 obj_0。",
    )
    io_group.add_argument(
        "--orientation-policy",
        default=ViewerConfig.orientation_policy,
        choices=["preserve_box", "upright_preserve_heading"],
        help=(
            "raw 物体姿态的显示策略。"
            "preserve_box 使用 box_for_spider 原始姿态；"
            "upright_preserve_heading 会强制直立显示。"
        ),
    )

    # ------------------------------------------------------------------
    # processed 叠加参数
    # ------------------------------------------------------------------
    overlay_group = parser.add_argument_group(
        "processed 叠加参数",
        "控制是否额外读取 example_datasets/processed/... 的转换结果做对照。",
    )
    overlay_group.add_argument(
        "--show-processed-overlay",
        type=_parse_bool,
        default=ViewerConfig.show_processed_overlay,
        help=(
            "是否叠加 processed 的 hand/object 结果。"
            "例如 true / false。默认 true。"
        ),
    )

    # ------------------------------------------------------------------
    # 可视化显示参数
    # ------------------------------------------------------------------
    display_group = parser.add_argument_group(
        "可视化显示参数",
        "控制哪些几何显示出来，以及点/坐标轴的显示大小。",
    )
    display_group.add_argument(
        "--show-raw-hand-vertices",
        type=_parse_bool,
        default=ViewerConfig.show_raw_hand_vertices,
        help="是否显示 raw 的整手顶点云。默认 true。",
    )
    display_group.add_argument(
        "--show-raw-object-mesh",
        type=_parse_bool,
        default=ViewerConfig.show_raw_object_mesh,
        help="是否显示 raw 的物体 mesh。默认 true。",
    )
    display_group.add_argument(
        "--show-processed-object-mesh",
        type=_parse_bool,
        default=ViewerConfig.show_processed_object_mesh,
        help="是否显示 processed 的物体 mesh。默认 false。",
    )
    display_group.add_argument(
        "--point-size-vertices",
        type=float,
        default=ViewerConfig.point_size_vertices,
        help="raw 手顶点云的显示大小。",
    )
    display_group.add_argument(
        "--point-size-keypoints",
        type=float,
        default=ViewerConfig.point_size_keypoints,
        help="手关键点 / 物体 bbox 角点的显示大小。",
    )
    display_group.add_argument(
        "--point-size-object",
        type=float,
        default=ViewerConfig.point_size_object,
        help="物体中心点的显示大小。",
    )
    display_group.add_argument(
        "--object-axes-length",
        type=float,
        default=ViewerConfig.object_axes_length,
        help="物体局部坐标轴的显示长度。",
    )

    # ------------------------------------------------------------------
    # 服务器参数
    # ------------------------------------------------------------------
    server_group = parser.add_argument_group(
        "服务器参数",
        "控制 Viser 服务监听地址、端口和日志输出。",
    )
    server_group.add_argument(
        "--host",
        default=ViewerConfig.host,
        help=(
            "Viser 监听地址。服务器环境通常使用 0.0.0.0，"
            "然后通过 SSH 端口转发在本地浏览器查看。"
        ),
    )
    server_group.add_argument(
        "--port",
        type=int,
        default=ViewerConfig.port,
        help="Viser 监听端口，默认 8080。",
    )
    server_group.add_argument(
        "--verbose",
        type=_parse_bool,
        default=ViewerConfig.verbose,
        help="是否打印 Viser 启动日志。默认 true。",
    )

    return parser


def parse_args() -> ViewerConfig:
    """Parse CLI args and convert them into a strongly-typed config."""
    parser = build_argparser()
    args = parser.parse_args()
    return ViewerConfig(
        workspace=args.workspace,
        dataset_dir=args.dataset_dir,
        dataset_name=args.dataset_name,
        task=args.task,
        data_id=args.data_id,
        embodiment_type=args.embodiment_type,
        object_id=args.object_id,
        orientation_policy=args.orientation_policy,
        show_processed_overlay=args.show_processed_overlay,
        show_raw_hand_vertices=args.show_raw_hand_vertices,
        show_raw_object_mesh=args.show_raw_object_mesh,
        show_processed_object_mesh=args.show_processed_object_mesh,
        host=args.host,
        port=args.port,
        point_size_vertices=args.point_size_vertices,
        point_size_keypoints=args.point_size_keypoints,
        point_size_object=args.point_size_object,
        object_axes_length=args.object_axes_length,
        verbose=args.verbose,
    )


def _wxyz_to_rotmat(quat_wxyz: np.ndarray) -> np.ndarray:
    quat_wxyz = np.asarray(quat_wxyz, dtype=np.float64)
    quat_wxyz = quat_wxyz / max(float(np.linalg.norm(quat_wxyz)), 1e-12)
    quat_xyzw = np.array(
        [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]],
        dtype=np.float64,
    )
    return R.from_quat(quat_xyzw).as_matrix()


def _obb_corners(
    center: np.ndarray,
    rotmat: np.ndarray,
    extents: np.ndarray,
) -> np.ndarray:
    center = np.asarray(center, dtype=np.float32)
    rotmat = np.asarray(rotmat, dtype=np.float32)
    half = 0.5 * np.asarray(extents, dtype=np.float32)
    corners: list[np.ndarray] = []
    for s0 in (-1.0, 1.0):
        for s1 in (-1.0, 1.0):
            for s2 in (-1.0, 1.0):
                corners.append(
                    center
                    + s0 * rotmat[:, 0] * half[0]
                    + s1 * rotmat[:, 1] * half[1]
                    + s2 * rotmat[:, 2] * half[2]
                )
    return np.asarray(corners, dtype=np.float32)


def _obb_segments(corners: np.ndarray) -> np.ndarray:
    return np.asarray(
        [[corners[a], corners[b]] for a, b in BBOX_EDGES],
        dtype=np.float32,
    )


def _hand_keypoint_segments(keypoints: np.ndarray) -> np.ndarray:
    wrist = keypoints[0]
    fingertips = keypoints[1:]
    return np.asarray(
        [[wrist, fingertip] for fingertip in fingertips],
        dtype=np.float32,
    )


def _make_line_colors(num_segments: int, color: np.ndarray) -> np.ndarray:
    color = np.asarray(color, dtype=np.uint8)
    return np.tile(color[None, None, :], (num_segments, 2, 1))


def _as_keypoints(qpos_wrist: np.ndarray, qpos_finger: np.ndarray) -> np.ndarray:
    return np.concatenate([qpos_wrist[:, None, :3], qpos_finger[:, :, :3]], axis=1)


def _set_mesh_color(
    mesh: trimesh.Trimesh,
    rgba: tuple[int, int, int, int],
) -> trimesh.Trimesh:
    out = mesh.copy()
    rgba_arr = np.asarray(rgba, dtype=np.uint8)
    if len(out.faces) > 0:
        out.visual.face_colors = np.tile(rgba_arr[None, :], (len(out.faces), 1))
    if len(out.vertices) > 0:
        out.visual.vertex_colors = np.tile(rgba_arr[None, :], (len(out.vertices), 1))
    return out


def _load_mesh_any(path: Path) -> trimesh.Trimesh:
    geometry = trimesh.load(path, process=False)
    if isinstance(geometry, trimesh.Scene):
        geometry = geometry.to_mesh()
    if isinstance(geometry, trimesh.points.PointCloud):
        if len(geometry.vertices) == 0:
            raise RuntimeError(f"point cloud at {path} is empty")
        geometry = geometry.convex_hull
    if not isinstance(geometry, trimesh.Trimesh):
        raise TypeError(f"unsupported geometry type from {path}: {type(geometry)}")
    if len(geometry.faces) == 0:
        geometry = geometry.convex_hull
    if geometry.is_empty:
        raise RuntimeError(f"failed to build a mesh from {path}")
    return geometry


def load_raw_scene(config: ViewerConfig) -> RawSceneData:
    workspace = Path(config.workspace).resolve()
    inputs = ourdata_utils._resolve_workspace_inputs(
        workspace_path=workspace,
        object_id=config.object_id,
    )

    mocap_data = np.load(inputs.world_mocap_path)
    camera_bundle = ourdata_utils._load_camera_bundle(inputs.camera_npz_path)
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
        raise RuntimeError("no overlapping frames found across raw workspace inputs")

    right_verts = right_verts[:num_frames]
    left_verts = left_verts[:num_frames]
    if depths is not None:
        depths = depths[:num_frames]
    cam_c2w = cam_c2w[:num_frames]

    qpos_wrist_right, qpos_finger_right = ourdata_utils._estimate_hand_trajectory(
        right_verts, side="right"
    )
    qpos_wrist_left, qpos_finger_left = ourdata_utils._estimate_hand_trajectory(
        left_verts, side="left"
    )

    initial_center_world = box_data["box_center_world"].astype(np.float64)
    if inputs.object_pose_dir is not None:
        object_centers_world, object_rotations_world, _ = (
            ourdata_utils._load_foundationpose_object_trajectory(
                pose_dir=inputs.object_pose_dir,
                cam_c2w=cam_c2w,
                num_frames=num_frames,
            )
        )
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
        object_centers_world, _ = ourdata_utils._estimate_object_trajectory(
            masks_dir=inputs.masks_dir,
            depths=depths,
            intrinsic=intrinsic,
            cam_c2w=cam_c2w,
            initial_center_world=initial_center_world,
            num_frames=num_frames,
        )
        object_rotation = ourdata_utils._resolve_object_rotation(
            box_rotation=box_data["box_rotation_R"].astype(np.float64),
            orientation_policy=config.orientation_policy,
        )
        object_rotations_world = np.tile(object_rotation[None, :, :], (num_frames, 1, 1))
    object_quats_wxyz = np.stack(
        [
            ourdata_utils._rotation_matrix_to_wxyz(object_rotation)
            for object_rotation in object_rotations_world
        ],
        axis=0,
    )
    object_extents = box_data["box_real_size_xyz_m"].astype(np.float32)

    object_scale_factor = float(box_data["scale_factor"])
    object_mesh, _ = ourdata_utils._prepare_object_mesh_for_bbox_frame(
        ourdata_utils._load_object_mesh(inputs.object_ply_path),
        scale_factor=object_scale_factor,
        bbox_size_xyz_m=object_extents,
        align_axes_to_bbox=True,
    )

    return RawSceneData(
        num_frames=num_frames,
        right_vertices=right_verts,
        left_vertices=left_verts,
        right_keypoints=_as_keypoints(qpos_wrist_right, qpos_finger_right),
        left_keypoints=_as_keypoints(qpos_wrist_left, qpos_finger_left),
        object_centers=object_centers_world.astype(np.float32),
        object_quats_wxyz=object_quats_wxyz.astype(np.float32),
        object_extents=object_extents,
        object_mesh=object_mesh,
    )


def load_processed_scene(config: ViewerConfig, raw: RawSceneData) -> ProcessedSceneData | None:
    dataset_dir = Path(config.dataset_dir).resolve()
    processed_dir = Path(
        get_processed_data_dir(
            dataset_dir=str(dataset_dir),
            dataset_name=config.dataset_name,
            robot_type="mano",
            embodiment_type=config.embodiment_type,
            task=config.task,
            data_id=config.data_id,
        )
    )
    traj_path = processed_dir / "trajectory_keypoints.npz"
    task_info_path = processed_dir.parent / "task_info.json"
    if not traj_path.exists() or not task_info_path.exists():
        loguru.logger.warning(
            "Processed overlay not found at {} / {}. Raw workspace only.",
            traj_path,
            task_info_path,
        )
        return None

    traj = np.load(traj_path)
    with task_info_path.open("r", encoding="utf-8") as f:
        task_info = json.load(f)

    qpos_wrist_right = traj["qpos_wrist_right"].astype(np.float32)
    qpos_finger_right = traj["qpos_finger_right"].astype(np.float32)
    qpos_wrist_left = traj["qpos_wrist_left"].astype(np.float32)
    qpos_finger_left = traj["qpos_finger_left"].astype(np.float32)
    qpos_obj_right = traj["qpos_obj_right"].astype(np.float32)

    object_mesh = None
    right_object_mesh_dir = task_info.get("right_object_mesh_dir")
    if right_object_mesh_dir:
        mesh_path = dataset_dir / right_object_mesh_dir / "visual.obj"
        if mesh_path.exists():
            object_mesh = _load_mesh_any(mesh_path)
        else:
            loguru.logger.warning(
                "Processed object mesh not found at {}. Falling back to raw mesh.",
                mesh_path,
            )
    if object_mesh is None:
        object_mesh = raw.object_mesh.copy()

    num_frames = min(
        len(qpos_wrist_right),
        len(qpos_finger_right),
        len(qpos_wrist_left),
        len(qpos_finger_left),
        len(qpos_obj_right),
    )
    return ProcessedSceneData(
        num_frames=num_frames,
        right_keypoints=_as_keypoints(
            qpos_wrist_right[:num_frames],
            qpos_finger_right[:num_frames],
        ),
        left_keypoints=_as_keypoints(
            qpos_wrist_left[:num_frames],
            qpos_finger_left[:num_frames],
        ),
        object_centers=qpos_obj_right[:num_frames, :3].astype(np.float32),
        object_quats_wxyz=qpos_obj_right[:num_frames, 3:].astype(np.float32),
        object_extents=raw.object_extents.copy(),
        object_mesh=object_mesh,
        task_info=task_info,
    )


def main(config: ViewerConfig) -> None:
    import viser  # type: ignore

    raw = load_raw_scene(config)
    processed = load_processed_scene(config, raw) if config.show_processed_overlay else None
    num_frames = raw.num_frames if processed is None else min(raw.num_frames, processed.num_frames)

    raw_mesh = _set_mesh_color(raw.object_mesh, (40, 220, 120, 120))
    processed_mesh = None
    if processed is not None and processed.object_mesh is not None:
        processed_mesh = _set_mesh_color(processed.object_mesh, (255, 160, 0, 120))

    server = viser.ViserServer(
        host=config.host,
        port=config.port,
        label="ourdata milk keypoints",
        verbose=config.verbose,
    )
    server.scene.set_up_direction("+z")
    server.scene.add_grid(
        "/world/grid",
        section_color=(0.75, 0.75, 0.75),
        cell_color=(0.88, 0.88, 0.88),
    )

    initial_look_at = raw.object_centers[0].astype(np.float64)
    initial_camera_position = initial_look_at + np.array([0.8, -1.1, 0.6], dtype=np.float64)

    @server.on_client_connect
    def _(client) -> None:
        client.camera.position = initial_camera_position
        client.camera.look_at = initial_look_at

    raw_right_vertices_handle = server.scene.add_point_cloud(
        "/raw/right_hand_vertices",
        raw.right_vertices[0],
        RIGHT_HAND_VERT_COLOR,
        point_size=config.point_size_vertices,
        point_shape="circle",
        visible=config.show_raw_hand_vertices,
    )
    raw_left_vertices_handle = server.scene.add_point_cloud(
        "/raw/left_hand_vertices",
        raw.left_vertices[0],
        LEFT_HAND_VERT_COLOR,
        point_size=config.point_size_vertices,
        point_shape="circle",
        visible=config.show_raw_hand_vertices,
    )
    raw_right_keypoints_handle = server.scene.add_point_cloud(
        "/raw/right_hand_keypoints",
        raw.right_keypoints[0],
        RIGHT_HAND_COLOR,
        point_size=config.point_size_keypoints,
        point_shape="sparkle",
        visible=True,
    )
    raw_left_keypoints_handle = server.scene.add_point_cloud(
        "/raw/left_hand_keypoints",
        raw.left_keypoints[0],
        LEFT_HAND_COLOR,
        point_size=config.point_size_keypoints,
        point_shape="sparkle",
        visible=True,
    )
    raw_right_segments = _hand_keypoint_segments(raw.right_keypoints[0])
    raw_left_segments = _hand_keypoint_segments(raw.left_keypoints[0])
    raw_right_lines_handle = server.scene.add_line_segments(
        "/raw/right_hand_links",
        raw_right_segments,
        _make_line_colors(len(raw_right_segments), RIGHT_HAND_COLOR),
        line_width=2.5,
        visible=True,
    )
    raw_left_lines_handle = server.scene.add_line_segments(
        "/raw/left_hand_links",
        raw_left_segments,
        _make_line_colors(len(raw_left_segments), LEFT_HAND_COLOR),
        line_width=2.5,
        visible=True,
    )

    raw_corners = _obb_corners(
        raw.object_centers[0],
        _wxyz_to_rotmat(raw.object_quats_wxyz[0]),
        raw.object_extents,
    )
    raw_bbox_segments = _obb_segments(raw_corners)
    raw_object_center_handle = server.scene.add_point_cloud(
        "/raw/object_center",
        raw.object_centers[0][None, :],
        RAW_OBJECT_CENTER_COLOR,
        point_size=config.point_size_object,
        point_shape="rounded",
        visible=True,
    )
    raw_object_corners_handle = server.scene.add_point_cloud(
        "/raw/object_bbox_corners",
        raw_corners,
        RAW_OBJECT_CORNER_COLOR,
        point_size=config.point_size_keypoints,
        point_shape="diamond",
        visible=True,
    )
    raw_object_bbox_handle = server.scene.add_line_segments(
        "/raw/object_bbox_edges",
        raw_bbox_segments,
        _make_line_colors(len(raw_bbox_segments), RAW_OBJECT_CORNER_COLOR),
        line_width=2.0,
        visible=True,
    )
    raw_object_frame_handle = server.scene.add_frame(
        "/raw/object_frame",
        axes_length=config.object_axes_length,
        axes_radius=0.004,
        origin_radius=0.01,
        position=raw.object_centers[0],
        wxyz=raw.object_quats_wxyz[0],
        visible=True,
    )
    raw_object_mesh_handle = server.scene.add_mesh_trimesh(
        "/raw/object_mesh",
        raw_mesh,
        position=raw.object_centers[0],
        wxyz=raw.object_quats_wxyz[0],
        visible=config.show_raw_object_mesh,
    )

    processed_handles: dict[str, object] = {}
    if processed is not None:
        proc_right = processed.right_keypoints[0]
        proc_left = processed.left_keypoints[0]
        proc_quat = processed.object_quats_wxyz[0]
        proc_corners = _obb_corners(
            processed.object_centers[0],
            _wxyz_to_rotmat(proc_quat),
            processed.object_extents,
        )
        proc_bbox_segments = _obb_segments(proc_corners)
        processed_handles["right_keypoints"] = server.scene.add_point_cloud(
            "/processed/right_hand_keypoints",
            proc_right,
            PROC_RIGHT_HAND_COLOR,
            point_size=config.point_size_keypoints,
            point_shape="sparkle",
            visible=True,
        )
        processed_handles["left_keypoints"] = server.scene.add_point_cloud(
            "/processed/left_hand_keypoints",
            proc_left,
            PROC_LEFT_HAND_COLOR,
            point_size=config.point_size_keypoints,
            point_shape="sparkle",
            visible=True,
        )
        proc_right_segments = _hand_keypoint_segments(proc_right)
        proc_left_segments = _hand_keypoint_segments(proc_left)
        processed_handles["right_lines"] = server.scene.add_line_segments(
            "/processed/right_hand_links",
            proc_right_segments,
            _make_line_colors(len(proc_right_segments), PROC_RIGHT_HAND_COLOR),
            line_width=2.0,
            visible=True,
        )
        processed_handles["left_lines"] = server.scene.add_line_segments(
            "/processed/left_hand_links",
            proc_left_segments,
            _make_line_colors(len(proc_left_segments), PROC_LEFT_HAND_COLOR),
            line_width=2.0,
            visible=True,
        )
        processed_handles["object_center"] = server.scene.add_point_cloud(
            "/processed/object_center",
            processed.object_centers[0][None, :],
            PROC_OBJECT_CENTER_COLOR,
            point_size=config.point_size_object,
            point_shape="rounded",
            visible=True,
        )
        processed_handles["object_corners"] = server.scene.add_point_cloud(
            "/processed/object_bbox_corners",
            proc_corners,
            PROC_OBJECT_CORNER_COLOR,
            point_size=config.point_size_keypoints,
            point_shape="diamond",
            visible=True,
        )
        processed_handles["object_bbox"] = server.scene.add_line_segments(
            "/processed/object_bbox_edges",
            proc_bbox_segments,
            _make_line_colors(len(proc_bbox_segments), PROC_OBJECT_CORNER_COLOR),
            line_width=1.8,
            visible=True,
        )
        processed_handles["object_frame"] = server.scene.add_frame(
            "/processed/object_frame",
            axes_length=config.object_axes_length * 0.9,
            axes_radius=0.0035,
            origin_radius=0.008,
            position=processed.object_centers[0],
            wxyz=proc_quat,
            visible=True,
        )
        if processed_mesh is not None:
            processed_handles["object_mesh"] = server.scene.add_mesh_trimesh(
                "/processed/object_mesh",
                processed_mesh,
                position=processed.object_centers[0],
                wxyz=proc_quat,
                visible=config.show_processed_object_mesh,
            )

    display_folder = server.gui.add_folder("Display")
    with display_folder:
        show_raw_vertices = server.gui.add_checkbox(
            "Raw Hand Vertices",
            initial_value=config.show_raw_hand_vertices,
        )
        show_raw_keypoints = server.gui.add_checkbox(
            "Raw Hand Keypoints",
            initial_value=True,
        )
        show_raw_object = server.gui.add_checkbox(
            "Raw Object BBox/Center",
            initial_value=True,
        )
        show_raw_object_mesh = server.gui.add_checkbox(
            "Raw Object Mesh",
            initial_value=config.show_raw_object_mesh,
        )
        show_processed = server.gui.add_checkbox(
            "Processed Overlay",
            initial_value=processed is not None,
            disabled=processed is None,
        )
        show_processed_mesh = server.gui.add_checkbox(
            "Processed Object Mesh",
            initial_value=processed is not None and config.show_processed_object_mesh,
            disabled=processed is None or "object_mesh" not in processed_handles,
        )

    timeline_folder = server.gui.add_folder("Timeline")
    with timeline_folder:
        frame_slider = server.gui.add_slider(
            "Frame",
            min=0,
            max=max(0, num_frames - 1),
            step=1,
            initial_value=0,
        )
        fps_slider = server.gui.add_slider(
            "FPS",
            min=1,
            max=60,
            step=1,
            initial_value=12,
        )
        btn_rev = server.gui.add_button("Play Backward")
        btn_pause = server.gui.add_button("Pause")
        btn_fwd = server.gui.add_button("Play Forward")

    playback_speed = {"value": 0}

    def apply_visibility() -> None:
        raw_right_vertices_handle.visible = show_raw_vertices.value
        raw_left_vertices_handle.visible = show_raw_vertices.value

        raw_keypoints_visible = show_raw_keypoints.value
        raw_right_keypoints_handle.visible = raw_keypoints_visible
        raw_left_keypoints_handle.visible = raw_keypoints_visible
        raw_right_lines_handle.visible = raw_keypoints_visible
        raw_left_lines_handle.visible = raw_keypoints_visible

        raw_object_visible = show_raw_object.value
        raw_object_center_handle.visible = raw_object_visible
        raw_object_corners_handle.visible = raw_object_visible
        raw_object_bbox_handle.visible = raw_object_visible
        raw_object_frame_handle.visible = raw_object_visible
        raw_object_mesh_handle.visible = raw_object_visible and show_raw_object_mesh.value

        if processed is None:
            return
        proc_visible = show_processed.value
        for key in [
            "right_keypoints",
            "left_keypoints",
            "right_lines",
            "left_lines",
            "object_center",
            "object_corners",
            "object_bbox",
            "object_frame",
        ]:
            processed_handles[key].visible = proc_visible
        if "object_mesh" in processed_handles:
            processed_handles["object_mesh"].visible = proc_visible and show_processed_mesh.value

    def render_frame(frame_idx: int) -> None:
        raw_idx = min(frame_idx, raw.num_frames - 1)
        raw_right_vertices_handle.points = raw.right_vertices[raw_idx]
        raw_left_vertices_handle.points = raw.left_vertices[raw_idx]
        raw_right_keypoints_handle.points = raw.right_keypoints[raw_idx]
        raw_left_keypoints_handle.points = raw.left_keypoints[raw_idx]
        raw_right_lines_handle.points = _hand_keypoint_segments(raw.right_keypoints[raw_idx])
        raw_left_lines_handle.points = _hand_keypoint_segments(raw.left_keypoints[raw_idx])

        raw_center = raw.object_centers[raw_idx]
        raw_quat = raw.object_quats_wxyz[raw_idx]
        raw_rot = _wxyz_to_rotmat(raw_quat)
        raw_corners_frame = _obb_corners(raw_center, raw_rot, raw.object_extents)
        raw_object_center_handle.points = raw_center[None, :]
        raw_object_corners_handle.points = raw_corners_frame
        raw_object_bbox_handle.points = _obb_segments(raw_corners_frame)
        raw_object_frame_handle.position = tuple(raw_center.tolist())
        raw_object_frame_handle.wxyz = tuple(raw_quat.tolist())
        raw_object_mesh_handle.position = tuple(raw_center.tolist())
        raw_object_mesh_handle.wxyz = tuple(raw_quat.tolist())

        if processed is not None:
            proc_idx = min(frame_idx, processed.num_frames - 1)
            proc_right_keypoints = processed.right_keypoints[proc_idx]
            proc_left_keypoints = processed.left_keypoints[proc_idx]
            proc_center = processed.object_centers[proc_idx]
            proc_quat = processed.object_quats_wxyz[proc_idx]
            proc_rot = _wxyz_to_rotmat(proc_quat)
            proc_corners_frame = _obb_corners(
                proc_center,
                proc_rot,
                processed.object_extents,
            )

            processed_handles["right_keypoints"].points = proc_right_keypoints
            processed_handles["left_keypoints"].points = proc_left_keypoints
            processed_handles["right_lines"].points = _hand_keypoint_segments(proc_right_keypoints)
            processed_handles["left_lines"].points = _hand_keypoint_segments(proc_left_keypoints)
            processed_handles["object_center"].points = proc_center[None, :]
            processed_handles["object_corners"].points = proc_corners_frame
            processed_handles["object_bbox"].points = _obb_segments(proc_corners_frame)
            processed_handles["object_frame"].position = tuple(proc_center.tolist())
            processed_handles["object_frame"].wxyz = tuple(proc_quat.tolist())
            if "object_mesh" in processed_handles:
                processed_handles["object_mesh"].position = tuple(proc_center.tolist())
                processed_handles["object_mesh"].wxyz = tuple(proc_quat.tolist())

        apply_visibility()
        try:
            server.flush()
        except Exception:
            pass

    @show_raw_vertices.on_update
    def _(_) -> None:
        apply_visibility()

    @show_raw_keypoints.on_update
    def _(_) -> None:
        apply_visibility()

    @show_raw_object.on_update
    def _(_) -> None:
        apply_visibility()

    @show_raw_object_mesh.on_update
    def _(_) -> None:
        apply_visibility()

    @show_processed.on_update
    def _(_) -> None:
        apply_visibility()

    @show_processed_mesh.on_update
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
                next_frame = max(0, min(int(frame_slider.max), next_frame))
                frame_slider.value = next_frame
            time.sleep(1.0 / max(1.0, float(fps_slider.value)))

    threading.Thread(target=playback_loop, daemon=True).start()

    render_frame(0)
    loguru.logger.info(
        "Viser viewer running at http://{}:{}/ (frames={}, processed_overlay={})",
        config.host,
        config.port,
        num_frames,
        processed is not None,
    )
    loguru.logger.info(
        "Raw object uses orientation_policy='{}'. Processed overlay, if present, uses the saved qpos from trajectory_keypoints.npz.",
        config.orientation_policy,
    )
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main(parse_args())
