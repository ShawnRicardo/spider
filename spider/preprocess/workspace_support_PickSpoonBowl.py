from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import mujoco
import numpy as np

SUPPORT_TABLE_CLEARANCE = 0.002
SUPPORT_TABLE_MARGIN = 0.10
SUPPORT_TABLE_HALF_THICKNESS = 0.02
SUPPORT_TABLE_COLLISION_MODE_OBJECT_ONLY = "object_only"
SUPPORT_TABLE_COLLISION_MODE_OBJECT_AND_HAND = "object_and_hand"
SUPPORT_TABLE_COLLISION_MODE_OBJECT_AND_MANIPULATOR = "object_and_manipulator"
ASM_PICK_SPOON_BOWL_REFERENCE_BACK_OFFSET_Y = -0.18


@dataclass(frozen=True)
class WorkspaceSupportSpec:
    robot_height: float
    table_surface_z: float
    workspace_z_offset: float
    workspace_xy_offset: np.ndarray
    workspace_yaw_rad: float
    object_first_frame_min_z: float
    table_center: np.ndarray
    table_size: np.ndarray
    collision_mode: str = SUPPORT_TABLE_COLLISION_MODE_OBJECT_ONLY

    def to_json_dict(self) -> dict[str, float | list[float] | str]:
        return {
            "robot_height": float(self.robot_height),
            "table_surface_z": float(self.table_surface_z),
            "workspace_z_offset": float(self.workspace_z_offset),
            "workspace_xy_offset": self.workspace_xy_offset.tolist(),
            "workspace_yaw_rad": float(self.workspace_yaw_rad),
            "object_first_frame_min_z": float(self.object_first_frame_min_z),
            "support_table_center": self.table_center.tolist(),
            "support_table_size": self.table_size.tolist(),
            "support_table_collision_mode": self.collision_mode,
        }


def quat_wxyz_to_rotmat(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    quat = quat / max(np.linalg.norm(quat), 1e-12)
    w, x, y, z = quat
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


@lru_cache(maxsize=None)
def load_obj_vertices(obj_path: str) -> np.ndarray:
    vertices: list[list[float]] = []
    with open(obj_path) as f:
        for line in f:
            if not line.startswith("v "):
                continue
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if not vertices:
        raise ValueError(f"No vertices found in OBJ: {obj_path}")
    return np.asarray(vertices, dtype=np.float64)


def transform_points(points: np.ndarray, pos: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    rot = quat_wxyz_to_rotmat(quat_wxyz)
    return points @ rot.T + np.asarray(pos, dtype=np.float64)


def _geom_local_bbox_points(model: mujoco.MjModel, geom_id: int) -> np.ndarray:
    geom_type = int(model.geom_type[geom_id])
    size = np.asarray(model.geom_size[geom_id], dtype=np.float64)
    if geom_type == mujoco.mjtGeom.mjGEOM_PLANE:
        return np.zeros((0, 3), dtype=np.float64)

    if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
        extent = size
    elif geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
        extent = np.array([size[0], size[0], size[0]], dtype=np.float64)
    elif geom_type == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
        extent = size
    elif geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
        extent = np.array([size[0], size[0], size[1] + size[0]], dtype=np.float64)
    elif geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
        extent = np.array([size[0], size[0], size[1]], dtype=np.float64)
    elif geom_type == mujoco.mjtGeom.mjGEOM_MESH:
        mesh_id = int(model.geom_dataid[geom_id])
        if mesh_id < 0:
            radius = float(model.geom_rbound[geom_id])
            extent = np.array([radius, radius, radius], dtype=np.float64)
        else:
            vert_adr = int(model.mesh_vertadr[mesh_id])
            vert_num = int(model.mesh_vertnum[mesh_id])
            return np.asarray(
                model.mesh_vert[vert_adr : vert_adr + vert_num],
                dtype=np.float64,
            )
    else:
        radius = float(model.geom_rbound[geom_id])
        extent = np.array([radius, radius, radius], dtype=np.float64)

    corners = []
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                corners.append([sx * extent[0], sy * extent[1], sz * extent[2]])
    return np.asarray(corners, dtype=np.float64)


def compute_robot_geom_bounds(robot_xml_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    model = mujoco.MjModel.from_xml_path(str(robot_xml_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    lower = np.full(3, np.inf, dtype=np.float64)
    upper = np.full(3, -np.inf, dtype=np.float64)
    for geom_id in range(model.ngeom):
        local_points = _geom_local_bbox_points(model, geom_id)
        if local_points.size == 0:
            continue
        geom_pos = np.asarray(data.geom_xpos[geom_id], dtype=np.float64)
        geom_rot = np.asarray(data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
        world_points = local_points @ geom_rot.T + geom_pos
        lower = np.minimum(lower, world_points.min(axis=0))
        upper = np.maximum(upper, world_points.max(axis=0))
    return lower, upper


def _iter_active_object_vertices(
    dataset_dir: str,
    task_info: dict,
    embodiment_type: str,
    qpos_obj_right_first: np.ndarray,
    qpos_obj_left_first: np.ndarray,
):
    object_entries = []
    if embodiment_type in ["right", "bimanual"]:
        right_mesh_dir = task_info.get("right_object_mesh_dir")
        if right_mesh_dir:
            object_entries.append(
                ("right", Path(dataset_dir) / right_mesh_dir / "visual.obj", qpos_obj_right_first)
            )
    if embodiment_type in ["left", "bimanual"]:
        left_mesh_dir = task_info.get("left_object_mesh_dir")
        if left_mesh_dir:
            object_entries.append(
                ("left", Path(dataset_dir) / left_mesh_dir / "visual.obj", qpos_obj_left_first)
            )

    for side, mesh_path, qpos in object_entries:
        if not mesh_path.exists():
            continue
        qpos = np.asarray(qpos, dtype=np.float64)
        vertices = load_obj_vertices(str(mesh_path))
        yield side, transform_points(vertices, qpos[:3], qpos[3:])


def compute_object_bounds(
    dataset_dir: str,
    task_info: dict,
    embodiment_type: str,
    qpos_obj_right_first: np.ndarray,
    qpos_obj_left_first: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    lower = np.full(3, np.inf, dtype=np.float64)
    upper = np.full(3, -np.inf, dtype=np.float64)
    per_object_min_z: dict[str, float] = {}
    found_any = False

    for side, vertices_world in _iter_active_object_vertices(
        dataset_dir,
        task_info,
        embodiment_type,
        qpos_obj_right_first,
        qpos_obj_left_first,
    ):
        found_any = True
        lower = np.minimum(lower, vertices_world.min(axis=0))
        upper = np.maximum(upper, vertices_world.max(axis=0))
        per_object_min_z[side] = float(vertices_world[:, 2].min())

    if not found_any:
        raise ValueError("No active object visual meshes found for workspace support.")

    return lower, upper, per_object_min_z


def compute_workspace_support_spec(
    dataset_dir: str,
    robot_xml_path: str | Path,
    task_info: dict,
    dataset_name: str,
    robot_type: str,
    embodiment_type: str,
    task: str,
    qpos_obj_right_first: np.ndarray,
    qpos_obj_left_first: np.ndarray,
    clearance: float = SUPPORT_TABLE_CLEARANCE,
    margin: float = SUPPORT_TABLE_MARGIN,
    half_thickness: float = SUPPORT_TABLE_HALF_THICKNESS,
    collision_mode: str = SUPPORT_TABLE_COLLISION_MODE_OBJECT_ONLY,
) -> tuple[WorkspaceSupportSpec, dict[str, float]]:
    robot_lower, robot_upper = compute_robot_geom_bounds(robot_xml_path)
    robot_height = float(robot_upper[2] - robot_lower[2])
    if robot_type == "asm":
        # ASM's base/support meshes can extend below the operational base frame.
        # Using the full bbox height then places the tabletop too high. Use the
        # upper robot reach height as the tabletop reference instead.
        table_surface_z = 0.5 * float(robot_upper[2])
    else:
        table_surface_z = 0.5 * robot_height

    object_lower, object_upper, per_object_min_z = compute_object_bounds(
        dataset_dir,
        task_info,
        embodiment_type,
        qpos_obj_right_first,
        qpos_obj_left_first,
    )
    object_min_z = float(object_lower[2])
    use_pick_spoon_bowl_front_workspace = (
        dataset_name == "oakink"
        and robot_type == "asm"
        and embodiment_type == "bimanual"
        and task == "pick_spoon_bowl"
    )
    effective_clearance = 0.0 if use_pick_spoon_bowl_front_workspace else float(clearance)
    workspace_z_offset = table_surface_z + effective_clearance - object_min_z

    xy_center = 0.5 * (object_lower[:2] + object_upper[:2])
    xy_half_extent = 0.5 * (object_upper[:2] - object_lower[:2]) + margin
    workspace_xy_offset = np.zeros(2, dtype=np.float64)
    workspace_yaw_rad = 0.0
    table_xy_center = xy_center.copy()
    table_xy_half_extent = xy_half_extent.copy()
    if use_pick_spoon_bowl_front_workspace:
        # The original OakInk object workspace sits behind/aside the ASM base in
        # the dataset frame. First recover the +Y workspace used by the normal
        # ASM path, then rotate that whole workspace by -90 deg around Z. This
        # preserves the relative table/object/hand layout instead of merely
        # translating object centers onto the X axis.
        reference_table_center_y = xy_center[1] + ASM_PICK_SPOON_BOWL_REFERENCE_BACK_OFFSET_Y
        mirrored_table_center_y = -reference_table_center_y
        workspace_xy_offset = np.array(
            [-xy_center[0], mirrored_table_center_y - xy_center[1]],
            dtype=np.float64,
        )
        workspace_yaw_rad = -0.5 * np.pi
        table_xy_center_y = xy_center + workspace_xy_offset
        table_xy_center = np.array(
            [table_xy_center_y[1], -table_xy_center_y[0]],
            dtype=np.float64,
        )
        table_xy_half_extent = np.array(
            [xy_half_extent[1], xy_half_extent[0]],
            dtype=np.float64,
        )
    table_center = np.array(
        [
            table_xy_center[0],
            table_xy_center[1],
            table_surface_z - half_thickness,
        ],
        dtype=np.float64,
    )
    table_size = np.array(
        [table_xy_half_extent[0], table_xy_half_extent[1], half_thickness],
        dtype=np.float64,
    )

    return (
        WorkspaceSupportSpec(
            robot_height=robot_height,
            table_surface_z=table_surface_z,
            workspace_z_offset=workspace_z_offset,
            workspace_xy_offset=workspace_xy_offset,
            workspace_yaw_rad=workspace_yaw_rad,
            object_first_frame_min_z=object_min_z,
            table_center=table_center,
            table_size=table_size,
            collision_mode=collision_mode,
        ),
        per_object_min_z,
    )


def workspace_support_spec_from_task_info(
    task_info: dict,
) -> WorkspaceSupportSpec | None:
    required = [
        "robot_height",
        "table_surface_z",
        "workspace_z_offset",
        "object_first_frame_min_z",
        "support_table_center",
        "support_table_size",
    ]
    if not all(key in task_info for key in required):
        return None
    return WorkspaceSupportSpec(
        robot_height=float(task_info["robot_height"]),
        table_surface_z=float(task_info["table_surface_z"]),
        workspace_z_offset=float(task_info["workspace_z_offset"]),
        workspace_xy_offset=np.asarray(
            task_info.get("workspace_xy_offset", [0.0, 0.0]),
            dtype=np.float64,
        ),
        workspace_yaw_rad=float(task_info.get("workspace_yaw_rad", 0.0)),
        object_first_frame_min_z=float(task_info["object_first_frame_min_z"]),
        table_center=np.asarray(task_info["support_table_center"], dtype=np.float64),
        table_size=np.asarray(task_info["support_table_size"], dtype=np.float64),
        collision_mode=task_info.get(
            "support_table_collision_mode",
            SUPPORT_TABLE_COLLISION_MODE_OBJECT_ONLY,
        ),
    )


def get_support_table_collision_targets(
    object_collision_names: list[str],
    hand_collision_names: list[str],
    collision_mode: str,
) -> list[str]:
    if collision_mode == SUPPORT_TABLE_COLLISION_MODE_OBJECT_ONLY:
        return list(object_collision_names)
    if collision_mode == SUPPORT_TABLE_COLLISION_MODE_OBJECT_AND_HAND:
        return list(object_collision_names) + list(hand_collision_names)
    if collision_mode == SUPPORT_TABLE_COLLISION_MODE_OBJECT_AND_MANIPULATOR:
        return list(object_collision_names) + list(hand_collision_names)
    raise ValueError(f"Unsupported support table collision mode: {collision_mode}")
