#!/usr/bin/env python3
"""Render Pick-Place parquet trajectories on the ASM MuJoCo robot.

The input parquet stores real robot data as:
    left_arm(7) + left_hand(20) + right_arm(7) + right_hand(20)

The ASM MJCF uses:
    right_arm(7) + right_hand(20) + left_arm(7) + left_hand(20)

This script renders four verification videos:
    kinematic_front.mp4  kinematic_d435.mp4
    dynamic_front.mp4    dynamic_d435.mp4

Use --asm-variant asm_2, asm_3, or asm_4 to generate the robot model from:
    spider/assets/robots/asm_description/urdf/asm_2.urdf
    spider/assets/robots/asm_description/urdf/asm_3.urdf
    spider/assets/robots/asm_description/urdf/asm_4.urdf

Collision can be switched at render-scene generation time:
    urdf_mesh: keep the current active URDF collision meshes
    none:      disable all contacts
    urdf_mesh_scaled: use colored, scaled copies of the URDF collision meshes
    inset_box: replace collision mesh geoms with smaller box proxies
    inset_capsule: replace collision mesh geoms with inset capsule proxies
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from contextlib import ExitStack
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

_START_TIME = time.perf_counter()


def _log(message: str) -> None:
    elapsed = time.perf_counter() - _START_TIME
    print(f"[pick-place-render {elapsed:8.1f}s] {message}", flush=True)


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_log("Importing imageio, mujoco, numpy, and pandas...")
import imageio
import mujoco
import numpy as np
import pandas as pd
_log("Python dependencies imported.")

DEFAULT_EPISODE_DIR = REPO_ROOT / "preprocessed/Pick-Place/episode_000001"
DEFAULT_MODEL_PATH = (
    REPO_ROOT / "example_datasets/processed/ourdata/assets/robots/asm/bimanual.xml"
)
ASM_VARIANT_DEFAULT = "asm"
ASM_VARIANT_ASM_2 = "asm_2"
ASM_VARIANT_ASM_3 = "asm_3"
ASM_VARIANT_ASM_4 = "asm_4"
ASM_VARIANT_ASM_5 = "asm_5"
ASM_VARIANTS = (
    ASM_VARIANT_DEFAULT,
    ASM_VARIANT_ASM_2,
    ASM_VARIANT_ASM_3,
    ASM_VARIANT_ASM_4,
    ASM_VARIANT_ASM_5
)
ASM_VARIANT_URDFS = {
    ASM_VARIANT_DEFAULT: REPO_ROOT / "spider/assets/robots/asm_description/urdf/asm.urdf",
    ASM_VARIANT_ASM_2: REPO_ROOT / "spider/assets/robots/asm_description/urdf/asm_2.urdf",
    ASM_VARIANT_ASM_3: REPO_ROOT / "spider/assets/robots/asm_description/urdf/asm_3.urdf",
    ASM_VARIANT_ASM_4: REPO_ROOT / "spider/assets/robots/asm_description/urdf/asm_4.urdf",
    ASM_VARIANT_ASM_5: REPO_ROOT / "spider/assets/robots/asm_description/urdf/asm_5.urdf",
}
D435_SITE_NAME = "d435_optical_frame"
FRONT_CAMERA_NAME = "front"
SIDE_Y_CAMERA_NAME = "side_y"
D435_RENDER_CAMERA_NAME = "d435_optical_render"
COLLISION_MODE_URDF_MESH = "urdf_mesh"
COLLISION_MODE_NONE = "none"
COLLISION_MODE_URDF_MESH_SCALED = "urdf_mesh_scaled"
COLLISION_MODE_INSET_BOX = "inset_box"
COLLISION_MODE_INSET_CAPSULE = "inset_capsule"
COLLISION_MODES = (
    COLLISION_MODE_URDF_MESH,
    COLLISION_MODE_NONE,
    COLLISION_MODE_URDF_MESH_SCALED,
    COLLISION_MODE_INSET_BOX,
    COLLISION_MODE_INSET_CAPSULE,
)
LINK_COLOR_HEX = {
    "Link_1": "#E53935",
    "Link_2": "#1E88E5",
    "Link_3": "#43A047",
    "Link_4": "#FB8C00",
    "Link_5": "#8E24AA",
    "Link_6": "#00ACC1",
    "Link_7": "#FDD835",
    "Palm": "#6D4C41",
}
FALLBACK_COLOR_HEX = (
    "#D81B60",
    "#3949AB",
    "#00897B",
    "#7CB342",
    "#C0CA33",
    "#F4511E",
    "#5E35B1",
    "#039BE5",
    "#A1887F",
    "#546E7A",
)


def _resolve_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def _default_output_dir(
    episode_dir: Path,
    *,
    collision_mode: str,
    asm_variant: str,
) -> Path:
    variant_suffix = "" if asm_variant == ASM_VARIANT_DEFAULT else f"_{asm_variant}"
    if collision_mode == COLLISION_MODE_URDF_MESH:
        return episode_dir / "verify" if not variant_suffix else episode_dir / "verify" / f"collision_urdf_mesh{variant_suffix}"
    return episode_dir / "verify" / f"collision_{collision_mode}{variant_suffix}"


def _resolve_source_urdf(asm_variant: str, source_urdf: str) -> Path:
    if source_urdf:
        return _resolve_path(source_urdf)
    try:
        return ASM_VARIANT_URDFS[asm_variant]
    except KeyError as exc:
        raise ValueError(f"Unsupported ASM variant: {asm_variant!r}") from exc


def _prepare_variant_model(
    output_dir: Path,
    *,
    asm_variant: str,
    source_urdf: Path,
    prepare_collision_mesh_scale: float,
) -> Path:
    if not source_urdf.is_file():
        raise FileNotFoundError(f"ASM source URDF not found: {source_urdf}")

    dataset_dir = output_dir / "_generated_robot_dataset"
    dataset_name = f"pick_place_{asm_variant}"
    model_path = dataset_dir / "processed" / dataset_name / "assets" / "robots" / "asm" / "bimanual.xml"
    cmd = [
        sys.executable,
        str(REPO_ROOT / "spider/preprocess/prepare_asm_mjcf.py"),
        "--dataset-dir",
        str(dataset_dir),
        "--dataset-name",
        dataset_name,
        "--source-urdf",
        str(source_urdf),
        "--robot-type",
        "asm",
        "--arm-kp",
        "800",
        "--hand-kp",
        "140",
        "--arm-damping",
        "3.0",
        "--hand-damping",
        "1.2",
        "--arm-armature",
        "0.05",
        "--hand-armature",
        "0.02",
        "--arm-frictionloss",
        "0.0",
        "--hand-frictionloss",
        "0.02",
        "--arm-force-scale",
        "8.0",
        "--hand-force-scale",
        "8.0",
        "--collision-mesh-scale",
        f"{float(prepare_collision_mesh_scale):.9g}",
        "--collision-geometry-mode",
        "urdf_mesh",
        "--variants",
        "bimanual",
    ]
    _log(
        "Preparing ASM variant model: "
        f"variant={asm_variant}, source_urdf={source_urdf}, "
        f"prepare_collision_mesh_scale={prepare_collision_mesh_scale}"
    )
    env = os.environ.copy()
    env.pop("LD_LIBRARY_PATH", None)
    subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, check=True)
    if not model_path.is_file():
        raise FileNotFoundError(f"Prepared ASM model was not created: {model_path}")
    return model_path


def _format_vec(values: np.ndarray) -> str:
    return " ".join(f"{float(v):.9g}" for v in np.asarray(values).reshape(-1))


def _parse_vec_attr(element: ET.Element, name: str, dim: int, default: np.ndarray) -> np.ndarray:
    raw = element.get(name)
    if not raw:
        return default.astype(np.float64, copy=True)
    values = np.fromstring(raw, sep=" ", dtype=np.float64)
    if values.shape != (dim,):
        raise ValueError(f"{element.tag} {element.get('name', '')!r} has invalid {name}={raw!r}")
    return values


def _quat_wxyz_to_mat(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    norm = np.linalg.norm(quat)
    if norm <= 0:
        return np.eye(3)
    w, x, y, z = quat / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _hex_to_rgba_values(color_hex: str, alpha: float = 0.85) -> list[float]:
    color_hex = color_hex.lstrip("#")
    if len(color_hex) != 6:
        raise ValueError(f"Invalid hex color: {color_hex!r}")
    return [
        int(color_hex[0:2], 16) / 255.0,
        int(color_hex[2:4], 16) / 255.0,
        int(color_hex[4:6], 16) / 255.0,
        float(alpha),
    ]


def _rgba_string(values: list[float]) -> str:
    return " ".join(f"{value:.6g}" for value in values)


def _fallback_color_hex(key: str) -> str:
    idx = sum((i + 1) * ord(ch) for i, ch in enumerate(key)) % len(FALLBACK_COLOR_HEX)
    return FALLBACK_COLOR_HEX[idx]


def _collision_proxy_metadata(geom_name: str, mesh_name: str = "") -> dict[str, object]:
    side = "none"
    if "_left_" in geom_name:
        side = "left"
    elif "_right_" in geom_name:
        side = "right"

    component = "other"
    link_key = mesh_name or geom_name
    link_label = link_key

    arm_match = re.search(r"collision_arm_(left|right)_link(\d+)", geom_name)
    if arm_match:
        side = arm_match.group(1)
        component = "arm"
        link_key = f"Link_{arm_match.group(2)}"
        link_label = link_key
    elif re.search(r"collision_hand_(left|right)_palm", geom_name):
        component = "hand"
        link_key = "Palm"
        link_label = "Palm"
    else:
        finger_match = re.search(
            r"collision_hand_(left|right)_(thumb|index|middle|ring|pinky)_(link\d+|tip_link)",
            geom_name,
        )
        if finger_match:
            side = finger_match.group(1)
            component = "hand"
            link_key = f"{finger_match.group(2)}_{finger_match.group(3)}"
            link_label = link_key
        elif geom_name.startswith("collision_body_"):
            component = "body"
            link_key = mesh_name or geom_name.replace("collision_body_", "Body_")
            link_label = link_key

    color_hex = LINK_COLOR_HEX.get(link_key, _fallback_color_hex(link_key))
    rgba_values = _hex_to_rgba_values(color_hex)
    return {
        "component": component,
        "side": side,
        "link_key": link_key,
        "link_label": link_label,
        "color_hex": color_hex,
        "rgba": rgba_values,
        "rgba_string": _rgba_string(rgba_values),
    }


def _collision_debug_rgba(geom_name: str, mesh_name: str = "") -> str:
    return str(_collision_proxy_metadata(geom_name, mesh_name)["rgba_string"])


def _load_parquet_trajectories(parquet_path: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    df = pd.read_parquet(parquet_path)
    required = [
        "frame_index",
        "timestamp",
        "observation.state",
        "action",
        "observation.state.arm_pos_left",
        "observation.state.hand_pos_left",
        "observation.state.arm_pos_right",
        "observation.state.hand_pos_right",
        "action.arm_left",
        "action.hand_left",
        "action.arm_right",
        "action.hand_right",
    ]
    missing = [name for name in required if name not in df.columns]
    if missing:
        raise KeyError(f"Missing required parquet column(s): {missing}")

    state_left_arm = _stack_column(df, "observation.state.arm_pos_left", 7)
    state_left_hand = _stack_column(df, "observation.state.hand_pos_left", 20)
    state_right_arm = _stack_column(df, "observation.state.arm_pos_right", 7)
    state_right_hand = _stack_column(df, "observation.state.hand_pos_right", 20)
    action_left_arm = _stack_column(df, "action.arm_left", 7)
    action_left_hand = _stack_column(df, "action.hand_left", 20)
    action_right_arm = _stack_column(df, "action.arm_right", 7)
    action_right_hand = _stack_column(df, "action.hand_right", 20)

    qpos_kinematic = np.concatenate(
        [state_right_arm, state_right_hand, state_left_arm, state_left_hand],
        axis=1,
    )
    ctrl_dynamic = np.concatenate(
        [action_right_arm, action_right_hand, action_left_arm, action_left_hand],
        axis=1,
    )

    if not np.isfinite(qpos_kinematic).all():
        raise ValueError("Mapped kinematic qpos contains non-finite values")
    if not np.isfinite(ctrl_dynamic).all():
        raise ValueError("Mapped dynamic ctrl contains non-finite values")
    return df, qpos_kinematic, ctrl_dynamic


def _stack_column(df: pd.DataFrame, column: str, expected_dim: int) -> np.ndarray:
    arr = np.stack(df[column].map(lambda x: np.asarray(x, dtype=np.float64)).to_numpy())
    if arr.ndim != 2 or arr.shape[1] != expected_dim:
        raise ValueError(f"{column} expected shape (T, {expected_dim}), got {arr.shape}")
    return arr


def _camera_site_pose(model_path: Path, site_name: str) -> tuple[np.ndarray, np.ndarray]:
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if site_id == -1:
        raise ValueError(f"Site {site_name!r} not found in {model_path}")
    pos = data.site_xpos[site_id].copy()
    rot = data.site_xmat[site_id].reshape(3, 3).copy()
    return pos, rot


def _mesh_vertices_by_name(model_path: Path) -> dict[str, np.ndarray]:
    model = mujoco.MjModel.from_xml_path(str(model_path))
    vertices: dict[str, np.ndarray] = {}
    for mesh_id in range(model.nmesh):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mesh_id)
        if not name:
            continue
        vert_adr = int(model.mesh_vertadr[mesh_id])
        vert_num = int(model.mesh_vertnum[mesh_id])
        if vert_num <= 0:
            continue
        verts = model.mesh_vert[vert_adr : vert_adr + vert_num]
        vertices[name] = verts.copy()
    return vertices


def _mesh_bounds_by_name(model_path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    bounds: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, verts in _mesh_vertices_by_name(model_path).items():
        bounds[name] = (verts.min(axis=0).copy(), verts.max(axis=0).copy())
    return bounds


def _parse_mesh_scale(mesh_element: ET.Element) -> np.ndarray:
    raw = mesh_element.get("scale")
    if not raw:
        return np.ones(3, dtype=np.float64)
    values = np.fromstring(raw, sep=" ", dtype=np.float64)
    if values.shape == (1,):
        return np.repeat(values[0], 3).astype(np.float64)
    if values.shape == (3,):
        return values.astype(np.float64)
    raise ValueError(
        f"Mesh asset {mesh_element.get('name', '')!r} has invalid scale={raw!r}"
    )


def _replace_collision_meshes_with_scaled_urdf_meshes(
    root: ET.Element,
    source_model_path: Path,
    *,
    mesh_scale: float,
) -> dict[str, object]:
    if mesh_scale <= 0.0:
        raise ValueError(f"URDF collision mesh scale must be > 0, got {mesh_scale}")

    asset = root.find("asset")
    if asset is None:
        raise ValueError("MJCF root has no <asset> section")
    mesh_assets = {
        mesh.get("name"): mesh
        for mesh in asset.findall("mesh")
        if mesh.get("name")
    }
    mesh_vertices = _mesh_vertices_by_name(source_model_path)
    converted = 0
    skipped = 0
    collision_meshes = []
    created_mesh_assets: set[str] = set()

    for geom in root.iter("geom"):
        geom_name = geom.get("name", "")
        mesh_name = geom.get("mesh")
        if not geom_name.startswith("collision_") or not mesh_name:
            continue
        source_mesh = mesh_assets.get(mesh_name)
        vertices = mesh_vertices.get(mesh_name)
        if source_mesh is None or vertices is None:
            skipped += 1
            continue

        scaled_mesh_name = f"{mesh_name}_collision_scaled"
        if scaled_mesh_name not in created_mesh_assets:
            scaled_mesh = copy.deepcopy(source_mesh)
            scaled_mesh.set("name", scaled_mesh_name)
            base_scale = _parse_mesh_scale(source_mesh)
            scaled_mesh.set("scale", _format_vec(base_scale * float(mesh_scale)))
            asset.append(scaled_mesh)
            created_mesh_assets.add(scaled_mesh_name)

        lower = vertices.min(axis=0)
        upper = vertices.max(axis=0)
        center = 0.5 * (lower + upper)
        original_pos = _parse_vec_attr(geom, "pos", 3, np.zeros(3))
        original_quat = _parse_vec_attr(
            geom,
            "quat",
            4,
            np.array([1.0, 0.0, 0.0, 0.0]),
        )
        original_rot = _quat_wxyz_to_mat(original_quat)
        scaled_pos = original_pos + original_rot @ ((1.0 - float(mesh_scale)) * center)

        metadata = _collision_proxy_metadata(geom_name, mesh_name)
        geom.set("mesh", scaled_mesh_name)
        geom.set("pos", _format_vec(scaled_pos))
        geom.set("contype", geom.get("contype", "1"))
        geom.set("conaffinity", geom.get("conaffinity", "1"))
        geom.set("condim", geom.get("condim", "3"))
        geom.set("group", geom.get("group", "3"))
        geom.set("rgba", str(metadata["rgba_string"]))

        collision_meshes.append(
            {
                "geom": geom_name,
                "mesh": mesh_name,
                "scaled_mesh": scaled_mesh_name,
                "scale": float(mesh_scale),
                "component": metadata["component"],
                "side": metadata["side"],
                "link_key": metadata["link_key"],
                "link_label": metadata["link_label"],
                "color_hex": metadata["color_hex"],
                "rgba": metadata["rgba"],
                "original_bbox_center": center.tolist(),
                "scaled_geom_pos": scaled_pos.tolist(),
            }
        )
        converted += 1

    return {
        "converted_collision_meshes": converted,
        "skipped_collision_meshes": skipped,
        "urdf_collision_mesh_scale": float(mesh_scale),
        "created_scaled_mesh_assets": len(created_mesh_assets),
        "collision_meshes": collision_meshes,
    }


def _remove_contact_pairs(root: ET.Element) -> int:
    removed = 0
    for contact in list(root.findall("contact")):
        removed += len(list(contact))
        root.remove(contact)
    return removed


def _disable_all_geom_contacts(root: ET.Element) -> int:
    count = 0
    for geom in root.iter("geom"):
        geom.set("contype", "0")
        geom.set("conaffinity", "0")
        count += 1
    return count


def _replace_collision_meshes_with_inset_boxes(
    root: ET.Element,
    source_model_path: Path,
    *,
    shrink: float,
    min_half_size: float,
) -> dict[str, int | float]:
    if not (0.0 < shrink <= 1.0):
        raise ValueError(f"primitive collision shrink must be in (0, 1], got {shrink}")
    if min_half_size <= 0.0:
        raise ValueError(f"primitive collision min half size must be > 0, got {min_half_size}")

    mesh_bounds = _mesh_bounds_by_name(source_model_path)
    converted = 0
    skipped = 0
    for geom in root.iter("geom"):
        geom_name = geom.get("name", "")
        mesh_name = geom.get("mesh")
        if not geom_name.startswith("collision_") or not mesh_name:
            continue
        bounds = mesh_bounds.get(mesh_name)
        if bounds is None:
            skipped += 1
            continue
        lower, upper = bounds
        center = 0.5 * (lower + upper)
        half_extents = 0.5 * (upper - lower) * float(shrink)
        half_extents = np.maximum(half_extents, float(min_half_size))

        original_pos = _parse_vec_attr(geom, "pos", 3, np.zeros(3))
        original_quat = _parse_vec_attr(geom, "quat", 4, np.array([1.0, 0.0, 0.0, 0.0]))
        box_pos = original_pos + _quat_wxyz_to_mat(original_quat) @ center

        geom.set("type", "box")
        geom.set("size", _format_vec(half_extents))
        geom.set("pos", _format_vec(box_pos))
        geom.set("quat", _format_vec(original_quat))
        geom.set("contype", geom.get("contype", "1"))
        geom.set("conaffinity", geom.get("conaffinity", "1"))
        geom.set("condim", geom.get("condim", "3"))
        geom.set("group", geom.get("group", "3"))
        geom.set("rgba", _collision_debug_rgba(geom_name, mesh_name))
        geom.attrib.pop("mesh", None)
        converted += 1

    return {
        "converted_collision_meshes": converted,
        "skipped_collision_meshes": skipped,
        "primitive_shrink": float(shrink),
        "primitive_min_half_size": float(min_half_size),
    }


def _capsule_axis_from_vertices(vertices: np.ndarray) -> np.ndarray:
    centered = vertices - vertices.mean(axis=0)
    if vertices.shape[0] >= 3 and np.linalg.norm(centered) > 0:
        try:
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
            axis = vh[0]
            if np.isfinite(axis).all() and np.linalg.norm(axis) > 0:
                return axis / np.linalg.norm(axis)
        except np.linalg.LinAlgError:
            pass
    extents = vertices.max(axis=0) - vertices.min(axis=0)
    axis = np.zeros(3, dtype=np.float64)
    axis[int(np.argmax(extents))] = 1.0
    return axis


def _replace_collision_meshes_with_inset_capsules(
    root: ET.Element,
    source_model_path: Path,
    *,
    length_quantile: float,
    radius_quantile: float,
    radius_scale: float,
    min_radius: float,
    min_half_length: float,
) -> dict[str, object]:
    if not (0.0 < length_quantile <= 1.0):
        raise ValueError(f"capsule length quantile must be in (0, 1], got {length_quantile}")
    if not (0.0 < radius_quantile <= 1.0):
        raise ValueError(f"capsule radius quantile must be in (0, 1], got {radius_quantile}")
    if radius_scale <= 0.0:
        raise ValueError(f"capsule radius scale must be > 0, got {radius_scale}")
    if min_radius <= 0.0:
        raise ValueError(f"capsule min radius must be > 0, got {min_radius}")
    if min_half_length <= 0.0:
        raise ValueError(f"capsule min half length must be > 0, got {min_half_length}")

    mesh_vertices = _mesh_vertices_by_name(source_model_path)
    converted = 0
    skipped = 0
    capsule_details = []
    radius_values = []
    length_values = []
    low_q = 0.5 * (1.0 - float(length_quantile))
    high_q = 1.0 - low_q

    for geom in root.iter("geom"):
        geom_name = geom.get("name", "")
        mesh_name = geom.get("mesh")
        if not geom_name.startswith("collision_") or not mesh_name:
            continue
        vertices = mesh_vertices.get(mesh_name)
        if vertices is None or vertices.shape[0] < 2:
            skipped += 1
            continue

        axis = _capsule_axis_from_vertices(vertices)
        center = vertices.mean(axis=0)
        centered = vertices - center
        projections = centered @ axis
        lower_proj = float(np.quantile(projections, low_q))
        upper_proj = float(np.quantile(projections, high_q))
        half_length = 0.5 * (upper_proj - lower_proj)
        degenerate = half_length < float(min_half_length)
        if degenerate:
            midpoint_proj = 0.5 * (lower_proj + upper_proj)
            lower_proj = midpoint_proj - float(min_half_length)
            upper_proj = midpoint_proj + float(min_half_length)
            half_length = float(min_half_length)

        radial = centered - np.outer(projections, axis)
        radial_dist = np.linalg.norm(radial, axis=1)
        radius = float(np.quantile(radial_dist, radius_quantile) * radius_scale)
        radius = max(radius, float(min_radius))

        endpoint_a = center + lower_proj * axis
        endpoint_b = center + upper_proj * axis
        original_pos = _parse_vec_attr(geom, "pos", 3, np.zeros(3))
        original_quat = _parse_vec_attr(geom, "quat", 4, np.array([1.0, 0.0, 0.0, 0.0]))
        original_rot = _quat_wxyz_to_mat(original_quat)
        endpoint_a_body = original_pos + original_rot @ endpoint_a
        endpoint_b_body = original_pos + original_rot @ endpoint_b

        geom.set("type", "capsule")
        geom.set("fromto", _format_vec(np.concatenate([endpoint_a_body, endpoint_b_body])))
        geom.set("size", _format_vec(np.array([radius], dtype=np.float64)))
        geom.set("contype", geom.get("contype", "1"))
        geom.set("conaffinity", geom.get("conaffinity", "1"))
        geom.set("condim", geom.get("condim", "3"))
        geom.set("group", geom.get("group", "3"))
        metadata = _collision_proxy_metadata(geom_name, mesh_name)
        geom.set("rgba", str(metadata["rgba_string"]))
        for attr in ("mesh", "pos", "quat", "euler", "axisangle", "xyaxes", "zaxis"):
            geom.attrib.pop(attr, None)

        length = 2.0 * half_length
        radius_values.append(radius)
        length_values.append(length)
        capsule_details.append(
            {
                "geom": geom_name,
                "mesh": mesh_name,
                "radius": radius,
                "length": length,
                "degenerate": bool(degenerate),
                "component": metadata["component"],
                "side": metadata["side"],
                "link_key": metadata["link_key"],
                "link_label": metadata["link_label"],
                "color_hex": metadata["color_hex"],
                "rgba": metadata["rgba"],
                "fromto": np.concatenate([endpoint_a_body, endpoint_b_body]).tolist(),
            }
        )
        converted += 1

    return {
        "converted_collision_meshes": converted,
        "skipped_collision_meshes": skipped,
        "capsule_length_quantile": float(length_quantile),
        "capsule_radius_quantile": float(radius_quantile),
        "capsule_radius_scale": float(radius_scale),
        "capsule_min_radius": float(min_radius),
        "capsule_min_half_length": float(min_half_length),
        "capsule_radius_min": float(min(radius_values)) if radius_values else 0.0,
        "capsule_radius_max": float(max(radius_values)) if radius_values else 0.0,
        "capsule_radius_mean": float(np.mean(radius_values)) if radius_values else 0.0,
        "capsule_length_min": float(min(length_values)) if length_values else 0.0,
        "capsule_length_max": float(max(length_values)) if length_values else 0.0,
        "capsule_length_mean": float(np.mean(length_values)) if length_values else 0.0,
        "capsules": capsule_details,
    }


def _apply_collision_mode(
    root: ET.Element,
    source_model_path: Path,
    *,
    collision_mode: str,
    primitive_shrink: float,
    primitive_min_half_size: float,
    urdf_collision_mesh_scale: float,
    capsule_length_quantile: float,
    capsule_radius_quantile: float,
    capsule_radius_scale: float,
    capsule_min_radius: float,
    capsule_min_half_length: float,
) -> dict[str, object]:
    if collision_mode == COLLISION_MODE_URDF_MESH:
        return {"collision_mode": collision_mode}
    if collision_mode == COLLISION_MODE_NONE:
        return {
            "collision_mode": collision_mode,
            "disabled_geom_contacts": _disable_all_geom_contacts(root),
            "removed_contact_entries": _remove_contact_pairs(root),
        }
    if collision_mode == COLLISION_MODE_URDF_MESH_SCALED:
        stats = _replace_collision_meshes_with_scaled_urdf_meshes(
            root,
            source_model_path,
            mesh_scale=urdf_collision_mesh_scale,
        )
        stats["collision_mode"] = collision_mode
        return stats
    if collision_mode == COLLISION_MODE_INSET_BOX:
        stats = _replace_collision_meshes_with_inset_boxes(
            root,
            source_model_path,
            shrink=primitive_shrink,
            min_half_size=primitive_min_half_size,
        )
        stats["collision_mode"] = collision_mode
        return stats
    if collision_mode == COLLISION_MODE_INSET_CAPSULE:
        stats = _replace_collision_meshes_with_inset_capsules(
            root,
            source_model_path,
            length_quantile=capsule_length_quantile,
            radius_quantile=capsule_radius_quantile,
            radius_scale=capsule_radius_scale,
            min_radius=capsule_min_radius,
            min_half_length=capsule_min_half_length,
        )
        stats["collision_mode"] = collision_mode
        return stats
    raise ValueError(f"Unsupported collision mode: {collision_mode!r}")


def _body_name(model: mujoco.MjModel, body_id: int) -> str:
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(body_id)) or f"body_{body_id}"


def _geom_name(model: mujoco.MjModel, geom_id: int) -> str:
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(geom_id)) or f"geom_{geom_id}"


def _set_qpos_zero(data: mujoco.MjData) -> None:
    model = data.model
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    if model.nu:
        data.ctrl[:] = 0.0
    for joint_id in range(model.njnt):
        joint_type = int(model.jnt_type[joint_id])
        qpos_addr = int(model.jnt_qposadr[joint_id])
        if joint_type == int(mujoco.mjtJoint.mjJNT_FREE):
            data.qpos[qpos_addr + 3] = 1.0
        elif joint_type == int(mujoco.mjtJoint.mjJNT_BALL):
            data.qpos[qpos_addr] = 1.0


def _collect_qpos0_penetrations(
    model: mujoco.MjModel,
    *,
    margin: float,
) -> dict[str, object]:
    data = mujoco.MjData(model)
    _set_qpos_zero(data)
    try:
        mujoco.mj_forward(model, data)
    except mujoco.FatalError as exc:
        return {
            "qpos_definition": "all hinge/slide qpos set to 0; free/ball quaternions set to identity",
            "margin": float(margin),
            "fatal_error": str(exc),
            "total_contact_count": None,
            "penetrating_contact_count": None,
            "unique_body_pair_count": None,
            "min_dist": None,
            "max_penetration": None,
            "contacts": [],
            "body_pairs": [],
        }

    contacts: list[dict[str, object]] = []
    grouped: dict[tuple[str, str], dict[str, object]] = {}
    for contact_idx in range(int(data.ncon)):
        contact = data.contact[contact_idx]
        dist = float(contact.dist)
        if dist >= -float(margin):
            continue
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)
        body1 = int(model.geom_bodyid[geom1])
        body2 = int(model.geom_bodyid[geom2])
        if body1 == body2:
            continue
        body_name1 = _body_name(model, body1)
        body_name2 = _body_name(model, body2)
        geom_name1 = _geom_name(model, geom1)
        geom_name2 = _geom_name(model, geom2)
        penetration = max(0.0, -dist)
        contact_entry = {
            "contact_index": int(contact_idx),
            "body1": body_name1,
            "body2": body_name2,
            "geom1": geom_name1,
            "geom2": geom_name2,
            "dist": dist,
            "penetration": penetration,
        }
        contacts.append(contact_entry)

        body_pair = tuple(sorted((body_name1, body_name2)))
        group = grouped.setdefault(
            body_pair,
            {
                "body1": body_pair[0],
                "body2": body_pair[1],
                "min_dist": dist,
                "max_penetration": penetration,
                "contact_count": 0,
                "contacts": [],
            },
        )
        group["min_dist"] = min(float(group["min_dist"]), dist)
        group["max_penetration"] = max(float(group["max_penetration"]), penetration)
        group["contact_count"] = int(group["contact_count"]) + 1
        group_contacts = group["contacts"]
        assert isinstance(group_contacts, list)
        group_contacts.append(
            {
                "geom1": geom_name1,
                "geom2": geom_name2,
                "dist": dist,
                "penetration": penetration,
            }
        )

    contacts.sort(key=lambda item: float(item["dist"]))
    body_pairs = sorted(grouped.values(), key=lambda item: float(item["min_dist"]))
    return {
        "qpos_definition": "all hinge/slide qpos set to 0; free/ball quaternions set to identity",
        "margin": float(margin),
        "fatal_error": None,
        "total_contact_count": int(data.ncon),
        "penetrating_contact_count": len(contacts),
        "unique_body_pair_count": len(body_pairs),
        "min_dist": float(contacts[0]["dist"]) if contacts else None,
        "max_penetration": float(contacts[0]["penetration"]) if contacts else 0.0,
        "contacts": contacts,
        "body_pairs": body_pairs,
    }


def _write_qpos0_penetration_report(
    output_dir: Path,
    model: mujoco.MjModel,
    *,
    collision_mode: str,
    asm_variant: str,
    margin: float,
) -> tuple[Path, Path, dict[str, object]]:
    report = _collect_qpos0_penetrations(model, margin=margin)
    report.update(
        {
            "asm_variant": asm_variant,
            "collision_mode": collision_mode,
        }
    )
    json_path = output_dir / "qpos0_penetration_pairs.json"
    txt_path = output_dir / "qpos0_penetration_pairs.txt"
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    lines = [
        f"asm_variant: {asm_variant}",
        f"collision_mode: {collision_mode}",
        f"qpos_definition: {report['qpos_definition']}",
        f"margin: {report['margin']}",
        f"fatal_error: {report['fatal_error']}",
        f"total_contact_count: {report['total_contact_count']}",
        f"penetrating_contact_count: {report['penetrating_contact_count']}",
        f"unique_body_pair_count: {report['unique_body_pair_count']}",
        f"min_dist: {report['min_dist']}",
        f"max_penetration: {report['max_penetration']}",
        "",
        "body_pairs sorted by min_dist:",
    ]
    for pair in report.get("body_pairs", []):
        if not isinstance(pair, dict):
            continue
        lines.append(
            f"- {pair.get('body1')} <-> {pair.get('body2')}: "
            f"min_dist={pair.get('min_dist')}, "
            f"max_penetration={pair.get('max_penetration')}, "
            f"contact_count={pair.get('contact_count')}"
        )
    lines.extend(["", "contacts sorted by dist:"])
    for contact in report.get("contacts", []):
        if not isinstance(contact, dict):
            continue
        lines.append(
            f"- {contact.get('geom1')} ({contact.get('body1')}) <-> "
            f"{contact.get('geom2')} ({contact.get('body2')}): "
            f"dist={contact.get('dist')}, penetration={contact.get('penetration')}"
        )
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, txt_path, report


def _ensure_contact_element(root: ET.Element) -> ET.Element:
    contact = root.find("contact")
    if contact is not None:
        return contact
    contact = ET.Element("contact")
    worldbody = root.find("worldbody")
    insert_idx = list(root).index(worldbody) + 1 if worldbody is not None else len(root)
    root.insert(insert_idx, contact)
    return contact


def _add_initial_penetration_excludes(
    root: ET.Element,
    render_scene_path: Path,
    *,
    margin: float,
) -> dict[str, object]:
    if margin < 0.0:
        raise ValueError(f"initial penetration exclude margin must be >= 0, got {margin}")

    temp_xml_path = render_scene_path.with_name(
        f"{render_scene_path.stem}_pre_exclude{render_scene_path.suffix}"
    )
    temp_xml_path.write_text(
        ET.tostring(root, encoding="unicode"),
        encoding="utf-8",
    )
    model = mujoco.MjModel.from_xml_path(str(temp_xml_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    excluded_body_pairs: dict[tuple[str, str], dict[str, object]] = {}
    for contact_idx in range(int(data.ncon)):
        contact = data.contact[contact_idx]
        dist = float(contact.dist)
        if dist >= -float(margin):
            continue
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)
        body1 = int(model.geom_bodyid[geom1])
        body2 = int(model.geom_bodyid[geom2])
        if body1 == body2:
            continue
        body_name1 = _body_name(model, body1)
        body_name2 = _body_name(model, body2)
        body_pair = tuple(sorted((body_name1, body_name2)))
        geom_pair = {
            "geom1": _geom_name(model, geom1),
            "geom2": _geom_name(model, geom2),
            "dist": dist,
            "penetration": max(0.0, -dist),
        }
        entry = excluded_body_pairs.setdefault(
            body_pair,
            {
                "body1": body_pair[0],
                "body2": body_pair[1],
                "min_dist": dist,
                "max_penetration": max(0.0, -dist),
                "contacts": [],
            },
        )
        entry["min_dist"] = min(float(entry["min_dist"]), dist)
        entry["max_penetration"] = max(float(entry["max_penetration"]), max(0.0, -dist))
        contacts = entry["contacts"]
        assert isinstance(contacts, list)
        contacts.append(geom_pair)

    contact_element = _ensure_contact_element(root)
    existing = {
        tuple(sorted((child.get("body1", ""), child.get("body2", ""))))
        for child in contact_element.findall("exclude")
    }
    added = 0
    for body1, body2 in sorted(excluded_body_pairs):
        if (body1, body2) in existing:
            continue
        contact_element.append(
            ET.Element(
                "exclude",
                {
                    "body1": body1,
                    "body2": body2,
                },
            )
        )
        added += 1

    try:
        temp_xml_path.unlink()
    except OSError:
        pass

    pair_list = sorted(
        excluded_body_pairs.values(),
        key=lambda item: float(item["min_dist"]),
    )
    return {
        "enabled": True,
        "margin": float(margin),
        "initial_contact_count": int(data.ncon),
        "excluded_body_pair_count": len(excluded_body_pairs),
        "added_exclude_count": added,
        "pairs": pair_list,
    }


def _make_render_scene_xml(
    source_model_path: Path,
    output_dir: Path,
    *,
    front_fovy: float,
    d435_fovy: float,
    collision_mode: str,
    primitive_shrink: float,
    primitive_min_half_size: float,
    urdf_collision_mesh_scale: float,
    capsule_length_quantile: float,
    capsule_radius_quantile: float,
    capsule_radius_scale: float,
    capsule_min_radius: float,
    capsule_min_half_length: float,
    exclude_initial_penetrations: bool,
    initial_penetration_exclude_margin: float,
) -> tuple[Path, dict[str, object]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    render_scene_path = output_dir / "render_scene.xml"

    camera_pos, camera_rot = _camera_site_pose(source_model_path, D435_SITE_NAME)
    optical_x_world = camera_rot[:, 0]
    optical_y_world = camera_rot[:, 1]
    # MuJoCo camera convention uses +X to the right, +Y up, and looks along -Z.
    # D435 optical uses +X right, +Y down, +Z forward.
    d435_xaxis = optical_x_world
    d435_yaxis = -optical_y_world

    tree = ET.parse(source_model_path)
    root = tree.getroot()
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.Element("compiler")
        root.insert(0, compiler)
    meshdir = compiler.get("meshdir", "")
    meshdir_abs = (
        (source_model_path.parent / meshdir).resolve()
        if meshdir and not Path(meshdir).is_absolute()
        else Path(meshdir).resolve()
    )
    compiler.set("meshdir", str(meshdir_abs))

    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"Model {source_model_path} has no worldbody")
    for child in list(worldbody):
        if child.tag == "camera" and child.get("name") in {
            FRONT_CAMERA_NAME,
            SIDE_Y_CAMERA_NAME,
            D435_RENDER_CAMERA_NAME,
        }:
            worldbody.remove(child)
        if child.tag == "light" and child.get("name") == "verify_key_light":
            worldbody.remove(child)
        if child.tag == "geom" and child.get("name") == "verify_floor":
            worldbody.remove(child)

    collision_stats = _apply_collision_mode(
        root,
        source_model_path,
        collision_mode=collision_mode,
        primitive_shrink=primitive_shrink,
        primitive_min_half_size=primitive_min_half_size,
        urdf_collision_mesh_scale=urdf_collision_mesh_scale,
        capsule_length_quantile=capsule_length_quantile,
        capsule_radius_quantile=capsule_radius_quantile,
        capsule_radius_scale=capsule_radius_scale,
        capsule_min_radius=capsule_min_radius,
        capsule_min_half_length=capsule_min_half_length,
    )

    worldbody.insert(
        0,
        ET.Element(
            "light",
            {
                "name": "verify_key_light",
                "pos": "1.2 -1.0 3.0",
                "dir": "-0.4 0.3 -1",
                "diffuse": "0.8 0.8 0.8",
            },
        ),
    )
    worldbody.insert(
        1,
        ET.Element(
            "geom",
            {
                "name": "verify_floor",
                "type": "plane",
                "size": "0 0 0.05",
                "pos": "0 0 0",
                "rgba": "0.12 0.14 0.16 1",
                "contype": "0",
                "conaffinity": "0",
            },
        ),
    )
    worldbody.insert(
        2,
        ET.Element(
            "camera",
            {
                "name": "front",
                "pos": "3.15 0 1.45",
                "quat": "0.555478 0.437544 0.437544 0.555478",
                "fovy": f"{float(front_fovy):.6g}",
            },
        ),
    )
    worldbody.insert(
        3,
        ET.Element(
            "camera",
            {
                "name": SIDE_Y_CAMERA_NAME,
                "pos": "0 3.15 1.45",
                # +Y side view, looking toward the robot with the same downward
                # pitch as the front camera.
                "xyaxes": "-1 0 0 0 -0.234222 0.972183",
                "fovy": f"{float(front_fovy):.6g}",
            },
        ),
    )
    worldbody.insert(
        4,
        ET.Element(
            "camera",
            {
                "name": D435_RENDER_CAMERA_NAME,
                "pos": _format_vec(camera_pos),
                "xyaxes": f"{_format_vec(d435_xaxis)} {_format_vec(d435_yaxis)}",
                "fovy": f"{float(d435_fovy):.6g}",
            },
        ),
    )

    if exclude_initial_penetrations and collision_mode != COLLISION_MODE_NONE:
        collision_stats["initial_penetration_excludes"] = _add_initial_penetration_excludes(
            root,
            render_scene_path,
            margin=initial_penetration_exclude_margin,
        )
    else:
        collision_stats["initial_penetration_excludes"] = {
            "enabled": False,
            "margin": float(initial_penetration_exclude_margin),
            "initial_contact_count": None,
            "excluded_body_pair_count": 0,
            "added_exclude_count": 0,
            "pairs": [],
        }

    try:
        ET.indent(tree, space="  ")
    except AttributeError:
        pass
    tree.write(render_scene_path, encoding="unicode")
    return render_scene_path, collision_stats


def _validate_model_mapping(model: mujoco.MjModel, qpos: np.ndarray, ctrl: np.ndarray) -> None:
    if model.nq != 54 or model.nu != 54:
        raise ValueError(f"Expected ASM model nq=nu=54, got nq={model.nq}, nu={model.nu}")
    if qpos.shape[1] != model.nq:
        raise ValueError(f"Mapped qpos dim {qpos.shape[1]} does not match model.nq={model.nq}")
    if ctrl.shape[1] != model.nu:
        raise ValueError(f"Mapped ctrl dim {ctrl.shape[1]} does not match model.nu={model.nu}")

    expected_joint_prefixes = [
        "Joint1_R",
        "Joint2_R",
        "Joint3_R",
        "Joint4_R",
        "Joint5_R",
        "Joint6_R",
        "Joint7_R",
        "right_finger",
        "right_finger",
        "right_finger",
        "right_finger",
        "right_finger",
        "right_finger",
        "right_finger",
        "right_finger",
        "right_finger",
        "right_finger",
        "right_finger",
        "right_finger",
        "right_finger",
        "right_finger",
        "right_finger",
        "right_finger",
        "right_finger",
        "right_finger",
        "right_finger",
        "right_finger",
        "Joint1_L",
    ]
    for idx, prefix in enumerate(expected_joint_prefixes):
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, idx) or ""
        if not joint_name.startswith(prefix):
            raise ValueError(
                f"Unexpected ASM joint order at {idx}: {joint_name!r}, expected prefix {prefix!r}"
            )


def _clip_ctrl_to_range(model: mujoco.MjModel, ctrl: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    limited = model.actuator_ctrllimited.astype(bool)
    if not np.any(limited):
        return ctrl.copy(), {"ctrl_clip_max_violation": 0.0, "ctrl_clip_count": 0}
    low = model.actuator_ctrlrange[:, 0]
    high = model.actuator_ctrlrange[:, 1]
    clipped = ctrl.copy()
    below = clipped[:, limited] < low[limited]
    above = clipped[:, limited] > high[limited]
    violation = np.maximum(low[limited] - clipped[:, limited], clipped[:, limited] - high[limited])
    violation = np.maximum(violation, 0.0)
    clipped[:, limited] = np.clip(clipped[:, limited], low[limited], high[limited])
    return clipped, {
        "ctrl_clip_max_violation": float(violation.max()) if violation.size else 0.0,
        "ctrl_clip_count": int(np.count_nonzero(below | above)),
    }


def _qpos_range_stats(model: mujoco.MjModel, qpos: np.ndarray) -> dict[str, float]:
    limited = model.jnt_limited.astype(bool)
    joint_ids = np.arange(model.njnt)[limited]
    if joint_ids.size == 0:
        return {"qpos_range_max_violation": 0.0, "qpos_range_count": 0}
    qpos_ids = model.jnt_qposadr[joint_ids]
    low = model.jnt_range[joint_ids, 0]
    high = model.jnt_range[joint_ids, 1]
    values = qpos[:, qpos_ids]
    below = values < low
    above = values > high
    violation = np.maximum(low - values, values - high)
    violation = np.maximum(violation, 0.0)
    return {
        "qpos_range_max_violation": float(violation.max()) if violation.size else 0.0,
        "qpos_range_count": int(np.count_nonzero(below | above)),
    }


def _render_frame(
    renderer: mujoco.Renderer,
    data: mujoco.MjData,
    camera_name: str,
    *,
    hide_helpers: bool,
) -> np.ndarray:
    options = mujoco.MjvOption()
    mujoco.mjv_defaultOption(options)
    if hide_helpers:
        options.geomgroup[3] = False
        options.sitegroup[3] = False
        options.sitegroup[4] = False
    mujoco.mj_forward(data.model, data)
    renderer.update_scene(data, camera_name, options)
    return renderer.render()


def _render_collision_debug_frame(
    renderer: mujoco.Renderer,
    data: mujoco.MjData,
    camera_name: str,
) -> np.ndarray:
    options = mujoco.MjvOption()
    mujoco.mjv_defaultOption(options)
    for group_idx in range(len(options.geomgroup)):
        options.geomgroup[group_idx] = False
    options.geomgroup[0] = True
    options.geomgroup[3] = True
    for group_idx in range(len(options.sitegroup)):
        options.sitegroup[group_idx] = False
    mujoco.mj_forward(data.model, data)
    renderer.update_scene(data, camera_name, options)
    return renderer.render()


def _write_collision_proxy_debug_video(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    output_dir: Path,
    *,
    collision_mode: str,
    fps: int,
    width: int,
    height: int,
    progress_interval: int,
) -> dict[str, Path]:
    model.vis.global_.offwidth = int(width)
    model.vis.global_.offheight = int(height)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=height, width=width)
    video_paths = {
        "front": output_dir / f"{collision_mode}_collision_front.mp4",
        "side_y": output_dir / f"{collision_mode}_collision_side_y.mp4",
    }
    total_frames = len(qpos)
    start_time = time.perf_counter()
    _log(
        f"Writing {collision_mode} collision debug videos: {total_frames} frames, "
        f"{width}x{height}, fps={fps}, views={list(video_paths)}"
    )
    with ExitStack() as stack:
        front_writer = stack.enter_context(imageio.get_writer(video_paths["front"], fps=fps))
        side_y_writer = stack.enter_context(imageio.get_writer(video_paths["side_y"], fps=fps))
        for frame_idx, frame_qpos in enumerate(qpos):
            data.qpos[:] = frame_qpos
            data.qvel[:] = 0.0
            front_writer.append_data(
                _render_collision_debug_frame(renderer, data, FRONT_CAMERA_NAME)
            )
            side_y_writer.append_data(
                _render_collision_debug_frame(renderer, data, SIDE_Y_CAMERA_NAME)
            )
            if _should_log_frame(frame_idx, total_frames, progress_interval):
                _log_progress(f"{collision_mode}_collision", frame_idx, total_frames, start_time)
    _log(
        f"Finished {collision_mode} collision debug videos: "
        f"{video_paths['front']}, {video_paths['side_y']}"
    )
    return video_paths


def _write_link_capsule_manifest(
    output_dir: Path,
    collision_stats: dict[str, object],
) -> Path | None:
    capsules = collision_stats.get("capsules")
    if not isinstance(capsules, list) or not capsules:
        return None

    grouped: dict[str, dict[str, object]] = {}
    for capsule in capsules:
        if not isinstance(capsule, dict):
            continue
        link_key = str(capsule.get("link_key", "unknown"))
        entry = grouped.setdefault(
            link_key,
            {
                "link_key": link_key,
                "link_label": capsule.get("link_label", link_key),
                "color_hex": capsule.get("color_hex", "#FFFFFF"),
                "rgba": capsule.get("rgba", [1.0, 1.0, 1.0, 0.85]),
                "component": capsule.get("component", "unknown"),
                "capsules": [],
            },
        )
        entry_capsules = entry["capsules"]
        assert isinstance(entry_capsules, list)
        entry_capsules.append(
            {
                "side": capsule.get("side", "none"),
                "geom": capsule.get("geom", ""),
                "mesh": capsule.get("mesh", ""),
                "radius": capsule.get("radius", 0.0),
                "length": capsule.get("length", 0.0),
                "fromto": capsule.get("fromto", []),
                "degenerate": capsule.get("degenerate", False),
            }
        )

    manifest = {
        "collision_mode": collision_stats.get("collision_mode"),
        "color_rule": (
            "Capsules with the same semantic link_key use the same color; "
            "left/right arm Link_1..Link_7 therefore share colors."
        ),
        "link_colors": list(grouped.values()),
        "capsules": capsules,
    }
    manifest_path = output_dir / "link_capsule_colors.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def _write_link_collision_manifest(
    output_dir: Path,
    collision_stats: dict[str, object],
) -> Path | None:
    entries = collision_stats.get("collision_meshes")
    entry_type = "urdf_collision_mesh"
    if not isinstance(entries, list) or not entries:
        entries = collision_stats.get("capsules")
        entry_type = "capsule"
    if not isinstance(entries, list) or not entries:
        return None

    grouped: dict[str, dict[str, object]] = {}
    for entry_data in entries:
        if not isinstance(entry_data, dict):
            continue
        link_key = str(entry_data.get("link_key", "unknown"))
        group = grouped.setdefault(
            link_key,
            {
                "link_key": link_key,
                "link_label": entry_data.get("link_label", link_key),
                "color_hex": entry_data.get("color_hex", "#FFFFFF"),
                "rgba": entry_data.get("rgba", [1.0, 1.0, 1.0, 0.85]),
                "component": entry_data.get("component", "unknown"),
                "entries": [],
            },
        )
        group_entries = group["entries"]
        assert isinstance(group_entries, list)
        group_entries.append(entry_data)

    manifest = {
        "collision_mode": collision_stats.get("collision_mode"),
        "entry_type": entry_type,
        "color_rule": (
            "Collision geoms with the same semantic link_key use the same color; "
            "left/right arm Link_1..Link_7 therefore share colors."
        ),
        "link_colors": list(grouped.values()),
        "entries": entries,
    }
    manifest_path = output_dir / "link_collision_colors.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def _write_initial_penetration_exclude_manifest(
    output_dir: Path,
    collision_stats: dict[str, object],
) -> Path | None:
    excludes = collision_stats.get("initial_penetration_excludes")
    if not isinstance(excludes, dict) or not excludes.get("enabled"):
        return None

    manifest = {
        "collision_mode": collision_stats.get("collision_mode"),
        "description": (
            "Body pairs excluded because they were already penetrating at model "
            "default qpos after applying the selected collision proxy mode."
        ),
        "exclude_type": "mujoco_contact_exclude_body_pair",
        "margin": excludes.get("margin", 0.0),
        "initial_contact_count": excludes.get("initial_contact_count", 0),
        "excluded_body_pair_count": excludes.get("excluded_body_pair_count", 0),
        "added_exclude_count": excludes.get("added_exclude_count", 0),
        "pairs": excludes.get("pairs", []),
    }
    manifest_path = output_dir / "initial_penetration_exclude_pairs.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def _write_kinematic_videos(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    output_dir: Path,
    *,
    fps: int,
    width: int,
    height: int,
    hide_helpers: bool,
    progress_interval: int,
    render_front: bool,
    render_head: bool,
) -> None:
    if not render_front and not render_head:
        _log("Skipping kinematic videos: front=false, head=false")
        return

    model.vis.global_.offwidth = int(width)
    model.vis.global_.offheight = int(height)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=height, width=width)
    front_path = output_dir / "kinematic_front.mp4"
    d435_path = output_dir / "kinematic_d435.mp4"
    total_frames = len(qpos)
    start_time = time.perf_counter()
    _log(
        f"Writing kinematic videos: {total_frames} frames, "
        f"{width}x{height}, fps={fps}, front={render_front}, head={render_head}"
    )
    with ExitStack() as stack:
        front_writer = (
            stack.enter_context(imageio.get_writer(front_path, fps=fps))
            if render_front
            else None
        )
        d435_writer = (
            stack.enter_context(imageio.get_writer(d435_path, fps=fps))
            if render_head
            else None
        )
        for frame_idx, frame_qpos in enumerate(qpos):
            data.qpos[:] = frame_qpos
            data.qvel[:] = 0.0
            if front_writer is not None:
                front_writer.append_data(
                    _render_frame(renderer, data, "front", hide_helpers=hide_helpers)
                )
            if d435_writer is not None:
                d435_writer.append_data(
                    _render_frame(renderer, data, "d435_optical_render", hide_helpers=hide_helpers)
                )
            if _should_log_frame(frame_idx, total_frames, progress_interval):
                _log_progress("kinematic", frame_idx, total_frames, start_time)
    finished = []
    if render_front:
        finished.append(str(front_path))
    if render_head:
        finished.append(str(d435_path))
    _log(f"Finished kinematic videos: {', '.join(finished)}")


def _write_dynamic_videos(
    model: mujoco.MjModel,
    initial_qpos: np.ndarray,
    ctrl: np.ndarray,
    output_dir: Path,
    *,
    fps: int,
    substeps_per_frame: int,
    width: int,
    height: int,
    hide_helpers: bool,
    progress_interval: int,
    render_front: bool,
    render_head: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not render_front and not render_head:
        _log("Skipping dynamic videos and dynamic rollout: front=false, head=false")
        return (
            np.empty((0, model.nq), dtype=np.float64),
            np.empty((0, model.nv), dtype=np.float64),
            np.empty((0, model.nu), dtype=np.float64),
            np.empty((0,), dtype=np.float64),
        )

    model.vis.global_.offwidth = int(width)
    model.vis.global_.offheight = int(height)
    data = mujoco.MjData(model)
    data.qpos[:] = initial_qpos
    data.qvel[:] = 0.0
    data.ctrl[:] = ctrl[0]
    data.time = 0.0
    mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(model, height=height, width=width)
    front_path = output_dir / "dynamic_front.mp4"
    d435_path = output_dir / "dynamic_d435.mp4"
    qpos_out = []
    qvel_out = []
    ctrl_out = []
    time_out = []
    total_frames = len(ctrl)
    start_time = time.perf_counter()
    _log(
        f"Writing dynamic videos: {total_frames} frames, "
        f"{substeps_per_frame} MuJoCo steps/frame, {width}x{height}, fps={fps}, "
        f"front={render_front}, head={render_head}"
    )

    with ExitStack() as stack:
        front_writer = (
            stack.enter_context(imageio.get_writer(front_path, fps=fps))
            if render_front
            else None
        )
        d435_writer = (
            stack.enter_context(imageio.get_writer(d435_path, fps=fps))
            if render_head
            else None
        )
        for frame_idx in range(len(ctrl)):
            if frame_idx > 0:
                data.ctrl[:] = ctrl[frame_idx - 1]
                for _ in range(substeps_per_frame):
                    mujoco.mj_step(model, data)
            qpos_out.append(data.qpos.copy())
            qvel_out.append(data.qvel.copy())
            ctrl_out.append(data.ctrl.copy())
            time_out.append(float(data.time))
            if front_writer is not None:
                front_writer.append_data(
                    _render_frame(renderer, data, "front", hide_helpers=hide_helpers)
                )
            if d435_writer is not None:
                d435_writer.append_data(
                    _render_frame(renderer, data, "d435_optical_render", hide_helpers=hide_helpers)
                )
            if _should_log_frame(frame_idx, total_frames, progress_interval):
                _log_progress("dynamic", frame_idx, total_frames, start_time)

    finished = []
    if render_front:
        finished.append(str(front_path))
    if render_head:
        finished.append(str(d435_path))
    _log(f"Finished dynamic videos: {', '.join(finished)}")
    return (
        np.asarray(qpos_out, dtype=np.float64),
        np.asarray(qvel_out, dtype=np.float64),
        np.asarray(ctrl_out, dtype=np.float64),
        np.asarray(time_out, dtype=np.float64),
    )


def _should_log_frame(frame_idx: int, total_frames: int, progress_interval: int) -> bool:
    if total_frames <= 0:
        return False
    if frame_idx == 0 or frame_idx == total_frames - 1:
        return True
    return progress_interval > 0 and (frame_idx + 1) % progress_interval == 0


def _log_progress(
    stage: str,
    frame_idx: int,
    total_frames: int,
    start_time: float,
) -> None:
    done = frame_idx + 1
    elapsed = max(time.perf_counter() - start_time, 1e-9)
    fps = done / elapsed
    remaining = max(total_frames - done, 0)
    eta = remaining / fps if fps > 0 else 0.0
    _log(
        f"{stage}: frame {done}/{total_frames} "
        f"({100.0 * done / total_frames:.1f}%), "
        f"{fps:.2f} frames/s, eta {eta:.1f}s"
    )


def _write_summary(
    output_dir: Path,
    *,
    episode_dir: Path,
    parquet_path: Path,
    source_model_path: Path,
    render_scene_path: Path,
    df: pd.DataFrame,
    qpos: np.ndarray,
    ctrl: np.ndarray,
    qpos_stats: dict[str, float],
    ctrl_stats: dict[str, float],
    fps: int,
    sim_dt: float,
    substeps_per_frame: int,
    render_kinematic_front: bool,
    render_kinematic_head: bool,
    render_dynamic_front: bool,
    render_dynamic_head: bool,
    dynamic_rollout_saved: bool,
    collision_stats: dict[str, object],
    collision_debug_videos: dict[str, Path] | None,
    link_capsule_manifest: Path | None,
    link_collision_manifest: Path | None,
    initial_penetration_exclude_manifest: Path | None,
    qpos0_penetration_json: Path | None,
    qpos0_penetration_txt: Path | None,
    qpos0_penetration_stats: dict[str, object] | None,
    asm_variant: str,
    source_urdf_path: Path | None,
) -> None:
    first_frame_mapping = {
        "parquet_order": "left_arm(7) + left_hand(20) + right_arm(7) + right_hand(20)",
        "asm_order": "right_arm(7) + right_hand(20) + left_arm(7) + left_hand(20)",
        "qpos_first_10": qpos[0, :10].tolist(),
        "ctrl_first_10": ctrl[0, :10].tolist(),
    }
    summary = {
        "episode_dir": str(episode_dir),
        "parquet_path": str(parquet_path),
        "source_model_path": str(source_model_path),
        "asm_variant": asm_variant,
        "source_urdf_path": str(source_urdf_path) if source_urdf_path else None,
        "render_scene_path": str(render_scene_path),
        "num_frames": int(len(df)),
        "fps": int(fps),
        "timestamp_start": float(df["timestamp"].iloc[0]),
        "timestamp_end": float(df["timestamp"].iloc[-1]),
        "qpos_shape": list(qpos.shape),
        "ctrl_shape": list(ctrl.shape),
        "qpos_range_stats": qpos_stats,
        "ctrl_clip_stats": ctrl_stats,
        "sim_dt": float(sim_dt),
        "substeps_per_frame": int(substeps_per_frame),
        "render_options": {
            "render_kinematic_front": bool(render_kinematic_front),
            "render_kinematic_head": bool(render_kinematic_head),
            "render_dynamic_front": bool(render_dynamic_front),
            "render_dynamic_head": bool(render_dynamic_head),
        },
        "videos": {
            "kinematic_front": (
                str(output_dir / "kinematic_front.mp4") if render_kinematic_front else None
            ),
            "kinematic_d435": (
                str(output_dir / "kinematic_d435.mp4") if render_kinematic_head else None
            ),
            "dynamic_front": (
                str(output_dir / "dynamic_front.mp4") if render_dynamic_front else None
            ),
            "dynamic_d435": (
                str(output_dir / "dynamic_d435.mp4") if render_dynamic_head else None
            ),
        },
        "dynamic_rollout": (
            str(output_dir / "dynamic_rollout.npz") if dynamic_rollout_saved else None
        ),
        "collision": collision_stats,
        "collision_debug_video": (
            str(collision_debug_videos.get("front"))
            if collision_debug_videos and collision_debug_videos.get("front")
            else None
        ),
        "collision_debug_videos": (
            {name: str(path) for name, path in collision_debug_videos.items()}
            if collision_debug_videos
            else {}
        ),
        "link_capsule_manifest": (
            str(link_capsule_manifest) if link_capsule_manifest else None
        ),
        "link_collision_manifest": (
            str(link_collision_manifest) if link_collision_manifest else None
        ),
        "initial_penetration_exclude_manifest": (
            str(initial_penetration_exclude_manifest)
            if initial_penetration_exclude_manifest
            else None
        ),
        "qpos0_penetration_report": {
            "json": str(qpos0_penetration_json) if qpos0_penetration_json else None,
            "txt": str(qpos0_penetration_txt) if qpos0_penetration_txt else None,
            "stats": qpos0_penetration_stats,
        },
        "mapping": first_frame_mapping,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render Pick-Place parquet trajectories on ASM."
    )
    parser.add_argument(
        "--episode-dir",
        default=str(DEFAULT_EPISODE_DIR),
        help="Episode directory containing timeseries.parquet.",
    )
    parser.add_argument(
        "--asm-variant",
        choices=ASM_VARIANTS,
        default=ASM_VARIANT_DEFAULT,
        help=(
            "ASM model variant. asm uses the existing processed bimanual.xml by "
            "default; asm_2/asm_3/asm_4 generate a temporary MJCF from the matching "
            "URDF under spider/assets/robots/asm_description/urdf."
        ),
    )
    parser.add_argument(
        "--source-urdf",
        default="",
        help="Optional source URDF override used when generating an ASM variant model.",
    )
    parser.add_argument(
        "--model-path",
        default="",
        help=(
            "ASM bimanual MJCF path. If omitted, asm uses the existing processed "
            "model and asm_2 is generated from --source-urdf."
        ),
    )
    parser.add_argument(
        "--prepare-collision-mesh-scale",
        type=float,
        default=1.0,
        help=(
            "Collision mesh scale used only while generating the temporary MJCF "
            "for generated ASM variants. The render-time collision mode can "
            "still apply its own --urdf-collision-mesh-scale."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help=(
            "Output directory. Defaults to <episode-dir>/verify for urdf_mesh, "
            "or <episode-dir>/verify/collision_<mode> for other collision modes."
        ),
    )
    parser.add_argument("--max-frames", type=int, default=-1, help="Debug frame limit.")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--sim-dt", type=float, default=1.0 / 300.0)
    parser.add_argument("--front-fovy", type=float, default=45.0)
    parser.add_argument("--d435-fovy", type=float, default=50.0)
    parser.add_argument(
        "--collision-mode",
        choices=COLLISION_MODES,
        default=COLLISION_MODE_URDF_MESH,
        help=(
            "Robot collision handling: urdf_mesh keeps current URDF collision meshes; "
            "none disables all contacts; urdf_mesh_scaled uses colored, scaled copies "
            "of the URDF collision meshes; inset_box replaces collision_* mesh geoms "
            "with small inset box proxies; inset_capsule replaces them with "
            "PCA-fitted inset capsule proxies."
        ),
    )
    parser.add_argument(
        "--urdf-collision-mesh-scale",
        type=float,
        default=0.7,
        help="Mesh scale used by --collision-mode urdf_mesh_scaled.",
    )
    parser.add_argument(
        "--primitive-collision-shrink",
        type=float,
        default=0.65,
        help="Shrink factor for --collision-mode inset_box primitive half extents.",
    )
    parser.add_argument(
        "--primitive-collision-min-half-size",
        type=float,
        default=0.003,
        help="Minimum box half size in meters for --collision-mode inset_box.",
    )
    parser.add_argument(
        "--capsule-length-quantile",
        type=float,
        default=0.80,
        help="Central mesh projection fraction used for --collision-mode inset_capsule.",
    )
    parser.add_argument(
        "--capsule-radius-quantile",
        type=float,
        default=0.35,
        help="Radial distance quantile used for --collision-mode inset_capsule.",
    )
    parser.add_argument(
        "--capsule-radius-scale",
        type=float,
        default=0.75,
        help="Extra radius shrink factor used for --collision-mode inset_capsule.",
    )
    parser.add_argument(
        "--capsule-min-radius",
        type=float,
        default=0.002,
        help="Minimum capsule radius in meters for --collision-mode inset_capsule.",
    )
    parser.add_argument(
        "--capsule-min-half-length",
        type=float,
        default=0.003,
        help="Minimum capsule half length in meters for --collision-mode inset_capsule.",
    )
    parser.add_argument(
        "--render-collision-debug",
        dest="render_collision_debug",
        type=lambda value: value.lower() == "true",
        default=True,
        help=(
            "When using urdf_mesh_scaled or primitive collision modes, render a "
            "front-view colored collision debug video."
        ),
    )
    parser.add_argument(
        "--render-inset-box-debug",
        dest="render_collision_debug",
        type=lambda value: value.lower() == "true",
        default=argparse.SUPPRESS,
        help="Backward-compatible alias for --render-collision-debug.",
    )
    parser.add_argument(
        "--exclude-initial-penetrations",
        type=lambda value: value.lower() == "true",
        default=False,
        help=(
            "After applying the selected collision mode, detect contacts at model "
            "default qpos and add body-level excludes for penetrating pairs."
        ),
    )
    parser.add_argument(
        "--initial-penetration-exclude-margin",
        type=float,
        default=1e-6,
        help="Exclude pairs whose default-qpos contact dist is below -margin.",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=50,
        help="Print render progress every N frames. Use 0 to only print first/last frame.",
    )
    parser.add_argument(
        "--render-kinematic-front",
        type=lambda value: value.lower() == "true",
        default=True,
        help="Whether to render kinematic_front.mp4.",
    )
    parser.add_argument(
        "--render-kinematic-head",
        type=lambda value: value.lower() == "true",
        default=True,
        help="Whether to render kinematic_d435.mp4 from the ASM head D435 camera.",
    )
    parser.add_argument(
        "--render-dynamic-front",
        type=lambda value: value.lower() == "true",
        default=True,
        help="Whether to render dynamic_front.mp4 after MuJoCo dynamics.",
    )
    parser.add_argument(
        "--render-dynamic-head",
        type=lambda value: value.lower() == "true",
        default=True,
        help="Whether to render dynamic_d435.mp4 from the ASM head D435 camera after MuJoCo dynamics.",
    )
    parser.add_argument(
        "--show-helpers",
        action="store_true",
        help="Show helper sites/collision groups if present.",
    )
    args = parser.parse_args()

    _log("Starting Pick-Place parquet ASM render.")
    _log(f"MUJOCO_GL={os.environ.get('MUJOCO_GL')}")
    _log(f"PYOPENGL_PLATFORM={os.environ.get('PYOPENGL_PLATFORM')}")
    _log(
        "Render options: "
        f"asm_variant={args.asm_variant}, "
        f"kinematic_front={args.render_kinematic_front}, "
        f"kinematic_head={args.render_kinematic_head}, "
        f"dynamic_front={args.render_dynamic_front}, "
        f"dynamic_head={args.render_dynamic_head}"
    )
    _log(
        "Collision options: "
        f"collision_mode={args.collision_mode}, "
        f"urdf_collision_mesh_scale={args.urdf_collision_mesh_scale}, "
        f"primitive_collision_shrink={args.primitive_collision_shrink}, "
        f"primitive_collision_min_half_size={args.primitive_collision_min_half_size}, "
        f"capsule_length_quantile={args.capsule_length_quantile}, "
        f"capsule_radius_quantile={args.capsule_radius_quantile}, "
        f"capsule_radius_scale={args.capsule_radius_scale}, "
        f"capsule_min_radius={args.capsule_min_radius}, "
        f"capsule_min_half_length={args.capsule_min_half_length}, "
        f"render_collision_debug={args.render_collision_debug}, "
        f"exclude_initial_penetrations={args.exclude_initial_penetrations}, "
        f"initial_penetration_exclude_margin={args.initial_penetration_exclude_margin}"
    )
    episode_dir = _resolve_path(args.episode_dir)
    parquet_path = episode_dir / "timeseries.parquet"
    if args.output_dir:
        output_dir = _resolve_path(args.output_dir)
    else:
        output_dir = _default_output_dir(
            episode_dir,
            collision_mode=args.collision_mode,
            asm_variant=args.asm_variant,
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    source_urdf_path: Path | None = None
    if args.model_path:
        model_path = _resolve_path(args.model_path)
    elif args.asm_variant == ASM_VARIANT_DEFAULT:
        model_path = DEFAULT_MODEL_PATH
    else:
        source_urdf_path = _resolve_source_urdf(args.asm_variant, args.source_urdf)
        model_path = _prepare_variant_model(
            output_dir,
            asm_variant=args.asm_variant,
            source_urdf=source_urdf_path,
            prepare_collision_mesh_scale=args.prepare_collision_mesh_scale,
        )
    _log(f"episode_dir={episode_dir}")
    _log(f"parquet_path={parquet_path}")
    _log(f"asm_variant={args.asm_variant}")
    if source_urdf_path is not None:
        _log(f"source_urdf={source_urdf_path}")
    _log(f"model_path={model_path}")
    _log(f"output_dir={output_dir}")
    if not parquet_path.is_file():
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")
    if not model_path.is_file():
        raise FileNotFoundError(f"ASM model not found: {model_path}")

    _log("Loading parquet and remapping joint order...")
    df, qpos_kinematic, ctrl_dynamic = _load_parquet_trajectories(parquet_path)
    _log(
        f"Loaded parquet: rows={len(df)}, columns={len(df.columns)}, "
        f"qpos_shape={qpos_kinematic.shape}, ctrl_shape={ctrl_dynamic.shape}"
    )
    if args.max_frames > 0:
        _log(f"Applying --max-frames={args.max_frames}")
        df = df.iloc[: args.max_frames].copy()
        qpos_kinematic = qpos_kinematic[: args.max_frames]
        ctrl_dynamic = ctrl_dynamic[: args.max_frames]
        _log(
            f"After frame limit: rows={len(df)}, "
            f"qpos_shape={qpos_kinematic.shape}, ctrl_shape={ctrl_dynamic.shape}"
        )

    _log("Creating temporary render scene XML...")
    render_scene_path, collision_stats = _make_render_scene_xml(
        model_path,
        output_dir,
        front_fovy=args.front_fovy,
        d435_fovy=args.d435_fovy,
        collision_mode=args.collision_mode,
        primitive_shrink=args.primitive_collision_shrink,
        primitive_min_half_size=args.primitive_collision_min_half_size,
        urdf_collision_mesh_scale=args.urdf_collision_mesh_scale,
        capsule_length_quantile=args.capsule_length_quantile,
        capsule_radius_quantile=args.capsule_radius_quantile,
        capsule_radius_scale=args.capsule_radius_scale,
        capsule_min_radius=args.capsule_min_radius,
        capsule_min_half_length=args.capsule_min_half_length,
        exclude_initial_penetrations=args.exclude_initial_penetrations,
        initial_penetration_exclude_margin=args.initial_penetration_exclude_margin,
    )
    _log(f"Render scene XML written: {render_scene_path}")
    collision_stats_for_log = {
        key: value
        for key, value in collision_stats.items()
        if key not in {"capsules", "collision_meshes"}
    }
    _log(f"Collision stats: {collision_stats_for_log}")
    link_capsule_manifest = _write_link_capsule_manifest(output_dir, collision_stats)
    if link_capsule_manifest is not None:
        _log(f"Link-capsule color manifest written: {link_capsule_manifest}")
    link_collision_manifest = _write_link_collision_manifest(output_dir, collision_stats)
    if link_collision_manifest is not None:
        _log(f"Link-collision color manifest written: {link_collision_manifest}")
    initial_penetration_exclude_manifest = _write_initial_penetration_exclude_manifest(
        output_dir,
        collision_stats,
    )
    if initial_penetration_exclude_manifest is not None:
        _log(
            "Initial penetration exclude manifest written: "
            f"{initial_penetration_exclude_manifest}"
        )
    _log("Loading MuJoCo model...")
    model = mujoco.MjModel.from_xml_path(str(render_scene_path))
    model.opt.timestep = float(args.sim_dt)
    _log(f"MuJoCo model loaded: nq={model.nq}, nv={model.nv}, nu={model.nu}")
    qpos0_json, qpos0_txt, qpos0_stats = _write_qpos0_penetration_report(
        output_dir,
        model,
        collision_mode=args.collision_mode,
        asm_variant=args.asm_variant,
        margin=args.initial_penetration_exclude_margin,
    )
    qpos0_log_stats = {
        key: qpos0_stats.get(key)
        for key in (
            "fatal_error",
            "total_contact_count",
            "penetrating_contact_count",
            "unique_body_pair_count",
            "min_dist",
            "max_penetration",
        )
    }
    _log(f"qpos=0 penetration report written: {qpos0_json}, {qpos0_txt}")
    _log(f"qpos=0 penetration stats: {qpos0_log_stats}")
    _log("Validating parquet-to-ASM joint mapping...")
    _validate_model_mapping(model, qpos_kinematic, ctrl_dynamic)
    _log("Joint mapping validated.")

    qpos_stats = _qpos_range_stats(model, qpos_kinematic)
    ctrl_dynamic, ctrl_stats = _clip_ctrl_to_range(model, ctrl_dynamic)
    _log(f"qpos range stats: {qpos_stats}")
    _log(f"ctrl clip stats: {ctrl_stats}")
    substeps_per_frame = max(1, int(round((1.0 / args.fps) / args.sim_dt)))
    model.opt.timestep = 1.0 / (args.fps * substeps_per_frame)
    _log(
        f"Timing: fps={args.fps}, requested_sim_dt={args.sim_dt:.9g}, "
        f"actual_sim_dt={model.opt.timestep:.9g}, substeps_per_frame={substeps_per_frame}"
    )

    _write_kinematic_videos(
        model,
        qpos_kinematic,
        output_dir,
        fps=args.fps,
        width=args.width,
        height=args.height,
        hide_helpers=not args.show_helpers,
        progress_interval=args.progress_interval,
        render_front=args.render_kinematic_front,
        render_head=args.render_kinematic_head,
    )
    collision_debug_videos = None
    if (
        args.collision_mode
        in {
            COLLISION_MODE_URDF_MESH_SCALED,
            COLLISION_MODE_INSET_BOX,
            COLLISION_MODE_INSET_CAPSULE,
        }
        and args.render_collision_debug
    ):
        collision_debug_videos = _write_collision_proxy_debug_video(
            model,
            qpos_kinematic,
            output_dir,
            collision_mode=args.collision_mode,
            fps=args.fps,
            width=args.width,
            height=args.height,
            progress_interval=args.progress_interval,
        )
    render_any_dynamic = args.render_dynamic_front or args.render_dynamic_head
    if render_any_dynamic:
        dynamic_qpos, dynamic_qvel, dynamic_ctrl, dynamic_time = _write_dynamic_videos(
            model,
            qpos_kinematic[0],
            ctrl_dynamic,
            output_dir,
            fps=args.fps,
            substeps_per_frame=substeps_per_frame,
            width=args.width,
            height=args.height,
            hide_helpers=not args.show_helpers,
            progress_interval=args.progress_interval,
            render_front=args.render_dynamic_front,
            render_head=args.render_dynamic_head,
        )
        _log("Saving dynamic_rollout.npz...")
        np.savez(
            output_dir / "dynamic_rollout.npz",
            qpos=dynamic_qpos,
            qvel=dynamic_qvel,
            ctrl=dynamic_ctrl,
            time=dynamic_time,
            source_parquet=str(parquet_path),
            source_model=str(model_path),
            asm_variant=str(args.asm_variant),
            source_urdf=str(source_urdf_path) if source_urdf_path else "",
            collision_mode=str(args.collision_mode),
        )
    else:
        _log("Skipping dynamic rollout: dynamic_front=false and dynamic_head=false")
    _log("Saving summary.json...")
    _write_summary(
        output_dir,
        episode_dir=episode_dir,
        parquet_path=parquet_path,
        source_model_path=model_path,
        render_scene_path=render_scene_path,
        df=df,
        qpos=qpos_kinematic,
        ctrl=ctrl_dynamic,
        qpos_stats=qpos_stats,
        ctrl_stats=ctrl_stats,
        fps=args.fps,
        sim_dt=model.opt.timestep,
        substeps_per_frame=substeps_per_frame,
        render_kinematic_front=args.render_kinematic_front,
        render_kinematic_head=args.render_kinematic_head,
        render_dynamic_front=args.render_dynamic_front,
        render_dynamic_head=args.render_dynamic_head,
        dynamic_rollout_saved=render_any_dynamic,
        collision_stats=collision_stats,
        collision_debug_videos=collision_debug_videos,
        link_capsule_manifest=link_capsule_manifest,
        link_collision_manifest=link_collision_manifest,
        initial_penetration_exclude_manifest=initial_penetration_exclude_manifest,
        qpos0_penetration_json=qpos0_json,
        qpos0_penetration_txt=qpos0_txt,
        qpos0_penetration_stats=qpos0_stats,
        asm_variant=args.asm_variant,
        source_urdf_path=source_urdf_path,
    )

    _log(f"Saved verification videos and metadata to {output_dir}")
    if args.render_kinematic_front:
        _log(f"  {output_dir / 'kinematic_front.mp4'}")
    if args.render_kinematic_head:
        _log(f"  {output_dir / 'kinematic_d435.mp4'}")
    if args.render_dynamic_front:
        _log(f"  {output_dir / 'dynamic_front.mp4'}")
    if args.render_dynamic_head:
        _log(f"  {output_dir / 'dynamic_d435.mp4'}")
    if collision_debug_videos is not None:
        for path in collision_debug_videos.values():
            _log(f"  {path}")
    if link_capsule_manifest is not None:
        _log(f"  {link_capsule_manifest}")
    if link_collision_manifest is not None:
        _log(f"  {link_collision_manifest}")
    if initial_penetration_exclude_manifest is not None:
        _log(f"  {initial_penetration_exclude_manifest}")
    _log(f"  {qpos0_json}")
    _log(f"  {qpos0_txt}")
    _log(f"  {output_dir / 'summary.json'}")
    if render_any_dynamic:
        _log(f"  {output_dir / 'dynamic_rollout.npz'}")


if __name__ == "__main__":
    main()
