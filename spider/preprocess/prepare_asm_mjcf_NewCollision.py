# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Prepare ASM robot MJCF assets for SPIDER.

The ASM source model is a ROS-style URDF package with package:// mesh paths.
SPIDER's MJWP pipeline expects robot assets under the processed dataset tree as
MuJoCo XML files named right.xml, left.xml, and bimanual.xml. This script converts
asm.urdf to MJCF, converts the URDF collision meshes into stable primitive
collision proxies by default, and adds the SPIDER-specific sites and position
actuators needed by generate_xml.py, ik.py, and run_mjwp.py.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import re
import shutil
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

from spider import ROOT

SUPPORTED_SOURCE_MESH_SUFFIXES = {".stl", ".obj", ".dae"}
MUJOCO_MESH_SUFFIXES = {".stl", ".obj"}

FINGER_MAP = {
    "thumb": "finger1",
    "index": "finger2",
    "middle": "finger3",
    "ring": "finger4",
    "pinky": "finger5",
}

GROUND_ASSETS = [
    ("texture", {"type": "skybox", "builtin": "gradient", "rgb1": "1 1 1", "rgb2": "1 1 1", "width": "512", "height": "3072"}),
    ("texture", {"type": "2d", "name": "right_groundplane", "builtin": "checker", "mark": "edge", "rgb1": "0.2 0.3 0.4", "rgb2": "0.1 0.2 0.3", "markrgb": "0.8 0.8 0.8", "width": "300", "height": "300"}),
    ("texture", {"type": "2d", "name": "left_groundplane", "builtin": "checker", "mark": "edge", "rgb1": "0.2 0.3 0.4", "rgb2": "0.1 0.2 0.3", "markrgb": "0.8 0.8 0.8", "width": "300", "height": "300"}),
    ("material", {"name": "right_groundplane", "texture": "right_groundplane", "texuniform": "true", "texrepeat": "5 5", "reflectance": "0.2"}),
    ("material", {"name": "left_groundplane", "texture": "left_groundplane", "texuniform": "true", "texrepeat": "5 5", "reflectance": "0.2"}),
]

ASM_ROOT_NAME = "asm_root"
ASM_ROOT_YAW_DEG = 0.0
HEAD_CAMERA_SITE_NAME = "head_camera_frame"
D435_OPTICAL_FRAME_SITE_NAME = "d435_optical_frame"
# The D435 mesh/visual frame currently appears as +X left, +Y up, +Z forward.
# Mega-SAM/OpenCV uses +X right, +Y down, +Z forward, so flip X/Y only.
D435_OPTICAL_FROM_VISUAL_ROT = np.diag([-1.0, -1.0, 1.0])
URDF_COLLISION_PREFIXES = ("collision_hand_", "collision_arm_", "collision_body_")
ASM_FINGER_TO_NAME = {value: key for key, value in FINGER_MAP.items()}
COLLISION_GEOMETRY_MODES = ("primitive", "urdf_mesh")
DEFAULT_COLLISION_GEOMETRY_MODE = "primitive"
COLLISION_COLOR_ALPHA = 0.45
COLLISION_COLOR_HEX = {
    "Link_1": "#e6194b",
    "Link_2": "#3cb44b",
    "Link_3": "#ffe119",
    "Link_4": "#4363d8",
    "Link_5": "#f58231",
    "Link_6": "#911eb4",
    "Link_7": "#46f0f0",
    "Palm": "#f032e6",
    "thumb_link1": "#bcf60c",
    "thumb_link2": "#fabebe",
    "thumb_link3": "#008080",
    "thumb_tip_link": "#e6beff",
    "index_link1": "#9a6324",
    "index_link2": "#fffac8",
    "index_link3": "#800000",
    "index_tip_link": "#aaffc3",
    "middle_link1": "#808000",
    "middle_link2": "#ffd8b1",
    "middle_link3": "#000075",
    "middle_tip_link": "#808080",
    "ring_link1": "#ff6f69",
    "ring_link2": "#88d8b0",
    "ring_link3": "#ffcc5c",
    "ring_tip_link": "#6b5b95",
    "pinky_link1": "#2a9d8f",
    "pinky_link2": "#e76f51",
    "pinky_link3": "#264653",
    "pinky_tip_link": "#f4a261",
    "Body_head": "#8ecae6",
    "Body_camera": "#219ebc",
    "Body_other": "#adb5bd",
}
FALLBACK_COLOR_HEX = (
    "#ff595e",
    "#ffca3a",
    "#8ac926",
    "#1982c4",
    "#6a4c93",
    "#00bbf9",
    "#00f5d4",
    "#f15bb5",
)
ARM_CAPSULE_RADIUS_AT_SCALE_1_BY_LINK = {
    1: 0.036,
    2: 0.036,
    3: 0.032,
    4: 0.032,
    5: 0.030,
    6: 0.026,
    7: 0.026,
}
ARM_CAPSULE_FALLBACK_LENGTH_BY_LINK = {
    1: 0.12,
    2: 0.24,
    3: 0.12,
    4: 0.24,
    5: 0.14,
    6: 0.08,
    7: 0.12,
}
ARM_CAPSULE_TRIM_MARGIN_M = 0.015
ARM_CAPSULE_MIN_HALF_LENGTH_M = 0.018
ARM_CAPSULE_CHILD_VECTOR_MIN_LENGTH_M = 0.06
ARM_CAPSULE_SCALE_MIN = 0.35
ARM_CAPSULE_SCALE_MAX = 1.30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        default=f"{ROOT}/../example_datasets",
        help="Dataset root containing processed/<dataset_name>.",
    )
    parser.add_argument("--dataset-name", default="oakink")
    parser.add_argument("--robot-type", default="asm")
    parser.add_argument(
        "--source-urdf",
        default=f"{ROOT}/assets/robots/asm_description/urdf/asm.urdf",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["bimanual", "right", "left"],
        choices=["bimanual", "right", "left"],
    )
    parser.add_argument("--arm-kp", type=float, default=300.0)
    parser.add_argument("--hand-kp", type=float, default=180.0)
    parser.add_argument("--arm-damping", type=float, default=2.0)
    parser.add_argument("--hand-damping", type=float, default=0.5)
    parser.add_argument("--arm-armature", type=float, default=0.05)
    parser.add_argument("--hand-armature", type=float, default=0.02)
    parser.add_argument("--arm-frictionloss", type=float, default=0.0)
    parser.add_argument("--hand-frictionloss", type=float, default=0.01)
    parser.add_argument("--arm-force-scale", type=float, default=1.0)
    parser.add_argument("--hand-force-scale", type=float, default=2.0)
    parser.add_argument(
        "--collision-mesh-scale",
        type=float,
        default=0.7,
        help=(
            "Collision proxy shrink/scale factor. In primitive mode, this shrinks "
            "generated capsule/box proxies. In urdf_mesh mode, active URDF collision "
            "meshes are duplicated and scaled around compiled mesh bbox centers."
        ),
    )
    parser.add_argument(
        "--collision-geometry-mode",
        choices=COLLISION_GEOMETRY_MODES,
        default=DEFAULT_COLLISION_GEOMETRY_MODE,
        help=(
            "Robot collision geometry mode. 'primitive' generates capsule/box "
            "collision proxies from the URDF collision meshes. 'urdf_mesh' keeps "
            "the URDF collision meshes active."
        ),
    )
    parser.add_argument(
        "--root-yaw-deg",
        type=float,
        default=ASM_ROOT_YAW_DEG,
        help=(
            "Yaw rotation, in degrees, applied to the wrapped ASM root body. "
            "Use 90 for scenes where the ASM front should face +Y."
        ),
    )
    return parser.parse_args()


def fmt(value: float) -> str:
    return f"{value:.6g}"


def fmt_vec(values: np.ndarray) -> str:
    return " ".join(fmt(float(value)) for value in np.asarray(values).reshape(-1))


def yaw_quat_str(yaw_deg: float) -> str:
    half = math.radians(yaw_deg) * 0.5
    return f"{math.cos(half):.8f} 0 0 {math.sin(half):.8f}"


def quat_str_from_matrix(rotation_matrix: np.ndarray) -> str:
    quat_xyzw = R.from_matrix(rotation_matrix).as_quat()
    quat_wxyz = np.array(
        [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]],
        dtype=np.float64,
    )
    return " ".join(f"{value:.8f}" for value in quat_wxyz)


def parse_xyz_rpy(origin_elem: ET.Element | None) -> tuple[np.ndarray, np.ndarray]:
    if origin_elem is None:
        return np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.float64)
    xyz = np.fromstring(origin_elem.get("xyz", "0 0 0"), sep=" ", dtype=np.float64)
    rpy = np.fromstring(origin_elem.get("rpy", "0 0 0"), sep=" ", dtype=np.float64)
    if xyz.shape != (3,) or rpy.shape != (3,):
        raise ValueError(f"invalid xyz/rpy origin element: {ET.tostring(origin_elem, encoding='unicode')}")
    return xyz, rpy


def pose_from_origin(origin_elem: ET.Element | None) -> np.ndarray:
    xyz, rpy = parse_xyz_rpy(origin_elem)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = R.from_euler("xyz", rpy).as_matrix()
    pose[:3, 3] = xyz
    return pose


def find_joint(root: ET.Element, name: str) -> ET.Element:
    for joint in root.findall("joint"):
        if joint.get("name") == name:
            return joint
    raise ValueError(f"joint {name} not found in source URDF")


def compute_head_camera_pose(source_urdf: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return the legacy D435 visual/mesh frame pose from the URDF."""
    urdf_root = ET.parse(source_urdf).getroot()
    neck_joint = find_joint(urdf_root, "neck_joint")
    d435_joint = find_joint(urdf_root, "joint_d435")
    d435_link = None
    for link in urdf_root.findall("link"):
        if link.get("name") == "link_435":
            d435_link = link
            break
    if d435_link is None:
        raise ValueError("link_435 not found in source URDF")
    d435_visual = d435_link.find("visual")
    if d435_visual is None:
        raise ValueError("link_435 has no visual origin to use as the camera frame")

    base_to_neck = pose_from_origin(neck_joint.find("origin"))
    neck_to_link435 = pose_from_origin(d435_joint.find("origin"))
    link435_to_camera = pose_from_origin(d435_visual.find("origin"))
    base_to_camera = base_to_neck @ neck_to_link435 @ link435_to_camera
    return base_to_camera[:3, 3], base_to_camera[:3, :3]


def compute_d435_optical_rotation_from_visual(source_urdf: Path) -> np.ndarray:
    """Return optical axes expressed in the legacy visual frame."""
    _ = source_urdf
    return D435_OPTICAL_FROM_VISUAL_ROT.copy()


def compute_d435_optical_pose(source_urdf: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return the D435 optical frame pose used for world->sim alignment."""
    visual_pos, visual_rot = compute_head_camera_pose(source_urdf)
    # `optical_from_visual` stores optical axes in the visual frame, so the
    # world/base rotation is obtained by right-multiplying the legacy visual
    # frame rotation.
    optical_from_visual = compute_d435_optical_rotation_from_visual(source_urdf)
    optical_rot = visual_rot @ optical_from_visual
    return visual_pos, optical_rot


def _load_mesh_with_fallbacks(src: Path):
    try:
        import trimesh

        mesh = trimesh.load(str(src), force="mesh")
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.to_mesh()
        if mesh is not None and len(mesh.vertices) > 0:
            return mesh
    except Exception:
        pass

    try:
        import open3d as o3d
        import trimesh

        mesh = o3d.io.read_triangle_mesh(str(src))
        if mesh is not None and len(mesh.vertices) > 0 and len(mesh.triangles) > 0:
            return trimesh.Trimesh(
                vertices=np.asarray(mesh.vertices),
                faces=np.asarray(mesh.triangles),
                process=False,
            )
    except Exception:
        pass

    raise ValueError(
        f"Failed to load mesh '{src}' as a MuJoCo-compatible triangle mesh. "
        "Install trimesh+pycollada or provide an OBJ/STL version."
    )


def _convert_mesh_for_mujoco(src: Path, dst: Path) -> None:
    mesh = _load_mesh_with_fallbacks(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(dst)


def copy_flat_meshes(source_root: Path, output_mesh_dir: Path) -> dict[str, str]:
    output_mesh_dir.mkdir(parents=True, exist_ok=True)
    seen: dict[str, Path] = {}
    filename_map: dict[str, str] = {}
    for src in sorted((source_root / "meshes").rglob("*")):
        if not src.is_file() or src.suffix.lower() not in SUPPORTED_SOURCE_MESH_SUFFIXES:
            continue
        out_name = src.name
        if src.suffix.lower() == ".dae":
            out_name = f"{src.stem}.obj"
        dst = output_mesh_dir / out_name
        if out_name in seen and seen[out_name] != src:
            raise ValueError(f"Duplicate ASM mesh basename after conversion: {out_name}")
        seen[out_name] = src
        if src.suffix.lower() in MUJOCO_MESH_SUFFIXES:
            shutil.copy2(src, dst)
        else:
            _convert_mesh_for_mujoco(src, dst)
        filename_map[src.name] = out_name
    return filename_map


def make_compile_urdf(
    source_urdf: Path,
    mesh_dir: Path,
    output_urdf: Path,
    filename_map: dict[str, str],
) -> None:
    tree = ET.parse(source_urdf)
    root = tree.getroot()

    for existing in list(root.findall("mujoco")):
        root.remove(existing)
    mujoco_elem = ET.Element("mujoco")
    ET.SubElement(
        mujoco_elem,
        "compiler",
        {
            "meshdir": str(mesh_dir),
            "balanceinertia": "true",
            "discardvisual": "false",
        },
    )
    root.insert(0, mujoco_elem)

    for link in root.findall("link"):
        for tag in ("visual", "collision"):
            for elem in list(link.findall(tag)):
                mesh = elem.find("./geometry/mesh")
                if mesh is None:
                    continue
                filename = mesh.get("filename", "")
                suffix = Path(filename).suffix.lower()
                if suffix not in SUPPORTED_SOURCE_MESH_SUFFIXES:
                    link.remove(elem)
                    continue
                basename = Path(filename).name
                if basename not in filename_map:
                    link.remove(elem)
                    continue
                mesh.set("filename", filename_map[basename])

    output_urdf.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_urdf, encoding="unicode")


def compile_urdf_to_xml(source_urdf: Path, output_xml: Path) -> ET.Element:
    model = mujoco.MjModel.from_xml_path(str(source_urdf))
    mujoco.mj_saveLastXML(str(output_xml), model)
    return ET.parse(output_xml).getroot()


def mesh_vertices_by_name(model_path: Path) -> dict[str, np.ndarray]:
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
        vertices[name] = model.mesh_vert[vert_adr : vert_adr + vert_num].copy()
    return vertices


def parse_vec_attr(
    element: ET.Element,
    name: str,
    dim: int,
    default: np.ndarray,
) -> np.ndarray:
    raw = element.get(name)
    if not raw:
        return default.astype(np.float64, copy=True)
    values = np.fromstring(raw, sep=" ", dtype=np.float64)
    if values.shape != (dim,):
        raise ValueError(
            f"{element.tag} {element.get('name', '')!r} has invalid {name}={raw!r}"
        )
    return values


def parse_mesh_scale(mesh_element: ET.Element) -> np.ndarray:
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


def quat_wxyz_to_matrix(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    norm = np.linalg.norm(quat)
    if norm <= 0:
        return np.eye(3)
    quat = quat / norm
    return R.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()


def child_names(parent: ET.Element, tag: str) -> set[str]:
    return {child.get("name", "") for child in parent.findall(tag)}


def ensure_child(parent: ET.Element, tag: str, attrs: dict[str, str]) -> ET.Element:
    name = attrs.get("name")
    if name:
        for child in parent.findall(tag):
            if child.get("name") == name:
                return child
    child = ET.SubElement(parent, tag, attrs)
    return child


def find_first(root: ET.Element, tag: str) -> ET.Element:
    elem = root.find(tag)
    if elem is None:
        elem = ET.Element(tag)
        root.insert(1, elem)
    return elem


def iter_bodies(root: ET.Element):
    worldbody = root.find("worldbody")
    if worldbody is None:
        return
    yield from worldbody.iter("body")


def find_body(root: ET.Element, name: str) -> ET.Element | None:
    for body in iter_bodies(root):
        if body.get("name") == name:
            return body
    return None


def add_default_and_assets(root: ET.Element) -> None:
    compiler = root.find("compiler")
    if compiler is not None:
        compiler.set("meshdir", "meshes")
        compiler.set("angle", "radian")

    option = root.find("option")
    if option is None:
        option = ET.Element("option")
        insert_idx = 1 if compiler is not None else 0
        root.insert(insert_idx, option)
    flag = option.find("flag")
    if flag is None:
        flag = ET.SubElement(option, "flag")
    flag.set("filterparent", "disable")

    if root.find("default") is None:
        default = ET.Element("default")
        default.extend(
            [
                ET.Element("geom", {"density": "800", "condim": "1", "contype": "0", "conaffinity": "0"}),
                ET.Element("joint", {"damping": "0", "armature": "0.01", "frictionloss": "0"}),
                ET.Element("position", {"kp": "120", "dampratio": "1", "inheritrange": "1"}),
                ET.Element("site", {"size": "0.01", "type": "sphere", "rgba": "1 0 0 1", "group": "3"}),
            ]
        )
        insert_idx = list(root).index(option) + 1
        root.insert(insert_idx, default)

    asset = find_first(root, "asset")
    existing = child_names(asset, "texture") | child_names(asset, "material")
    for tag, attrs in GROUND_ASSETS:
        name = attrs.get("name")
        if name and name in existing:
            continue
        asset.insert(0, ET.Element(tag, attrs))
        if name:
            existing.add(name)


def _is_mesh_geom(geom: ET.Element) -> bool:
    return geom.get("type") == "mesh" or "mesh" in geom.attrib


def _is_explicitly_non_colliding_geom(geom: ET.Element) -> bool:
    return geom.get("contype") == "0" and geom.get("conaffinity") == "0"


def _is_urdf_collision_geom(geom: ET.Element) -> bool:
    # MuJoCo's URDF importer emits visual geoms with explicit 0/0 contact
    # masks. Collision geoms may be meshes or primitive cylinders/boxes/spheres,
    # and their contact masks are active or omitted. We intentionally key off
    # the imported per-geom attributes, not the <default> values added later.
    return not _is_explicitly_non_colliding_geom(geom)


def _sanitize_name(value: str) -> str:
    sanitized = re.sub(r"[^0-9A-Za-z]+", "_", value).strip("_").lower()
    return sanitized or "unnamed"


def _side_from_name(value: str) -> str | None:
    if value.startswith("right_") or value.endswith("_R"):
        return "right"
    if value.startswith("left_") or value.endswith("_L"):
        return "left"
    return None


def _hand_detail_from_mesh(mesh_name: str, side: str) -> str:
    detail = mesh_name
    if detail.startswith(f"{side}_"):
        detail = detail[len(side) + 1 :]
    for asm_finger, finger_name in ASM_FINGER_TO_NAME.items():
        detail = detail.replace(asm_finger, finger_name)
    return _sanitize_name(detail)


def _collision_name_base(body_name: str, mesh_name: str) -> str:
    side = _side_from_name(body_name) or _side_from_name(mesh_name)
    if side and ("finger" in mesh_name or "palm" in mesh_name):
        return f"collision_hand_{side}_{_hand_detail_from_mesh(mesh_name, side)}"

    if re.fullmatch(r"Link[0-9]+_[RL]", body_name):
        side = "right" if body_name.endswith("_R") else "left"
        return f"collision_arm_{side}_{_sanitize_name(body_name)}_{_sanitize_name(mesh_name)}"

    return f"collision_body_{_sanitize_name(body_name)}_{_sanitize_name(mesh_name)}"


def _hex_to_rgba_values(color_hex: str, alpha: float = COLLISION_COLOR_ALPHA) -> list[float]:
    value = color_hex.strip().lstrip("#")
    if len(value) != 6:
        raise ValueError(f"Expected 6-digit hex color, got {color_hex!r}")
    return [
        int(value[0:2], 16) / 255.0,
        int(value[2:4], 16) / 255.0,
        int(value[4:6], 16) / 255.0,
        float(alpha),
    ]


def _rgba_string(values: list[float]) -> str:
    return " ".join(fmt(float(value)) for value in values)


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

    arm_match = re.search(r"collision_arm_(left|right)_link([0-9]+)", geom_name)
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
            r"collision_hand_(left|right)_(thumb|index|middle|ring|pinky)_(link[0-9]+|tip_link)",
            geom_name,
        )
        if finger_match:
            side = finger_match.group(1)
            component = "hand"
            link_key = f"{finger_match.group(2)}_{finger_match.group(3)}"
            link_label = link_key
        elif geom_name.startswith("collision_body_"):
            component = "body"
            if "d435" in geom_name or "camera" in geom_name:
                link_key = "Body_camera"
                link_label = "Camera"
            elif "head" in geom_name:
                link_key = "Body_head"
                link_label = "Head"
            else:
                link_key = "Body_other"
                link_label = mesh_name or geom_name.replace("collision_body_", "Body_")

    color_hex = COLLISION_COLOR_HEX.get(link_key, _fallback_color_hex(link_key))
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


def activate_urdf_collision_geoms(root: ET.Element) -> None:
    """Name URDF collision geoms as active collision candidates.

    The URDF contains both visual and collision entries. We keep the imported
    visual geoms non-colliding and make every imported collision geom explicit
    and named, including mesh collisions and primitive cylinder/box/sphere
    collisions. Later, mesh candidates are either kept as mesh collisions or
    replaced by generated primitive proxies; imported primitive collisions stay
    as their original primitive type.
    """

    name_counts: dict[str, int] = {}
    collision_count = 0
    for body in iter_bodies(root):
        body_name = body.get("name", "body")
        for geom in body.findall("geom"):
            if _is_urdf_collision_geom(geom):
                geom_label = geom.get("mesh") or geom.get("type") or "geom"
                name_base = _collision_name_base(body_name, geom_label)
                name_idx = name_counts.get(name_base, 0)
                name_counts[name_base] = name_idx + 1
                geom.set("name", f"{name_base}_{name_idx}")
                geom.set("contype", "1")
                geom.set("conaffinity", "1")
                geom.set("condim", geom.get("condim", "3"))
                geom.set("density", "0")
                geom.set("group", "3")
                geom.set(
                    "rgba",
                    str(_collision_proxy_metadata(geom.get("name", ""), geom_label)["rgba_string"]),
                )
                collision_count += 1
            else:
                geom.set("contype", "0")
                geom.set("conaffinity", "0")
                geom.set("density", "0")
                geom.set("group", geom.get("group", "1"))
    if collision_count == 0:
        raise RuntimeError("No active URDF collision geoms found after URDF import")


def scale_urdf_collision_meshes(
    root: ET.Element,
    mesh_vertices: dict[str, np.ndarray],
    collision_mesh_scale: float,
) -> dict[str, int | float]:
    if collision_mesh_scale <= 0.0:
        raise ValueError(
            f"collision_mesh_scale must be positive, got {collision_mesh_scale}"
        )

    asset = root.find("asset")
    if asset is None:
        raise RuntimeError("Compiled ASM MJCF has no <asset> section")
    mesh_assets = {
        mesh.get("name"): mesh
        for mesh in asset.findall("mesh")
        if mesh.get("name")
    }

    converted = 0
    skipped = 0
    non_mesh_active = 0
    created_mesh_assets: set[str] = set()
    for geom in root.iter("geom"):
        geom_name = geom.get("name", "")
        if not geom_name.startswith(URDF_COLLISION_PREFIXES):
            continue
        mesh_name = geom.get("mesh")
        if not mesh_name:
            non_mesh_active += 1
            continue
        source_mesh = mesh_assets.get(mesh_name)
        vertices = mesh_vertices.get(mesh_name)
        if source_mesh is None or vertices is None or len(vertices) == 0:
            skipped += 1
            continue

        scaled_mesh_name = f"{mesh_name}_collision_scaled"
        if scaled_mesh_name not in mesh_assets:
            scaled_mesh = copy.deepcopy(source_mesh)
            scaled_mesh.set("name", scaled_mesh_name)
            base_scale = parse_mesh_scale(source_mesh)
            scaled_mesh.set(
                "scale",
                fmt_vec(base_scale * float(collision_mesh_scale)),
            )
            asset.append(scaled_mesh)
            mesh_assets[scaled_mesh_name] = scaled_mesh
            created_mesh_assets.add(scaled_mesh_name)

        lower = vertices.min(axis=0)
        upper = vertices.max(axis=0)
        center = 0.5 * (lower + upper)
        original_pos = parse_vec_attr(geom, "pos", 3, np.zeros(3, dtype=np.float64))
        original_quat = parse_vec_attr(
            geom,
            "quat",
            4,
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        )
        original_rot = quat_wxyz_to_matrix(original_quat)
        scaled_pos = (
            original_pos
            + original_rot @ ((1.0 - float(collision_mesh_scale)) * center)
        )

        geom.set("mesh", scaled_mesh_name)
        geom.set("pos", fmt_vec(scaled_pos))
        converted += 1

    return {
        "collision_mesh_scale": float(collision_mesh_scale),
        "converted_collision_mesh_geoms": converted,
        "skipped_collision_mesh_geoms": skipped,
        "non_mesh_active_collision_geoms": non_mesh_active,
        "created_scaled_mesh_assets": len(created_mesh_assets),
    }


def _collision_proxy_kind(geom_name: str) -> str:
    if geom_name.startswith("collision_body_"):
        return "body_box"
    if geom_name.startswith("collision_hand_") and "_palm" in geom_name:
        return "palm_box"
    if geom_name.startswith("collision_hand_"):
        return "finger_capsule"
    if geom_name.startswith("collision_arm_"):
        return "arm_capsule"
    return "capsule"


def _pca_frame(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = vertices.mean(axis=0)
    centered = vertices - center
    if vertices.shape[0] >= 3 and np.linalg.norm(centered) > 0:
        try:
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
            axes = vh.T
            if np.linalg.det(axes) < 0:
                axes[:, -1] *= -1.0
            variances = centered.var(axis=0)
            if np.isfinite(axes).all():
                return center, axes, variances
        except np.linalg.LinAlgError:
            pass
    axes = np.eye(3, dtype=np.float64)
    variances = centered.var(axis=0) if vertices.size else np.zeros(3, dtype=np.float64)
    return center, axes, variances


def _capsule_axis_from_vertices(vertices: np.ndarray) -> np.ndarray:
    _, axes, _ = _pca_frame(vertices)
    axis = axes[:, 0]
    if np.isfinite(axis).all() and np.linalg.norm(axis) > 0:
        return axis / np.linalg.norm(axis)
    extents = vertices.max(axis=0) - vertices.min(axis=0)
    axis = np.zeros(3, dtype=np.float64)
    axis[int(np.argmax(extents))] = 1.0
    return axis


def _arm_link_index_from_geom_name(geom_name: str) -> int | None:
    match = re.search(r"collision_arm_(left|right)_link([0-9]+)", geom_name)
    if not match:
        return None
    return int(match.group(2))


def _find_body_containing_geom(root: ET.Element, target_geom: ET.Element) -> ET.Element | None:
    target_name = target_geom.get("name")
    for body in iter_bodies(root):
        for geom in body.findall("geom"):
            if geom is target_geom:
                return body
            if target_name and geom.get("name") == target_name:
                return body
    return None


def _parse_body_pos(body: ET.Element | None) -> np.ndarray:
    if body is None:
        return np.zeros(3, dtype=np.float64)
    return parse_vec_attr(body, "pos", 3, np.zeros(3, dtype=np.float64))


def _same_side_arm_child_vector(body: ET.Element | None) -> tuple[ET.Element | None, np.ndarray | None]:
    if body is None:
        return None, None
    body_name = body.get("name", "")
    match = re.fullmatch(r"Link([0-9]+)_([RL])", body_name)
    if not match:
        return None, None
    suffix = f"_{match.group(2)}"
    for child in body.findall("body"):
        child_name = child.get("name", "")
        if not re.fullmatch(r"Link[0-9]+_[RL]", child_name):
            continue
        if not child_name.endswith(suffix):
            continue
        vector = _parse_body_pos(child)
        if np.linalg.norm(vector) >= ARM_CAPSULE_CHILD_VECTOR_MIN_LENGTH_M:
            return child, vector
    return None, None


def _terminal_hand_vector(body: ET.Element | None) -> tuple[str | None, np.ndarray | None]:
    if body is None:
        return None, None
    candidates: list[np.ndarray] = []
    names: list[str] = []
    for geom in body.findall("geom"):
        mesh_name = geom.get("mesh", "")
        geom_name = geom.get("name", "")
        if "palm" not in mesh_name and "palm" not in geom_name:
            continue
        vector = parse_vec_attr(geom, "pos", 3, np.zeros(3, dtype=np.float64))
        if np.linalg.norm(vector) >= ARM_CAPSULE_CHILD_VECTOR_MIN_LENGTH_M:
            candidates.append(vector)
            names.append(geom_name or mesh_name or "palm_geom")
    for child in body.findall("body"):
        child_name = child.get("name", "")
        if "finger" not in child_name:
            continue
        vector = _parse_body_pos(child)
        if np.linalg.norm(vector) >= ARM_CAPSULE_CHILD_VECTOR_MIN_LENGTH_M:
            candidates.append(vector)
            names.append(child_name)
    if not candidates:
        return None, None
    return "+".join(names[:4]), np.mean(np.stack(candidates, axis=0), axis=0)


def _mesh_capsule_frame_in_body(
    geom: ET.Element,
    vertices: np.ndarray,
    *,
    length_quantile: float = 0.86,
) -> tuple[np.ndarray, np.ndarray, float]:
    axis_mesh = _capsule_axis_from_vertices(vertices)
    line_center = vertices.mean(axis=0)
    centered = vertices - line_center
    projections = centered @ axis_mesh
    low_q = 0.5 * (1.0 - float(length_quantile))
    high_q = 1.0 - low_q
    lower_proj = float(np.quantile(projections, low_q))
    upper_proj = float(np.quantile(projections, high_q))
    midpoint_proj = 0.5 * (lower_proj + upper_proj)
    center_mesh = line_center + midpoint_proj * axis_mesh
    original_pos = parse_vec_attr(geom, "pos", 3, np.zeros(3, dtype=np.float64))
    original_quat = parse_vec_attr(
        geom,
        "quat",
        4,
        np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
    )
    original_rot = quat_wxyz_to_matrix(original_quat)
    axis_body = original_rot @ axis_mesh
    axis_norm = np.linalg.norm(axis_body)
    if axis_norm <= 0:
        axis_body = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        axis_body = axis_body / axis_norm
    center_body = original_pos + original_rot @ center_mesh
    return center_body, axis_body, max(0.0, upper_proj - lower_proj)


def _arm_capsule_length_scale(proxy_scale: float) -> float:
    return float(
        np.clip(
            math.sqrt(max(float(proxy_scale), 1e-6)),
            ARM_CAPSULE_SCALE_MIN,
            ARM_CAPSULE_SCALE_MAX,
        )
    )


def _set_geom_as_capsule(
    geom: ET.Element,
    endpoint_a_body: np.ndarray,
    endpoint_b_body: np.ndarray,
    radius: float,
) -> None:
    for attr in ("mesh", "pos", "quat", "euler", "axisangle", "xyaxes", "zaxis"):
        geom.attrib.pop(attr, None)
    geom.set("type", "capsule")
    geom.set("fromto", fmt_vec(np.concatenate([endpoint_a_body, endpoint_b_body])))
    geom.set("size", fmt_vec(np.array([radius], dtype=np.float64)))
    _set_collision_geom_common_attrs(geom)


def _set_collision_geom_common_attrs(geom: ET.Element) -> None:
    geom.set("contype", geom.get("contype", "1"))
    geom.set("conaffinity", geom.get("conaffinity", "1"))
    geom.set("condim", geom.get("condim", "3"))
    geom.set("density", "0")
    geom.set("group", geom.get("group", "3"))
    geom.set("rgba", geom.get("rgba", "0 0.8 1 0.28"))


def _disable_collision_proxy(geom: ET.Element) -> None:
    geom.set("contype", "0")
    geom.set("conaffinity", "0")
    geom.set("density", "0")
    geom.set("group", "6")
    geom.set("rgba", "0 0 0 0")


def _matrix_to_quat_wxyz(rotation_matrix: np.ndarray) -> np.ndarray:
    quat_xyzw = R.from_matrix(rotation_matrix).as_quat()
    return np.array(
        [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]],
        dtype=np.float64,
    )


def _convert_collision_geom_to_box(
    geom: ET.Element,
    vertices: np.ndarray,
    proxy_scale: float,
    min_half_size: float,
    *,
    max_half_extent: float | None = None,
) -> dict[str, object]:
    center, axes, _ = _pca_frame(vertices)
    local = (vertices - center) @ axes
    lower = local.min(axis=0)
    upper = local.max(axis=0)
    local_center = 0.5 * (lower + upper)
    half_extents = 0.5 * (upper - lower) * float(proxy_scale)
    half_extents = np.maximum(half_extents, float(min_half_size))
    if max_half_extent is not None:
        half_extents = np.minimum(half_extents, float(max_half_extent))
    center = center + axes @ local_center

    original_pos = parse_vec_attr(geom, "pos", 3, np.zeros(3, dtype=np.float64))
    original_quat = parse_vec_attr(
        geom,
        "quat",
        4,
        np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
    )
    original_rot = quat_wxyz_to_matrix(original_quat)
    box_pos = original_pos + original_rot @ center
    box_rot = original_rot @ axes
    box_quat = _matrix_to_quat_wxyz(box_rot)

    for attr in ("mesh", "fromto", "euler", "axisangle", "xyaxes", "zaxis"):
        geom.attrib.pop(attr, None)
    geom.set("type", "box")
    geom.set("size", fmt_vec(half_extents))
    geom.set("pos", fmt_vec(box_pos))
    geom.set("quat", fmt_vec(box_quat))
    _set_collision_geom_common_attrs(geom)
    return {
        "proxy_type": "box",
        "center": box_pos.tolist(),
        "half_extents": half_extents.tolist(),
        "axis": box_rot[:, 0].tolist(),
        "reason": "oriented_bbox",
    }


def _convert_collision_geom_to_capsule(
    geom: ET.Element,
    vertices: np.ndarray,
    proxy_scale: float,
    *,
    length_quantile: float,
    radius_quantile: float,
    radius_scale: float,
    min_radius: float,
    min_half_length: float,
    max_radius: float | None = None,
    fallback_aspect_ratio: float = 1.25,
) -> dict[str, object]:
    axis = _capsule_axis_from_vertices(vertices)
    line_center = vertices.mean(axis=0)
    centered = vertices - line_center
    projections = centered @ axis
    low_q = 0.5 * (1.0 - float(length_quantile))
    high_q = 1.0 - low_q
    lower_proj = float(np.quantile(projections, low_q))
    upper_proj = float(np.quantile(projections, high_q))
    half_length = 0.5 * (upper_proj - lower_proj) * float(proxy_scale)
    midpoint_proj = 0.5 * (lower_proj + upper_proj)
    degenerate = half_length < float(min_half_length)
    if degenerate:
        half_length = float(min_half_length)

    in_middle = (projections >= lower_proj) & (projections <= upper_proj)
    selected_centered = centered[in_middle] if np.any(in_middle) else centered
    selected_projections = projections[in_middle] if np.any(in_middle) else projections
    radial = selected_centered - np.outer(selected_projections, axis)
    radial_dist = np.linalg.norm(radial, axis=1)
    radius = float(np.quantile(radial_dist, radius_quantile))
    radius *= float(radius_scale) * float(proxy_scale)
    radius = max(radius, float(min_radius))
    if max_radius is not None:
        radius = min(radius, float(max_radius))

    if half_length / max(radius, 1e-9) < fallback_aspect_ratio:
        return _convert_collision_geom_to_box(
            geom,
            vertices,
            proxy_scale,
            min_half_size=max(min_radius, 0.001),
            max_half_extent=max_radius,
        )

    endpoint_a = line_center + (midpoint_proj - half_length) * axis
    endpoint_b = line_center + (midpoint_proj + half_length) * axis
    original_pos = parse_vec_attr(geom, "pos", 3, np.zeros(3, dtype=np.float64))
    original_quat = parse_vec_attr(
        geom,
        "quat",
        4,
        np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
    )
    original_rot = quat_wxyz_to_matrix(original_quat)
    endpoint_a_body = original_pos + original_rot @ endpoint_a
    endpoint_b_body = original_pos + original_rot @ endpoint_b

    for attr in ("mesh", "pos", "quat", "euler", "axisangle", "xyaxes", "zaxis"):
        geom.attrib.pop(attr, None)
    geom.set("type", "capsule")
    geom.set("fromto", fmt_vec(np.concatenate([endpoint_a_body, endpoint_b_body])))
    geom.set("size", fmt_vec(np.array([radius], dtype=np.float64)))
    _set_collision_geom_common_attrs(geom)
    return {
        "proxy_type": "capsule",
        "axis": (original_rot @ axis).tolist(),
        "center": (0.5 * (endpoint_a_body + endpoint_b_body)).tolist(),
        "radius": float(radius),
        "length": float(2.0 * half_length),
        "trim_percentile": [float(low_q), float(high_q)],
        "reason": "trimmed_pca_capsule",
        "degenerate": bool(degenerate),
    }


def _convert_arm_collision_geom_to_capsule(
    root: ET.Element,
    geom: ET.Element,
    vertices: np.ndarray,
    proxy_scale: float,
) -> dict[str, object]:
    """Use a chain-aware capsule for ASM arm links instead of a PCA box.

    The arm collision meshes often include joint hubs, so their oriented boxes
    overlap heavily. For arm links we prefer a trimmed capsule along the next
    structural child link; if that vector is unavailable, we fall back to the
    mesh principal axis while still keeping a capsule proxy.
    """

    geom_name = geom.get("name", "")
    link_index = _arm_link_index_from_geom_name(geom_name) or 0
    body = _find_body_containing_geom(root, geom)
    body_name = body.get("name", "") if body is not None else ""
    length_scale = _arm_capsule_length_scale(proxy_scale)
    radius_scale = float(np.clip(float(proxy_scale), 0.50, ARM_CAPSULE_SCALE_MAX))
    base_radius = ARM_CAPSULE_RADIUS_AT_SCALE_1_BY_LINK.get(link_index, 0.030)
    radius = max(0.008, base_radius * radius_scale)

    axis_source = "mesh_pca"
    to_body_name: str | None = None
    child_body, child_vector = _same_side_arm_child_vector(body)
    if child_vector is not None:
        source_vector = child_vector
        axis_source = "child_link"
        to_body_name = child_body.get("name", "") if child_body is not None else None
    else:
        terminal_name, terminal_vector = _terminal_hand_vector(body)
        if terminal_vector is not None:
            source_vector = terminal_vector
            axis_source = "terminal_hand"
            to_body_name = terminal_name
        else:
            source_vector = None

    if source_vector is not None:
        vector_norm = float(np.linalg.norm(source_vector))
        axis = source_vector / max(vector_norm, 1e-9)
        center = 0.5 * source_vector
        half_length = max(
            ARM_CAPSULE_MIN_HALF_LENGTH_M,
            (0.5 * vector_norm - ARM_CAPSULE_TRIM_MARGIN_M) * length_scale,
        )
        endpoint_a_body = center - axis * half_length
        endpoint_b_body = center + axis * half_length
    else:
        center, axis, mesh_length = _mesh_capsule_frame_in_body(geom, vertices)
        fallback_length = max(
            float(mesh_length),
            ARM_CAPSULE_FALLBACK_LENGTH_BY_LINK.get(link_index, 0.10),
        )
        half_length = max(
            ARM_CAPSULE_MIN_HALF_LENGTH_M,
            (0.5 * fallback_length - ARM_CAPSULE_TRIM_MARGIN_M) * length_scale,
        )
        endpoint_a_body = center - axis * half_length
        endpoint_b_body = center + axis * half_length

    _set_geom_as_capsule(geom, endpoint_a_body, endpoint_b_body, radius)
    center_body = 0.5 * (endpoint_a_body + endpoint_b_body)
    axis_body = endpoint_b_body - endpoint_a_body
    length = float(np.linalg.norm(axis_body))
    if length > 0:
        axis_body = axis_body / length
    return {
        "proxy_type": "capsule",
        "axis": axis_body.tolist(),
        "center": center_body.tolist(),
        "radius": float(radius),
        "length": length,
        "fromto": np.concatenate([endpoint_a_body, endpoint_b_body]).tolist(),
        "link_index": int(link_index),
        "from_body": body_name,
        "to_body": to_body_name,
        "axis_source": axis_source,
        "trim_margin_m": float(ARM_CAPSULE_TRIM_MARGIN_M),
        "length_scale": float(length_scale),
        "radius_scale": float(radius_scale),
        "reason": f"arm_{axis_source}_capsule",
        "degenerate": False,
    }


def _convert_new_collision_proxy(
    root: ET.Element,
    geom: ET.Element,
    vertices: np.ndarray,
    proxy_scale: float,
    min_half_size: float,
) -> dict[str, object]:
    geom_name = geom.get("name", "")
    kind = _collision_proxy_kind(geom_name)

    if kind == "body_box":
        if any(token in geom_name for token in ("base_link1", "base_link2", "base_r", "base_l")):
            _disable_collision_proxy(geom)
            return {
                "proxy_type": "skipped",
                "reason": "large_fixed_base_proxy_hidden",
            }
        return _convert_collision_geom_to_box(
            geom,
            vertices,
            proxy_scale,
            min_half_size,
            max_half_extent=0.08,
        )

    if kind == "palm_box":
        return _convert_collision_geom_to_box(
            geom,
            vertices,
            proxy_scale=min(float(proxy_scale), 0.92),
            min_half_size=min_half_size,
            max_half_extent=0.07,
        )

    if kind == "finger_capsule":
        return _convert_collision_geom_to_capsule(
            geom,
            vertices,
            proxy_scale=proxy_scale,
            length_quantile=0.84,
            radius_quantile=0.68,
            radius_scale=0.95,
            min_radius=0.0015,
            min_half_length=0.003,
            max_radius=0.011,
            fallback_aspect_ratio=1.15,
        )

    if kind == "arm_capsule":
        return _convert_arm_collision_geom_to_capsule(root, geom, vertices, proxy_scale)

    return _convert_collision_geom_to_capsule(
        geom,
        vertices,
        proxy_scale=proxy_scale,
        length_quantile=0.8,
        radius_quantile=0.7,
        radius_scale=1.0,
        min_radius=0.002,
        min_half_length=0.004,
        max_radius=0.04,
    )


def replace_collision_meshes_with_primitive_proxies(
    root: ET.Element,
    mesh_vertices: dict[str, np.ndarray],
    proxy_scale: float,
    *,
    min_half_size: float = 0.003,
    capsule_length_quantile: float = 0.8,
    capsule_radius_quantile: float = 0.35,
    capsule_radius_scale: float = 0.75,
    capsule_min_radius: float = 0.002,
    capsule_min_half_length: float = 0.003,
) -> dict[str, int | float]:
    if proxy_scale <= 0.0:
        raise ValueError(f"primitive collision proxy scale must be > 0, got {proxy_scale}")
    if min_half_size <= 0.0:
        raise ValueError(f"primitive collision min half size must be > 0, got {min_half_size}")

    converted = 0
    skipped = 0
    hidden = 0
    box_count = 0
    capsule_count = 0
    degenerate_capsules = 0
    proxy_records: list[dict[str, object]] = []
    radius_values: list[float] = []
    length_values: list[float] = []
    for geom in root.iter("geom"):
        geom_name = geom.get("name", "")
        if not geom_name.startswith(URDF_COLLISION_PREFIXES):
            continue
        mesh_name = geom.get("mesh")
        vertices = mesh_vertices.get(mesh_name or "")
        if mesh_name is None or vertices is None or vertices.shape[0] < 2:
            skipped += 1
            continue

        record = _convert_new_collision_proxy(root, geom, vertices, proxy_scale, min_half_size)
        record["geom_name"] = geom_name
        record["source_mesh"] = mesh_name
        proxy_records.append(record)
        if record.get("proxy_type") == "box":
            box_count += 1
        elif record.get("proxy_type") == "capsule":
            capsule_count += 1
            degenerate_capsules += int(bool(record.get("degenerate", False)))
            radius_values.append(float(record.get("radius", 0.0)))
            length_values.append(float(record.get("length", 0.0)))
        elif record.get("proxy_type") == "skipped":
            hidden += 1
        converted += 1

    return {
        "primitive_proxy_scale": float(proxy_scale),
        "converted_collision_geoms": converted,
        "skipped_collision_geoms": skipped,
        "hidden_collision_geoms": hidden,
        "box_collision_geoms": box_count,
        "capsule_collision_geoms": capsule_count,
        "degenerate_capsule_geoms": degenerate_capsules,
        "capsule_radius_min": float(min(radius_values)) if radius_values else 0.0,
        "capsule_radius_max": float(max(radius_values)) if radius_values else 0.0,
        "capsule_length_min": float(min(length_values)) if length_values else 0.0,
        "capsule_length_max": float(max(length_values)) if length_values else 0.0,
        "proxy_records": proxy_records,
    }


def _body_has_active_urdf_collision_geom(body: ET.Element) -> bool:
    for geom in body.findall("geom"):
        name = geom.get("name", "")
        if not name.startswith(URDF_COLLISION_PREFIXES):
            continue
        if geom.get("contype") == "0" and geom.get("conaffinity") == "0":
            continue
        return True
    return False


def wrap_worldbody_in_root(root: ET.Element, yaw_deg: float) -> None:
    worldbody = root.find("worldbody")
    if worldbody is None:
        return

    for child in worldbody.findall("body"):
        if child.get("name") == ASM_ROOT_NAME:
            child.set("quat", yaw_quat_str(yaw_deg))
            return

    children = list(worldbody)
    if not children:
        return

    root_body = ET.Element(
        "body",
        {
            "name": ASM_ROOT_NAME,
            "quat": yaw_quat_str(yaw_deg),
        },
    )
    for child in children:
        worldbody.remove(child)
        root_body.append(child)
    worldbody.append(root_body)


def add_camera_sites(root: ET.Element, source_urdf: Path) -> None:
    asm_root = find_body(root, ASM_ROOT_NAME)
    if asm_root is None:
        raise RuntimeError(f"{ASM_ROOT_NAME} body not found after wrapping worldbody")

    camera_pos, camera_rot = compute_head_camera_pose(source_urdf)
    ensure_child(
        asm_root,
        "site",
        {
            "name": HEAD_CAMERA_SITE_NAME,
            "type": "box",
            "size": "0.012 0.008 0.006",
            "pos": " ".join(fmt(value) for value in camera_pos),
            "quat": quat_str_from_matrix(camera_rot),
            "rgba": "1 0 1 1",
            "group": "3",
        },
    )
    optical_pos, optical_rot = compute_d435_optical_pose(source_urdf)
    ensure_child(
        asm_root,
        "site",
        {
            "name": D435_OPTICAL_FRAME_SITE_NAME,
            "type": "box",
            "size": "0.01 0.006 0.004",
            "pos": " ".join(fmt(value) for value in optical_pos),
            "quat": quat_str_from_matrix(optical_rot),
            "rgba": "1 0.45 0 1",
            "group": "3",
        },
    )



def find_body_with_mesh(root: ET.Element, mesh_name: str) -> tuple[ET.Element | None, ET.Element | None]:
    for body in iter_bodies(root):
        for geom in body.findall("geom"):
            if geom.get("mesh") == mesh_name:
                return body, geom
    return None, None


def geom_pos(geom: ET.Element | None) -> str:
    if geom is None:
        return "0 0 0"
    return geom.get("pos", "0 0 0")


def geom_quat(geom: ET.Element | None) -> str | None:
    if geom is None:
        return None
    return geom.get("quat")


def _is_arm_joint_name(name: str) -> bool:
    return bool(name) and name.startswith("Joint") and name.endswith(("_R", "_L"))


def _is_hand_joint_name(name: str) -> bool:
    return bool(name) and "_finger" in name and "_joint" in name


def _parse_range(range_text: str) -> tuple[float, float] | None:
    try:
        low_text, high_text = range_text.split()
        return float(low_text), float(high_text)
    except ValueError:
        return None


def add_hand_sites(root: ET.Element) -> None:
    for side in ("right", "left"):
        palm, palm_geom = find_body_with_mesh(root, f"{side}_palm_link")
        if palm is not None:
            palm_pos = geom_pos(palm_geom)
            palm_quat = geom_quat(palm_geom)
            palm_site_attrs = {
                "name": f"{side}_palm",
                "type": "box",
                "size": "0.015 0.025 0.035",
                "pos": palm_pos,
                "rgba": "1 1 0 1",
                "group": "3",
            }
            if palm_quat is not None:
                palm_site_attrs["quat"] = palm_quat
            ensure_child(palm, "site", palm_site_attrs)

        for finger, asm_finger in FINGER_MAP.items():
            body, tip_geom = find_body_with_mesh(root, f"{side}_{asm_finger}_tip_link")
            if body is None:
                body = find_body(root, f"{side}_{asm_finger}_link4")
            if body is None:
                continue
            tip_pos = geom_pos(tip_geom)
            for site_name, rgba in (
                (f"{side}_{finger}_tip", "0 1 1 1"),
                (f"track_hand_{side}_{finger}_tip", "0 0 1 1"),
                (f"trace_hand_{side}_{finger}_tip", "1 0 0 1"),
            ):
                ensure_child(
                    body,
                    "site",
                    {
                        "name": site_name,
                        "type": "sphere",
                        "size": "0.008",
                        "pos": tip_pos,
                        "rgba": rgba,
                        "group": "3",
                    },
                )


def joint_names(root: ET.Element) -> list[str]:
    return [joint.get("name") for joint in root.iter("joint") if joint.get("name")]


def tune_joint_dynamics(
    root: ET.Element,
    arm_damping: float,
    hand_damping: float,
    arm_armature: float,
    hand_armature: float,
    arm_frictionloss: float,
    hand_frictionloss: float,
    arm_force_scale: float,
    hand_force_scale: float,
) -> None:
    for joint in root.iter("joint"):
        name = joint.get("name", "")
        if _is_arm_joint_name(name):
            joint.set("damping", fmt(arm_damping))
            joint.set("armature", fmt(arm_armature))
            joint.set("frictionloss", fmt(arm_frictionloss))
            force_range_text = joint.get("actuatorfrcrange")
            if force_range_text:
                force_range = _parse_range(force_range_text)
                if force_range is not None:
                    low, high = force_range
                    joint.set(
                        "actuatorfrcrange",
                        f"{fmt(low * arm_force_scale)} {fmt(high * arm_force_scale)}",
                    )
        elif _is_hand_joint_name(name):
            joint.set("damping", fmt(hand_damping))
            joint.set("armature", fmt(hand_armature))
            joint.set("frictionloss", fmt(hand_frictionloss))
            force_range_text = joint.get("actuatorfrcrange")
            if force_range_text:
                force_range = _parse_range(force_range_text)
                if force_range is not None:
                    low, high = force_range
                    joint.set(
                        "actuatorfrcrange",
                        f"{fmt(low * hand_force_scale)} {fmt(high * hand_force_scale)}",
                    )


def add_position_actuators(root: ET.Element, arm_kp: float, hand_kp: float) -> None:
    actuator = root.find("actuator")
    if actuator is None:
        actuator = ET.SubElement(root, "actuator")
    existing = child_names(actuator, "position")
    for joint_name in joint_names(root):
        name = f"{joint_name}_position"
        if name in existing:
            continue
        is_arm = joint_name.startswith("Joint") and joint_name.endswith(("_R", "_L"))
        kp = arm_kp if is_arm else hand_kp
        ET.SubElement(
            actuator,
            "position",
            {
                "name": name,
                "joint": joint_name,
                "kp": fmt(kp),
                "dampratio": "1",
                "inheritrange": "1",
            },
        )
        existing.add(name)


def remove_body_named(parent: ET.Element, body_name: str) -> bool:
    for child in list(parent):
        if child.tag == "body" and child.get("name") == body_name:
            parent.remove(child)
            return True
        if remove_body_named(child, body_name):
            return True
    return False


def remove_side_fixed_geoms(root: ET.Element, side: str) -> None:
    suffix = "_R" if side == "right" else "_L"
    asm_root = find_body(root, ASM_ROOT_NAME)
    if asm_root is None:
        return
    for geom in list(asm_root.findall("geom")):
        mesh_name = geom.get("mesh", "")
        geom_name = geom.get("name", "")
        if mesh_name.endswith(suffix) or geom_name.endswith(f"_{side}_base_0"):
            asm_root.remove(geom)


def remove_invalid_actuators(root: ET.Element) -> None:
    actuator = root.find("actuator")
    if actuator is None:
        return
    valid_joints = set(joint_names(root))
    for child in list(actuator):
        joint = child.get("joint")
        if joint and joint not in valid_joints:
            actuator.remove(child)


def prune_variant(root: ET.Element, variant: str) -> ET.Element:
    if variant == "bimanual":
        return copy.deepcopy(root)
    pruned = copy.deepcopy(root)
    worldbody = pruned.find("worldbody")
    if worldbody is None:
        return pruned
    remove_left = variant == "right"
    remove_names = ["Base_L", "Link1_L"] if remove_left else ["Base_R", "Link1_R"]
    for remove_name in remove_names:
        remove_body_named(worldbody, remove_name)
    remove_side_fixed_geoms(pruned, "left" if remove_left else "right")
    remove_invalid_actuators(pruned)
    return pruned


def validate_model(xml_path: Path, variant: str) -> None:
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    required: list[str] = [HEAD_CAMERA_SITE_NAME, D435_OPTICAL_FRAME_SITE_NAME]
    sides = ["right", "left"] if variant == "bimanual" else [variant]
    for side in sides:
        required.append(f"{side}_palm")
        required.extend(f"{side}_{finger}_tip" for finger in FINGER_MAP)
    missing = [
        name
        for name in required
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name) == -1
    ]
    if missing:
        raise RuntimeError(f"Missing required sites in {xml_path}: {missing}")
    if model.nu <= 0:
        raise RuntimeError(f"No actuators in {xml_path}")
    collision_count = 0
    disabled_collision_count = 0
    hand_collision_count = 0
    arm_collision_count = 0
    body_collision_count = 0
    collision_type_counts: dict[str, int] = {}
    geom_type_names = {
        int(mujoco.mjtGeom.mjGEOM_PLANE): "plane",
        int(mujoco.mjtGeom.mjGEOM_HFIELD): "hfield",
        int(mujoco.mjtGeom.mjGEOM_SPHERE): "sphere",
        int(mujoco.mjtGeom.mjGEOM_CAPSULE): "capsule",
        int(mujoco.mjtGeom.mjGEOM_ELLIPSOID): "ellipsoid",
        int(mujoco.mjtGeom.mjGEOM_CYLINDER): "cylinder",
        int(mujoco.mjtGeom.mjGEOM_BOX): "box",
        int(mujoco.mjtGeom.mjGEOM_MESH): "mesh",
    }
    for gid in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
        if not name or not name.startswith(URDF_COLLISION_PREFIXES):
            continue
        geom_type = geom_type_names.get(int(model.geom_type[gid]), str(int(model.geom_type[gid])))
        collision_type_counts[geom_type] = collision_type_counts.get(geom_type, 0) + 1
        if model.geom_contype[gid] == 0 or model.geom_conaffinity[gid] == 0:
            disabled_collision_count += 1
            continue
        collision_count += 1
        if name.startswith("collision_hand_"):
            hand_collision_count += 1
        elif name and name.startswith("collision_arm_"):
            arm_collision_count += 1
        elif name and name.startswith("collision_body_"):
            body_collision_count += 1
    if collision_count == 0:
        raise RuntimeError(f"No active ASM collision geoms in {xml_path}")
    if hand_collision_count == 0:
        raise RuntimeError(f"No collision_hand_ geoms in {xml_path}")
    if variant in {"bimanual", "right"} and not any(
        (
            (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or "").startswith(
                ("collision_hand_right_", "collision_arm_right_")
            )
            and model.geom_contype[gid] != 0
            and model.geom_conaffinity[gid] != 0
        )
        for gid in range(model.ngeom)
    ):
        raise RuntimeError(f"No right-side URDF collision meshes in {xml_path}")
    if variant in {"bimanual", "left"} and not any(
        (
            (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or "").startswith(
                ("collision_hand_left_", "collision_arm_left_")
            )
            and model.geom_contype[gid] != 0
            and model.geom_conaffinity[gid] != 0
        )
        for gid in range(model.ngeom)
    ):
        raise RuntimeError(f"No left-side URDF collision meshes in {xml_path}")
    if arm_collision_count == 0:
        raise RuntimeError(f"No collision_arm_ geoms in {xml_path}")
    print(
        f"{variant}: nq={model.nq} nv={model.nv} nu={model.nu} "
        f"nsite={model.nsite} ngeom={model.ngeom} "
        f"nexclude={getattr(model, 'nexclude', 0)} "
        f"collision_asm={collision_count} "
        f"collision_disabled={disabled_collision_count} "
        f"collision_types={collision_type_counts} "
        f"collision_hand={hand_collision_count} "
        f"collision_arm={arm_collision_count} "
        f"collision_body={body_collision_count}"
    )


def write_xml(root: ET.Element, path: Path) -> None:
    try:
        ET.indent(root, space="  ")
    except AttributeError:
        pass
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(path, encoding="unicode")


def _parse_rgba_string(value: str | None, fallback: list[float]) -> list[float]:
    if not value:
        return list(fallback)
    try:
        parts = [float(part) for part in value.split()]
    except ValueError:
        return list(fallback)
    if len(parts) == 3:
        parts.append(1.0)
    if len(parts) != 4:
        return list(fallback)
    return parts


def write_collision_proxy_color_manifest(
    root: ET.Element,
    path: Path,
    *,
    collision_mode: str,
    collision_mesh_scale: float,
    source_urdf: Path,
    proxy_records: list[dict[str, object]] | None = None,
) -> None:
    record_map = {
        str(record.get("geom_name")): record
        for record in (proxy_records or [])
        if record.get("geom_name")
    }
    entries: list[dict[str, object]] = []
    for geom in root.iter("geom"):
        geom_name = geom.get("name", "")
        if not geom_name.startswith(URDF_COLLISION_PREFIXES):
            continue
        mesh_name = geom.get("mesh", "")
        metadata = _collision_proxy_metadata(geom_name, mesh_name)
        fallback_rgba = list(metadata["rgba"])  # type: ignore[arg-type]
        record = record_map.get(geom_name, {})
        geom_type = geom.get("type", "mesh" if mesh_name else "unknown")
        proxy_type = record.get("proxy_type", geom_type)
        entries.append(
            {
                "geom_name": geom_name,
                "geom_type": geom_type,
                "proxy_type": proxy_type,
                "mesh_name": mesh_name,
                "source_mesh": record.get("source_mesh", mesh_name),
                "component": metadata["component"],
                "side": metadata["side"],
                "link_key": metadata["link_key"],
                "link_label": metadata["link_label"],
                "color_hex": metadata["color_hex"],
                "rgba": _parse_rgba_string(geom.get("rgba"), fallback_rgba),
                "axis": record.get("axis"),
                "center": record.get("center"),
                "radius": record.get("radius"),
                "length": record.get("length"),
                "fromto": record.get("fromto"),
                "half_extents": record.get("half_extents"),
                "trim_percentile": record.get("trim_percentile"),
                "trim_margin_m": record.get("trim_margin_m"),
                "link_index": record.get("link_index"),
                "from_body": record.get("from_body"),
                "to_body": record.get("to_body"),
                "axis_source": record.get("axis_source"),
                "length_scale": record.get("length_scale"),
                "radius_scale": record.get("radius_scale"),
                "reason": record.get("reason"),
            }
        )

    payload = {
        "collision_mode": collision_mode,
        "collision_mesh_scale": float(collision_mesh_scale),
        "source_urdf": str(source_urdf),
        "num_entries": len(entries),
        "proxy_algorithm": "NewCollision chain-aware arm capsules / trimmed PCA hand capsules / oriented bbox",
        "entries": entries,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir).resolve()
    source_urdf = Path(args.source_urdf).resolve()
    source_root = source_urdf.parents[1]
    output_dir = dataset_dir / "processed" / args.dataset_name / "assets" / "robots" / args.robot_type
    mesh_dir = output_dir / "meshes"

    filename_map = copy_flat_meshes(source_root, mesh_dir)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        compile_urdf = tmp_dir / "asm_compile.urdf"
        compiled_xml = tmp_dir / "asm_compiled.xml"
        make_compile_urdf(source_urdf, mesh_dir, compile_urdf, filename_map)
        root = compile_urdf_to_xml(compile_urdf, compiled_xml)
        compiled_mesh_vertices = mesh_vertices_by_name(compiled_xml)

    add_default_and_assets(root)
    wrap_worldbody_in_root(root, args.root_yaw_deg)
    activate_urdf_collision_geoms(root)
    if args.collision_geometry_mode == "primitive":
        collision_proxy_stats = replace_collision_meshes_with_primitive_proxies(
            root,
            compiled_mesh_vertices,
            args.collision_mesh_scale,
        )
        print(
            "Generated ASM NewCollision primitive proxies: "
            f"scale={collision_proxy_stats['primitive_proxy_scale']} "
            f"converted={collision_proxy_stats['converted_collision_geoms']} "
            f"skipped={collision_proxy_stats['skipped_collision_geoms']} "
            f"hidden={collision_proxy_stats['hidden_collision_geoms']} "
            f"boxes={collision_proxy_stats['box_collision_geoms']} "
            f"capsules={collision_proxy_stats['capsule_collision_geoms']} "
            f"degenerate_capsules={collision_proxy_stats['degenerate_capsule_geoms']}"
        )
    elif args.collision_geometry_mode == "urdf_mesh":
        collision_scale_stats = scale_urdf_collision_meshes(
            root,
            compiled_mesh_vertices,
            args.collision_mesh_scale,
        )
        print(
            "Scaled ASM URDF collision meshes: "
            f"scale={collision_scale_stats['collision_mesh_scale']} "
            f"converted={collision_scale_stats['converted_collision_mesh_geoms']} "
            f"skipped={collision_scale_stats['skipped_collision_mesh_geoms']} "
            f"non_mesh_active={collision_scale_stats['non_mesh_active_collision_geoms']} "
            f"created_assets={collision_scale_stats['created_scaled_mesh_assets']}"
        )
    else:
        raise ValueError(f"Unsupported collision geometry mode: {args.collision_geometry_mode!r}")
    add_camera_sites(root, source_urdf)
    add_hand_sites(root)
    tune_joint_dynamics(
        root,
        arm_damping=args.arm_damping,
        hand_damping=args.hand_damping,
        arm_armature=args.arm_armature,
        hand_armature=args.hand_armature,
        arm_frictionloss=args.arm_frictionloss,
        hand_frictionloss=args.hand_frictionloss,
        arm_force_scale=args.arm_force_scale,
        hand_force_scale=args.hand_force_scale,
    )
    add_position_actuators(root, args.arm_kp, args.hand_kp)
    manifest_path = output_dir / "collision_proxy_colors.json"
    proxy_records = (
        collision_proxy_stats.get("proxy_records", [])
        if args.collision_geometry_mode == "primitive"
        else []
    )
    write_collision_proxy_color_manifest(
        root,
        manifest_path,
        collision_mode=args.collision_geometry_mode,
        collision_mesh_scale=args.collision_mesh_scale,
        source_urdf=source_urdf,
        proxy_records=proxy_records,  # type: ignore[arg-type]
    )

    for variant in args.variants:
        variant_root = prune_variant(root, variant)
        xml_name = "bimanual.xml" if variant == "bimanual" else f"{variant}.xml"
        xml_path = output_dir / xml_name
        write_xml(variant_root, xml_path)
        validate_model(xml_path, variant)

    print(f"Saved ASM collision color manifest to {manifest_path}")
    print(f"Saved ASM robot assets to {output_dir}")


if __name__ == "__main__":
    main()
