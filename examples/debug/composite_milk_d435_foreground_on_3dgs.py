#!/usr/bin/env python3
"""Composite MuJoCo D435 foreground over pre-rendered 3DGS background frames."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import cv2
import mujoco
import numpy as np
from PIL import Image


DEFAULT_PROCESSED_ROOT = Path(
    "example_datasets/processed/ourdata/asm/bimanual/milk"
)
DEFAULT_BG_FRAME_DIR = Path("preprocessed/milk/3dgs_bg_videos/frames")
DEFAULT_OUTPUT_DIR = Path("preprocessed/milk/3dgs_composite_videos")
DEFAULT_CAMERA_NAME = "d435_optical_render"


def _load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def _save_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(image, dtype=np.uint8), mode="RGB").save(path)


def _clear_frame_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for frame in path.glob("frame_*.png"):
        if frame.is_file():
            frame.unlink()


def _trajectory_filename(mode: str) -> str:
    if mode == "mjwp":
        return "trajectory_mjwp.npz"
    if mode == "ikrollout":
        return "trajectory_ikrollout.npz"
    return "trajectory_kinematic.npz"


def _resolve_trajectory_path(
    processed_root: Path,
    data_id: str,
    mode: str,
    explicit_path: str | None,
) -> Path:
    if explicit_path:
        path = Path(explicit_path)
        if not path.exists():
            raise FileNotFoundError(f"trajectory path does not exist: {path}")
        return path.resolve()

    filename = _trajectory_filename(mode)
    direct = processed_root / data_id / filename
    if direct.exists():
        return direct.resolve()

    candidates = sorted(
        processed_root.glob(f"*/{filename}"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"Could not find {filename} under {processed_root}; pass --trajectory-path."
        )
    print(
        f"  WARNING: {direct} not found; using most recent candidate {candidates[0]}"
    )
    return candidates[0].resolve()


def _resolve_model_path(processed_root: Path, mode: str, explicit_path: str | None) -> Path:
    if explicit_path:
        path = Path(explicit_path)
    else:
        path = processed_root / ("scene_act.xml" if mode == "mjwp" else "scene.xml")
    if not path.exists():
        raise FileNotFoundError(f"MuJoCo model XML does not exist: {path}")
    return path.resolve()


def _load_qpos(trajectory_path: Path, qpos_key: str) -> np.ndarray:
    data = np.load(trajectory_path)
    if qpos_key not in data.files:
        raise KeyError(
            f"{trajectory_path} does not contain key '{qpos_key}'. Available keys: {data.files}"
        )
    qpos = np.asarray(data[qpos_key], dtype=np.float64)
    if qpos.ndim != 2:
        raise ValueError(f"qpos must have shape (T, nq), got {qpos.shape}")
    return qpos


def _is_foreground_geom(model: mujoco.MjModel, geom_id: int) -> bool:
    geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
    body_id = int(model.geom_bodyid[geom_id])
    body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
    name = geom_name.lower()
    body = body_name.lower()

    if name in {"floor", "support_table"}:
        return False
    if "ground" in name or "table" in name:
        return False
    if name.startswith("debug_") or body.startswith("debug_"):
        return False
    if "trace_" in name or "trace_" in body:
        return False
    if "bbox" in name or "axis" in name or "axes" in name:
        return False
    if name.startswith("collision_"):
        return False

    if body in {"right_object", "left_object"}:
        return "visual" in name
    if name.startswith("right_object") or name.startswith("left_object"):
        return "visual" in name

    # Remaining visible geoms are treated as robot foreground.
    return body not in {"world", "worldbody"}


def _foreground_geom_mask(model: mujoco.MjModel) -> tuple[np.ndarray, list[dict[str, object]]]:
    mask = np.zeros(model.ngeom, dtype=bool)
    records: list[dict[str, object]] = []
    for geom_id in range(model.ngeom):
        selected = _is_foreground_geom(model, geom_id)
        mask[geom_id] = selected
        if selected:
            body_id = int(model.geom_bodyid[geom_id])
            records.append(
                {
                    "geom_id": int(geom_id),
                    "geom_name": mujoco.mj_id2name(
                        model, mujoco.mjtObj.mjOBJ_GEOM, geom_id
                    )
                    or "",
                    "body_id": body_id,
                    "body_name": mujoco.mj_id2name(
                        model, mujoco.mjtObj.mjOBJ_BODY, body_id
                    )
                    or "",
                }
            )
    return mask, records


def _make_scene_option() -> mujoco.MjvOption:
    option = mujoco.MjvOption()
    mujoco.mjv_defaultOption(option)
    # Hide collision helpers. The model still has collision geoms; they are not
    # part of the visual foreground mask.
    option.geomgroup[3] = False
    for site_group in range(len(option.sitegroup)):
        option.sitegroup[site_group] = False
    return option


def _mask_from_segmentation(seg: np.ndarray, foreground_geoms: np.ndarray) -> np.ndarray:
    if seg.ndim != 3 or seg.shape[2] < 2:
        raise ValueError(f"Unexpected segmentation image shape: {seg.shape}")

    seg = np.asarray(seg, dtype=np.int32)
    geom_type = int(mujoco.mjtObj.mjOBJ_GEOM)

    def candidate(obj_ids: np.ndarray, obj_types: np.ndarray) -> np.ndarray:
        valid = (
            (obj_types == geom_type)
            & (obj_ids >= 0)
            & (obj_ids < len(foreground_geoms))
        )
        out = np.zeros(obj_ids.shape, dtype=bool)
        out[valid] = foreground_geoms[obj_ids[valid]]
        return out

    mask_a = candidate(seg[:, :, 0], seg[:, :, 1])
    mask_b = candidate(seg[:, :, 1], seg[:, :, 0])
    return mask_a if int(mask_a.sum()) >= int(mask_b.sum()) else mask_b


def _alpha_from_mask(
    mask: np.ndarray,
    close_kernel: int,
    dilate_iterations: int,
    feather_radius: int,
) -> np.ndarray:
    mask_u8 = (mask.astype(np.uint8) * 255)
    if close_kernel > 1:
        kernel_size = int(close_kernel)
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_size, kernel_size),
        )
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
    if dilate_iterations > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_u8 = cv2.dilate(mask_u8, kernel, iterations=int(dilate_iterations))
    if feather_radius > 0:
        radius = int(feather_radius)
        kernel_size = radius * 2 + 1
        mask_u8 = cv2.GaussianBlur(mask_u8, (kernel_size, kernel_size), 0)
    return (mask_u8.astype(np.float32) / 255.0)[:, :, None]


def _render_rgb_and_mask(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    rgb_renderer: mujoco.Renderer,
    seg_renderer: mujoco.Renderer,
    scene_option: mujoco.MjvOption,
    camera_name: str,
    foreground_geoms: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mujoco.mj_forward(model, data)

    rgb_renderer.update_scene(data=data, camera=camera_name, scene_option=scene_option)
    rgb = rgb_renderer.render().copy()

    seg_renderer.update_scene(data=data, camera=camera_name, scene_option=scene_option)
    seg = seg_renderer.render().copy()
    mask = _mask_from_segmentation(seg, foreground_geoms)
    return rgb, mask


def _write_video_from_frames(frame_dir: Path, output_video: Path, fps: float) -> None:
    output_video.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(frame_dir / "frame_%05d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "18",
            str(output_video),
        ],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Composite milk MuJoCo D435 foreground over 3DGS background frames.",
    )
    parser.add_argument("--processed-root", default=str(DEFAULT_PROCESSED_ROOT))
    parser.add_argument("--data-id", default="0")
    parser.add_argument("--mode", choices=["kinematic", "ikrollout", "mjwp"], default="kinematic")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--trajectory-path", default="example_datasets/processed/ourdata/asm/bimanual/milk/0/trajectory_mjwp.npz")
    parser.add_argument("--qpos-key", default="qpos")
    parser.add_argument("--bg-frame-dir", default=str(DEFAULT_BG_FRAME_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--output-video", default=None)
    parser.add_argument("--camera-name", default=DEFAULT_CAMERA_NAME)
    parser.add_argument("--fps", type=float, default=50.0)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--strict-frame-count", action="store_true")
    parser.add_argument("--clean-output", action="store_true", default=True)
    parser.add_argument("--no-clean-output", dest="clean_output", action="store_false")
    parser.add_argument("--mask-close-kernel", type=int, default=3)
    parser.add_argument("--mask-dilate-iterations", type=int, default=1)
    parser.add_argument("--feather-radius", type=int, default=1)
    parser.add_argument("--debug-frame-count", type=int, default=5)
    args = parser.parse_args()

    processed_root = Path(args.processed_root)
    bg_frame_dir = Path(args.bg_frame_dir)
    output_dir = Path(args.output_dir)
    output_frame_dir = output_dir / "frames"
    debug_dir = output_dir / "debug"
    output_video = (
        Path(args.output_video)
        if args.output_video
        else output_dir / f"composite_d435_{args.mode}.mp4"
    )

    model_path = _resolve_model_path(processed_root, args.mode, args.model_path)
    trajectory_path = _resolve_trajectory_path(
        processed_root,
        args.data_id,
        args.mode,
        args.trajectory_path,
    )
    bg_frames = sorted(bg_frame_dir.glob("frame_*.png"))
    if not bg_frames:
        raise FileNotFoundError(f"No background frames found in {bg_frame_dir}")

    qpos = _load_qpos(trajectory_path, args.qpos_key)
    first_bg = _load_rgb(bg_frames[0])
    height, width = first_bg.shape[:2]

    total_frames = min(len(bg_frames), len(qpos))
    if args.max_frames is not None:
        total_frames = min(total_frames, int(args.max_frames))
    if args.strict_frame_count and len(bg_frames) != len(qpos):
        raise ValueError(
            f"Frame count mismatch: bg_frames={len(bg_frames)}, qpos={len(qpos)}"
        )
    if total_frames <= 0:
        raise RuntimeError("No frames to composite")

    print("Composite settings")
    print(f"  model:      {model_path}")
    print(f"  trajectory: {trajectory_path} key={args.qpos_key} frames={len(qpos)}")
    print(f"  bg frames:  {bg_frame_dir} frames={len(bg_frames)}")
    print(f"  output:     {output_video}")
    print(f"  size/fps:   {width}x{height} @ {args.fps}")
    print(f"  frames:     {total_frames}")

    if args.clean_output:
        _clear_frame_dir(output_frame_dir)
        if debug_dir.exists():
            shutil.rmtree(debug_dir)
    output_frame_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    model = mujoco.MjModel.from_xml_path(str(model_path))
    if qpos.shape[1] != model.nq:
        raise ValueError(
            f"trajectory qpos width {qpos.shape[1]} does not match model.nq {model.nq}"
        )
    if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, args.camera_name) < 0:
        raise ValueError(f"camera '{args.camera_name}' not found in {model_path}")

    model.vis.global_.offwidth = int(width)
    model.vis.global_.offheight = int(height)
    data = mujoco.MjData(model)
    rgb_renderer = mujoco.Renderer(model, height=height, width=width)
    seg_renderer = mujoco.Renderer(model, height=height, width=width)
    seg_renderer.enable_segmentation_rendering()
    scene_option = _make_scene_option()
    foreground_geoms, foreground_records = _foreground_geom_mask(model)

    with (output_dir / "foreground_geoms.json").open("w", encoding="utf-8") as file:
        json.dump(foreground_records, file, indent=2, ensure_ascii=False)
    print(f"  foreground geoms: {len(foreground_records)}")

    for frame_idx in range(total_frames):
        bg = _load_rgb(bg_frames[frame_idx])
        if bg.shape[:2] != (height, width):
            raise ValueError(
                f"Background frame size changed at {bg_frames[frame_idx]}: "
                f"{bg.shape[1]}x{bg.shape[0]} vs expected {width}x{height}"
            )

        data.qpos[:] = qpos[frame_idx]
        data.qvel[:] = 0.0
        rgb, mask = _render_rgb_and_mask(
            model=model,
            data=data,
            rgb_renderer=rgb_renderer,
            seg_renderer=seg_renderer,
            scene_option=scene_option,
            camera_name=args.camera_name,
            foreground_geoms=foreground_geoms,
        )
        alpha = _alpha_from_mask(
            mask,
            close_kernel=args.mask_close_kernel,
            dilate_iterations=args.mask_dilate_iterations,
            feather_radius=args.feather_radius,
        )
        composite = (
            rgb.astype(np.float32) * alpha
            + bg.astype(np.float32) * (1.0 - alpha)
        )
        composite = np.clip(composite, 0, 255).astype(np.uint8)
        _save_rgb(output_frame_dir / f"frame_{frame_idx:05d}.png", composite)

        if frame_idx < args.debug_frame_count:
            _save_rgb(debug_dir / "foreground_rgb" / f"frame_{frame_idx:05d}.png", rgb)
            _save_rgb(
                debug_dir / "mask" / f"frame_{frame_idx:05d}.png",
                np.repeat((mask.astype(np.uint8) * 255)[:, :, None], 3, axis=2),
            )
            _save_rgb(debug_dir / "background" / f"frame_{frame_idx:05d}.png", bg)

        if (frame_idx + 1) % 30 == 0 or frame_idx == total_frames - 1:
            print(f"  [{frame_idx + 1:>4d}/{total_frames}]")

    _write_video_from_frames(output_frame_dir, output_video, fps=args.fps)
    print(f"Saved composite frames to {output_frame_dir}")
    print(f"Saved composite video to {output_video}")


if __name__ == "__main__":
    main()
