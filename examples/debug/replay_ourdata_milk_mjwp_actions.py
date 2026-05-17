#!/usr/bin/env python3
"""Replay saved milk MJWP actions and render the resulting trajectory.

This is a debug utility for checking whether trajectory_mjwp.npz contains
enough action information to reproduce the original MJWP rollout video.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
os.environ.setdefault("WARP_CACHE_PATH", "/tmp/spider_warp_cache")
Path(os.environ["WARP_CACHE_PATH"]).mkdir(parents=True, exist_ok=True)

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import imageio
import mujoco
import numpy as np
import torch
import warp as wp

from spider.config import Config, filter_config_fields, load_config_yaml, process_config
from spider.io import load_data
from spider.simulators.mjwp import get_qpos, get_qvel, setup_env, setup_mj_model, step_env
from spider.viewers import render_image, setup_renderer

DEFAULT_TRIAL_DIR = (
    REPO_ROOT / "example_datasets/processed/ourdata/asm/bimanual/milk/0"
)
DEFAULT_NPZ_PATH = DEFAULT_TRIAL_DIR / "trajectory_mjwp.npz"


def _repo_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _load_config(override: str, device: str, num_samples: int) -> Config:
    override_path = _repo_path(f"examples/config/override/{override}.yaml")
    config_dict = filter_config_fields(load_config_yaml(str(override_path)))
    config_dict["device"] = _resolve_device(device)
    config_dict["num_samples"] = int(num_samples)
    config_dict["viewer"] = "none"
    config_dict["show_viewer"] = False
    config_dict["save_video"] = True
    config_dict["save_info"] = True
    config_dict["wait_on_finish"] = False
    config_dict["use_torch_compile"] = False
    if "pair_margin_range" in config_dict:
        config_dict["pair_margin_range"] = tuple(config_dict["pair_margin_range"])
    if "xy_offset_range" in config_dict:
        config_dict["xy_offset_range"] = tuple(config_dict["xy_offset_range"])
    return process_config(Config(**config_dict))


def _flatten_saved_arrays(data: np.lib.npyio.NpzFile) -> dict[str, np.ndarray]:
    required = ("qpos", "qvel", "ctrl")
    missing = [name for name in required if name not in data]
    if missing:
        raise KeyError(f"Missing required array(s) in trajectory npz: {missing}")

    flattened = {}
    for name in required:
        arr = np.asarray(data[name])
        if arr.ndim == 3:
            arr = arr.reshape(-1, arr.shape[-1])
        elif arr.ndim != 2:
            raise ValueError(f"{name} must be 2D or 3D, got shape {arr.shape}")
        if not np.isfinite(arr).all():
            raise ValueError(f"{name} contains non-finite values")
        flattened[name] = arr

    if "time" in data:
        time = np.asarray(data["time"])
        if time.ndim == 2:
            time = time.reshape(-1)
        elif time.ndim != 1:
            raise ValueError(f"time must be 1D or 2D, got shape {time.shape}")
        flattened["time"] = time

    return flattened


def _default_output_path(mode: str) -> Path:
    if mode == "render_saved_qpos":
        return DEFAULT_TRIAL_DIR / "visualization_mjwp_saved_qpos_render.mp4"
    return DEFAULT_TRIAL_DIR / "visualization_mjwp_action_replay.mp4"


def _render_stride(config: Config) -> int:
    return max(1, int(np.round(config.render_dt / config.sim_dt)))


def _render_rollout(
    config: Config,
    mj_model: mujoco.MjModel,
    qpos_ref_np: np.ndarray,
    qpos_rollout: np.ndarray,
    qvel_rollout: np.ndarray,
    ctrl_rollout: np.ndarray,
    output_video: Path,
    include_helpers: bool,
) -> None:
    renderer = setup_renderer(config, mj_model)
    if renderer is None:
        raise RuntimeError("Renderer was not created; config.save_video must be true")

    mj_data = mujoco.MjData(mj_model)
    mj_data_ref = mujoco.MjData(mj_model)
    images = []
    stride = _render_stride(config)

    for step_idx, qpos in enumerate(qpos_rollout):
        if step_idx % stride != 0:
            continue
        mj_data.qpos[:] = qpos
        if step_idx < len(qvel_rollout):
            mj_data.qvel[:] = qvel_rollout[step_idx]
        if step_idx < len(ctrl_rollout):
            mj_data.ctrl[:] = ctrl_rollout[step_idx]
        mj_data.time = (step_idx + 1) * config.sim_dt

        ref_idx = min(step_idx + 1, len(qpos_ref_np) - 1)
        mj_data_ref.qpos[:] = qpos_ref_np[ref_idx]
        if mj_data_ref.qvel.shape[0] > 0:
            mj_data_ref.qvel[:] = 0.0

        images.append(
            render_image(
                config,
                renderer,
                mj_model,
                mj_data,
                mj_data_ref,
                include_helpers=include_helpers,
            )
        )

    if not images:
        raise RuntimeError("No frames rendered; check max_steps/render_dt/sim_dt")
    output_video.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output_video, images, fps=int(round(1.0 / config.render_dt)))


def _rollout_actions(
    config: Config,
    ref_data: tuple[torch.Tensor, ...],
    ctrl_flat: np.ndarray,
    max_steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if ctrl_flat.shape[1] != config.nu:
        raise ValueError(
            f"Saved ctrl has dim {ctrl_flat.shape[1]}, but model expects nu={config.nu}"
        )
    if max_steps > 0:
        ctrl_flat = ctrl_flat[:max_steps]

    env = setup_env(config, ref_data)
    qpos_list = []
    qvel_list = []
    ctrl_list = []

    for ctrl_np in ctrl_flat:
        ctrl_torch = torch.from_numpy(ctrl_np).to(config.device, dtype=torch.float32)
        step_env(config, env, ctrl_torch)
        qpos_list.append(get_qpos(config, env)[0].detach().cpu().numpy().copy())
        qvel_list.append(get_qvel(config, env)[0].detach().cpu().numpy().copy())
        ctrl_list.append(ctrl_np.copy())

    return (
        np.asarray(qpos_list, dtype=np.float64),
        np.asarray(qvel_list, dtype=np.float64),
        np.asarray(ctrl_list, dtype=np.float64),
    )


def _write_metrics(
    metrics_path: Path,
    *,
    mode: str,
    npz_path: Path,
    output_video: Path,
    output_npz: Path,
    saved_shapes: dict[str, tuple[int, ...]],
    replay_qpos: np.ndarray,
    replay_qvel: np.ndarray,
    saved_qpos: np.ndarray,
    saved_qvel: np.ndarray,
) -> None:
    steps = min(len(replay_qpos), len(saved_qpos))
    qpos_err = np.linalg.norm(replay_qpos[:steps] - saved_qpos[:steps], axis=1)
    qvel_steps = min(len(replay_qvel), len(saved_qvel))
    qvel_err = np.linalg.norm(replay_qvel[:qvel_steps] - saved_qvel[:qvel_steps], axis=1)

    lines = [
        f"mode: {mode}",
        f"input_npz: {npz_path}",
        f"output_video: {output_video}",
        f"output_npz: {output_npz}",
        f"saved_shapes: {saved_shapes}",
        f"replay_qpos_shape: {tuple(replay_qpos.shape)}",
        f"replay_qvel_shape: {tuple(replay_qvel.shape)}",
        f"qpos_error_mean: {float(qpos_err.mean()) if steps else float('nan'):.8f}",
        f"qpos_error_max: {float(qpos_err.max()) if steps else float('nan'):.8f}",
        f"qvel_error_mean: {float(qvel_err.mean()) if qvel_steps else float('nan'):.8f}",
        f"qvel_error_max: {float(qvel_err.max()) if qvel_steps else float('nan'):.8f}",
    ]
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay or render saved ourdata/milk MJWP actions."
    )
    parser.add_argument(
        "--npz-path",
        default=str(DEFAULT_NPZ_PATH),
        help="Path to trajectory_mjwp.npz.",
    )
    parser.add_argument(
        "--override",
        default="ourdata_asm",
        help="Override YAML name under examples/config/override, without .yaml.",
    )
    parser.add_argument(
        "--mode",
        choices=("rollout", "render_saved_qpos"),
        default="rollout",
        help="rollout re-simulates saved ctrl; render_saved_qpos only renders saved qpos.",
    )
    parser.add_argument("--device", default="auto", help="cuda:0, cpu, or auto.")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Number of MJWP worlds for replay. One world is enough for deterministic action replay.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="Optional debug limit after flattening saved steps.",
    )
    parser.add_argument(
        "--output-video",
        default="",
        help="Output mp4 path. Defaults to the milk trial directory.",
    )
    parser.add_argument(
        "--output-npz",
        default="",
        help="Output replay npz path. Defaults to trajectory_mjwp_action_replay.npz.",
    )
    parser.add_argument(
        "--metrics-path",
        default="",
        help="Output metrics txt path. Defaults to action_replay_metrics.txt.",
    )
    parser.add_argument(
        "--no-helpers",
        action="store_true",
        help="Hide helper collision geoms, sites, and contact markers in the rendered video.",
    )
    args = parser.parse_args()

    try:
        wp.init()
    except RuntimeError:
        pass

    npz_path = _repo_path(args.npz_path)
    if not npz_path.is_file():
        raise FileNotFoundError(f"MJWP trajectory npz not found: {npz_path}")

    config = _load_config(args.override, args.device, args.num_samples)
    qpos_ref, qvel_ref, ctrl_ref, contact, contact_pos = load_data(
        config, config.data_path
    )
    ref_data = (qpos_ref, qvel_ref, ctrl_ref, contact, contact_pos)
    qpos_ref_np = qpos_ref.detach().cpu().numpy()

    raw = np.load(npz_path)
    saved_shapes = {name: tuple(raw[name].shape) for name in raw.files}
    flat = _flatten_saved_arrays(raw)
    saved_qpos = flat["qpos"]
    saved_qvel = flat["qvel"]
    saved_ctrl = flat["ctrl"]
    if args.max_steps > 0:
        saved_qpos = saved_qpos[: args.max_steps]
        saved_qvel = saved_qvel[: args.max_steps]
        saved_ctrl = saved_ctrl[: args.max_steps]

    mj_model = setup_mj_model(config)
    if args.mode == "rollout":
        replay_qpos, replay_qvel, replay_ctrl = _rollout_actions(
            config, ref_data, flat["ctrl"], args.max_steps
        )
    else:
        replay_qpos = saved_qpos.copy()
        replay_qvel = saved_qvel.copy()
        replay_ctrl = saved_ctrl.copy()

    output_video = (
        _repo_path(args.output_video) if args.output_video else _default_output_path(args.mode)
    )
    output_npz = (
        _repo_path(args.output_npz)
        if args.output_npz
        else DEFAULT_TRIAL_DIR / "trajectory_mjwp_action_replay.npz"
    )
    metrics_path = (
        _repo_path(args.metrics_path)
        if args.metrics_path
        else DEFAULT_TRIAL_DIR / "action_replay_metrics.txt"
    )

    _render_rollout(
        config,
        mj_model,
        qpos_ref_np,
        replay_qpos,
        replay_qvel,
        replay_ctrl,
        output_video,
        include_helpers=not args.no_helpers,
    )

    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_npz,
        qpos=replay_qpos,
        qvel=replay_qvel,
        ctrl=replay_ctrl,
        time=np.arange(1, len(replay_qpos) + 1, dtype=np.float64) * config.sim_dt,
        source_npz=str(npz_path),
        mode=args.mode,
    )
    _write_metrics(
        metrics_path,
        mode=args.mode,
        npz_path=npz_path,
        output_video=output_video,
        output_npz=output_npz,
        saved_shapes=saved_shapes,
        replay_qpos=replay_qpos,
        replay_qvel=replay_qvel,
        saved_qpos=saved_qpos,
        saved_qvel=saved_qvel,
    )


if __name__ == "__main__":
    main()
