#!/usr/bin/env python3
"""Roll out milk IK joint targets directly through MuJoCo actuators.

This debug utility bypasses MJWP sampling completely:

1. load the processed milk `trajectory_kinematic.npz`
2. use the IK robot qpos as the position-actuator ctrl target
3. step MuJoCo dynamics once per interpolated sim frame
4. save rollout arrays and a ref-vs-sim video under `ik_to_act`
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import fields
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import imageio
import mujoco
import numpy as np
import torch

from omegaconf import OmegaConf

from spider.config import Config, filter_config_fields, load_config_yaml, process_config
from spider.io import load_data
from spider.viewers import render_image, setup_renderer


DEFAULT_TRIAL_DIR = (
    REPO_ROOT / "example_datasets/processed/ourdata/asm/bimanual/milk/0"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "example_datasets/processed/ourdata/asm/bimanual/milk/ik_to_act"
)


def _repo_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def _load_config(override: str, device: str) -> Config:
    override_path = _repo_path(f"examples/config/override/{override}.yaml")
    config_dict = filter_config_fields(load_config_yaml(str(override_path)))
    config_dict["device"] = device
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


def _clamp_ctrl_to_range(model: mujoco.MjModel, ctrl: np.ndarray) -> np.ndarray:
    ctrl = np.asarray(ctrl, dtype=np.float64).copy()
    limited = model.actuator_ctrllimited.astype(bool)
    if np.any(limited):
        ctrl[:, limited] = np.clip(
            ctrl[:, limited],
            model.actuator_ctrlrange[limited, 0],
            model.actuator_ctrlrange[limited, 1],
        )
    return ctrl


def _setup_mj_model(config: Config) -> mujoco.MjModel:
    model = mujoco.MjModel.from_xml_path(config.model_path)
    model.opt.timestep = float(config.sim_dt)
    model.opt.iterations = 32
    model.opt.ls_iterations = 80
    if hasattr(model.opt, "ccd_iterations"):
        model.opt.ccd_iterations = max(int(model.opt.ccd_iterations), 64)
    model.opt.o_solref = [0.02, 1.0]
    model.opt.o_solimp = [0.0, 0.95, 0.03, 0.5, 2.0]
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    return model


def _quat_angle_distance(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    q1 = np.asarray(q1, dtype=np.float64)
    q2 = np.asarray(q2, dtype=np.float64)
    dot = np.abs(np.sum(q1 * q2, axis=-1))
    dot = np.clip(dot, 0.0, 1.0)
    return 2.0 * np.arccos(dot)


def _compute_metrics(
    qpos_sim: np.ndarray,
    qvel_sim: np.ndarray,
    qpos_ref: np.ndarray,
    qvel_ref: np.ndarray,
    nu: int,
) -> dict[str, float]:
    steps = min(len(qpos_sim), len(qpos_ref))
    qpos_sim = qpos_sim[:steps]
    qvel_sim = qvel_sim[:steps]
    qpos_ref = qpos_ref[:steps]
    qvel_ref = qvel_ref[:steps]

    qpos_err = np.linalg.norm(qpos_sim - qpos_ref, axis=1)
    robot_qpos_err = np.linalg.norm(qpos_sim[:, :nu] - qpos_ref[:, :nu], axis=1)
    qvel_err = np.linalg.norm(qvel_sim - qvel_ref, axis=1)

    metrics = {
        "steps": int(steps),
        "qpos_error_mean": float(qpos_err.mean()),
        "qpos_error_max": float(qpos_err.max()),
        "robot_qpos_error_mean": float(robot_qpos_err.mean()),
        "robot_qpos_error_max": float(robot_qpos_err.max()),
        "qvel_error_mean": float(qvel_err.mean()),
        "qvel_error_max": float(qvel_err.max()),
    }
    if qpos_sim.shape[1] >= nu + 7:
        right_obj_sim = qpos_sim[:, -14:-7] if qpos_sim.shape[1] >= nu + 14 else qpos_sim[:, -7:]
        right_obj_ref = qpos_ref[:, -14:-7] if qpos_ref.shape[1] >= nu + 14 else qpos_ref[:, -7:]
        obj_pos_err = np.linalg.norm(right_obj_sim[:, :3] - right_obj_ref[:, :3], axis=1)
        obj_rot_err = _quat_angle_distance(right_obj_sim[:, 3:7], right_obj_ref[:, 3:7])
        metrics.update(
            {
                "right_object_pos_error_mean": float(obj_pos_err.mean()),
                "right_object_pos_error_max": float(obj_pos_err.max()),
                "right_object_rot_error_mean": float(obj_rot_err.mean()),
                "right_object_rot_error_max": float(obj_rot_err.max()),
            }
        )
    return metrics


def _write_metrics(path: Path, metrics: dict[str, float], metadata: dict[str, object]) -> None:
    lines = ["IK to actuator direct rollout", "=" * 80, ""]
    lines.append("metadata:")
    for key, value in metadata.items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    lines.append("metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            lines.append(f"- {key}: {value:.8f}")
        else:
            lines.append(f"- {key}: {value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _rollout(
    model: mujoco.MjModel,
    qpos_ref: np.ndarray,
    qvel_ref: np.ndarray,
    ctrl_ref: np.ndarray,
    *,
    max_steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    steps = min(len(qpos_ref), len(qvel_ref), len(ctrl_ref))
    if max_steps > 0:
        steps = min(steps, int(max_steps))
    if steps <= 1:
        raise ValueError(f"Need at least 2 rollout steps, got {steps}")

    data = mujoco.MjData(model)
    data.qpos[:] = qpos_ref[0]
    data.qvel[:] = qvel_ref[0]
    data.ctrl[:] = ctrl_ref[0]
    data.time = 0.0
    mujoco.mj_forward(model, data)

    qpos_out = [data.qpos.copy()]
    qvel_out = [data.qvel.copy()]
    ctrl_out = [data.ctrl.copy()]
    time_out = [float(data.time)]

    for step_idx in range(1, steps):
        data.ctrl[:] = ctrl_ref[step_idx - 1]
        mujoco.mj_step(model, data)
        qpos_out.append(data.qpos.copy())
        qvel_out.append(data.qvel.copy())
        ctrl_out.append(data.ctrl.copy())
        time_out.append(float(data.time))

    return (
        np.asarray(qpos_out, dtype=np.float64),
        np.asarray(qvel_out, dtype=np.float64),
        np.asarray(ctrl_out, dtype=np.float64),
        np.asarray(time_out, dtype=np.float64),
    )


def _render_video(
    config: Config,
    model: mujoco.MjModel,
    qpos_ref: np.ndarray,
    qpos_sim: np.ndarray,
    qvel_sim: np.ndarray,
    ctrl_sim: np.ndarray,
    output_video: Path,
    *,
    include_helpers: bool,
) -> None:
    renderer = setup_renderer(config, model)
    if renderer is None:
        raise RuntimeError("Renderer was not created")

    data = mujoco.MjData(model)
    data_ref = mujoco.MjData(model)
    stride = max(1, int(round(config.render_dt / config.sim_dt)))
    images = []
    for step_idx in range(0, len(qpos_sim), stride):
        data.qpos[:] = qpos_sim[step_idx]
        data.qvel[:] = qvel_sim[step_idx]
        data.ctrl[:] = ctrl_sim[step_idx]
        data.time = step_idx * config.sim_dt

        ref_idx = min(step_idx, len(qpos_ref) - 1)
        data_ref.qpos[:] = qpos_ref[ref_idx]
        data_ref.qvel[:] = 0.0
        data_ref.time = data.time
        images.append(
            render_image(
                config,
                renderer,
                model,
                data,
                data_ref,
                include_helpers=include_helpers,
            )
        )

    if not images:
        raise RuntimeError("No video frames rendered")
    imageio.mimsave(output_video, images, fps=int(round(1.0 / config.render_dt)))


def _save_config_snapshot(config: Config, output_path: Path) -> None:
    skip = {"noise_scale", "env_params_list", "viewer_body_entity_and_ids"}
    data = {}
    for field in fields(config):
        if field.name in skip:
            continue
        value = getattr(config, field.name)
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().tolist()
        elif isinstance(value, np.ndarray):
            value = value.tolist()
        elif isinstance(value, tuple):
            value = list(value)
        data[field.name] = value
    OmegaConf.save(OmegaConf.create(data), str(output_path))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Execute milk IK qpos directly as MuJoCo actuator targets."
    )
    parser.add_argument("--override", default="ourdata_asm")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--no-save-video", action="store_true")
    parser.add_argument("--show-helpers", action="store_true")
    args = parser.parse_args()

    output_dir = _repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = _load_config(args.override, args.device)
    model = _setup_mj_model(config)
    qpos_ref_t, qvel_ref_t, ctrl_ref_t, _contact, _contact_pos = load_data(
        config, config.data_path
    )
    qpos_ref = qpos_ref_t.detach().cpu().numpy()
    qvel_ref = qvel_ref_t.detach().cpu().numpy()
    ctrl_ref = ctrl_ref_t.detach().cpu().numpy()
    if ctrl_ref.shape[1] != model.nu:
        ctrl_ref = qpos_ref[:, : model.nu]
    ctrl_ref = _clamp_ctrl_to_range(model, ctrl_ref)

    rollout_steps = (
        config.max_sim_steps
        if config.max_sim_steps > 0
        else qpos_ref.shape[0] - config.horizon_steps - config.ctrl_steps
    )
    if args.max_steps > 0:
        rollout_steps = min(rollout_steps, args.max_steps)

    print(f"model_path: {config.model_path}")
    print(f"data_path: {config.data_path}")
    print(f"output_dir: {output_dir}")
    print(
        f"rollout_steps: {rollout_steps}, nq={model.nq}, nv={model.nv}, "
        f"nu={model.nu}, sim_dt={config.sim_dt}, ref_dt={config.ref_dt}"
    )

    qpos_sim, qvel_sim, ctrl_sim, time_sim = _rollout(
        model,
        qpos_ref,
        qvel_ref,
        ctrl_ref,
        max_steps=rollout_steps,
    )

    metrics = _compute_metrics(qpos_sim, qvel_sim, qpos_ref, qvel_ref, model.nu)
    metadata = {
        "model_path": config.model_path,
        "data_path": config.data_path,
        "override": args.override,
        "sim_dt": config.sim_dt,
        "ref_dt": config.ref_dt,
        "horizon_steps": config.horizon_steps,
        "ctrl_steps": config.ctrl_steps,
        "rollout_definition": "ctrl[t] = interpolated IK qpos_ref[t, :nu], no noise, no optimizer",
    }

    npz_path = output_dir / "trajectory_ik_to_act.npz"
    np.savez(
        npz_path,
        qpos=qpos_sim,
        qvel=qvel_sim,
        ctrl=ctrl_sim,
        time=time_sim,
        qpos_ref=qpos_ref[: len(qpos_sim)],
        qvel_ref=qvel_ref[: len(qvel_sim)],
        ctrl_ref=ctrl_ref[: len(ctrl_sim)],
        metrics_json=json.dumps(metrics, indent=2),
        metadata_json=json.dumps(metadata, indent=2),
    )
    print(f"saved npz: {npz_path}")

    metrics_path = output_dir / "metrics_ik_to_act.txt"
    _write_metrics(metrics_path, metrics, metadata)
    print(f"saved metrics: {metrics_path}")

    _save_config_snapshot(config, output_dir / "config_ik_to_act.yaml")

    if not args.no_save_video:
        video_path = output_dir / "visualization_ik_to_act.mp4"
        _render_video(
            config,
            model,
            qpos_ref,
            qpos_sim,
            qvel_sim,
            ctrl_sim,
            video_path,
            include_helpers=args.show_helpers,
        )
        print(f"saved video: {video_path}")


if __name__ == "__main__":
    main()
