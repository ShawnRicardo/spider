# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""A standalone script to run DIAL MPC with Mujoco + Warp

Up to now, domain randomization is not supported. Will add it later.

Author: Chaoyi Pan
Date: 2025-08-11
"""

from __future__ import annotations

import atexit
import datetime as _datetime
import os
import sys
import threading
import time
from dataclasses import fields
from pathlib import Path

import hydra
import imageio
import loguru
import mujoco
import numpy as np
import torch
import warp as wp
from omegaconf import DictConfig, OmegaConf

from spider.config import (
    Config,
    filter_config_fields,
    load_config_yaml,
    process_config,
)
from spider.interp import get_slice
from spider.io import load_data
from spider.optimizers.sampling import (
    make_optimize_fn,
    make_optimize_once_fn,
    make_rollout_fn,
)
from spider.postprocess.get_success_rate import compute_object_tracking_error
from spider.simulators.mjwp import (
    compute_contact_point_delta,
    copy_sample_state,
    get_qpos,
    get_qvel,
    get_reward,
    get_terminal_reward,
    get_terminate,
    get_trace,
    load_env_params,
    load_state,
    save_env_params,
    save_state,
    setup_env,
    setup_mj_model,  # mjwp specific
    step_env,
    sync_env,
)
from spider.viewers import (
    log_frame,
    render_image,
    setup_renderer,
    setup_viewer,
    update_viewer,
)

_CONFIG_SKIP_FIELDS = {
    "noise_scale",
    "env_params_list",
    "viewer_body_entity_and_ids",
}
_D435_RENDER_CAMERA_NAME = "d435_optical_render"

_CONSOLE_LOG_STARTED = False


def _start_console_log(config: Config) -> Path:
    """Mirror stdout/stderr to a timestamped log file in the output directory."""
    global _CONSOLE_LOG_STARTED
    if _CONSOLE_LOG_STARTED:
        return Path(config.output_dir)
    _CONSOLE_LOG_STARTED = True

    timestamp = _datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = "_act" if config.contact_guidance else ""
    log_path = Path(config.output_dir) / f"run_mjwp{suffix}_{timestamp}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    read_fd, write_fd = os.pipe()
    stdout_fd = os.dup(1)
    stderr_fd = os.dup(2)
    log_file = open(log_path, "ab", buffering=0)

    def _reader() -> None:
        while True:
            chunk = os.read(read_fd, 8192)
            if not chunk:
                break
            os.write(stdout_fd, chunk)
            log_file.write(chunk)

    reader = threading.Thread(target=_reader, name="console-log-tee", daemon=True)
    reader.start()
    os.dup2(write_fd, 1)
    os.dup2(write_fd, 2)
    os.close(write_fd)

    def _restore() -> None:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        finally:
            os.dup2(stdout_fd, 1)
            os.dup2(stderr_fd, 2)
            reader.join(timeout=1.0)
            os.close(stdout_fd)
            os.close(stderr_fd)
            os.close(read_fd)
            log_file.close()

    atexit.register(_restore)
    print(f"Saving console log to {log_path}", flush=True)
    return log_path


def _has_camera(mj_model: mujoco.MjModel, camera_name: str) -> bool:
    return mujoco.mj_name2id(
        mj_model,
        mujoco.mjtObj.mjOBJ_CAMERA,
        camera_name,
    ) >= 0


def _parse_override_tokens(tokens: list[str]) -> dict:
    allowed = {field.name for field in fields(Config)}
    override_dict: dict = {}
    for item in tokens:
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        key = key.lstrip("+")
        if key not in allowed:
            continue
        parsed = OmegaConf.to_container(
            OmegaConf.from_dotlist([f"{key}={value}"]), resolve=True
        )
        if isinstance(parsed, dict) and key in parsed:
            override_dict[key] = parsed[key]
    return override_dict


def _extract_cli_overrides(cfg: DictConfig) -> dict:
    """Extract CLI overrides so they can be applied on top of a loaded config."""
    overrides = OmegaConf.select(cfg, "hydra.overrides.task") or []
    override_dict = _parse_override_tokens(overrides)
    if override_dict:
        return override_dict
    return _parse_override_tokens(sys.argv[1:])


def _assert_object_actuator_gains_zero(
    env, config: Config, stage: str, atol: float = 1e-4
) -> None:
    if not config.contact_guidance or not config.object_actuator_ids:
        return
    actuator_ids = np.asarray(config.object_actuator_ids, dtype=int)
    if not hasattr(env, "model_wp") or not hasattr(env.model_wp, "actuator_gainprm"):
        raise AssertionError("MJWarp model does not expose actuator_gainprm.")
    gainprm = wp.to_torch(env.model_wp.actuator_gainprm).detach().cpu().numpy()
    biasprm = wp.to_torch(env.model_wp.actuator_biasprm).detach().cpu().numpy()
    if gainprm.ndim == 3:
        gainprm = gainprm[0]
    if biasprm.ndim == 3:
        biasprm = biasprm[0]
    kp = gainprm[actuator_ids, 0]
    kd = -biasprm[actuator_ids, 1]
    assert np.allclose(kp, 0.0, atol=atol), (
        f"Object actuator Kp not near zero at {stage}: max={np.max(np.abs(kp))}"
    )
    assert np.allclose(kd, 0.0, atol=atol), (
        f"Object actuator Kd not near zero at {stage}: max={np.max(np.abs(kd))}"
    )


def _normalize_yaml_value(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return list(value)
    return value


def _save_config_yaml(config: Config) -> None:
    if not config.save_config:
        return
    config_dict = {}
    for field in fields(config):
        if field.name in _CONFIG_SKIP_FIELDS:
            continue
        config_dict[field.name] = _normalize_yaml_value(getattr(config, field.name))
    output_path = (
        Path(config.output_dir)
        / f"config{'_act' if config.contact_guidance else ''}.yaml"
    )
    OmegaConf.save(config=OmegaConf.create(config_dict), f=str(output_path))
    loguru.logger.info(f"Saved config to {output_path}")


def _get_bimanual_hand_indices(config: Config) -> tuple[list[int], list[int]]:
    robot_nu = int(config.nu)
    if config.contact_guidance:
        obj_dims = (
            int(config.object_action_dims) if config.object_action_dims > 0 else 12
        )
        robot_nu = max(robot_nu - obj_dims, 0)
    half = robot_nu // 2
    right_ids = list(range(0, half))
    left_ids = list(range(half, robot_nu))
    return right_ids, left_ids


def _apply_noise_mask(
    base_noise_scale: torch.Tensor, zero_indices: list[int]
) -> torch.Tensor:
    noise_scale = base_noise_scale.clone()
    if zero_indices:
        idx = torch.as_tensor(
            zero_indices, device=base_noise_scale.device, dtype=torch.long
        )
        noise_scale[:, :, idx] *= 0.0
    return noise_scale


def _get_object_positions_from_qpos(
    config: Config, qpos_world: torch.Tensor
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if config.embodiment_type != "bimanual":
        return None, None
    if config.nq_obj == 12:
        return qpos_world[-12:-9], qpos_world[-6:-3]
    if config.nq_obj == 14:
        return qpos_world[-14:-11], qpos_world[-7:-4]
    return None, None


def _apply_grasp_bias(
    ctrls_for_opt: torch.Tensor,
    ctrl_low: torch.Tensor,
    ctrl_high: torch.Tensor,
    ctrl_limited: torch.Tensor,
    ctrl_ids: list[int],
    ctrl_bias: list[float],
) -> torch.Tensor:
    if not ctrl_ids or not ctrl_bias:
        return ctrls_for_opt
    ctrl_ids_tensor = torch.as_tensor(
        ctrl_ids, device=ctrls_for_opt.device, dtype=torch.long
    )
    ctrl_bias_tensor = torch.as_tensor(
        ctrl_bias, device=ctrls_for_opt.device, dtype=ctrls_for_opt.dtype
    )
    ctrls_for_opt[:, ctrl_ids_tensor] = (
        ctrls_for_opt[:, ctrl_ids_tensor] + ctrl_bias_tensor
    )
    limited_mask = ctrl_limited[ctrl_ids_tensor]
    if torch.any(limited_mask):
        limited_ids = ctrl_ids_tensor[limited_mask]
        ctrls_for_opt[:, limited_ids] = torch.clamp(
            ctrls_for_opt[:, limited_ids],
            ctrl_low[limited_ids],
            ctrl_high[limited_ids],
        )
    return ctrls_for_opt


def _clamp_ctrls_to_range(
    ctrls: torch.Tensor,
    ctrl_low: torch.Tensor,
    ctrl_high: torch.Tensor,
    ctrl_limited: torch.Tensor,
) -> torch.Tensor:
    if not bool(torch.any(ctrl_limited).item()):
        return ctrls
    clamped = ctrls.clone()
    clamped[..., ctrl_limited] = torch.clamp(
        clamped[..., ctrl_limited],
        ctrl_low[ctrl_limited],
        ctrl_high[ctrl_limited],
    )
    return clamped


def _log_ctrl_range_violations(
    label: str,
    ctrls: torch.Tensor,
    ctrl_low: torch.Tensor,
    ctrl_high: torch.Tensor,
    ctrl_limited: torch.Tensor,
) -> None:
    if not bool(torch.any(ctrl_limited).item()):
        return
    limited_ctrls = ctrls[..., ctrl_limited]
    low = ctrl_low[ctrl_limited]
    high = ctrl_high[ctrl_limited]
    below = limited_ctrls < low
    above = limited_ctrls > high
    violation_mask = below | above
    if not bool(torch.any(violation_mask).item()):
        loguru.logger.info("{} controls are within actuator ctrlrange.", label)
        return
    low_violation = torch.where(
        below,
        low - limited_ctrls,
        torch.zeros_like(limited_ctrls),
    )
    high_violation = torch.where(
        above,
        limited_ctrls - high,
        torch.zeros_like(limited_ctrls),
    )
    max_violation = torch.max(torch.maximum(low_violation, high_violation)).item()
    loguru.logger.warning(
        "{} controls exceed actuator ctrlrange: values={}, max_violation={:.4f}",
        label,
        int(torch.count_nonzero(violation_mask).item()),
        max_violation,
    )


def _compute_site_ref_positions(
    mj_model: mujoco.MjModel,
    qpos_ref: torch.Tensor,
    site_ids: list[int],
    device: str,
) -> torch.Tensor:
    if not site_ids:
        return torch.empty((qpos_ref.shape[0], 0, 3), device=device)

    mj_data = mujoco.MjData(mj_model)
    qpos_np = qpos_ref.detach().cpu().numpy()
    site_pos = np.empty((qpos_np.shape[0], len(site_ids), 3), dtype=np.float32)
    for i, qpos in enumerate(qpos_np):
        mj_data.qpos[:] = qpos
        mujoco.mj_forward(mj_model, mj_data)
        site_pos[i] = mj_data.site_xpos[site_ids]
    return torch.from_numpy(site_pos).to(device=device, dtype=torch.float32)


def _final_iteration_value(info: dict, key: str) -> float:
    if key not in info:
        return float("nan")
    value = np.asarray(info[key])
    if value.size == 0:
        return float("nan")
    opt_steps = int(np.asarray(info.get("opt_steps", [value.size])).reshape(-1)[0])
    idx = max(0, min(opt_steps - 1, value.reshape(-1).shape[0] - 1))
    return float(value.reshape(-1)[idx])


def _collect_final_series(info_list: list[dict], key: str) -> np.ndarray:
    values = [_final_iteration_value(info, key) for info in info_list]
    return np.asarray(values, dtype=np.float64)


def _write_reward_breakdown_txt(
    config: Config,
    info_list: list[dict],
    errors: dict | None,
) -> Path | None:
    if not info_list:
        return None

    suffix = "_act" if config.contact_guidance else ""
    output_path = Path(config.output_dir) / f"reward_breakdown_mjwp{suffix}.txt"
    components = [
        ("qpos_rew", "关节/物体位姿跟踪 reward"),
        ("right_arm_qpos_rew", "右臂 7 个关节的 qpos 跟踪 reward"),
        ("left_arm_qpos_rew", "左臂 7 个关节的 qpos 跟踪 reward"),
        ("right_hand_qpos_rew", "右手手指关节 qpos 跟踪 reward"),
        ("left_hand_qpos_rew", "左手手指关节 qpos 跟踪 reward"),
        ("right_object_pos_rew", "真实 milk 物体位置跟踪 reward"),
        ("right_object_rot_rew", "真实 milk 物体旋转跟踪 reward"),
        ("qvel_rew", "速度跟踪 reward"),
        ("contact_rew", "接触点 reward"),
        ("fingertip_rew", "指尖位置跟踪 reward"),
        ("right_fingertip_rew", "右手 5 个指尖位置跟踪 reward"),
        ("left_fingertip_rew", "左手 5 个指尖位置跟踪 reward"),
        ("total_rew", "上述分量相加后的总 reward"),
        ("rew", "采样 rollout 的平均总 reward"),
    ]
    diagnostics = [
        ("right_arm_qpos_dist", "右臂 7 个关节 qpos 加权前误差范数"),
        ("left_arm_qpos_dist", "左臂 7 个关节 qpos 加权前误差范数"),
        ("right_hand_qpos_dist", "右手手指关节 qpos 加权前误差范数"),
        ("left_hand_qpos_dist", "左手手指关节 qpos 加权前误差范数"),
        ("right_arm_qvel_dist", "右臂 7 个关节速度误差范数"),
        ("left_arm_qvel_dist", "左臂 7 个关节速度误差范数"),
        ("right_hand_qvel_dist", "右手手指关节速度误差范数"),
        ("left_hand_qvel_dist", "左手手指关节速度误差范数"),
        ("right_object_pos_dist", "真实 milk 物体位置误差，单位近似为米"),
        ("right_object_rot_dist", "真实 milk 物体旋转误差，来自 quat_sub"),
        ("fingertip_dist", "左右手 10 个指尖位置误差总和，单位米"),
        ("right_fingertip_dist", "右手 5 个指尖位置误差总和，单位米"),
        ("left_fingertip_dist", "左手 5 个指尖位置误差总和，单位米"),
        ("qvel_dist", "整体 qvel 误差范数"),
    ]

    lines: list[str] = []
    lines.append("MJWP reward breakdown")
    lines.append("=" * 80)
    lines.append("")
    lines.append("解释:")
    lines.append("1. 这里的 reward 都是负数形式的 penalty。数值越接近 0，说明误差越小、跟踪越好。")
    lines.append("2. qpos_rew: 机器人关节角 + 有效物体位姿的加权跟踪项；milk 的空 left_object 已被屏蔽。")
    lines.append("3. arm/hand/object 分项是诊断项，单独看哪段机械链没有跟上；它们不是 total_rew 的逐项求和定义。")
    lines.append("4. qvel_rew: 速度跟踪项，属于间接平滑约束，不是单独的动作平滑 reward。")
    lines.append("5. fingertip_rew: 左右手 10 个指尖 site 的世界坐标跟踪项；不包含 palm/wrist 的显式位置 reward。")
    lines.append("6. contact_rew: 接触点 reward；milk 当前没有接触点数据，通常 contact_guidance=false 且该项为 0。")
    lines.append("7. total_rew = qpos_rew + qvel_rew + contact_rew + fingertip_rew。")
    lines.append("8. rew 是优化器 rollout 后对候选轨迹的总 reward 统计，优化器会选择更大的 rew。")
    lines.append("")
    lines.append("配置权重:")
    lines.append(f"- pos_rew_scale: {config.pos_rew_scale}")
    lines.append(f"- rot_rew_scale: {config.rot_rew_scale}")
    lines.append(f"- joint_rew_scale: {config.joint_rew_scale}")
    lines.append(f"- arm_joint_rew_scale: {config.arm_joint_rew_scale}")
    lines.append(f"- hand_joint_rew_scale: {config.hand_joint_rew_scale}")
    lines.append(f"- fingertip_rew_scale: {config.fingertip_rew_scale}")
    lines.append(f"- vel_rew_scale: {config.vel_rew_scale}")
    lines.append(f"- contact_rew_scale: {config.contact_rew_scale}")
    lines.append(f"- contact_guidance: {config.contact_guidance}")
    if errors is not None:
        lines.append("")
        lines.append("最终物体跟踪误差:")
        lines.append(f"- obj_pos_err: {errors.get('obj_pos_err', float('nan')):.6f}")
        lines.append(f"- obj_quat_err: {errors.get('obj_quat_err', float('nan')):.6f}")

    lines.append("")
    lines.append("全程统计: 每个控制步取最后一次 optimizer iteration 的 mean 值")
    for name, desc in components:
        key = f"{name}_mean"
        series = _collect_final_series(info_list, key)
        finite = series[np.isfinite(series)]
        if finite.size == 0:
            continue
        lines.append(
            f"- {name:14s} ({desc}): mean={finite.mean(): .6f}, "
            f"median={np.median(finite): .6f}, min={finite.min(): .6f}, "
            f"max={finite.max(): .6f}, last={finite[-1]: .6f}"
        )

    lines.append("")
    lines.append("诊断误差统计: 数值越小越好")
    for name, desc in diagnostics:
        key = f"{name}_mean"
        series = _collect_final_series(info_list, key)
        finite = series[np.isfinite(series)]
        if finite.size == 0:
            continue
        lines.append(
            f"- {name:22s} ({desc}): mean={finite.mean(): .6f}, "
            f"median={np.median(finite): .6f}, min={finite.min(): .6f}, "
            f"max={finite.max(): .6f}, last={finite[-1]: .6f}"
        )

    lines.append("")
    lines.append("逐控制步 reward 分量:")
    header = (
        "step sim_time qpos_rew_mean qvel_rew_mean contact_rew_mean "
        "fingertip_rew_mean right_arm_qpos_rew_mean left_arm_qpos_rew_mean "
        "right_hand_qpos_rew_mean left_hand_qpos_rew_mean "
        "right_fingertip_rew_mean left_fingertip_rew_mean "
        "right_object_pos_rew_mean total_rew_mean rew_mean"
    )
    lines.append(header)
    for step_idx, info in enumerate(info_list):
        time_arr = np.asarray(info.get("time", [float("nan")])).reshape(-1)
        sim_time = float(time_arr[-1]) if time_arr.size > 0 else float("nan")
        row_values = [
            _final_iteration_value(info, "qpos_rew_mean"),
            _final_iteration_value(info, "qvel_rew_mean"),
            _final_iteration_value(info, "contact_rew_mean"),
            _final_iteration_value(info, "fingertip_rew_mean"),
            _final_iteration_value(info, "right_arm_qpos_rew_mean"),
            _final_iteration_value(info, "left_arm_qpos_rew_mean"),
            _final_iteration_value(info, "right_hand_qpos_rew_mean"),
            _final_iteration_value(info, "left_hand_qpos_rew_mean"),
            _final_iteration_value(info, "right_fingertip_rew_mean"),
            _final_iteration_value(info, "left_fingertip_rew_mean"),
            _final_iteration_value(info, "right_object_pos_rew_mean"),
            _final_iteration_value(info, "total_rew_mean"),
            _final_iteration_value(info, "rew_mean"),
        ]
        lines.append(
            f"{step_idx:04d} {sim_time: .4f} "
            + " ".join(f"{value: .6f}" for value in row_values)
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    summary = {
        name: _collect_final_series(info_list, f"{name}_mean")
        for name, _desc in components
    }
    summary_msg = []
    for name in [
        "qpos_rew",
        "right_arm_qpos_rew",
        "left_arm_qpos_rew",
        "right_hand_qpos_rew",
        "left_hand_qpos_rew",
        "qvel_rew",
        "contact_rew",
        "fingertip_rew",
        "right_fingertip_rew",
        "left_fingertip_rew",
        "total_rew",
        "rew",
    ]:
        finite = summary[name][np.isfinite(summary[name])]
        if finite.size > 0:
            summary_msg.append(f"{name}={finite.mean():.4f}")
    loguru.logger.info("Reward summary mean: {}", ", ".join(summary_msg))
    loguru.logger.info("Saved reward breakdown to {}", output_path)
    return output_path


def main(config: Config):
    """Run the SPIDER using MuJoCo Warp backend"""

    # 1. 配置补全+加载参考轨迹
    # process config, set defaults and derived fields
    # 这个函数干的事在 spider/config.py 中，根据输入的配置给出真正的文件路径
    config = process_config(config)
    _start_console_log(config)
    if config.contact_guidance and config.improvement_threshold > 0.0:
        loguru.logger.warning(
            "contact_guidance requires improvement_threshold <= 0; overriding to 0.0."
        )
        config.improvement_threshold = 0.0

    # load reference data (already interpolated and extended)
    # load_data 读 trajectory_kinematic.npz，得到下面这些东西：
    #   qpos_ref: (T, nq) 每个时刻机器人 + 物体的参考位姿（来自 IK 的结果）
    #   qvel_ref: (T, nv) 对应的速度
    #   ctrl_ref: (T, nu) 初步的参考控制量
    #   contact: (T, contact_dim) 预测出来的接触点信息（可选，用于接触引导）
    #   contact_pos: (T, contact_dim, 3) 预测出来的接触点信息（可选，用于接触引导）
    qpos_ref, qvel_ref, ctrl_ref, contact, contact_pos = load_data(
        config, config.data_path
    )
    if (config.contact_guidance
        and ctrl_ref.shape[1] != config.nu
        and qpos_ref.shape[1] >= config.nu):
        loguru.logger.info(
            "Using qpos as ctrl reference for contact guidance (ctrl dims: {} -> {}).",
            ctrl_ref.shape[1],
            config.nu,
        )
        ctrl_ref = qpos_ref[:, : config.nu]
    if config.contact_guidance and torch.all(contact <= 0):
        raise ValueError("contact_guidance is enabled, but contact mask is all zeros.")
    # hack: start from step 500
    # qpos_ref = qpos_ref[500:]
    # qvel_ref = qvel_ref[500:]
    # ctrl_ref = ctrl_ref[500:]
    # contact = contact[500:]
    # contact_pos = contact_pos[500:]
    config.max_sim_steps = (config.max_sim_steps
        if config.max_sim_steps > 0
        else qpos_ref.shape[0] - config.horizon_steps - config.ctrl_steps)

    # 建立 MuJoCo Warp 仿真环境
    # setup mujoco model early so reference controls can be made legal before
    # MJWarp initialization and optimizer seeding.
    mj_model = setup_mj_model(config)
    ctrl_low = torch.from_numpy(mj_model.actuator_ctrlrange[:, 0]).to(config.device, dtype=torch.float32)
    ctrl_high = torch.from_numpy(mj_model.actuator_ctrlrange[:, 1]).to(config.device, dtype=torch.float32)
    ctrl_limited = torch.from_numpy(mj_model.actuator_ctrllimited.astype(bool)).to(config.device)
    _log_ctrl_range_violations("Reference",ctrl_ref,ctrl_low,ctrl_high,ctrl_limited,)
    ctrl_ref = _clamp_ctrls_to_range(ctrl_ref, ctrl_low, ctrl_high, ctrl_limited)
    fingertip_pos_ref = _compute_site_ref_positions(
        mj_model, qpos_ref, config.fingertip_site_ids, config.device
    )
    if config.fingertip_rew_scale > 0.0:
        loguru.logger.info(
            "Prepared fingertip reference trajectory with shape {}.",
            tuple(fingertip_pos_ref.shape),
        )
    ref_data = (qpos_ref, qvel_ref, ctrl_ref, contact, contact_pos, fingertip_pos_ref)

    # 2. 建仿真环境
    # setup env with initial state from first sim qpos
    '''
    放入下面这些东西
        - scene.xml 里的完整 MuJoCo model
        - IK 第 0 帧 qpos_ref[0]
        - IK 第 0 帧速度 qvel_ref[0]
        - 初始控制 ctrl_ref[0]
        - 并行仿真世界数量 num_samples
        - contact buffer 大小 nconmax_per_env
        - joint/contact constraint buffer 大小 njmax_per_env
    '''
    
    env = setup_env(config, ref_data)   # 建立 MJWarp 的批量仿真环境

    # setup mujoco (for viewer only)
    # 用 IK 的第 0 帧初始化机器人和物体（这似乎是我加的）
    mj_data = mujoco.MjData(mj_model)   # 仅用于 viewer 和记录
    mj_data_ref = mujoco.MjData(mj_model)
    mj_data.qpos[:] = qpos_ref[0].detach().cpu().numpy()
    mj_data.qvel[:] = qvel_ref[0].detach().cpu().numpy()
    mj_data.ctrl[:] = ctrl_ref[0].detach().cpu().numpy()
    mj_data.time = 0.0
    mujoco.mj_forward(mj_model, mj_data)
    env_qpos0 = get_qpos(config, env)[0].detach().cpu()
    env_qvel0 = get_qvel(config, env)[0].detach().cpu()
    qpos_init_error = torch.max(
        torch.abs(env_qpos0 - qpos_ref[0].detach().cpu())
    ).item()
    qvel_init_error = torch.max(
        torch.abs(env_qvel0 - qvel_ref[0].detach().cpu())
    ).item()
    loguru.logger.info("MJWP initial state error vs IK frame 0: qpos={:.3e}, qvel={:.3e}",qpos_init_error,qvel_init_error,)
    _assert_object_actuator_gains_zero(env, config, "start")
    images = []
    images_clean = []
    images_d435 = []
    images_d435_clean = []
    object_trace_site_ids = []
    robot_trace_site_ids = []
    for sid in range(mj_model.nsite):
        name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_SITE, sid)
        if name is not None:
            if name.startswith("trace"):
                if "object" in name:
                    object_trace_site_ids.append(sid)
                else:
                    robot_trace_site_ids.append(sid)
    config.trace_site_ids = object_trace_site_ids + robot_trace_site_ids
    contact_guidance_enabled = (config.contact_guidance and len(config.object_actuator_ids) > 0)
    grasp_bias_enabled = (
        config.robot_type == "asm"
        and config.embodiment_type == "bimanual"
        and (
            len(config.right_grasp_ctrl_ids) > 0 or len(config.left_grasp_ctrl_ids) > 0
        )
    )
    ctrl_low = torch.from_numpy(mj_model.actuator_ctrlrange[:, 0]).to(config.device, dtype=torch.float32)
    ctrl_high = torch.from_numpy(mj_model.actuator_ctrlrange[:, 1]).to(config.device, dtype=torch.float32)
    ctrl_limited = torch.from_numpy(mj_model.actuator_ctrllimited.astype(bool)).to(config.device)
    if config.contact_guidance and not contact_guidance_enabled:
        loguru.logger.warning(
            "contact_guidance is enabled but no object actuators were resolved."
        )
    contact_offset = 0
    if contact_guidance_enabled:
        config.contact_len = int(
            min(contact.shape[1], contact_pos.shape[1], len(config.contact_order))
        )
        if (
            config.contact_len != len(config.contact_order)
            or config.contact_len != contact.shape[1]):
            loguru.logger.warning(
                "Contact length mismatch (mask={}, pos={}, expected={}); truncating to {}.",
                contact.shape[1],
                contact_pos.shape[1],
                len(config.contact_order),
                config.contact_len,
            )
        config.contact_order = config.contact_order[: config.contact_len]
        config.hand_contact_site_ids = config.hand_contact_site_ids[
            : config.contact_len
        ]
        contact_offset = max(contact.shape[1] - config.contact_len, 0)

    # setup env params
    env_params_list = []
    if config.num_dr == 0:
        xy_offset_list = [0.0]
        pair_margin_list = [0.0]
    else:
        xy_offset_list = np.linspace(
            config.xy_offset_range[0], config.xy_offset_range[1], config.num_dr
        )
        pair_margin_list = np.linspace(
            config.pair_margin_range[0], config.pair_margin_range[1], config.num_dr
        )
    kp_schedule = []
    kd_schedule = []
    if contact_guidance_enabled and config.max_num_iterations > 0:
        actuator_names = config.object_actuator_names
        if not actuator_names:
            actuator_names = [
                mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, int(aid))
                for aid in config.object_actuator_ids
            ]
        base_kp = np.array(
            [
                (
                    config.init_rot_actuator_gain
                    if ("_rot_" in (name or ""))
                    else config.init_pos_actuator_gain
                )
                for name in actuator_names
            ],
            dtype=np.float32,
        )
        base_kd = np.array(
            [
                (
                    config.init_rot_actuator_bias
                    if ("_rot_" in (name or ""))
                    else config.init_pos_actuator_bias
                )
                for name in actuator_names
            ],
            dtype=np.float32,
        )
        for i in range(config.max_num_iterations):
            decay = float(config.guidance_decay_ratio) ** i
            kp_i = base_kp * decay
            kd_i = base_kd * decay
            if i == config.max_num_iterations - 1:
                kp_i = np.zeros_like(base_kp, dtype=np.float32)
                kd_i = np.zeros_like(base_kd, dtype=np.float32)
            kp_schedule.append(kp_i)
            kd_schedule.append(kd_i)

    # 3. 批量环境参数+接触引导日程表，创建 32 个并行世界
    # 每一次优化迭代，每一个 domain randomization 分组各自的仿真参数。
    for i in range(config.max_num_iterations):
        env_params = []
        for j in range(config.num_dr):
            params = {
                "xy_offset": xy_offset_list[j],
                "pair_margin": pair_margin_list[j],
            }
            if contact_guidance_enabled and kp_schedule:
                params["kp"] = kp_schedule[i]
                params["kd"] = kd_schedule[i]
            env_params.append(params)
        env_params_list.append(env_params)
    config.env_params_list = env_params_list
    _save_config_yaml(config)

    # 4. 搭优化器
    '''
    注意，在这里，优化器并不是像 Adam 这种优化器，而是一个采样式的 MPC 优化算法。
    它的核心是一个 rollout 函数，可以在仿真环境里执行一串控制指令，得到对应的奖励和轨迹信息。然后 optimize_once 就是在这个基础上做一次优化（比如 CEM 的一次采样和更新），make_optimize_fn 则是把这个 optimize_once 包裹成一个循环，直到收敛或者达到最大迭代次数。
    
    它不训练神经网络。它做的是：
    1. 当前有一串控制序列 ctrls。
    2. 给这串控制加随机噪声，生成很多候选控制序列。
    3. 每个候选控制序列都在 MuJoCo/Warp 里跑一遍。
    4. 看哪个候选能让机器人/物体更接近 IK reference。
    5. 把得分高的候选加权平均，得到新的控制序列。
    6. 只执行前几个控制 step，然后重新规划。
    '''
    
    # setup viewer and renderer
    run_viewer = setup_viewer(config, mj_model, mj_data)
    renderer = setup_renderer(config, mj_model)
    render_d435_video = (
        config.save_video
        and renderer is not None
        and _has_camera(mj_model, _D435_RENDER_CAMERA_NAME)
    )
    if config.save_video and renderer is not None and not render_d435_video:
        loguru.logger.warning(
            "D435 render camera '{}' not found in model; MJWP D435 videos will not be saved.",
            _D435_RENDER_CAMERA_NAME,
        )
    elif render_d435_video:
        loguru.logger.info(
            "MJWP D435 videos will be rendered from camera '{}'.",
            _D435_RENDER_CAMERA_NAME,
        )

    # setup optimizer
    # 把环境接口（step、save_state、load_state、get_reward 等）打包成一个可以在 1024 个并行环境里跑完整个 horizon的函数。
    # rollout 的意思是给一串控制指令，实际在仿真中跑一遍，然后计算这串控制好不好
    '''
        1. 对当前控制加随机噪声，生成很多候选动作。
        2. 每个候选动作都 rollout 一遍。
        3. 根据 reward 挑好的。
        4. 把好的动作加权平均成新的控制。
    '''
    rollout = make_rollout_fn(
        step_env,
        save_state,
        load_state,
        get_reward,
        get_terminal_reward,
        get_terminate,
        get_trace,
        save_env_params,
        load_env_params,
        copy_sample_state,
    )
    optimize_once = make_optimize_once_fn(rollout)
    # make_optimize_fn 则把这个 rollout 套进采样式 MPC 的外循环
    '''
    1. 从当前控制 ctrls 加噪声，生成 1024 条候选
    2. 并行跑 rollout，拿到每条的总奖励
    3. 用 softmax（温度 = temperature）做加权平均，得到更好的 ctrls
    4. 重复 max_num_iterations=32 次或直到改进小于 improvement_threshold

    这一套算法就是 DIAL-MPC （Diffusion-Annealed Local Search MPC），是 SPIDER 的核心。
    '''
    optimize = make_optimize_fn(optimize_once)
    base_noise_scale = config.noise_scale.clone()
    gibbs_enabled = config.gibbs_sampling and config.embodiment_type == "bimanual"
    if config.gibbs_sampling and not gibbs_enabled:
        loguru.logger.warning(
            "gibbs_sampling is enabled but embodiment_type is {}, disabling.",
            config.embodiment_type,
        )
    if gibbs_enabled:
        right_ids, left_ids = _get_bimanual_hand_indices(config)
        right_only_zero = left_ids
        left_only_zero = right_ids

    # initial controls 初始化 optimizer 的控制序列
    ctrls = _clamp_ctrls_to_range(
        ctrl_ref[: config.horizon_steps],
        ctrl_low,
        ctrl_high,
        ctrl_limited,
    )
    # buffers for saving info and trajectory
    info_list = []

    # run viewer + control loop
    t_start = time.perf_counter()

    # 5. 主控制循环
    with run_viewer() as viewer:
        right_grasp_locked = False
        left_grasp_locked = False
        while viewer.is_running():
            t0 = time.perf_counter()    # 有 N 个并行世界，主线的真实执行状态是看第 0 个世界

            # 算当前处于参考轨迹的第几帧
            # optimize using future reference window at control-rate (+1 lookahead)
            sim_step = int(np.round(mj_data.time / config.sim_dt))
            # 优化控制时，看未来 t+1 到 t+Horizon 的参考轨迹，得到更好的控制序列
            ref_slice = get_slice(
                ref_data, sim_step + 1, sim_step + config.horizon_steps + 1
            )
            ctrls_for_opt = ctrls
            if grasp_bias_enabled:
                qpos_world = wp.to_torch(env.data_wp.qpos)[0]
                site_xpos = wp.to_torch(env.data_wp.site_xpos)[0]
                right_object_pos, left_object_pos = _get_object_positions_from_qpos(
                    config, qpos_world
                )
                if (
                    not right_grasp_locked
                    and right_object_pos is not None
                    and config.right_palm_site_id >= 0
                ):
                    right_grasp_locked = bool(
                        torch.norm(
                            site_xpos[config.right_palm_site_id] - right_object_pos, p=2
                        ).item()
                        < 0.09
                    )
                if (
                    not left_grasp_locked
                    and left_object_pos is not None
                    and config.left_palm_site_id >= 0
                ):
                    left_grasp_locked = bool(
                        torch.norm(
                            site_xpos[config.left_palm_site_id] - left_object_pos, p=2
                        ).item()
                        < 0.09
                    )
                if right_grasp_locked:
                    if ctrls_for_opt is ctrls:
                        ctrls_for_opt = ctrls_for_opt.clone()
                    ctrls_for_opt = _apply_grasp_bias(
                        ctrls_for_opt,
                        ctrl_low,
                        ctrl_high,
                        ctrl_limited,
                        config.right_grasp_ctrl_ids,
                        config.right_grasp_ctrl_bias,
                    )
                if left_grasp_locked:
                    if ctrls_for_opt is ctrls:
                        ctrls_for_opt = ctrls_for_opt.clone()
                    ctrls_for_opt = _apply_grasp_bias(
                        ctrls_for_opt,
                        ctrl_low,
                        ctrl_high,
                        ctrl_limited,
                        config.left_grasp_ctrl_ids,
                        config.left_grasp_ctrl_bias,
                    )
            if contact_guidance_enabled and config.contact_len > 0:
                contact_mask_step = contact[sim_step][
                    contact_offset : contact_offset + config.contact_len
                ]
                contact_pos_ref_step = contact_pos[sim_step]
                site_xpos = wp.to_torch(env.data_wp.site_xpos)[0]

                right_delta = compute_contact_point_delta(
                    contact_mask_step,
                    contact_pos_ref_step,
                    site_xpos,
                    config.hand_contact_site_ids,
                    config.right_contact_indices,
                )
                left_delta = compute_contact_point_delta(
                    contact_mask_step,
                    contact_pos_ref_step,
                    site_xpos,
                    config.hand_contact_site_ids,
                    config.left_contact_indices,
                )
                if (
                    right_delta is not None
                    and config.right_pos_ctrl_ids
                    and sim_step + ctrls.shape[0] <= ctrl_ref.shape[0]
                ):
                    ctrls_for_opt = ctrls_for_opt.clone()
                    ref_ctrl_slice = ctrl_ref[sim_step : sim_step + ctrls.shape[0]]
                    ctrls_for_opt[:, config.right_pos_ctrl_ids] = ref_ctrl_slice[
                        :, config.right_pos_ctrl_ids
                    ] + torch.clip(right_delta, -0.01, 0.01)
                if (
                    left_delta is not None
                    and config.left_pos_ctrl_ids
                    and sim_step + ctrls.shape[0] <= ctrl_ref.shape[0]
                ):
                    if ctrls_for_opt is ctrls:
                        ctrls_for_opt = ctrls_for_opt.clone()
                        ref_ctrl_slice = ctrl_ref[sim_step : sim_step + ctrls.shape[0]]
                    ctrls_for_opt[:, config.left_pos_ctrl_ids] = ref_ctrl_slice[
                        :, config.left_pos_ctrl_ids
                    ] + torch.clip(left_delta, -0.01, 0.01)
            ctrls_for_opt = _clamp_ctrls_to_range(
                ctrls_for_opt,
                ctrl_low,
                ctrl_high,
                ctrl_limited,
            )
            if gibbs_enabled:
                config.noise_scale = _apply_noise_mask(
                    base_noise_scale, right_only_zero
                )
                ctrls, infos = optimize(config, env, ctrls_for_opt, ref_slice)
                config.noise_scale = _apply_noise_mask(base_noise_scale, left_only_zero)
                ctrls, infos = optimize(config, env, ctrls, ref_slice)
                config.noise_scale = base_noise_scale
            else:
                # 这个 retargeting 是 MJWP 的核心
                config.noise_scale = base_noise_scale
                ctrls, infos = optimize(config, env, ctrls_for_opt, ref_slice)
            ctrls = _clamp_ctrls_to_range(ctrls, ctrl_low, ctrl_high, ctrl_limited)

            # Compute trace_ref from reference qpos over the horizon
            if len(config.trace_site_ids) > 0:
                trace_ref = []
                qpos_ref_horizon = ref_slice[0]
                for h in range(config.horizon_steps):
                    mj_data_ref.qpos[:] = qpos_ref_horizon[h].detach().cpu().numpy()
                    mujoco.mj_kinematics(mj_model, mj_data_ref)
                    site_xpos = np.array(
                        [mj_data_ref.site_xpos[sid] for sid in config.trace_site_ids]
                    )
                    trace_ref.append(site_xpos)
                # (H, K, 3) -> (1, 1, H, K, 3) to match trace_sample shape
                trace_ref_np = np.stack(trace_ref, axis=0)[None, None, :, :, :]
                infos["trace_ref"] = trace_ref_np

            # step environment for ctrl_steps
            step_info = {"qpos": [], "qvel": [], "time": [], "ctrl": []}
            for i in range(config.ctrl_steps):
                ctrl_step = _clamp_ctrls_to_range(
                    ctrls[i],
                    ctrl_low,
                    ctrl_high,
                    ctrl_limited,
                )

                # option 1: use mujoco step
                # mj_data.ctrl[:] = ctrls[i].detach().cpu().numpy()
                # mujoco.mj_step(mj_model, mj_data)
                # option 2: use warp step
                step_env(config, env, ctrl_step)
                mj_data.qpos[:] = get_qpos(config, env)[0].detach().cpu().numpy()
                mj_data.qvel[:] = get_qvel(config, env)[0].detach().cpu().numpy()
                mj_data.ctrl[:] = ctrl_step.detach().cpu().numpy()
                mj_data.time += config.sim_dt
                if config.save_video and renderer is not None:
                    if i % int(np.round(config.render_dt / config.sim_dt)) == 0:
                        mj_data_ref.qpos[:] = (
                            qpos_ref[sim_step + i].detach().cpu().numpy()
                        )
                        image = render_image(
                            config,
                            renderer,
                            mj_model,
                            mj_data,
                            mj_data_ref,
                            include_helpers=True,
                        )
                        image_clean = render_image(
                            config,
                            renderer,
                            mj_model,
                            mj_data,
                            mj_data_ref,
                            include_helpers=False,
                        )
                        images.append(image)
                        images_clean.append(image_clean)
                        if render_d435_video:
                            image_d435 = render_image(
                                config,
                                renderer,
                                mj_model,
                                mj_data,
                                mj_data_ref,
                                include_helpers=True,
                                camera=_D435_RENDER_CAMERA_NAME,
                            )
                            image_d435_clean = render_image(
                                config,
                                renderer,
                                mj_model,
                                mj_data,
                                mj_data_ref,
                                include_helpers=False,
                                camera=_D435_RENDER_CAMERA_NAME,
                            )
                            images_d435.append(image_d435)
                            images_d435_clean.append(image_d435_clean)
                if "rerun" in config.viewer or "viser" in config.viewer:
                    mj_data_ref.qpos[:] = qpos_ref[sim_step + i].detach().cpu().numpy()
                    mujoco.mj_kinematics(mj_model, mj_data_ref)
                    log_frame(
                        mj_data,
                        sim_time=mj_data.time,
                        viewer_body_entity_and_ids=config.viewer_body_entity_and_ids,
                        data_ref=mj_data_ref,
                    )
                step_info["qpos"].append(mj_data.qpos.copy())
                step_info["qvel"].append(mj_data.qvel.copy())
                step_info["time"].append(mj_data.time)
                step_info["ctrl"].append(mj_data.ctrl.copy())
            for k in step_info:
                step_info[k] = np.stack(step_info[k], axis=0)
            infos.update(step_info)
            # sync env state
            sync_env(config, env, mj_data)

            # receding horizon update
            sim_step = int(np.round(mj_data.time / config.sim_dt))
            prev_ctrl = ctrls[config.ctrl_steps :]
            new_ctrl = ctrl_ref[
                sim_step + prev_ctrl.shape[0] : sim_step
                + prev_ctrl.shape[0]
                + config.ctrl_steps
            ]
            ctrls = _clamp_ctrls_to_range(
                torch.cat([prev_ctrl, new_ctrl], dim=0),
                ctrl_low,
                ctrl_high,
                ctrl_limited,
            )

            # sync viewer state and render
            mj_data.qpos[:] = get_qpos(config, env)[0].detach().cpu().numpy()
            mj_data.qvel[:] = get_qvel(config, env)[0].detach().cpu().numpy()
            mj_data_ref.qpos[:] = qpos_ref[sim_step].detach().cpu().numpy()
            update_viewer(config, viewer, mj_model, mj_data, mj_data_ref, infos)

            # progress
            t1 = time.perf_counter()
            rtr = config.ctrl_dt / (t1 - t0)
            print(
                f"Realtime rate: {rtr:.2f}, plan time: {t1 - t0:.4f}s, sim_steps: {sim_step}/{config.max_sim_steps}, opt_steps: {infos['opt_steps'][0]}",
                end="\n",
                flush=True,
            )

            # record info/trajectory at control tick
            # rule out "trace"
            info_list.append({k: v for k, v in infos.items() if k != "trace_sample"})

            if sim_step >= config.max_sim_steps:
                break

        t_end = time.perf_counter()
        print(f"Total time: {t_end - t_start:.4f}s")

    # 6. 保存结果
    # save retargeted trajectory
    if config.save_info and len(info_list) > 0:
        info_aggregated = {}
        for k in info_list[0].keys():
            info_aggregated[k] = np.stack([info[k] for info in info_list], axis=0)
        # npz，包含每一步的 qpos、qvel、ctrl、reward、trace_sample 等等
        np.savez(
            f"{config.output_dir}/trajectory_mjwp{'_act' if config.contact_guidance else ''}.npz",
            **info_aggregated,
        )
        loguru.logger.info(
            f"Saved info to {config.output_dir}/trajectory_mjwp{'_act' if config.contact_guidance else ''}.npz"
        )

    # save video
    # 写 mp4，文件名里带不带 _act 取决于是否用了 contact guidance
    if config.save_video and len(images) > 0:
        video_path = f"{config.output_dir}/visualization_mjwp{'_act' if config.contact_guidance else ''}.mp4"
        imageio.mimsave(
            video_path,
            images,
            fps=int(1 / config.render_dt),
        )
        loguru.logger.info(f"Saved video to {video_path}")
    if config.save_video and len(images_clean) > 0:
        video_path_clean = f"{config.output_dir}/visualization_mjwp_clean{'_act' if config.contact_guidance else ''}.mp4"
        imageio.mimsave(
            video_path_clean,
            images_clean,
            fps=int(1 / config.render_dt),
        )
        loguru.logger.info(f"Saved clean video to {video_path_clean}")
    if config.save_video and len(images_d435) > 0:
        video_path_d435 = f"{config.output_dir}/visualization_mjwp_d435{'_act' if config.contact_guidance else ''}.mp4"
        imageio.mimsave(
            video_path_d435,
            images_d435,
            fps=int(1 / config.render_dt),
        )
        loguru.logger.info(f"Saved D435 video to {video_path_d435}")
    if config.save_video and len(images_d435_clean) > 0:
        video_path_d435_clean = f"{config.output_dir}/visualization_mjwp_d435_clean{'_act' if config.contact_guidance else ''}.mp4"
        imageio.mimsave(
            video_path_d435_clean,
            images_d435_clean,
            fps=int(1 / config.render_dt),
        )
        loguru.logger.info(f"Saved clean D435 video to {video_path_d435_clean}")

    # 算成功率指标
    errors = None
    if info_list:
        qpos_traj = np.concatenate([info["qpos"] for info in info_list], axis=0)
        qpos_ref_np = qpos_ref[: qpos_traj.shape[0]].detach().cpu().numpy()
        data_type = "mjwp_act" if config.contact_guidance else "mjwp"
        errors = compute_object_tracking_error(
            qpos_traj, qpos_ref_np, config.embodiment_type, data_type
        )
        loguru.logger.info(
            "Final object tracking error: pos={:.4f}, quat={:.4f}",
            errors["obj_pos_err"],
            errors["obj_quat_err"],
        )
    _write_reward_breakdown_txt(config, info_list, errors)

    _assert_object_actuator_gains_zero(env, config, "end")

    if "viser" in config.viewer and config.wait_on_finish:
        loguru.logger.info(
            "Optimization complete! Keeping Viser server alive. Press Ctrl+C to exit."
        )
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass

    return errors


@hydra.main(version_base=None, config_path="config", config_name="default")
def run_main(cfg: DictConfig) -> None:
    """Entry point for Hydra configuration runner."""
    # Convert DictConfig to Config dataclass, handling special fields
    config_dict = dict(cfg)

    # Optionally load a saved config YAML and merge; CLI overrides take priority.
    # 解析配置文件，有上层配置文件和覆盖配置文件
    # 默认运行的话，就是 configs/default.yaml，没有覆盖配置文件
    # 如果指定了覆盖配置文件，比如 configs/override/oakink.yaml，就会用这个覆盖配置文件中的字段覆盖默认配置文件中的字段
    load_config_path = config_dict.get("load_config_path", "")
    if load_config_path:
        loaded_config = load_config_yaml(load_config_path)
        cli_overrides = _extract_cli_overrides(cfg)
        config_dict = {**loaded_config, **cli_overrides}
    else:
        config_dict = filter_config_fields(config_dict)

    # Handle special conversions
    if "noise_scale" in config_dict and config_dict["noise_scale"] is None:
        config_dict.pop("noise_scale")  # Let the default factory handle it

    # Convert lists to tuples where needed
    if "pair_margin_range" in config_dict:
        config_dict["pair_margin_range"] = tuple(config_dict["pair_margin_range"])
    if "xy_offset_range" in config_dict:
        config_dict["xy_offset_range"] = tuple(config_dict["xy_offset_range"])

    config = Config(**config_dict)
    main(config)


if __name__ == "__main__":
    run_main()
