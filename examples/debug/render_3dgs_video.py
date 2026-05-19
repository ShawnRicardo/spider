#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
render_custom_trajectory.py
===========================
从 RoboSimGS++ output 文件夹中读取已对齐的 3DGS 场景（gaussians_aligned.ply）
和 DA3 估计的相机内参，配合 **任意输入的相机外参序列** 渲染成视频。

核心函数
--------
render_trajectory(
    output_dir   : str | Path,      # RoboSimGS++ workspace 根目录（含 result/ 和 da3.npz）
    cam_c2w      : np.ndarray,      # 相机外参序列，shape = (T, 4, 4)，camera-to-world，float32/64
    video_path   : str | Path,      # 输出视频路径，如 "output/render.mp4"
    width        : int = 720,       # 渲染宽度（像素），默认对齐 milk D435 视频
    height       : int = 480,       # 渲染高度（像素），默认对齐 milk D435 视频
    fps          : int = 50,        # 输出视频帧率，默认对齐 milk D435 视频
    frame_dir    : str | Path,      # 输出逐帧 PNG，供后续前景合成使用
)

相机外参格式说明
----------------
cam_c2w 是 **camera-to-world** 变换矩阵序列，shape = (T, 4, 4)。
每一帧的 4×4 矩阵含义：

    [R | t]   其中 R (3×3) 是相机朝向（列向量为相机 x/y/z 轴在世界坐标系下的方向），
    [0 | 1]         t (3,)  是相机光心在世界坐标系下的位置。

坐标系约定（与 DA3 / AnySplat 一致）：
  - 世界坐标系：右手系，Y 轴向上（或视场景而定，由 DA3 输出决定）
  - 相机坐标系：X 右、Y 下、Z 前（OpenCV 标准）

快速构造示例
------------
# 静止相机（直接用 DA3 第 0 帧外参重复 T 帧）
cam_c2w = np.tile(da3_cam_c2w[0:1], (T, 1, 1))

# 圆弧绕场景运动（伪代码）
for i in range(T):
    theta = 2 * np.pi * i / T
    R = rotation_matrix_y(theta)
    t = np.array([r*cos(theta), height, r*sin(theta)])
    cam_c2w[i] = np.eye(4); cam_c2w[i,:3,:3]=R; cam_c2w[i,:3,3]=t

main() 中的 test 使用 DA3 估出的真实外参序列渲染 milk 背景视频，
默认保存到 preprocessed/milk/3dgs_bg_videos/frames 和
preprocessed/milk/3dgs_videos/render_da3_traj.mp4。

依赖
----
  conda env: anysplat（即 CONDA_ENVS["anysplat"]）
  需要安装：numpy, opencv-python, plyfile, torch, scipy, Pillow
  AnySplat 路径：ANYSPLAT_ROOT（见下方常量，按需修改）
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# ★ 修改这两行以匹配你的环境
# ---------------------------------------------------------------------------
ANYSPLAT_ROOT = "/data/zhaohaoyu/lcy-embodiedai/AnySplat"
ANYSPLAT_PYTHON = "/data/zhaohaoyu/condaenv/envs/anysplat/bin/python"
# ---------------------------------------------------------------------------

DEFAULT_WORKSPACE = Path("preprocessed/milk")
DEFAULT_BG_FRAME_DIR = DEFAULT_WORKSPACE / "3dgs_bg_videos" / "frames"
DEFAULT_BG_VIDEO_PATH = DEFAULT_WORKSPACE / "3dgs_videos" / "render_da3_traj.mp4"
DEFAULT_RENDER_WIDTH = 720
DEFAULT_RENDER_HEIGHT = 480
DEFAULT_RENDER_FPS = 50
DEFAULT_MAX_FRAMES = 120


# ── PLY 工具 ────────────────────────────────────────────────────────────────

def _load_gaussians_from_ply(ply_path: Path):
    """
    读取 3DGS PLY 文件，返回 CUDA tensors。
    PLY 字段：x y z  scale_0/1/2  rot_0/1/2/3  opacity  f_dc_0/1/2
    """
    from plyfile import PlyData

    ply  = PlyData.read(str(ply_path))
    v    = ply.elements[0]

    means  = torch.tensor(
        np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float32), device="cuda")
    scales = torch.exp(torch.tensor(
        np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], 1).astype(np.float32),
        device="cuda"))
    rots   = torch.tensor(
        np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], 1).astype(np.float32),
        device="cuda")
    opacs  = torch.tensor(
        np.array(v["opacity"], dtype=np.float32), device="cuda").clamp(0, 1)
    harms  = torch.tensor(
        np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], 1).astype(np.float32)[:, :, None],
        device="cuda")

    print(f"  高斯数量: {len(means):,}   opacity [{opacs.min():.3f}, {opacs.max():.3f}]")
    return means, scales, rots, opacs, harms


# ── 协方差工具 ───────────────────────────────────────────────────────────────

def _build_covariance(rots: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """四元数 + 缩放 → 3D 协方差矩阵  Σ = R·diag(s²)·Rᵀ"""
    w, x, y, z = rots[:, 0], rots[:, 1], rots[:, 2], rots[:, 3]
    R = torch.stack([
        torch.stack([1-2*(y*y+z*z),   2*(x*y-w*z),   2*(x*z+w*y)], -1),
        torch.stack([  2*(x*y+w*z), 1-2*(x*x+z*z),   2*(y*z-w*x)], -1),
        torch.stack([  2*(x*z-w*y),   2*(y*z+w*x), 1-2*(x*x+y*y)], -1),
    ], dim=-2)
    return R @ torch.diag_embed(scales ** 2) @ R.transpose(-1, -2)


# ── DA3 工具 ─────────────────────────────────────────────────────────────────

def _load_da3_mean_intrinsic(da3_npz: Path, render_width: int, render_height: int):
    """
    从 da3.npz 读取所有帧内参并取平均，然后按渲染分辨率缩放。

    返回:
      K_render : (3, 3) ndarray，真实像素单位的内参矩阵
      K_norm   : (3, 3) ndarray，归一化内参（AnySplat Decoder 所需）
      da3_w, da3_h : DA3 原始分辨率
    """
    da3 = np.load(str(da3_npz), allow_pickle=True)

    if "intrinsics" in da3:
        Ks = da3["intrinsics"].astype(np.float64)   # (T, 3, 3)
        K  = Ks.mean(axis=0)
    elif "intrinsic" in da3:
        K  = da3["intrinsic"].astype(np.float64)    # (3, 3)
    else:
        raise KeyError(f"da3.npz 缺少 intrinsic/intrinsics，keys={da3.files}")

    # 读原始分辨率
    if "height" in da3 and "width" in da3:
        da3_h, da3_w = int(da3["height"]), int(da3["width"])
    elif "depths" in da3:
        da3_h, da3_w = da3["depths"].shape[1:3]
    elif "images" in da3:
        da3_h, da3_w = da3["images"].shape[1:3]
    else:
        da3_h, da3_w = render_height, render_width
        print("  ⚠ 无法从 da3.npz 读取分辨率，假设与渲染分辨率一致")

    # 按渲染分辨率缩放
    sx = render_width  / float(da3_w)
    sy = render_height / float(da3_h)
    K_render = K.copy()
    K_render[0, 0] *= sx;  K_render[0, 2] *= sx
    K_render[1, 1] *= sy;  K_render[1, 2] *= sy

    K_norm = np.array([
        [K_render[0, 0] / render_width,  0.0, K_render[0, 2] / render_width],
        [0.0, K_render[1, 1] / render_height,  K_render[1, 2] / render_height],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)

    print(f"  DA3 原始分辨率: {da3_w}×{da3_h}  →  渲染分辨率: {render_width}×{render_height}")
    print(f"  平均内参 K: fx={K_render[0,0]:.1f}  fy={K_render[1,1]:.1f}  "
          f"cx={K_render[0,2]:.1f}  cy={K_render[1,2]:.1f}")
    return K_render, K_norm, da3_w, da3_h


def _load_da3_cam_c2w(da3_npz: Path) -> np.ndarray:
    """从 da3.npz 读取 camera-to-world 外参序列，shape = (T, 4, 4)。"""
    da3 = np.load(str(da3_npz), allow_pickle=True)

    if "cam_c2w" in da3:
        return da3["cam_c2w"].astype(np.float32)

    if "extrinsics" in da3:
        ext = da3["extrinsics"].astype(np.float32)
        if ext.shape[-2:] == (3, 4):
            T = ext.shape[0]
            w2c = np.tile(np.eye(4, dtype=np.float32)[None], (T, 1, 1))
            w2c[:, :3, :4] = ext
        elif ext.shape[-2:] == (4, 4):
            w2c = ext
        else:
            raise ValueError(f"extrinsics shape 不支持: {ext.shape}")
        return np.linalg.inv(w2c).astype(np.float32)

    raise KeyError(f"da3.npz 缺少 cam_c2w / extrinsics，keys={da3.files}")


# ── c2w → extrinsics（world-to-camera 4×4）────────────────────────────────

def _c2w_to_extrinsic(cam_c2w: np.ndarray) -> np.ndarray:
    """
    cam_c2w: (T, 4, 4) camera-to-world
    返回:    (T, 4, 4) world-to-camera  （渲染器所需）
    """
    T = len(cam_c2w)
    R_c2w = cam_c2w[:, :3, :3]
    t_c2w = cam_c2w[:, :3,  3]
    R_w2c = R_c2w.transpose(0, 2, 1)
    t_w2c = -np.einsum("bij,bj->bi", R_w2c, t_c2w)

    ext = np.zeros((T, 4, 4), dtype=np.float32)
    ext[:, :3, :3] = R_w2c
    ext[:, :3,  3] = t_w2c
    ext[:,  3,  3] = 1.0
    return ext


def _read_reference_video_metadata(reference_video: Path) -> tuple[int, int, float, int]:
    """Read width, height, fps, and frame count from a reference video."""
    reference_video = Path(reference_video).resolve()
    cap = cv2.VideoCapture(str(reference_video))
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开参考视频: {reference_video}")
    width = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    cap.release()
    if width <= 0 or height <= 0 or fps <= 0.0 or frame_count <= 0:
        raise ValueError(
            f"参考视频元数据无效: {reference_video}, "
            f"width={width}, height={height}, fps={fps}, frames={frame_count}"
        )
    return width, height, fps, frame_count


# ── 主渲染函数 ───────────────────────────────────────────────────────────────

def render_trajectory(
    output_dir: "str | Path",
    cam_c2w:    np.ndarray,
    video_path: "str | Path",
    width:      int = DEFAULT_RENDER_WIDTH,
    height:     int = DEFAULT_RENDER_HEIGHT,
    fps:        int = DEFAULT_RENDER_FPS,
    frame_dir:  "str | Path | None" = DEFAULT_BG_FRAME_DIR,
    keep_frames: bool = True,
    clean_frames: bool = True,
) -> Path:
    """
    用任意相机外参序列渲染 3DGS 场景并输出视频。

    参数
    ----
    output_dir : RoboSimGS++ workspace 根目录。
                 必须包含：
                   result/gaussians_aligned.ply   ← Stage 6 产物
                   da3.npz                        ← Stage 2 产物

    cam_c2w    : np.ndarray，shape = (T, 4, 4)，dtype float32 或 float64。
                 camera-to-world 变换矩阵序列（T 帧）。

                 每帧矩阵格式：
                   [[r00 r01 r02 tx]
                    [r10 r11 r12 ty]
                    [r20 r21 r22 tz]
                    [  0   0   0  1]]
                 其中 R[:3,:3] 是旋转，t[:3,3] 是相机光心世界坐标。

    video_path : 输出视频路径（父目录不存在会自动创建）。

    width, height : 渲染分辨率（像素）。

    fps        : 输出视频帧率。

    frame_dir  : 输出逐帧 PNG 的目录。用于后续与 MuJoCo 前景逐帧合成。

    keep_frames: 是否在视频编码完成后保留逐帧 PNG。

    clean_frames: 渲染前是否清理 frame_dir 中旧的 frame_*.png。

    返回
    ----
    Path : 输出视频的绝对路径。
    """
    output_dir = Path(output_dir).resolve()
    video_path = Path(video_path).resolve()
    video_path.parent.mkdir(parents=True, exist_ok=True)
    if frame_dir is None:
        frame_dir = video_path.parent / "frames"
    frame_dir = Path(frame_dir).resolve()
    frame_dir.mkdir(parents=True, exist_ok=True)
    if clean_frames:
        for old_frame in sorted(frame_dir.glob("frame_*.png")):
            if old_frame.is_file():
                old_frame.unlink()

    # ── 输入校验 ──
    cam_c2w = np.asarray(cam_c2w, dtype=np.float32)
    if cam_c2w.ndim != 3 or cam_c2w.shape[1:] != (4, 4):
        raise ValueError(
            f"cam_c2w 形状必须为 (T, 4, 4)，实际得到 {cam_c2w.shape}\n"
            "每帧为 camera-to-world 4×4 齐次变换矩阵。"
        )
    T = len(cam_c2w)

    # ── 路径检查 ──
    ply_path = output_dir / "result" / "gaussians_aligned.ply"
    da3_path = output_dir / "da3.npz"

    for p in (ply_path, da3_path):
        if not p.exists():
            raise FileNotFoundError(
                f"找不到必要文件: {p}\n"
                f"请确认 output_dir={output_dir} 是 RoboSimGS++ workspace 根目录，"
                f"且 Stage 2 (DA3) 和 Stage 6 (Sim3+AnySplat) 已完成。"
            )

    print(f"\n{'='*60}")
    print(f"  render_trajectory")
    print(f"{'='*60}")
    print(f"  PLY:       {ply_path}")
    print(f"  DA3:       {da3_path}")
    print(f"  输入帧数:  {T}")
    print(f"  分辨率:    {width}×{height} @ {fps}fps")
    print(f"  输出视频:  {video_path}")
    print(f"  输出帧目录: {frame_dir}")

    # ── 加载 AnySplat decoder ──
    sys.path.insert(0, ANYSPLAT_ROOT)
    from src.model.types import Gaussians
    from src.model.decoder.decoder_splatting_cuda import (
        DecoderSplattingCUDA, DecoderSplattingCUDACfg,
    )

    # ── 加载内参（DA3 所有帧均值，按渲染分辨率缩放）──
    _, K_norm, _, _ = _load_da3_mean_intrinsic(da3_path, width, height)

    intr_t = torch.tensor(K_norm, dtype=torch.float32, device="cuda").unsqueeze(0).unsqueeze(0)
    near_t = torch.tensor([[0.01]], dtype=torch.float32, device="cuda")
    far_t  = torch.tensor([[100.0]], dtype=torch.float32, device="cuda")

    # ── 加载高斯 ──
    means, scales, rots, opacs, harms = _load_gaussians_from_ply(ply_path)
    covs = _build_covariance(rots, scales)

    gaussians = Gaussians(
        means      = means.unsqueeze(0),
        covariances= covs.unsqueeze(0),
        harmonics  = harms.unsqueeze(0),
        opacities  = opacs.unsqueeze(0),
        scales     = scales.unsqueeze(0),
        rotations  = rots.unsqueeze(0),
    )

    # ── 初始化 Decoder ──
    cfg = DecoderSplattingCUDACfg(
        name="splatting_cuda",
        background_color=[0.05, 0.05, 0.05],
        make_scale_invariant=False,
    )
    decoder = DecoderSplattingCUDA(cfg).cuda().eval()

    # ── c2w → w2c extrinsics ──
    ext_np = _c2w_to_extrinsic(cam_c2w)  # (T, 4, 4)

    # ── 逐帧渲染 ──
    print(f"  开始渲染 {T} 帧...")

    for t_idx in range(T):
        ext_t = torch.tensor(
            ext_np[t_idx:t_idx+1][None],   # (1, 1, 4, 4)
            dtype=torch.float32, device="cuda",
        )

        with torch.no_grad():
            output = decoder.forward(gaussians, ext_t, intr_t, near_t, far_t, (height, width))

        # 兼容不同版本 decoder 的输出格式
        try:
            img_t = output.color[0, 0]
        except AttributeError:
            img_t = output[0][0, 0]

        img_np = (img_t.permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(img_np).save(str(frame_dir / f"frame_{t_idx:05d}.png"))

        if (t_idx + 1) % 30 == 0 or t_idx == T - 1:
            print(f"  [{t_idx+1:>4d}/{T}]")

    # ── 编码视频 ──
    subprocess.run([
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i",         str(frame_dir / "frame_%05d.png"),
        "-c:v",       "libx264",
        "-pix_fmt",   "yuv420p",
        "-crf",       "18",
        str(video_path),
    ], check=True)

    if not keep_frames:
        shutil.rmtree(str(frame_dir), ignore_errors=True)

    print(f"\n  ✓ 视频已保存: {video_path}  ({T}帧 @ {fps}fps)")
    if keep_frames:
        print(f"  ✓ 视频帧已保存: {frame_dir}")
    return video_path


# ── main：用 DA3 真实外参做 test ────────────────────────────────────────────

def main():
    """测试用途：直接读取 DA3 估出的相机外参序列渲染整段视频。

    用法：
        python render_custom_trajectory.py --output-dir /path/to/workspace

    可选参数：
        --output-dir    RoboSimGS++ workspace 根目录（默认 preprocessed/milk）
        --video-out     输出视频路径（默认 preprocessed/milk/3dgs_videos/render_da3_traj.mp4）
        --frames-out-dir 输出逐帧 PNG 目录（默认 preprocessed/milk/3dgs_bg_videos/frames）
        --width         渲染宽度（默认 720）
        --height        渲染高度（默认 480）
        --fps           帧率（默认 50）
        --max-frames    最多渲染多少帧（默认 120）
        --reference-video 参考 D435 视频；提供后自动对齐宽高、fps、帧数
    """
    parser = argparse.ArgumentParser(
        description="用 DA3 外参序列测试渲染 RoboSimGS++ 场景",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_WORKSPACE),
                        help="RoboSimGS++ workspace 根目录")
    parser.add_argument("--video-out",  default=None,
                        help=f"输出视频路径（默认 {DEFAULT_BG_VIDEO_PATH}）")
    parser.add_argument("--frames-out-dir", default=str(DEFAULT_BG_FRAME_DIR),
                        help=f"输出逐帧 PNG 目录（默认 {DEFAULT_BG_FRAME_DIR}）")
    parser.add_argument("--width",      type=int, default=DEFAULT_RENDER_WIDTH)
    parser.add_argument("--height",     type=int, default=DEFAULT_RENDER_HEIGHT)
    parser.add_argument("--fps",        type=float, default=DEFAULT_RENDER_FPS)
    parser.add_argument("--max-frames", type=int, default=DEFAULT_MAX_FRAMES,
                        help="最多渲染帧数；默认对齐当前 milk D435 IK 视频")
    parser.add_argument("--reference-video", default=None,
                        help="参考 D435 视频路径；提供后自动覆盖 width/height/fps/max-frames")
    parser.add_argument("--no-clean-frames", action="store_true",
                        help="不清理 frames-out-dir 中已有的 frame_*.png")
    parser.add_argument("--delete-frames-after-video", action="store_true",
                        help="视频编码完成后删除逐帧 PNG；默认保留")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    da3_path   = output_dir / "da3.npz"

    if not da3_path.exists():
        raise FileNotFoundError(
            f"找不到 da3.npz: {da3_path}\n"
            f"请先完成 Stage 2 (DA3)，或手动指定正确的 --output-dir。"
        )

    width = int(args.width)
    height = int(args.height)
    fps = float(args.fps)
    max_frames = args.max_frames
    if args.reference_video:
        width, height, fps, frame_count = _read_reference_video_metadata(
            Path(args.reference_video)
        )
        max_frames = frame_count
        print(
            f"\n  使用参考视频参数: {args.reference_video}\n"
            f"  width={width}, height={height}, fps={fps}, frames={frame_count}"
        )

    # ── 读取 DA3 真实外参（即 test 输入）──
    cam_c2w = _load_da3_cam_c2w(da3_path)   # (T, 4, 4)
    print(f"\n  从 DA3 读取外参序列: {cam_c2w.shape}  dtype={cam_c2w.dtype}")
    print(f"  示例（第 0 帧 c2w）:\n{cam_c2w[0]}")

    if max_frames is not None:
        cam_c2w = cam_c2w[: max_frames]
        print(f"  截断至前 {max_frames} 帧")

    # ── 输出路径 ──
    if args.video_out:
        video_path = Path(args.video_out)
    else:
        video_path = DEFAULT_BG_VIDEO_PATH
    frame_dir = Path(args.frames_out_dir)

    # ── 调用核心渲染函数 ──
    render_trajectory(
        output_dir = output_dir,
        cam_c2w    = cam_c2w,
        video_path = video_path,
        width      = width,
        height     = height,
        fps        = fps,
        frame_dir  = frame_dir,
        keep_frames= not args.delete_frames_after_video,
        clean_frames= not args.no_clean_frames,
    )

    print(f"\n  [test OK]  输出视频: {video_path}")


if __name__ == "__main__":
    main()
