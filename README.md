# Preprocessed 数据交付与 IK 运行说明

当前这套 ourdata/ASM 流程的目标是：把你生成的一个场景目录放到 `preprocessed/<scene_name>/` 下，然后通过 `examples/asm_ourdata_watermelon` 中的脚本只运行 IK 阶段，得到 ASM 机器人的 kinematic 轨迹和调试视频。

## 1. 数据应该放在哪里

每个新场景单独放在：

```text
preprocessed/<scene_name>/
```

例如当前 watermelon 脚本默认读取：

```text
preprocessed/watermelon_server/
```

如果你生成的是一个新场景，比如 `robot`，建议放成：

```text
preprocessed/robot/
```

后续只需要把脚本里的 `preprocessed/watermelon_server` 改成 `preprocessed/robot`。

## 2. 必需目录结构

IK 流程目前按“单个被操作物体”设计，默认物体目录名是 `obj_0`。最小可运行结构如下：

```text
preprocessed/<scene_name>/
├── da3.npz
├── hawor/
│   └── world_mocap.npz
├── sam3d/
│   └── obj_0/
│       ├── obj_mesh_final.glb          # 推荐
│       └── obj_3d_final.ply            # 可选，用于点云/颜色调试
└── result/
    └── obj_0/
        ├── box_for_spider.npz
        └── foundationpose_debug/
            └── center_pose/
                ├── 00000.txt
                ├── 00001.txt
                └── ...
```

也可以把物体 pose 放在：

```text
preprocessed/<scene_name>/result/obj_0/poses/*.txt
```

但如果两个目录都存在，当前处理代码优先读取：

```text
result/obj_0/foundationpose_debug/center_pose/*.txt
```

## 3. 每个文件里需要有什么

### `da3.npz`

这是相机和 DA3 world 坐标系的核心文件，至少需要包含：

```text
cam_c2w      (N, 4, 4) float
cam_w2c      (N, 4, 4) float
intrinsic    (3, 3) 或 intrinsics (N, 3, 3)
depths       (N, H, W) float，建议保留
images       (N, H, W, 3) uint8，建议保留
```

`cam_c2w[t]` 表示第 `t` 帧相机坐标系到 DA3 world 坐标系的变换。

### `hawor/world_mocap.npz`

这是左右手的 3D 重建结果，至少需要包含：

```text
right_verts   (N, 778, 3) float32
left_verts    (N, 778, 3) float32
right_joints  (N, 21, 3) float32
left_joints   (N, 21, 3) float32
right_faces   (1552, 3)
left_faces    (1552, 3)
pred_space    "world"
```

要求：

- 帧数 `N` 要和 `da3.npz`、物体 pose 文件数量一致。
- 左右手数据必须尽量连续。
- 不要有大段 `NaN`。IK 不能处理大段手部缺失。
- 坐标系必须是 DA3 world，也就是和 `da3.npz` 的 `cam_c2w` 对齐。

### `result/obj_0/box_for_spider.npz`

这是物体真实尺寸和初始 bbox 信息，必须包含：

```text
box_center_world      (3,)
box_rotation_R        (3, 3)
box_pose_4x4          (4, 4)
box_real_size_xyz_m   (3,)
scale_factor          scalar
sam3d_model_size      (3,)
```

其中最重要的是：

```text
box_real_size_xyz_m
```

它必须是真实物体尺寸，单位是米。MuJoCo 里的 bbox、桌子高度、物体大小都会依赖它。

### `result/obj_0/foundationpose_debug/center_pose/*.txt`

每一帧一个 `4x4` 矩阵文本文件：

```text
00000.txt
00001.txt
...
```

每个 txt 的含义应该是：

```text
T_cam_obj
```

也就是“物体在当前相机坐标系下的位置和朝向”。

处理脚本会做：

```text
T_world_obj = da3.cam_c2w[t] @ T_cam_obj[t]
```

然后再通过 `d435_optical` 对齐到 ASM/MuJoCo 仿真坐标系。

### `sam3d/obj_0/obj_mesh_final.glb`

推荐提供 `glb`。也支持：

```text
obj_mesh_final.obj
obj_mesh_final.ply
obj_3d_final.ply
```

注意：mesh 的原始单位和大小可以不完全可靠，但 `box_for_spider.npz` 里的 `box_real_size_xyz_m` 必须正确。脚本会以 bbox 真实尺寸为准。

## 4. 坐标系要求

请保证以下三类数据在同一个 DA3 world 下自洽：

```text
world_mocap.npz 里的手
da3.npz 里的 cam_c2w/cam_w2c
center_pose/*.txt 通过 cam_c2w 转出来的物体 pose
```

一个快速检查方式是：把手和物体放到 Viser 里看。

```bash
conda run -n spider python visualize_watermelon_server_da3_viser.py \
  --workspace preprocessed/<scene_name> \
  --object-id obj_0 \
  --port 8080
```

如果手和物体在 DA3 world 里明显离得很远，IK/MJWP 后面大概率也不会正常。

## 5. 如何修改 `examples/asm_ourdata_watermelon` 脚本

只跑 IK 时，入口是：

```text
examples/asm_ourdata_watermelon/run_ik_ourdata_watermelon_asm_URDFCollision.sh
```

它会依次调用：

```text
run_ik_ourdata_watermelon_asm_URDFCollision.sh
└── generate_scene_ourdata_watermelon_asm_URDFCollision.sh
    └── process_ourdata_watermelon_PickSpoonBowlParams.sh
```

不需要运行：

```text
examples/asm_ourdata_watermelon/run_mjwp_ourdata_watermelon_asm_URDFCollision.sh
```

### 5.1 修改 `process_ourdata_watermelon_PickSpoonBowlParams.sh`

找到 Python 调用里的这些参数：

```bash
--workspace preprocessed/watermelon_server
--task watermelon_server
--object-name watermelon
--object-id obj_0
```

如果你的数据目录是 `preprocessed/robot`，目标物体仍是 `obj_0`，可以改成：

```bash
--workspace preprocessed/robot
--task robot
--object-name robot
--object-id obj_0
```

如果目标物体是 `obj_1`，则改成：

```bash
--object-id obj_1
```

同时你的数据里也必须有：

```text
result/obj_1/box_for_spider.npz
result/obj_1/foundationpose_debug/center_pose/*.txt
sam3d/obj_1/obj_mesh_final.glb
```

### 5.2 修改 `generate_scene_ourdata_watermelon_asm_URDFCollision.sh`

找到：

```bash
--task watermelon_server
```

改成和上一步一致的 task，例如：

```bash
--task robot
```

其他参数一般先不要改。

### 5.3 修改 `run_ik_ourdata_watermelon_asm_URDFCollision.sh`

找到：

```bash
--task watermelon_server
```

改成同一个 task，例如：

```bash
--task robot
```

如果需要换 GPU，修改或运行时覆盖：

```bash
CUDA_VISIBLE_DEVICES=0
```

## 6. 运行 IK

进入项目根目录后运行：

```bash
conda activate spider
DATA_ID=0 CUDA_VISIBLE_DEVICES=0 \
bash examples/asm_ourdata_watermelon/run_ik_ourdata_watermelon_asm_URDFCollision.sh
```

如果不想改脚本里的 `CUDA_VISIBLE_DEVICES`，也可以直接：

```bash
bash examples/asm_ourdata_watermelon/run_ik_ourdata_watermelon_asm_URDFCollision.sh
```

## 7. IK 输出位置

假设 `--task robot`、`DATA_ID=0`，主要输出在：

```text
example_datasets/processed/ourdata/asm/bimanual/robot/0/
```

常用文件：

```text
trajectory_kinematic.npz
trajectory_ikrollout.npz
visualization_ik.mp4
visualization_trajectory_kinematic.mp4
ik_collision_diagnostics.json
ik_collision_diagnostics.txt
```

中间转换结果在：

```text
example_datasets/processed/ourdata/mano/bimanual/robot/0/
```

场景 XML 在：

```text
example_datasets/processed/ourdata/asm/bimanual/robot/
```

## 8. 常见失败原因

### 手部数据有大量 NaN

如果 `world_mocap.npz` 里左右手大段为 `NaN`，IK 无法正常运行。需要重新生成或修复 HAWOR/world mocap。

### 缺少 `box_for_spider.npz`

没有真实 bbox 尺寸时，脚本不知道物体大小，也无法生成桌子高度和物体 bbox。

### 缺少物体逐帧 pose

必须提供：

```text
result/obj_0/foundationpose_debug/center_pose/*.txt
```

或者：

```text
result/obj_0/poses/*.txt
```

每一帧一个 `4x4` 矩阵。

### 多物体场景

当前 watermelon 脚本按单物体 `obj_0` 处理。如果场景里有多个物体，请先明确要操作哪个物体，并只把该物体作为 `--object-id` 输入。多物体同时进入 IK/MJWP 需要额外扩展代码。

## 9. 提交数据前的最小检查清单

请确认：

- `da3.npz` 存在，且 `cam_c2w` 是 `(N, 4, 4)`。
- `hawor/world_mocap.npz` 存在，左右手帧数是 `N`。
- `right_verts/left_verts/right_joints/left_joints` 不存在大段 `NaN`。
- `result/obj_0/box_for_spider.npz` 存在。
- `result/obj_0/foundationpose_debug/center_pose/*.txt` 数量至少是 `N`。
- `sam3d/obj_0/obj_mesh_final.glb` 或等价 mesh 文件存在。
- 用 Viser 看时，手和物体在 DA3 world 中位置关系合理。
