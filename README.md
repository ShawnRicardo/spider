# Preprocessed 数据交付与 IK 运行说明
把每个新场景放到项目根目录下的：

```text
preprocessed/<scene_name>/
```

例如：

```text
preprocessed/watermelon_server/
preprocessed/robot/
```

当前 ourdata/ASM 流程分两类入口：

- 单物体：一个真实交互物体，使用 `examples/asm_ourdata_watermelon/`。
- 双物体：两只手分别交互两个物体，使用 `examples/asm_ourdata_2objs/`。

## 单物体场景

单物体流程默认只处理一个物体槽位，通常是 `obj_0`。最小目录结构如下：

```text
preprocessed/<scene_name>/
├── da3.npz                         # 推荐放这里
├── da3/
│   └── da3.npz                     # 如果根目录没有 da3.npz，则读取这里
├── hawor/
│   └── world_mocap.npz
├── sam3d/
│   └── obj_0/
│       ├── obj_mesh_final.glb          # 推荐
│       ├── obj_mesh_final.obj          # 可选
│       ├── obj_mesh_final.ply          # 可选
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

物体 pose 也可以放在：

```text
preprocessed/<scene_name>/result/obj_0/poses/*.txt
```

如果 `foundationpose_debug/center_pose/*.txt` 和 `poses/*.txt` 都存在，当前处理代码优先读取 `foundationpose_debug/center_pose/*.txt`。

关键文件要求：

- `da3.npz`：优先读取 `preprocessed/<scene_name>/da3.npz`；如果不存在，则读取 `preprocessed/<scene_name>/da3/da3.npz`。至少包含 `cam_c2w (N,4,4)`、`intrinsic (3,3)` 或 `intrinsics (N,3,3)`；建议保留 `cam_w2c`、`depths`、`images`。
- `hawor/world_mocap.npz`：至少包含 `right_verts`、`left_verts`、`right_joints`、`left_joints`、`right_faces`、`left_faces`。左右手数据不要有大段 `NaN`。
- `result/obj_0/box_for_spider.npz`：必须包含 `box_center_world`、`box_rotation_R`、`box_pose_4x4`、`box_real_size_xyz_m`、`scale_factor`、`sam3d_model_size`。
- `result/obj_0/.../*.txt`：每帧一个 `4x4` 的 `T_cam_obj`，处理脚本会计算 `T_world_obj = da3.cam_c2w[t] @ T_cam_obj[t]`。
- `sam3d/obj_0/obj_mesh_final.glb`：推荐提供；也支持 `obj_mesh_final.obj`、`obj_mesh_final.ply`、`obj_3d_final.ply`。

只运行 IK 时，入口是：

```text
examples/asm_ourdata_watermelon/run_ik_ourdata_watermelon_asm_URDFCollision.sh
```

它会依次调用：

```text
run_ik_ourdata_watermelon_asm_URDFCollision.sh
└── generate_scene_ourdata_watermelon_asm_URDFCollision.sh
    └── process_ourdata_watermelon_PickSpoonBowlParams.sh
```

如果新场景叫 `robot`，需要在 `examples/asm_ourdata_watermelon/process_ourdata_watermelon_PickSpoonBowlParams.sh` 中把：

```bash
--workspace preprocessed/watermelon_server
--task watermelon_server
--object-name watermelon
--object-id obj_0
```

改成：

```bash
--workspace preprocessed/robot
--task robot
--object-name robot
--object-id obj_0
```

同时在 `generate_scene_ourdata_watermelon_asm_URDFCollision.sh` 和 `run_ik_ourdata_watermelon_asm_URDFCollision.sh` 中把 `--task watermelon_server` 改成同一个 task，例如 `--task robot`。

运行：

```bash
conda activate spider
DATA_ID=0 CUDA_VISIBLE_DEVICES=0 \
bash examples/asm_ourdata_watermelon/run_ik_ourdata_watermelon_asm_URDFCollision.sh
```

假设 `task=robot`、`DATA_ID=0`，主要输出在：

```text
example_datasets/processed/ourdata/asm/bimanual/robot/0/
```

常用文件：

```text
trajectory_kinematic.npz
trajectory_ikrollout.npz
visualization_ik.mp4
visualization_ik_d435.mp4
visualization_trajectory_kinematic.mp4
visualization_trajectory_kinematic_d435.mp4
ik_collision_diagnostics.json
ik_collision_diagnostics.txt
```

## 双物体场景

双物体流程用于“两只手分别抓两个物体”的场景。SPIDER 内部使用两个固定物体槽位：

```text
left_object
right_object
```

当前默认映射是：

```text
left_object  <- obj_0
right_object <- obj_1
```

如果需要交换槽位，可以在运行脚本时覆盖：

```bash
LEFT_OBJECT_ID=obj_1 RIGHT_OBJECT_ID=obj_0
```

最小目录结构如下：

```text
preprocessed/<scene_name>/
├── da3.npz                         # 推荐放这里
├── da3/
│   └── da3.npz                     # 如果根目录没有 da3.npz，则读取这里
├── hawor/
│   └── world_mocap.npz
├── sam3d/
│   ├── obj_0/
│   │   └── obj_mesh_final.glb
│   └── obj_1/
│       └── obj_mesh_final.glb
└── result/
    ├── obj_0/
    │   ├── box_for_spider.npz
    │   └── foundationpose_debug/
    │       └── center_pose/
    │           ├── 00000.txt
    │           └── ...
    └── obj_1/
        ├── box_for_spider.npz
        └── foundationpose_debug/
            └── center_pose/
                ├── 00000.txt
                └── ...
```

每个物体都必须有自己的：

```text
result/obj_x/box_for_spider.npz
result/obj_x/foundationpose_debug/center_pose/*.txt
sam3d/obj_x/obj_mesh_final.glb
```

`poses/*.txt` 也可以作为 pose 目录：

```text
result/obj_x/poses/*.txt
```

双物体数据转换会输出：

```text
qpos_obj_left   # 默认来自 obj_0
qpos_obj_right  # 默认来自 obj_1
```

新的双物体脚本都在：

```text
examples/asm_ourdata_2objs/
```

只运行 IK：

```bash
conda activate spider
WORKSPACE=preprocessed/robot \
TASK=robot \
LEFT_OBJECT_ID=obj_0 \
RIGHT_OBJECT_ID=obj_1 \
LEFT_OBJECT_NAME=left_obj \
RIGHT_OBJECT_NAME=right_obj \
DATA_ID=0 \
CUDA_VISIBLE_DEVICES=0 \
bash examples/asm_ourdata_2objs/run_ik_ourdata_2objs_asm_URDFCollision.sh
```

如果要继续运行 MJWP：

```bash
WORKSPACE=preprocessed/robot \
TASK=robot \
LEFT_OBJECT_ID=obj_0 \
RIGHT_OBJECT_ID=obj_1 \
LEFT_OBJECT_NAME=left_obj \
RIGHT_OBJECT_NAME=right_obj \
DATA_ID=0 \
CUDA_VISIBLE_DEVICES=0 \
bash examples/asm_ourdata_2objs/run_mjwp_ourdata_2objs_asm_URDFCollision.sh
```

双物体脚本会依次调用：

```text
run_ik_ourdata_2objs_asm_URDFCollision.sh
└── generate_scene_ourdata_2objs_asm_URDFCollision.sh
    └── process_ourdata_2objs.sh
        └── spider/process_datasets/ourdata_textured_2objs.py
```

`trajectory_keypoints.npz` 必须同时包含：

```text
qpos_wrist_right
qpos_finger_right
qpos_obj_right
qpos_wrist_left
qpos_finger_left
qpos_obj_left
contact_right
contact_left
```

假设 `TASK=robot`、`DATA_ID=0`，输出位置仍是：

```text
example_datasets/processed/ourdata/asm/bimanual/robot/0/
```

中间转换结果在：

```text
example_datasets/processed/ourdata/mano/bimanual/robot/0/
```

场景 XML 在：

```text
example_datasets/processed/ourdata/asm/bimanual/robot/
```

### 坐标系和检查

单物体和双物体都要求以下数据在同一个 DA3 world 下自洽：

```text
world_mocap.npz 里的左右手
da3.npz 里的 cam_c2w/cam_w2c
每个 obj_x 的 T_cam_obj 通过 cam_c2w 转出的物体 pose
```

推荐先用 Viser 检查手和物体是否在同一坐标系：

```bash
conda run -n spider python visualize_watermelon_server_da3_viser.py \
  --workspace preprocessed/<scene_name> \
  --object-id obj_0 \
  --port 8080
```

双物体场景需要分别检查 `obj_0` 和 `obj_1`。

### 提交数据前检查清单

- 根目录 `da3.npz` 或 `da3/da3.npz` 存在，且 `cam_c2w` 是 `(N, 4, 4)`。
- `hawor/world_mocap.npz` 存在，左右手帧数是 `N`。
- `right_verts/left_verts/right_joints/left_joints` 不存在大段 `NaN`。
- 每个交互物体都有 `box_for_spider.npz`。
- 每个交互物体都有逐帧 `4x4` pose txt，数量至少是 `N`。
- 每个交互物体都有 `obj_mesh_final.glb` 或等价 mesh 文件。
- `box_real_size_xyz_m` 是真实尺寸，单位是米。
- 用 Viser 看时，左右手和所有物体的位置关系合理。

第三个动态物体目前不支持。如果第三个物体只是静态背景道具，可以后续作为普通场景 mesh 处理；如果它也需要被抓取、跟踪、参与 reward，则需要把当前 `left_object/right_object` 双槽位重构成对象列表。
