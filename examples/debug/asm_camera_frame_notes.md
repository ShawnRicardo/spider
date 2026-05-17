# ASM 相机与仿真坐标系说明

这份文档说明当前 `ASM + ourdata + IK/Viser` 调试链路里的坐标系约定。后续排查手部红蓝点是否和机器人对齐时，以这里为准。

## 1. 颜色约定

所有坐标轴使用同一套颜色：

- 红色：`+X`
- 绿色：`+Y`
- 蓝色：`+Z`

其他调试颜色：

- 红色点：右手输入关键点
- 蓝色点：左手输入关键点
- 黄色半透明框：物体 bbox
- 绿色小球：物体中心
- 洋红色半透明球：`head_camera_frame` 的原点，只作 legacy visual frame 位置参考
- 橙色球：`d435_optical_frame` 的原点

## 2. 机器人根坐标系

当前 `asm_root` 已经改回和 MuJoCo 世界坐标系对齐。

重新生成 XML 后：

- `asm_root` 位置：`[0, 0, 0]`
- `asm_root` 朝向：单位四元数
- `asm_root +X` 对齐 `world +X`
- `asm_root +Y` 对齐 `world +Y`
- `asm_root +Z` 对齐 `world +Z`

也就是说，之前的 `+90 deg yaw` 已经取消。后续如果世界坐标系里看到机器人方向不对，优先检查相机外参或输入数据，不再先怀疑 `asm_root` 的固定旋转。

## 3. D435 mesh

D435 相机外壳模型来自：

- `spider/assets/robots/asm_description/meshes/arm/d435.dae`

MuJoCo 不能直接稳定读取 `.dae`，所以资产生成阶段会把它转换成：

- `d435.obj`

然后写入 processed robot assets。

D435 mesh 只是外观模型，不等于最终用于坐标对齐的相机坐标系。

## 4. `head_camera_frame`

`head_camera_frame` 是 legacy visual/mesh frame。

它来自 URDF 里的这条固定链：

```text
base_link
-> neck_joint
-> joint_d435
-> link_435.visual.origin
```

对应含义是：

- D435 外壳 mesh 在机器人上如何摆放
- 用于可视化检查 mesh 和 URDF visual origin 是否一致

当前在 Viser 里观察到的这个 frame 约定是：

- `+X` 朝左
- `+Y` 朝上
- `+Z` 沿镜头向前

这个约定和 `da3.npz` / OpenCV 常用相机坐标系不一致，所以它不再作为 `ourdata` 的 world-to-sim 对齐基准。

## 5. `d435_optical_frame`

`d435_optical_frame` 是当前真正用于 `ourdata` 相机对齐的 frame。

它和 `head_camera_frame` 同源，但额外乘了一个固定旋转：

```text
R_optical_from_visual = diag(-1, -1, 1)
d435_optical_frame = head_camera_frame * R_optical_from_visual
```

这个旋转做的事情很明确：

- visual `+X` 原本朝左，所以 optical `+X = - visual +X`，朝右
- visual `+Y` 原本朝上，所以 optical `+Y = - visual +Y`，朝下
- visual `+Z` 已经沿镜头向前，所以 optical `+Z = visual +Z`，保持向前

因此 `d435_optical_frame` 的目标约定是：

- `+X`：图像向右
- `+Y`：图像向下
- `+Z`：深度向前

这和 `da3.npz` 深度反投影使用的相机坐标系一致。

## 6. `ourdata` 如何做相机对齐

`ourdata.py` 的命令行模式名是：

- `d435_optical`

但内部实际读取的是：

- `d435_optical_frame`

对齐公式是：

```text
T_sim_world = T_sim_d435_optical * inverse(T_world_cam0)
```

其中：

- `T_world_cam0` 来自 `preprocessed/milk/da3.npz` 的 `cam_c2w[0]`
- `T_sim_d435_optical` 来自 MuJoCo XML 里的 `d435_optical_frame`

这个固定变换会统一作用到：

- 右手 vertices
- 左手 vertices
- 物体中心轨迹
- 物体旋转

## 7. 取消 workspace 抬升

对 `ourdata + d435_optical` 这条路径，当前已经取消 ASM workspace support 的二次平移。

也就是说，IK 阶段不会再自动加：

- `workspace_z_offset`
- `workspace_xy_offset`

原因是：如果视频相机已经和机器人 D435 optical frame 对齐，后续再把手和物体整体抬升到桌面，会破坏相机对齐结果。

OakInk/ASM 原始数据路径仍然保留 workspace support，不受这次修改影响。

## 8. 在 Viser 中如何判断

打开 IK Viser 调试后，应该同时看到：

- D435 mesh
- 洋红色 `head_camera_frame` 原点球
- 橙色 `d435_optical_frame` 原点球和 RGB 正方向轴
- 世界坐标轴
- 机器人 base 坐标轴
- 红/蓝手部输入关键点
- 物体 bbox 和中心

判断顺序：

1. 看 `asm_root` 轴是否和世界轴一致。
2. 看 D435 mesh 是否和 `head_camera_frame` 原点位置一致。
3. 看 `d435_optical_frame` 的 RGB 轴是否满足 `+X` 右、`+Y` 下、`+Z` 前。
4. 看红蓝手点是否进入机器人双手工作空间。
5. 看物体 bbox 是否和手处在同一相机坐标变换后的空间。

为了避免把 legacy visual frame 和 optical frame 的相反轴混在一起，当前浏览器调试只给 `d435_optical_frame` 画 RGB 轴。所有轴 geom 都是从原点开始，只朝对应正方向伸出；原点用球形 marker 标出。

## 9. 当前关键结论

当前链路里：

- `head_camera_frame` 只表示 D435 mesh/visual frame
- `d435_optical_frame` 才是 `ourdata` 的相机对齐基准
- `asm_root` 不再保留 `+90 deg yaw`
- `ourdata + d435_optical` 不再做 workspace 高度抬升

后续如果仍然错位，优先检查：

- `da3.npz` 的 `cam_c2w[0]` 是否真的是视频第一帧相机坐标
- `d435_optical_frame` 是否正好对应实际拍摄相机安装位置
- 手和物体是否在上游已经处于同一个 `da3.npz` 世界坐标系


已修改。

你看到“两头都有”的主要原因是我们同时画了 head_camera_frame 和 d435_optical_frame，而它们的 X/Y 方向正好相反，所以红色、绿色看起来像从两边都伸出来。现在改成：

head_camera_frame 只显示洋红色半透明原点球，不再画 RGB 轴
d435_optical_frame 显示橙色原点球 + RGB 正方向轴
所有调试坐标轴都会在原点加一个明显圆球
轴 geom 仍然是从中心开始，只朝正方向伸出
代码改在 ik.py (line 52)，文档同步在 asm_camera_frame_notes.md (line 147)。
