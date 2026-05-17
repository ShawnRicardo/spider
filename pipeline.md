可以，我们这次完全换一种讲法：先不把它当成“代码”，而把它当成一条生产线。

你的例子是：

```text
视频里，一个人手拿着杯子。
杯子从 A = (3, 4, 5)
移动到 B = (6, 7, 8)
```

SPIDER 要做的事情不是直接复制这个视频，而是回答一个更难的问题：

```text
如果换成 xhand 机械手，它应该怎么动关节、怎么接触杯子、怎么施加控制，才能在物理仿真中把杯子也从 A 移到 B？
```

整个 pipeline 可以理解成四个文件阶段：

```text
原始视频 / pkl
    ↓
trajectory_keypoints.npz
    ↓
trajectory_kinematic.npz
    ↓
trajectory_mjwp.npz
```

---

**一、最开始的 pkl 是什么？**
先说一个容易误解的地方：SPIDER 不是直接从普通 RGB 视频里读像素，然后自动知道手和杯子在哪里。

它期望前面已经有人帮它做了“感知”：

```text
视频
 ↓
手部姿态估计
物体 6D pose 估计
手指关键点提取
物体 mesh 路径
 ↓
pkl
```

对于 OakInk，这个 pkl 的读取在 [oakink.py](/mnt/data3/guolijun/DexterousHand/spider/spider/process_datasets/oakink.py):64。

代码期望的原始文件大概是：

```text
example_datasets/raw/oakink/pick_spoon_bowl_bimanual.pkl
```

它里面大概长这样：

```python
data = {
    "right": {
        "wrist_pos":      (T, 3),
        "wrist_rot":      (T, 3),   # 3*3 旋转矩阵
        "mano_joints":    (T, 5, 3),# 5 个手指的关键点位置，可能是指尖
        "obj_mesh_path":  list[str],
        "obj_trajectory": (T, 4, 4),# 4*4 旋转+平移矩阵
    },
    "left": {
        "wrist_pos":      (T, 3),
        "wrist_rot":      (T, 3),
        "mano_joints":    (T, 5, 3),
        "obj_mesh_path":  list[str],
        "obj_trajectory": (T, 4, 4),
    }
}
```

如果用你的杯子例子，最重要的是 `obj_trajectory`。

它是每一帧杯子的 4x4 位姿矩阵：

```python
T_obj[t] = [
    [R11, R12, R13, x],
    [R21, R22, R23, y],
    [R31, R32, R33, z],
    [0,   0,   0,   1],
]
```

如果杯子只是平移，从 A 到 B，没有旋转，那么概念上：

```python
alpha = t / (T - 1)
p_t = (1 - alpha) * A + alpha * B

# 第 0 帧
p_0 = (3, 4, 5)

# 最后一帧
p_T = (6, 7, 8)
```

于是：

```python
obj_trajectory[0][:3, 3] = [3, 4, 5]
obj_trajectory[-1][:3, 3] = [6, 7, 8]
```

手的数据也是类似的：

```text
wrist_pos: 每一帧手腕位置
wrist_rot: 每一帧手腕旋转，axis-angle
mano_joints: 每一帧 5 个手指关键点位置
```

所以 pkl 的本质是：

```text
原始人手和物体的时序姿态数据
```

不是机器人数据。

---

**二、pkl 如何变成 trajectory_keypoints.npz？**
转换发生在 [oakink.py](/mnt/data3/guolijun/DexterousHand/spider/spider/process_datasets/oakink.py):87 到 [oakink.py](/mnt/data3/guolijun/DexterousHand/spider/spider/process_datasets/oakink.py):265。

它做了几件事。

第一，读右手：

```python
right_wrist_pos = data_right["wrist_pos"]
right_wrist_rot = data_right["wrist_rot"]
right_mano_joints = data_right["mano_joints"]
```

第二，读左手：

```python
left_wrist_pos = data_left["wrist_pos"]
left_wrist_rot = data_left["wrist_rot"]
left_mano_joints = data_left["mano_joints"]
```

第三，读物体：

```python
right_obj_trajectory = data_right["obj_trajectory"]
left_obj_trajectory = data_left["obj_trajectory"]
```

第四，把这些姿态统一转换成 MuJoCo 使用的坐标系。

代码里有：

```python
r_global = R.from_euler("xyz", [np.pi / 2, 0, 0])
```

所以你概念里的 A = `(3,4,5)`，经过这里以后，保存到 keypoints 里的坐标可能不是原封不动的 `(3,4,5)`，而是被转到 MuJoCo 世界坐标系后的结果。

第五，把旋转统一转成四元数 `wxyz` 格式。

最终保存：

```python
np.savez(
    trajectory_keypoints.npz,
    qpos_wrist_right=...,
    qpos_finger_right=...,
    qpos_obj_right=...,
    qpos_wrist_left=...,
    qpos_finger_left=...,
    qpos_obj_left=...,
)
```

保存代码在 [oakink.py](/mnt/data3/guolijun/DexterousHand/spider/spider/process_datasets/oakink.py):257。

你当前文件实际是：

```text
trajectory_keypoints.npz

qpos_wrist_right:  (1744, 7)
qpos_finger_right: (1744, 5, 7)
qpos_obj_right:    (1744, 7)
qpos_wrist_left:   (1744, 7)
qpos_finger_left:  (1744, 5, 7)
qpos_obj_left:     (1744, 7)
```

这里的 `7` 是：

```text
x, y, z, qw, qx, qy, qz
```

所以：

```text
qpos_obj_right[t] = 杯子在第 t 帧的位置 + 姿态
qpos_wrist_right[t] = 右手腕在第 t 帧的位置 + 姿态
qpos_finger_right[t, 0] = 右手大拇指关键点位置 + 姿态
qpos_finger_right[t, 1] = 右手食指关键点位置 + 姿态
...
```

这时它仍然不是机器人数据。

它只是：

```text
人手关键点 + 物体位姿
```

可以理解为：

```text
视频里的“目标轨迹”
```

如果用杯子例子：

```text
trajectory_keypoints.npz 里记录的是：

第 0 帧：
    杯子在 A 附近
    人手手腕在哪里
    人手五个指尖在哪里

中间帧：
    杯子逐渐从 A 移向 B
    手指跟着杯子运动

最后一帧：
    杯子在 B 附近
    手指也在对应位置
```

---

**三、contact 是怎么来的？**
有些 keypoints 文件还会被额外加上接触信息。

这个逻辑在 [detect_contact.py](/mnt/data3/guolijun/DexterousHand/spider/spider/preprocess/detect_contact.py):52 和 [detect_contact.py](/mnt/data3/guolijun/DexterousHand/spider/spider/preprocess/detect_contact.py):424。

它会判断：

```text
哪根手指在什么时候接触物体
接触点大概在哪里
```

可能保存：

```text
contact_right:     (T, 5)   # 1 or 0 的 bool 矩阵
contact_left:      (T, 5)   # 1 or 0 的 bool 矩阵
contact_pos_right: (5, 3)   # 接触点的 3D 位置坐标，注意没有 T，这个坐标是直接加在 right_object 上的，是局部坐标，不是检测的，是利用距离算出来的
contact_pos_left:  (5, 3)
```

其中 5 通常是：

```text
thumb, index, middle, ring, pinky
```

不过你当前的 `trajectory_keypoints.npz` 里没有 contact 字段。后面的 IK 如果读不到 contact，会默认全 1：

```python
contact_left = np.ones(...)
contact_right = np.ones(...)
```

这段在 [ik.py](/mnt/data3/guolijun/DexterousHand/spider/spider/preprocess/ik.py):228。

---

**四、trajectory_keypoints.npz 如何进入 IK？**
IK 阶段在 [ik.py](/mnt/data3/guolijun/DexterousHand/spider/spider/preprocess/ik.py):220 读取 keypoints：

```python
loaded_data = np.load(file_path)

qpos_finger_right = loaded_data["qpos_finger_right"]
qpos_finger_left = loaded_data["qpos_finger_left"]
qpos_wrist_right = loaded_data["qpos_wrist_right"]
qpos_wrist_left = loaded_data["qpos_wrist_left"]
qpos_obj_right = loaded_data["qpos_obj_right"]
qpos_obj_left = loaded_data["qpos_obj_left"]
```

然后它拼成一个更大的 `qpos_ref`：

```python
qpos_ref = np.concatenate(
    [
        qpos_wrist_right[:, None],
        qpos_finger_right,
        qpos_wrist_left[:, None],
        qpos_finger_left,
        qpos_obj_right[:, None],
        qpos_obj_left[:, None],
    ],
    axis=1,
)
```

代码在 [ik.py](/mnt/data3/guolijun/DexterousHand/spider/spider/preprocess/ik.py):253。

对于双手，它的语义顺序是：

```text
0  right_wrist
1  right_thumb_tip
2  right_index_tip
3  right_middle_tip
4  right_ring_tip
5  right_pinky_tip
6  left_wrist
7  left_thumb_tip
8  left_index_tip
9  left_middle_tip
10 left_ring_tip
11 left_pinky_tip
12 right_object
13 left_object
```

所以：

```text
qpos_ref.shape = (T, 14, 7)
```

这 14 个点是“人手/物体语义点”。

对于你的杯子例子，可以想成：

```text
第 t 帧：
    qpos_ref[t, 0]  = 右手腕目标位姿
    qpos_ref[t, 1]  = 右手拇指目标位姿
    qpos_ref[t, 2]  = 右手食指目标位姿
    ...
    qpos_ref[t, 12] = 杯子目标位姿
```

---

**五、IK 的核心比喻：木偶线**
IK 阶段做的事情可以用“木偶线”理解。

它有两类东西：

```text
目标点：来自人手视频，可以瞬移
机器人：真实 xhand，有关节结构，不能随便变形
```

代码会给每个目标点创建一个 `mocap body`。

这个 mocap body 可以理解成：

```text
漂浮在空间中的绿色小球/小盒子
它每一帧直接被放到人手关键点的位置
```

然后它把机器人上的 site 和这些 mocap body 用 equality constraint 绑起来。

添加 mocap body 和 equality constraint 的函数在 [ik.py](/mnt/data3/guolijun/DexterousHand/spider/spider/preprocess/ik.py):31。

比如：

```text
人手大拇指目标点
    ↓ 一根弹簧/约束
xhand 的大拇指 tip site
```

```text
杯子目标位姿
    ↓ 一根更硬的约束
MuJoCo 里的 object body
```

这样 MuJoCo 每 step 一下，就会试图让机器人关节调整到一种姿态：

```text
机器人指尖尽量贴近人手指尖
机器人手腕尽量贴近人手手腕
物体尽量贴近视频里的物体位置
```

这就是 IK 的本质。

**它不是在优化力，也不是在考虑真实抓取动力学。**

它是在问：

```text
xhand 有没有一个关节角度配置，可以让它的指尖看起来像视频里的人手指尖？
```

其实也就是说，xhand 是利用 MuJoCo 迭代，看这些手指 mesh 能不能有一个角度，来达到目标的位置。他是理论上机械手能够达到的目标。但是他没有考虑这个角度是不是真的能够被电机执行，没有考虑电机刚度、电机摩擦、遮挡碰撞等因素。比如他的 IK 算出来某个关节角需要设置为 180 度，但是实际上这个关节角的范围是 0～90 度，所以理论上能够达到的点实际上达不到。这就是 IK 的劣势，也是为什么还需要

---

**六、IK 得到的 trajectory_kinematic.npz 是什么？**
IK 最终保存发生在 [ik.py](/mnt/data3/guolijun/DexterousHand/spider/spider/preprocess/ik.py):796。

保存字段是：

```python
np.savez(
    out_npz,
    qpos=qpos_list,
    qpos_rollout=qpos_rollout,
    qvel=qvel_list,
    contact=contact_list,
    contact_pos=contact_pos_list,
    frequency=1 / ref_dt,
)
```

你当前的文件是：

```text
trajectory_kinematic.npz

qpos:         (1740, 50)
qpos_rollout: (1740, 50)
qvel:         (1740, 48)
contact:      (1742, 20)
contact_pos:  (1742, 20, 3)
frequency:    scalar
```

我们逐个解释。

---

**七、`qpos: (1740, 50)` 是什么？**
`1740` 是时间帧数。

它不是原始 `1744`，因为 IK 后面做了 moving average filter，并且为了计算速度会丢掉一些边界帧。

`50` 是 xhand bimanual 场景里的 MuJoCo `qpos` 维度。

你可以把它理解成：

```text
qpos[t] = 第 t 帧，整个机器人 + 物体的完整位置状态
```

对于你当前 xhand bimanual，大致是：

```text
右手: 18
左手: 18
右物体: 7
左物体: 7
总计: 50
```

其中物体的 7 是：

```text
x, y, z, qw, qx, qy, qz
```

所以如果是单个杯子从 A 到 B，概念上：

```text
qpos[0, object_pos_slice]  接近 A
qpos[-1, object_pos_slice] 接近 B
```

但注意：这是 IK 后放进机器人 MuJoCo 模型里的 object pose，不再是原始视频坐标，而是 MuJoCo 坐标系下的机器人场景坐标。

---

**八、`qvel: (1740, 48)` 是什么？**
`qvel` 是速度。

它由 `qpos` 相邻帧差分得到，代码在 [ik.py](/mnt/data3/guolijun/DexterousHand/spider/spider/preprocess/ik.py):759。

它不是简单 `qpos[i] - qpos[i-1]`，而是用 MuJoCo 的：

```python
mujoco.mj_differentiatePos(...)
```

为什么 `qpos` 是 50 维，但 `qvel` 是 48 维？

因为 freejoint 的姿态在 `qpos` 里用四元数 4 维表示，但在速度空间里旋转速度是 3 维。

所以每个 freejoint：

```text
qpos: position 3 + quaternion 4 = 7
qvel: linear velocity 3 + angular velocity 3 = 6
```

两个物体的话：

```text
qpos object 部分: 7 + 7 = 14
qvel object 部分: 6 + 6 = 12
```

因此：

```text
50 - 2 = 48
```

---

**九、`qpos_rollout` 是什么？**
`qpos_rollout` 是 IK 后做的一个检查性 rollout。

代码在 [ik.py](/mnt/data3/guolijun/DexterousHand/spider/spider/preprocess/ik.py):773。

它做的事情是：

```text
把 IK 得到的 qpos 的机器人部分当成控制 ctrl
放进普通 MuJoCo 里跑一下
看看仅靠这些控制，系统会 rollout 成什么样

ctrl 是控制信号，是 actuator 执行器的输入，不是实际上的目标位置，比如 ctrl[k]=1.10，说明现在是希望这个 k 关节移动 1.10 rad，但是不是真的一定能移动到。真正是否能移动到，取决于 actuator 参数，碰撞，摩擦等因素
```

所以：

```text
qpos:
    IK 直接求出来的目标状态

qpos_rollout:
    用这些目标控制跑一遍 MuJoCo 后得到的状态
```

注意：MJWP 阶段的 `load_data()` 并不读取 `qpos_rollout`。

MJWP 主要读取：

```text
qpos
qvel
contact
contact_pos
```

读取逻辑在 [io.py](/mnt/data3/guolijun/DexterousHand/spider/spider/io.py):33。

所以 `qpos_rollout` 更像是一个**辅助检查结果**，而不是 MJWP 的核心输入。

---

**十、`contact: (1742, 20)` 是什么？**
`contact` 是接触参考。

它告诉 MJWP：

```text
第 t 帧哪些接触点应该处于接触状态
```

你这里是 20 个接触相关 site，而不是 10 个手指，是因为 XML 里的 `track_site_ids` 包含了更多成对的 hand/object tracking sites。

可以通俗理解为：

```text
contact[t, i] = 1
表示第 t 帧第 i 个接触参考点应该有效

contact[t, i] = 0
表示这个点不用追踪接触
```

如果是杯子例子：

```text
前几帧，手还没碰杯子：
    contact 可能比较少或者为 0

抓住杯子以后：
    大拇指、食指、中指附近的 contact 变成 1

移动杯子时：
    这些 contact 持续为 1
```

---

**十一、`contact_pos: (1742, 20, 3)` 是什么？**
`contact_pos` 是每个接触参考点的位置。

可以理解成：

```text
第 t 帧，第 i 个接触点，应该在世界坐标的哪里
```

如果 `contact[t, i] = 1`，那么 MJWP 可以用它来鼓励机器人对应 site 靠近这个位置。

例如：

```text
contact_pos[t, 0] = 杯子表面某个拇指接触点的位置
contact_pos[t, 1] = 杯子表面某个食指接触点的位置
...
```

在代码里，contact reward 会比较：

```text
当前仿真中的 site_xpos
和 contact_pos_ref
```

奖励逻辑在 [mjwp.py](/mnt/data3/guolijun/DexterousHand/spider/spider/simulators/mjwp.py):312。

---

**十二、`frequency` 是什么？**
`frequency = 1 / ref_dt`。

如果 `ref_dt = 0.02`，那么：

```text
frequency = 50
```

意思是这条参考轨迹是 50 FPS。

---

**十三、从 kinematic 到 MJWP：核心思想**
现在进入最关键的 MJWP。

`trajectory_kinematic.npz` 还不是最终答案。

它只是说：

```text
理想情况下，每一帧机器人和物体应该在这里。
```

但是它不保证：

```text
机器人真的能通过控制做到
物体真的能被接触推动
抓取过程真的物理稳定
接触力真的合理
```

MJWP 要解决的是：

```text
**给我一串控制 ctrl，**
让 MuJoCo/MJWarp 里的真实物理系统跑出来的轨迹，
尽量接近 trajectory_kinematic.npz。

所以输入是 ctrl，也就是每个关键点希望他去到的位置
```

所以它从“姿态参考”变成了“物理控制问题”。

---

**十四、MJWP 读取 kinematic**
在 [run_mjwp.py](/mnt/data3/guolijun/DexterousHand/spider/examples/run_mjwp.py):258：

```python
qpos_ref, qvel_ref, ctrl_ref, contact, contact_pos = load_data(
    config, config.data_path)
```

`config.data_path` 默认就是：

```text
trajectory_kinematic.npz
```

`load_data()` 在 [io.py](/mnt/data3/guolijun/DexterousHand/spider/spider/io.py):28。

它读取：

```python
qpos_ref = raw_data["qpos"]
qvel_ref = raw_data["qvel"]
contact = raw_data["contact"]
contact_pos = raw_data["contact_pos"]
```

如果文件里没有 `ctrl`，它会这样做：

```python
ctrl_ref = qpos_ref[:, :-config.nq_obj]
```

也就是：

```text
把 qpos 里去掉 object 的部分，当成初始控制猜测
```

对于你当前 xhand bimanual：

```text
qpos_ref: (1740, 50)
object: 14
ctrl_ref: (1740, 36)
```

因为：

```text
50 - 14 = 36
```

这 36 维大致就是左右手的控制目标。

---

**十五、MJWP 构建并行仿真世界**
环境构建在 [mjwp.py](/mnt/data3/guolijun/DexterousHand/spider/spider/simulators/mjwp.py):110。

它先拿第一帧初始化：

```python
qpos_init = qpos_ref[0]

data_cpu.qpos[:] = qpos_init
data_cpu.qvel[:] = qvel_ref[0]
data_cpu.ctrl[:] = ctrl_ref[0]
```

也就是说：

```text
MJWP 从 IK 的第一帧状态开始
```

然后它创建 MJWarp 批量仿真：

```python
default_data_wp = mjwarp.put_data(
    model_cpu,
    data_cpu,
    nworld=config.num_samples,
)
```

如果：

```text
num_samples = 1024
```

那就等于创建了：

```text
1024 个一模一样的杯子 + 机器人仿真世界
```

为什么要 1024 个？

因为 MJWP 会同时尝试 1024 种控制方案。

这就像你在脑子里同时模拟 1024 种抓杯子的方式：

```text
方案 1：手指稍微往左
方案 2：手指稍微往右
方案 3：手腕抬高一点
方案 4：手腕低一点
...
```

然后看哪种让杯子更接近目标轨迹。

---

**十六、MJWP 每一轮在干什么？**
主循环在 [run_mjwp.py](/mnt/data3/guolijun/DexterousHand/spider/examples/run_mjwp.py):463。

每一轮它做：

```text
1. 看当前真实仿真状态
2. 从 qpos_ref 里取未来一段参考轨迹
3. 优化一段控制 ctrls
4. 只执行前 ctrl_steps 步
5. 再重新优化
```

这就是 MPC：

```text
计划未来一大段
只走眼前一小步
再重新计划
```

---

**十七、用杯子例子解释 MJWP**
假设当前杯子在 A：

```text
当前仿真杯子位置 = (3, 4, 5)
```

参考轨迹告诉 MJWP：

```text
未来 1.6 秒内，杯子应该逐渐移动到 B = (6, 7, 8)
```

MJWP 不会直接把杯子瞬移到 B。

它会尝试很多控制序列：

```text
ctrl candidate 1:
    大拇指压一点，食指收一点，手腕向右

ctrl candidate 2:
    大拇指松一点，手腕向上

ctrl candidate 3:
    手腕更快移动，指尖闭合更多


左手 ctrl_ref[t] = [
  tx, ty, tz, roll, pitch, yaw,
  thumb1, thumb2, thumb3,
  index1, index2, index3,
  mid1, mid2,
  ring1, ring2,
  pinky1, pinky2
]
...
```

所以其实这是一个类似于强化学习的方法，给定一个 ctrl 后，会自己去初始化多个机械手指的运动方式，后面再用 RL 训练，计算 reward。相当于多个扰动。


每个 candidate 都在一个 MJWarp world 里真实物理 step。

step 的代码在 [mjwp.py](/mnt/data3/guolijun/DexterousHand/spider/spider/simulators/mjwp.py):639：

```python
wp.copy(env.data_wp.ctrl, ctrl_mujoco)
wp.capture_launch(env.graph)
```

也就是：

```text
把控制写进去
让物理仿真往前走一步
```

走完之后，MJWP 检查：

```text
这个 candidate 里的杯子是不是更接近参考杯子位置？
机器人姿态是不是更接近 IK 姿态？
速度是不是更接近 IK 速度？
接触点是不是更接近参考接触点？
```

奖励计算在 [mjwp.py](/mnt/data3/guolijun/DexterousHand/spider/spider/simulators/mjwp.py):285。

核心是：

```python
qpos_dist = || qpos_sim - qpos_ref ||
qvel_dist = || qvel_sim - qvel_ref ||
reward = -qpos_dist - vel_scale * qvel_dist - contact_dist
```

所以：

```text
越接近参考，reward 越高
越偏离参考，reward 越低
```

最后它选出 reward 高的控制方案，把它们加权平均，得到新的 `ctrls`。

这个采样优化逻辑在 [sampling.py](/mnt/data3/guolijun/DexterousHand/spider/spider/optimizers/sampling.py):27 和 [sampling.py](/mnt/data3/guolijun/DexterousHand/spider/spider/optimizers/sampling.py):247。

---

**十八、为什么 MJWP 输出和 IK 不一样？**
这是最关键的直觉。

IK 阶段像是在说：

```text
我希望机械手第 t 帧长这样。
我希望杯子第 t 帧在这里。
```

MJWP 阶段像是在说：

```text
你希望归希望，但物理世界不一定允许。
我现在要找一串电机控制，让真实仿真尽量做到。
```

所以如果 IK 参考中杯子从 A 到 B 是完美直线：

```text
A -> B
```

MJWP 里的杯子可能会变成：

```text
A -> 稍微偏一下 -> 接触稳定后 -> 接近 B
```

这就是为什么 `trajectory_mjwp.npz` 是更接近“真实机器人可执行”的结果。

---

**十九、MJWP 最后保存 trajectory_mjwp.npz**
保存发生在 [run_mjwp.py](/mnt/data3/guolijun/DexterousHand/spider/examples/run_mjwp.py):629。

```python
np.savez(
    trajectory_mjwp.npz,
    **info_aggregated,
)
```

你当前的 `trajectory_mjwp.npz` 字段是：

```text
qpos: (87, 40, 50)
qvel: (87, 40, 48)
ctrl: (87, 40, 36)
time: (87, 40)
```

还有很多优化统计：

```text
qpos_dist_max:    (87, 32)
qpos_dist_min:    (87, 32)
qpos_dist_median: (87, 32)
qpos_dist_mean:   (87, 32)

qvel_dist_max:    (87, 32)
qvel_dist_min:    (87, 32)
qvel_dist_median: (87, 32)
qvel_dist_mean:   (87, 32)

qpos_rew_max:     (87, 32)
qpos_rew_min:     (87, 32)
qpos_rew_median:  (87, 32)
qpos_rew_mean:    (87, 32)

qvel_rew_max:     (87, 32)
qvel_rew_min:     (87, 32)
qvel_rew_median:  (87, 32)
qvel_rew_mean:    (87, 32)

rew_max:          (87, 32)
rew_min:          (87, 32)
rew_median:       (87, 32)
rew_mean:         (87, 32)

improvement:      (87, 32)
opt_steps:        (87, 1)
trace_cost:       (87, 32, 6)
trace_ref:        (87, 1, 1, 200, 12, 3)
```

---

**二十、为什么 qpos 是 `(87, 40, 50)`，不是 `(1740, 50)`？**
这个很重要。

`trajectory_kinematic.npz` 是按帧保存的：

```text
qpos: (1740, 50)
```

但 `trajectory_mjwp.npz` 是按 MPC 控制块保存的：

```text
qpos: (num_mpc_iterations, ctrl_steps, nq)
```

你这里是：

```text
num_mpc_iterations = 87
ctrl_steps = 40
nq = 50
```

所以：

```text
qpos.shape = (87, 40, 50)
```

如果你想把它变成普通时间序列，可以这样理解：

```python
qpos_flat = qpos.reshape(-1, 50)
```

那么：

```text
qpos_flat.shape = (87 * 40, 50)
```

也就是：

```text
3480 帧物理仿真状态
```

为什么会比 `1740` 多？

因为你的 `sim_dt = 0.01`，而原始 `ref_dt = 0.02`。

也就是：

```text
IK 参考 50Hz
MJWP 仿真 100Hz
```

所以 MJWP 的仿真帧数大约是参考帧数的两倍。

---

**二十一、trajectory_mjwp.npz 里面每个核心字段是什么意思？**
`qpos`：

```text
MJWP 真实物理仿真执行出来的机器人 + 物体位置状态
```

对于你的杯子例子：

```text
qpos[..., object_pos_slice]
就是 MJWP 里杯子实际移动的轨迹
```

`qvel`：

```text
MJWP 真实仿真状态的速度
```

`ctrl`：

```text
优化器最终选择并执行的控制
```

这是非常重要的字段。

`trajectory_kinematic.npz` 的重点是：

```text
参考姿态 qpos
```

`trajectory_mjwp.npz` 的重点是：

```text
真实执行出的 qpos
以及导致它的 ctrl
```

`time`：

```text
每个仿真步的时间
```

`qpos_dist_*`：

```text
候选轨迹的 qpos 跟 IK 参考 qpos 的距离统计
```

`qvel_dist_*`：

```text
候选轨迹的速度跟 IK 参考速度的距离统计
```

`qpos_rew_*`：

```text
由 qpos 距离得到的 reward 统计
```

`qvel_rew_*`：

```text
由 qvel 距离得到的 reward 统计
```

`rew_*`：

```text
总 reward 统计
```

`improvement`：

```text
这一轮优化相对初始候选提升了多少
```

`opt_steps`：

```text
这一轮 MPC 实际优化了多少次
```

比如最大允许 32 次，但可能提前停止。

`trace_ref`：

```text
用于可视化的参考轨迹点
```

`trace_cost`：

```text
被选出来显示的 sample 轨迹成本
```

---

**二十二、把四个阶段放在同一个杯子故事里**
现在我们完整走一遍。

原始视频里：

```text
第 0 帧：
    人手抓着杯子
    杯子在 A = (3, 4, 5)

第 T 帧：
    人手把杯子移动到 B = (6, 7, 8)
```

第一阶段，pkl：

```text
保存每一帧：
    人手腕位置
    人手腕旋转
    五个手指关键点位置
    杯子 4x4 位姿矩阵
    杯子 mesh 路径
```

第二阶段，`trajectory_keypoints.npz`：

```text
把 pkl 转成统一格式：

qpos_wrist_right[t]  = 右手腕 xyz + quat
qpos_finger_right[t] = 右手 5 个指尖 xyz + quat
qpos_obj_right[t]    = 杯子 xyz + quat
...
```

这时还是人手数据，不是机器人数据。

第三阶段，`trajectory_kinematic.npz`：

```text
通过 IK 把人手关键点映射到 xhand：

输入：
    人手手腕、指尖、杯子位姿

输出：
    xhand 的关节 qpos
    物体 qpos
    qvel
    contact
    contact_pos
```

这时已经是机器人状态，但主要是运动学结果，不保证物理控制真的能做到。

第四阶段，`trajectory_mjwp.npz`：

```text
通过 MJWarp 物理优化：

输入：
    IK 的 qpos_ref / qvel_ref / contact

优化：
    采样很多 ctrl
    并行物理仿真
    选 reward 高的控制

输出：
    真实仿真执行出来的 qpos / qvel / ctrl / time
```

这时才是 SPIDER 真正想要的机器人数据。

---

**最后用一句话压缩**
`trajectory_keypoints.npz` 是：

```text
人手和物体在视频里的目标点
```

`trajectory_kinematic.npz` 是：

```text
xhand 通过 IK 模仿这些目标点得到的参考状态
```

`trajectory_mjwp.npz` 是：

```text
xhand 在物理仿真里通过优化控制真正跑出来的可执行轨迹
```

所以你的杯子从 A 到 B，在三个文件里的身份分别是：

```text
keypoints:
    “视频中杯子在哪里”

kinematic:
    “如果让 xhand 模仿这个动作，理想的机器人和杯子状态是什么”

mjwp:
    “在物理仿真里，xhand 实际用什么控制把杯子移动过去，最后真实跑出了什么轨迹”
```
