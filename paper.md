## 初印象

SPIDER 本质上是一个数据生成 pipeline，目标是大规模、高效地把人类动作数据转成机器人可执行的物理可行轨迹。这类工作在 robotics 领域很常见，学界叫 retargeting（动作迁移/重定向）。

但它有实质性的技术贡献，不是简单的"直接映射"

"直接映射"的方法早就有了（就是 IK retargeting），问题是生成的轨迹物理上不可行——手会穿进物体、抓不住、姿势违反力学。
SPIDER 主要也是为了解决这些物理碰撞的问题，此外，其增量贡献在于：
    1. 退火采样核：让采样优化在非凸接触空间里能可靠收敛（这不是 trivial 的，标准 MPPI 在这里会失败）；
    2. 虚拟接触引导：解决了一个之前被忽视的问题——即使物体轨迹对了，机器人的接触模式可能是错的（比如人用拇指+食指捏，机器人用食指+中指捏，物体运动一样但接触意图不同）。这个约束设计有一定新意。
    3. 规模：9种机器人形态 + 6个数据集 + 2.4M帧，是目前这类工作里覆盖最广的。

这个代码库基本上和深度学习没有关系，偏向自动控制+物理仿真，基本上没有深度学习的内容

SPIDER 做的事情是：把人手的动作"翻译"成机器人的动作。所以它既需要"人怎么动"的原始数据，也需要处理后能直接喂给仿真器的格式。

pipeline 如下：

```
原始人手视频/动捕数据
    │
    │ SPIDER 流水线（IK + run_mjwp.py）
    │ 把每一条都"翻译"成对应机器人的物理合法轨迹
    ▼
N 万条 trajectory_mjwp.npz
    │
    │ 交给下游（可选）
    ▼
 ├─ 行为克隆（直接拿轨迹当监督数据训神经网络）
 ├─ RL 训练（把 SPIDER 轨迹当 reference motion，配 RL 学一个 policy）
 └─ 直接部署到真机（zero-shot，不训 RL）
```

用一句话：把"人手怎么操作物体"这个问题，翻译成"机器人手怎么操作同样的物体"。

再展开一点：

问题起点。我们有海量"人手操作物体"的数据（GigaHands、OakInk、Hot3D 等公开数据集里几百小时的视频），但没有对应的机器人数据——因为没人会拿真机器人挨个复刻一遍。机器人手跟人手形状差异巨大（自由度不同、关节范围不同、指节长度不同），人手数据根本不能直接喂给机器人。

SPIDER 的核心主张：用"物理仿真 + 优化"取代"数据收集"。只要你给我一段人手轨迹和一个机器人模型，我可以在仿真里算出一段让这个机器人也能完成同样任务的物理合法动作序列。整个过程不需要任何人工示教、不需要训练网络、不需要真机交互。

算法核心：两阶段

阶段 A：几何对齐（IK） —— 只看"手指指尖在哪儿"，找一组机器人关节角让指尖对齐。快但粗糙，不保证物理可行
阶段 B：物理优化（run_mjwp.py） —— 在 GPU 上同时跑 1024 个仿真，每次加不同的噪声扰动 IK 控制序列，看哪条扰动后的结果最接近参考轨迹（奖励最高），然后用 softmax 做加权得到下一次的搜索中心，反复 32 次。这本质上是一个零阶优化器，不需要梯度，所以可以直接优化穿过不可导的接触物理


example_datasets 里面有 raw 和 processed：
    raw 是各家的原始数据，只读，永远不改，有 OakInk（动捕+视觉），GigaHands（多视角相机），Hot3D（头戴相机）以及其他的；这些数据的格式各不相同
    processed 是经过处理后的统一能够输入到 spider pipeline 的数据，这些数据都经过了标准化的处理。

```
processed/oakink/
├── assets/          ← 共享资源
│   ├── objects/     ← 物体的 3D 网格（茶壶、烧杯等）
│   └── robots/      ← 机器人模型（见下文）
├── mano/            ← 人手运动数据（MANO 是标准的参数化手模型）
├── ability/         ← 已经把 mano 重定向到 Ability Hand 的结果 # 已经把 mano 重定向到 Ability Hand 的结果 机器人型号 1
├── allegro/         ← 重定向到 Allegro Hand    # 机器人型号 2
├── inspire/         ← 重定向到 Inspire Hand    # 机器人型号 3
├── schunk/          ← 重定向到 Schunk SVH      # 机器人型号 4
└── xhand/           ← 重定向到 XHand           # 机器人型号 5
```

```
xhand/bimanual 双手任务，对应的还有 right left 单手
    stir_beaker 具体任务名，
    ...
```

    但是还有一个不一样的点，就是发现 `examples` 文件夹中有不同的仿真器环境，run_dexmachina，run_hdmi, run_maniptrans, run_mjwp 等。而数据集中，也有 hdmi, maniptrans, dexmachina(raw/)，这是因为这些名字代指的是外部的第三方项目，就是其他的方法，这些数据的组织形式并不是本项目的组织形式，而是他那个项目的组织形式。

```
example_datasets/processed/<dataset>/<robot>/<embodiment>/<task>/
│
├── scene.xml          ← 默认物理场景（MJWP 用）
├── scene_eq.xml       ← MJWP-EQ 用（已废弃）
├── scene_act.xml      ← MJWP-ACT 用（接触引导）
├── task_info.json     ← 任务元信息（物体路径、时间步长等）
│
├── 0/                 ← data_id=0 的这条轨迹
│   ├── trajectory_kinematic.npz    ← IK 产物
│   ├── trajectory_ikrollout.npz    ← IK+物理 rollout 产物
│   ├── trajectory_mjwp.npz         ← MJWP 最终产物 ★
│   ├── trajectory_mjwpeq.npz       ← MJWP-EQ 产物（废弃）
│   ├── trajectory_mjwp_act.npz     ← MJWP-ACT 产物
│   ├── visualization_*.mp4         ← 对应可视化视频
│   └── config[_act].yaml           ← 跑这条轨迹时用的配置
├── 1/                 ← data_id=1
├── 2/                 ← data_id=2
└── ...
```

```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT: 人手数据                            │
│    (raw MANO / 视频 / 动捕) 在这个项目中是 pkl，人手+物体的动捕数据，每 1/30 秒手腕位置、手腕朝向、指尖位置、勺子和碗位置，指尖离物体距离│
└──────────────────────────┬──────────────────────────────────┘
                           │ process_datasets/*.py
                           │ (格式统一化)
                           ▼
              processed/<dataset>/mano/.../trajectory_keypoint.npz
                           │
                           │ preprocess/decompose_fast.py
                           │ (物体凸分解)
                           ▼
                 processed/<dataset>/assets/objects/
                           │
                           │ preprocess/generate_xml.py
                           │ (生成 MuJoCo 场景)
                           ▼
        processed/<dataset>/<robot>/.../scene.xml   ← model_path 最终指向这里
                           │
                           │ preprocess/ik_fast.py
                           │ (几何对齐，IK 求解)
                           ▼
    processed/<dataset>/<robot>/.../<task>/<id>/trajectory_kinematic.npz 只保证指尖位置对，但是不保证指尖和物体的物理关系，这个时候已经变成了机械手的电机角度了
                           │                      ↑
                           │                   data_path 指向这里
                           │
                           │ examples/run_mjwp.py ← 你跑的就是这一步
                           │ (采样式 MPC 物理优化)
                           ▼
    processed/<dataset>/<robot>/.../<task>/<id>/trajectory_mjwp.npz  ★
                           │
                           │ postprocess/read_to_robot.py (可选)
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              OUTPUT: 真机可执行的机器人轨迹                     │
│     (或作为 RL / BC 训练的监督数据)                            │
└─────────────────────────────────────────────────────────────┘

```

```
qpos [1740, 50] 这个 50 维怎么拆的？
MuJoCo 的 qpos 是整个场景里所有自由度的拼接。对于 pick_spoon_bowl 双手任务 + Xhand 机械手，50 维可能是这样拆的（大致——具体顺序由 scene.xml 决定）：


qpos [50] =
    ┌──────────────────────────────────────┐
    │ 右手腕 6D pose         [7]   xyz+四元数  │  ← 机器人右手腕
    │ 右手 Xhand 关节角      [12]            │  ← 机器人右手 12 个电机角度
    │ 左手腕 6D pose         [7]             │  ← 机器人左手腕
    │ 左手 Xhand 关节角      [12]            │  ← 机器人左手 12 个电机角度
    │ 右手物体 pose          [7]   xyz+四元数  │  ← 物体 1（勺子）
    │ 左手物体 pose          [7]   xyz+四元数  │  ← 物体 2（碗）
    └──────────────────────────────────────┘
    总计 7+12+7+12+7+7 = 52  （实际 50，可能有 free joint 合并）
```


```
人手数据（pkl）                    机器人轨迹（npz）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
wrist_pos    [T, 3]     ━━IK━▶    qpos 里的机器人手腕 xyz  [T, 3]
wrist_rot    [T, 3]     ━━IK━▶    qpos 里的机器人手腕四元数 [T, 4]
mano_joints  [T, 5, 3]  ━━IK━▶    qpos 里的机器人指节角度   [T, 12]
                         （求解器要问："电机要转多少才能让指尖到你说的位置"）
obj_trajectory [T, 4, 4] ━━直接复用━▶ qpos 里的物体 pose   [T, 7]
                         （物体不需要 IK，就是 copy 过去）
tips_distance [T, 5]    ━━阈值化━▶  contact [T, 20]
                                    contact_pos [T, 20, 3]

```


## 人手数据到 IK 输出技术路线梳理

两条并行逻辑
    手部：关键点目标（site）-> IK 约束求解 ->机器人关节（qpos）
    物体：  ik.py 物体目标(site) -> mocap + WELD 强约束 -> 物体 freejoint 跟随
            ik_fast.py 每帧直接写 qpos 尾部

### 1 IK 输入数据怎么组织

Oakink 预处理把 wrist/fingertip/object  都转成 [x,y,z,qw,qx,qy,qz]，并保存为 trajectory_keypoints.npz：oakink.py (line 256)

同时拼成统一顺序（14个目标）：
[R_wrist, R_5tips, L_wrist, L_5tips, R_obj, L_obj]：oakink.py (line 268)

这里的输出本质就是，每一帧中右手腕、右手五指尖、左手腕、左手五指尖、右物体、左物体这14个关键向量，他们分别在哪里，朝向如何（7维），这个格式的统一非常关键（14*7）

### 2 机器人和物体在场景里的“承载位置”

xhand 的手部目标 site 名字在 robot 中的 xml 文件中进行定义，比如 `spider/assets/robots/xhand` 中的 left.xml 和 right.xml。这些 xml 文件不是数据文件，是本体说明书。

trajectory 中存的是目标轨迹，他是不知道本体本身的连接方式的。这些 xml 文件就是用来进行本体说明的。比如机器人有几个关节、手指和手指之间如何连接，某个指尖的目标落到模型的哪个位置、想改拇指尖的位置应该变换哪个关节。

具体的 xml 分析我已经放到了 xml 中。

但是同一个文件夹下还有 urdf 文件，xml 偏向的是 mujoco 仿真，urdf 更偏向通用描述

URDF 是“机器人说明书（通用版）”
MuJoCo XML 是“把这个机器人放进 MuJoCo 里能跑起来的执行版”

preprocess/generate_xml.py 文件是把“机器人模型 + 任务物体 + 接触/碰撞规则 + 参考点/约束”组装成一个可供后续 IK / 物理优化直接使用的 MuJoCo 场景文件。除了读取前面的 xml 文件外，还要加载物体 mesh，最后生成 scene.xml 或者是 _eq.xml，为后续的 IK 做准备

### 3 ik.py 的映射

这个文件就是输入 traj_kp.npz 和 scene.xml，生成 traj_kinematic.npz，整体的流程和输入输出非常复杂。

举一个例子来说明，比如现在“我们用左手将杯子从（3，4，5）这个点移动到（6，7，8）”
它做的是：

“我告诉机器人：
左手掌这一帧应该在哪，
左手指尖这一帧应该在哪，
杯子这一帧应该在哪；
然后让 MuJoCo 自己去求一组关节角，使这些点尽量对齐目标。”

也就是：

输入：目标点/目标位姿轨迹
输出：机器人真实关节状态轨迹

具体流程如下：
1. 先读取 traj_kp.npz （可以理解为人手）文件
比如，第 0 帧，左手掌的目标位置在（2.9, 4, 5），杯子在（3, 4, 5），左手指尖目标位置在合适地方；在第 1 帧，左手掌目标在 (5.9, 7, 8)，左手五个指尖也移动到了新的位置，杯子目标在（6, 7, 8）。然后拼接成 `q_pos_ref[0], q_pos_ref[1], ... ,q_pos_ref[N]`，这样程序就知道了每一帧手和物体的位置应该在哪里。

2. 真实数据结构的创建
mujoco 场景里有 left_palm, left_thumb_tip, left_object 这些 site，挂在机器人的左手 body，杯子 body 上，他们会随着机器人关节和物体状态变化而移动。这些是真实模型里被跟踪的对象，就是那些关键点所在的 body（mesh）。

3. 虚拟中间量数据结构的创建
再额外创建一套 mocap target。程序会新建 target_mocap_body_left_palm, target_mocap_body_left_thumb_tip, target_mocap_body_left_object，这些 mocap_body 不受机器人关节限制，可以直接将他们瞬移到任意位置。

那么在这一步之后，场景有两套东西。一套是 2 中的真实 site，mujoco 中的真实 body，第二套是 3 中的虚拟 site，也就是目标 site，是不受控制可以瞬移的 target_site。

4. 用 equality constraint 把这两套东西绑起来。程序会给每一对的 site 建立约束映射

```
left_parm -> target_mocap_left_palm
left_thumb_tip -> target_mocap_left_thumb_tip
... -> target_mocap_...
left_object -> target_mocap_left_object
```

这一步的意思就是机器人和物体身上的真实 site（2）要和 target mocap site（3） 跟着走

5. 现在我们有了 q_pos(真人数据的位置)，left_palm(真实机器人和物体的位置)，target_mocap(中间量，可随意移动的真实机器人和物体位置)
然后我们从 q_pos 列表里取出第 0 帧的所有数据，把它放到 target_mocap 中，如下：

```
mj_data_ik.mocap_pos[left_object 的 mocap_idx] = [3, 4, 5]
mj_data_ik.mocap_pos[left_palm 的 mocap_idx] = [2.9, 4, 5]
```

这一步结束后，相当于中间量已经有了确定的目标

6. MuJoCo 开始求解
中间量已经有了明确的目标位置，但是我们的真实机械臂、真实物体的位置还没有到那里。
执行

```
mujoco.mj_step(mj_model_ik, mj_data_ik)
```

执行多步。由于有 equality constraint, MuJoCo 会自动尝试调整系统状态，使得真实机器臂的各个位置逼近目标位置

这个时候，由于 muJoCo 会严格执行，将真实数据映射到真实机械臂上去，所以 qpos 这个时候存的就是当前整套系统的真实状态了。

7. 整体

在你的杯子例子里，整个 IK 过程就是：

先告诉系统：
第 0 帧时左手掌、指尖、杯子应该在 A 附近；
第 1 帧时它们应该在 B 附近；
然后在 MuJoCo 里创建一组可直接摆位的 mocap 目标点，
再用约束把真实手掌/指尖/杯子 site 拉向这些目标点，
让仿真器自动解出每一帧对应的机器人关节状态和物体状态。

这是我觉得最直观的类比。

真实机器人 site：狗
mocap body：主人手里的牵引点
equality constraint：狗绳
qpos_ref：主人这一刻想走到哪
qpos：狗真实跑到哪、四肢怎么摆

于是过程就是：

主人先看地图（qpos_ref）
决定这帧左手掌目标在哪、杯子目标在哪
把牵引点（mocap body）放到这些位置
狗绳（constraint）拉着真实手掌/指尖/杯子去跟上
MuJoCo 自动求解狗该怎么迈腿（机器人关节怎么调）
最终得到真实姿态（qpos）

8. 杯子自己动？

是的，在这个过程中，杯子 tmd 就是自己动的，就是直接映射的，根本就不是机械手让他驱动着动的。
因为是造数据，所以也算合理，而且这是 IK ，第一阶段。第二阶段的 MJWP 可能才是有物理碰撞。

9. 变量梳理

```
第 i 帧
q_pose_left_obj = [3, 4, 5, qw, qx, qy, qz]
q_pose_left_wrist = [2.9, 4, 5, qw, qx, qy, qz]
q_pose_left_finger_thumb = [x, y, z]

q_pose[i, idx_1] = q_pose_left_obj
q_pose[i, idx_m] = q_pose_left_wrist
q_pose[i, idx_n] = q_pose_left_finger_thumb
```

```
target_mocap_bodies={
    "target_mocap_body_left_palm",
    "target_mocap_body_left_thumb_tip",
    ...
    "target_mocap_body_left_object",
}
```

site_for_mimic 本次真实机械结构要逼近的是
```
{
    "left_palm",
    "left_thumb_tip",
    "left_index_tip",
    "left_middle_tip",
    "left_ring_tip",
    "left_pinky_tip",
    "left_object",
}
```

index_map：index_map 是将 qpos_index 和 mocap_index 的 body site 对应起来的一张表
```
index_map["left_palm"]={
    "qpos_idx": id_m,
    "mocap_idx": id_n
}
```

10. traj_kinematic.npz 保存的结构

最后得到的 traj_kinematic.npz 并不是存的每个关节点的每一帧的位置，和之前关节可动物体一样，其实存的是变换。具体来说，他存的是每一帧 手腕6DoF，食指关节 1 的变换，食指关节 2 的变换，...，物体位姿。通过这些，可以使用 mj_forward 和 mj_kinematics 算出每个手指在每一帧的位置。具体结构如下：

```
qpos: [1740, 50] 1740 帧，每帧 50
    左手 18（手腕 6 + 拇指和食指 3*2 + 其他四指 2*3 = 18），
    右手 18，
    左右物体都是 7 （位置+四元数）
    18+18+7+7=50

qpos_rollout: [1740, 50] qpos 是直接用 IK 约束求解出来的状态，都是理论计算的。不是直接控制灵巧手关节得到的结果。而这个 rollout 就是把 qpos 当做 actuator 的控制目标后，真实的动力学系统再跑出来的状态。
给定当前 qpos，利用 mj_forward 把各种派生量算出来，比如 body 位姿，site 位置，传感器值。这都是计算，不是真正控制系统去执行任务。
而 rollout 是把 qpos 的前一部分（机器人关节部分）当作 position actuator 的目标，然后用正常的 mujoco 模型去跑，看系统跑到了哪里。
具体例子：
比如现在通过 MuJoCo 通过 mocap + equality constraints 求解出一帧的解：
```
qpos[t] = [
  左手腕tx = 5.85,
  左手腕ty = 6.95,
  左手腕tz = 8.02,
  左手腕roll = 0.10,
  左手腕pitch = -0.05,
  左手腕yaw = 0.20,
  ...
  左食指joint1 = 1.10,
  左食指joint2 = 0.85,
  ...
  左物体 = [6.0, 7.0, 8.0, 1,0,0,0]
]
```
然后再把这个当做命令给到 MuJoCo 按照 actuator 规则执行一步，真实执行会有很多问题，比如 actuator 刚度有限，接触扰动，动力学积分不到位，随机噪声等等，结果和 qpos 不完全一样，得到的 qpos—rollout 如下：
```
qpos_rollout[t] = [
  左手腕tx = 5.80,
  左手腕ty = 6.90,
  左手腕tz = 8.00,
  左手腕roll = 0.09,
  左手腕pitch = -0.04,
  左手腕yaw = 0.18,
  ...
  左食指joint1 = 1.03,
  左食指joint2 = 0.80,
  ...
  左物体 = [5.96, 6.98, 7.95, 0.999,...]
]
```


qvel:[1740, 48] 1740 帧，每帧 48，表示的是速度
    左右手 18+18
    左物体和右物体，是 3 维线速度+3 维角速度，不是位置+四元数了

contact: (1742, 20)，双手 10 个指头+左右两个物体接触点 10 个，里面是 0 or 1。
0 表示这一帧这个点不接触，1 表示接触

contact_pose：(1742, 20，3) 每一帧每一个参考点的 3D 空间位置

多两帧是因为前面用 window_len=3 做了滑动平均

```

## Retargeting 技术路线梳理

spider/simulators/mjwp.py

这个文件不是主程序，是被外层优化器调用的脚本，从而间接生成最终的 npz

1）把 MuJoCo 场景和 IK 参考轨迹装进 MJWarp，
2）并行推进很多个 world，
3）计算 reward / terminate / trace / state copy 等。

它做的事情有点像 Gym 里的 environment backend：
    建环境
    重置环境
    批量 step
    读状态
    算奖励
    判断终止
    保存/恢复状态
    同步 MuJoCo 和 Warp 状态

MjWP 是 MuJoCoWarp 的缩写，实际上就是将之前模拟出来的 IK 轨迹，放到真实的动力学/接触/控制下，怎样能够更接近理论数值。MJWP 是物理采样与评估阶段的底层环境，有很多接口函数，setup_env(), step_env(), get/set_qpos(), get_reward(), get_terminal_reward()

这个文件里面没有主函数，都是准备的一些被调用的函数，主函数在 run_mjwp.py

MJWP 阶段不是把 IK 的 qpos_ref 直接复制成最终轨迹，而是把 IK 结果当作“参考轨迹”，然后在物理仿真里优化 ctrl，让真实机器人和物体在接触、动力学、约束下尽量跟踪这条参考轨迹。最终保存的是物理仿真跑出来的状态。

而 MJWP 主要用到的输入有两个部分，一个是 config，另一个是 ref 数据。
config 对象就是一些配置文件，有 model_path, sim_dt, device, num_samples, 各种 reward_scale，各种阈值等等，就是基本上所有的配置和超参数
ref 数据其实就是上一步我们通过 IK 生成的一些机器灵巧手在每一帧的数据情况：(qpos_ref, qvel_ref, ctrl_ref, contact_ref, contact_pos_ref)，一些 MJWP 的配置函数都会用到这些数据

```
trajectory_kinematic.npz
        ↓
读取 qpos_ref / qvel_ref / contact_ref
        ↓
构建 MuJoCo + MJWarp 批量仿真环境
        ↓
采样很多组控制 ctrl 候选
        ↓
在 MJWarp 里并行 rollout
        ↓
用 reward 评估谁更接近 IK 参考轨迹
        ↓
选出更好的 ctrl
        ↓
真正执行一小段仿真
        ↓
保存 trajectory_mjwp.npz
```

### 1 入口 main(config), `run_mjwp.py`

处理 Hydra 读进来的配置。也包括命令行中的配置，都读到这个 config 里面去，有下面几个作用（非常重要且基础的功能）：

1. 根据机器人类型判断 object 的 qpos 维度
2. 拼出 model_path，也就是 MuJoCo XML 路径
3. 拼出 data_path，也就是 trajectory_kinematic.npz 路径
4. 读取 MuJoCo model，得到 nq / nv / nu / npair
5. 计算 horizon_steps / ctrl_steps / ref_steps
6. 设置输出目录 output_dir
7. 如果 task_info.json 里有 ref_dt，就覆盖 config.ref_dt

### 2 读取 IK 结果

`run_mjwp.py` line 258 数据加载函数，这里这个 datapath 就是 kinematic.npz ，读取 qpos，qvel 这些信息，以及接触信息 contact

IK 参考轨迹的频率比较低，MJWP 的频率更高，所以需要插值到 sim_dt 对应步长

数据打包，就是 qpos_ref, qvel_ref, ctrl_ref, contact, contect_pos 这五个东西会贯穿 MJWP 全流程 line 280

### 3 构建 MJWP 环境
mjwp.py line 289 setup_env

然后在 mjwp.py 中 line 110，用 IK 轨迹的第一帧作为仿真的初始状态。

创建普通 MuJoCo CPU model，line 118

然后对于 mjwp.py line 78 的 setup_mj_model 函数，加载最终用于物理仿真的 xml，除了读取 xml 里面的参数，还会设计一些 MuJoCo 接触求解和积分器参数。

mjwp.py line 121，用 IK 第一帧初始化仿真，qpos 是 IK 第一帧，qvel 是 IK 第一帧速度，ctrl 是 IK 第一帧的初始控制

...

### 构建 optimizer 的控制序列

run_mjwp.py line 455

ctrls = ctrl_ref[: config.horizon_steps]

ctrls.shape = (40, 36) 表示 从当前时刻开始，未来 40 个仿真步的控制初始猜测。但是它后面会被优化，不是固定不变。

### 主循环
run_mjwp.py line:463

核心优化部分是 line 527 左右的 optimize 函数，是 retargeting 的核心

优化器的定义在 `sampling.py` line 366


### 总流程

1. main(config)
   读取并处理配置

2. process_config(config)
   确定 XML、数据路径、nq/nv/nu、horizon_steps、ctrl_steps

3. load_data(config, trajectory_kinematic.npz)
   读取 IK 阶段输出的 qpos_ref/qvel_ref/contact/contact_pos
   如果没有 ctrl，就用 qpos_ref 去掉 object 部分作为 ctrl_ref

4. setup_env(config, ref_data)
   用 qpos_ref[0] 初始化 MuJoCo
   转成 MJWarp model/data
   创建 num_samples 个并行 world
   捕获一次 mjwarp.step CUDA graph

5. ctrls = ctrl_ref[:horizon_steps]
   用 IK 控制作为初始控制猜测

6. while loop
   当前真实状态是 env.data_wp 的 world 0
   从 ref_data 中取未来 horizon 的参考轨迹

7. optimize(config, env, ctrls, ref_slice)
   对 ctrls 加噪声采样
   得到 num_samples 条候选控制轨迹
   在 MJWarp 中并行 rollout
   reward 比较 qpos/qvel/contact 是否接近 IK 参考
   用 elite 样本加权平均得到新的 ctrls

8. 执行 ctrls 前 ctrl_steps 步
   这一步才真正推进真实仿真

9. sync_env()
   把 world 0 的真实状态复制到所有 world
   准备下一轮采样优化

10. 滑动控制窗口
    保留未执行的控制
    从 ctrl_ref 追加新的未来控制

11. 保存 trajectory_mjwp.npz
    保存物理仿真后的 qpos/qvel/ctrl/sim_time

## 数据流的产生

数据流分为 物体/人手数据流 和 机器人数据流
最开始用的是 `example_datasets/raw/oakink` 这里面的 meshes 和 pkl 文件，搭配 `spider/process_datasets/oakink.py`，生成的是：
    - processed/oakink/mano/.../trajectory_keypoints.npz
    - processed/oakink/mano/.../task_info.json
    - processed/oakink/assets/objects/... objs
    这里面的 npz 包含 是人手关键点的位置（手腕+手掌）+物体逐帧位姿，物体 mesh 资产，任务元信息。

后面是 `spider/assets/robots/new_asm_description` 中的 urdf 和 机器人手臂和手指的 meshes，搭配 `prepare_asm_mjcf.py` 和 `spider/preprocess/generate_xml` 生成 `left.xml`，`right.xml` 等机器手的 xml 以及带上了物体的 `scene.xml`。
