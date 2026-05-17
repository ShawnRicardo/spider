• 我刚刚主要改了 4 类东西，都是为了让 MJWP 更容易按 IK 的关节角去跑。

  1. PD / 电机相关
  在 examples/asm_ourdata/generate_scene_ourdata_milk_asm.sh:27 里改了：

  arm-kp: 500 -> 1200
  hand-kp: 220 -> 300
  arm-damping: 2.5 -> 4.0
  hand-damping: 0.4 -> 0.6
  arm-force-scale: 6 -> 20
  hand-force-scale: 30 -> 50

  意思是：MJWP 阶段的位控电机更“用力”，更愿意把关节拉到目标角度。
  如果大臂仍然不到位，说明还可能有剩余碰撞、关节限位、目标姿态本身物理不可
  达，或者力矩/控制器还不够。

  2. 物体动力学
  同一个脚本里还改了：

  object-density: 20000
  object-armature: 0.01
  object-frictionloss: 0.02

  原来牛奶盒太轻、freejoint 太“飘”，容易飞。这个修改让物体更稳定，不那么容
  易被数值误差或手指撞飞。

  3. ref_dt / sim_dt 是什么

  sim_dt 是仿真每一步的时间间隔。现在是：

  sim_dt = 0.01

  意思是 MuJoCo 每一步推进 0.01s，也就是 100Hz。

  ref_dt 是 IK 参考轨迹每一帧之间的真实时间间隔。milk 视频是 30Hz，所以：

  ref_dt = 0.033333...

  意思是 IK 轨迹原始每帧间隔约 1/30s。

  MJWP 要在 100Hz 仿真里跟踪 30Hz 的 IK 参考，所以要把 IK 轨迹插值到仿真时
  间轴上。正确应该是：

  0.033333 / 0.01 ≈ 3

  也就是每两帧 IK 中间插到大约 3 个仿真步。

  之前代码的问题是：它先用默认：

  ref_dt = 0.02

  算出了：

  ref_steps = 2

  然后才读取 milk 的真实 ref_dt=0.033333。这样等于把 30Hz 的视频当成 50Hz
  来跑，动作会更快，PD 更难追上。

  现在改成先读真实 ref_dt，再算：

  ref_steps = 3

  效果是：MJWP 看到的 target 时间轴变慢、更符合原始视频速度。

  4. 忽略了什么碰撞

  我忽略的是：

  collision_body_shoulder_hub
  和
  collision_arm_right_upper / collision_arm_left_upper

  也就是 ASM 机器人肩部中心横梁 proxy 和左右大臂上臂 proxy 的碰撞。

  修改位置：

  spider/preprocess/generate_xml.py:1009

  之前诊断发现这组碰撞会把 Joint2 附近卡住，导致大臂无法抬到 IK 角度。现在
  只跳过这一类：

  shoulder_hub <-> upper arm

  没有全局关闭碰撞。手、物体、桌子、物体 bbox 仍然有碰撞。

  另外我还把桌子碰撞从：

  object_and_manipulator

  改成：

  object_and_hand

  也就是桌子现在和物体、手碰撞，但不再和大臂/小臂碰撞，避免桌子挡住机械
  臂。

  5. 当前 reward 有哪些

  你现在跑的 run_mjwp_ourdata_milk_asm.sh 用的是：

  examples/config/override/ourdata_asm.yaml:1

  核心 reward 在：

  spider/simulators/mjwp.py:515

  当前主要有：

  qpos_rew
  qvel_rew
  contact_rew
  fingertip_rew
  terminal_rew

  其中：

  qpos_rew

  是最核心的，表示当前 MJWP 机器人/物体 qpos 和 IK reference 的差距。里面
  又细分为：

  right_arm_qpos_rew
  left_arm_qpos_rew
  right_hand_qpos_rew
  left_hand_qpos_rew
  right_object_pos/rot
  left_object_pos/rot

  当前配置权重大致是：

  joint_rew_scale: 1.0
  arm_joint_rew_scale: 2.0
  hand_joint_rew_scale: 0.5
  pos_rew_scale: 0.3
  rot_rew_scale: 0.3
  fingertip_rew_scale: 1.0
  vel_rew_scale: 0.0001
  contact_rew_scale: 0.0

  所以当前实际上：

  手臂关节角 reward：开启，权重较高
  手指关节角 reward：开启
  物体位置/旋转 reward：开启，但比手臂低
  指尖位置 reward：开启
  速度 reward：很弱
  contact reward：关闭

  如果“大臂还是不到位”，我下一步会优先怀疑还有其他固定 body proxy 或左右臂
  互相碰撞 pair 在卡大臂，而不是 reward 没写。