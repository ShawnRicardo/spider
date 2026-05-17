#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

class JointStatePublisher(Node):
    def __init__(self):
        super().__init__('joint_state_publisher')
        
        # 创建关节状态发布者
        self.joint_state_pub = self.create_publisher(
            JointState, 
            '/joint_states', 
            10
        )
        
        # 关节名称列表
        self.joint_names = [
            'Joint1_R', 'Joint2_R', 'Joint3_R', 'Joint4_R', 'Joint5_R', 'Joint6_R', 'Joint7_R',
            'right_finger1_joint1', 'right_finger1_joint2', 'right_finger1_joint3', 'right_finger1_joint4',
            'right_finger2_joint1', 'right_finger2_joint2', 'right_finger2_joint3', 'right_finger2_joint4',
            'right_finger3_joint1', 'right_finger3_joint2', 'right_finger3_joint3', 'right_finger3_joint4',
            'right_finger4_joint1', 'right_finger4_joint2', 'right_finger4_joint3', 'right_finger4_joint4',
            'right_finger5_joint1', 'right_finger5_joint2', 'right_finger5_joint3', 'right_finger5_joint4',
            'Joint1_L', 'Joint2_L', 'Joint3_L', 'Joint4_L', 'Joint5_L', 'Joint6_L', 'Joint7_L',
            'left_finger1_joint1', 'left_finger1_joint2', 'left_finger1_joint3', 'left_finger1_joint4',
            'left_finger2_joint1', 'left_finger2_joint2', 'left_finger2_joint3', 'left_finger2_joint4',
            'left_finger3_joint1', 'left_finger3_joint2', 'left_finger3_joint3', 'left_finger3_joint4',
            'left_finger4_joint1', 'left_finger4_joint2', 'left_finger4_joint3', 'left_finger4_joint4',
            'left_finger5_joint1', 'left_finger5_joint2', 'left_finger5_joint3', 'left_finger5_joint4'
        ]
        
        # 初始化全部关节为 0.0
        self.joint_positions = [0.0] * len(self.joint_names)

        # ===================== 1. 双臂关节（角度 → 自动转弧度）=====================
        # 右臂 7个关节 (索引 0~6)：-85，-85，75，-70，-150，0，0
        right_arm_deg = [-85, -85, 75, -70, -150, 0, 0]
        self.joint_positions[0:7] = [math.radians(deg) for deg in right_arm_deg]

        # 左臂 7个关节 (索引 27~33)：85，-85，-60，-50，110，0，0
        left_arm_deg = [85, -85, -60, -50, 110, 0, 0]
        self.joint_positions[27:34] = [math.radians(deg) for deg in left_arm_deg]

        # ===================== 2. 手指关节（直接使用弧度）=====================
        # 右手 [7:27]  你写的7:24笔误，统一用20个长度7:27
        self.joint_positions[7:27] = [
            1.143, 0.354, 0.243, 0.197, 1.172, -0.126, 1.038, 1.256, 1.154, -0.340,
            1.330, 1.626, 0.986, -0.310, 1.628, 1.510, 0.451, -0.221, 1.407, 1.625
        ]

        # 左手 [34:54]
        self.joint_positions[34:54] = [
            1.107, 0.495, 0.070, 0.270, 1.010, 0.078, 1.195, 1.205, 1.037, 0.197,
            1.509, 1.629, 0.973, 0.256, 1.618, 1.630, 0.448, 0.190, 1.481, 1.632
        ]
        
        # 速度和力保持0
        self.joint_velocities = [0.0] * len(self.joint_names)
        self.joint_efforts = [0.0] * len(self.joint_names)
        
        # 只发布一次
        self.publish_joint_state()
        self.get_logger().info('✅ 所有关节已正确赋值，姿势发布完成！')

    def publish_joint_state(self):
        joint_state_msg = JointState()
        joint_state_msg.header.stamp = self.get_clock().now().to_msg()
        joint_state_msg.header.frame_id = ''
        joint_state_msg.name = self.joint_names
        joint_state_msg.position = self.joint_positions
        joint_state_msg.velocity = self.joint_velocities
        joint_state_msg.effort = self.joint_efforts
        
        self.joint_state_pub.publish(joint_state_msg)

def main(args=None):
    rclpy.init(args=args)
    node = JointStatePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

