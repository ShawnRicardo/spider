#!/usr/bin/env python3
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
        
        # 关节名称列表 (保持不变)
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
        
        # ===================== 核心修改 =====================
        # 直接在这里定义固定的关节位置数组
        # 总共有 56 个关节，数组长度必须严格等于 56
        # 你可以自由修改这里的数值，格式：[0.0, 0.0, 0.1, ...]
        # ====================================================
        self.joint_positions = [
            # 右臂 7个关节
            -1.57, -1.57, 1.57, 0.0, -1.57, 0.0, 0.0,
            # 右手5根手指 (每根4个关节)
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            # 左臂 7个关节
            1.57, -1.57, -1.57, 0.0, 1.57, 0.0, 0.0,
            # 左手5根手指 (每根4个关节)
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0
        ]
        
        # 初始化关节速度和力数据 (固定为0)
        self.joint_velocities = [0.0] * len(self.joint_names)
        self.joint_efforts = [0.0] * len(self.joint_names)
        
        # 发布一次关节状态
        self.publish_joint_state()
        
        self.get_logger().info('关节状态已发布完成，节点已启动')

    def publish_joint_state(self):
        """发布固定的关节状态消息"""
        joint_state_msg = JointState()
        joint_state_msg.header.stamp = self.get_clock().now().to_msg()
        joint_state_msg.header.frame_id = ''
        joint_state_msg.name = self.joint_names
        joint_state_msg.position = self.joint_positions
        joint_state_msg.velocity = self.joint_velocities
        joint_state_msg.effort = self.joint_efforts
        
        # 发布消息
        self.joint_state_pub.publish(joint_state_msg)
        self.get_logger().info(f'已发布关节位置，长度: {len(self.joint_positions)}')

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

