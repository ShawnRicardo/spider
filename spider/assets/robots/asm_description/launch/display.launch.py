import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import Command

def generate_launch_description():
    # 1. 获取功能包路径
    pkg_share = get_package_share_directory('asm_description')
    
    # 2. 定义 URDF 文件路径
    urdf_file_path = os.path.join(pkg_share, 'urdf', 'asm.urdf')
    
    # 3. 加载 URDF 内容（支持直接加载 URDF 或 Xacro 转换）
    # 如果用 Xacro，替换为：Command(['xacro ', urdf_file_path])
    robot_description_content = Command(['cat ', urdf_file_path])
    
    # 4. 声明机器人描述参数（核心：将 URDF 传给 ROS 2 参数服务器）
    robot_description = {'robot_description': robot_description_content}

    # 5. 定义节点
    # 机器人状态发布器（发布 TF 变换）
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[robot_description],
        output='screen'
    )

    # 关节状态发布器（带 GUI，可调节关节角度）
    joint_state_publisher_gui_node = Node(
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
        output='screen'
    )

    # RViz2 节点（加载预设配置）
    rviz2_node = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', os.path.join(pkg_share, 'config', 'asm_right.rviz')],
        output='screen'
    )

    # 6. 组装 Launch 描述
    ld = LaunchDescription()
    ld.add_action(robot_state_publisher_node)
    ld.add_action(joint_state_publisher_gui_node)
    ld.add_action(rviz2_node)

    return ld
