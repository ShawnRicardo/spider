import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():
    # 1. 获取功能包路径
    pkg_asm_right = get_package_share_directory('asm_description')
    
    # 2. 定义URDF文件路径
    urdf_file_path = os.path.join(pkg_asm_right, 'urdf', 'asm.urdf')
    
    # 3. 启动Gazebo空世界
    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('gazebo_ros'), 'launch', 'gazebo.launch.py')
        ),
        launch_arguments={'world': 'empty.world'}.items()  # 使用空世界
    )
    
    # 4. 将URDF模型生成到Gazebo中
    spawn_robot_node = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-file', urdf_file_path,  # URDF文件路径
            '-entity', 'my_asm_robot',  # 模型在Gazebo中的名称（自定义）
            '-x', '0.0', '-y', '0.0', '-z', '0.5',  # 模型生成的初始位置
            '-R', '0.0', '-P', '0.0', '-Y', '0.0'   # 初始姿态（rpy）
        ],
        output='screen'
    )
    
    # 5. 启动机器人状态发布器（发布URDF和TF）
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': open(urdf_file_path).read()}],
        output='screen'
    )

    # 组装Launch描述
    return LaunchDescription([
        gazebo_launch,
        robot_state_publisher_node,
        spawn_robot_node
    ])

