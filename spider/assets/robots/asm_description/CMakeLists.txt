cmake_minimum_required(VERSION 3.8)
project(asm_description)

# 强制要求 C++17（ROS 2 推荐）
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# 查找 ROS 2 依赖包
find_package(ament_cmake REQUIRED)
find_package(urdf REQUIRED)
find_package(xacro REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)  # Python 依赖必须保留
find_package(gazebo_ros REQUIRED)
find_package(sensor_msgs REQUIRED)

add_executable(demo_cpp src/demo.cpp)
ament_target_dependencies(
  demo_cpp
  rclcpp
  sensor_msgs
)

# 安装C++可执行文件（ROS 2标准路径）
install(TARGETS
  demo_cpp
  DESTINATION lib/${PROJECT_NAME}
)

# ========== 关键修复：确保所有模型文件都被安装 ==========
# 1. 安装 launch 文件（保留）
install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
  PATTERN ".svn" EXCLUDE)

# 2. 安装 URDF 文件（保留）
install(DIRECTORY urdf/
  DESTINATION share/${PROJECT_NAME}/urdf
  PATTERN ".svn" EXCLUDE)

# 3. 安装 meshes 模型文件（修复：移除 FILES_MATCHING 限制）
install(DIRECTORY meshes/
  DESTINATION share/${PROJECT_NAME}/meshes
  PATTERN ".svn" EXCLUDE)

# 4. 安装 config 配置文件（保留）
install(DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
  PATTERN ".svn" EXCLUDE)

# ========== 新增：安装 Python 脚本并配置可运行 ==========
# 1. 安装 scripts 目录下的所有 Python 脚本
install(DIRECTORY scripts/
  DESTINATION share/${PROJECT_NAME}/scripts
  PATTERN ".svn" EXCLUDE
  # 确保 Python 文件有可执行权限
  PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
              GROUP_READ GROUP_EXECUTE
              WORLD_READ WORLD_EXECUTE)

# 2. 声明 demo.py 为 ROS 2 可执行文件（核心步骤）
install(PROGRAMS
  scripts/demo.py
  DESTINATION lib/${PROJECT_NAME}  # ROS 2 可执行文件默认安装路径
)

# 可选：打印安装路径（调试用）
message(STATUS "Meshes will be installed to: ${CMAKE_INSTALL_PREFIX}/share/${PROJECT_NAME}/meshes")
message(STATUS "Python scripts will be installed to: ${CMAKE_INSTALL_PREFIX}/lib/${PROJECT_NAME}")

# 声明包依赖（必须）
ament_package()
