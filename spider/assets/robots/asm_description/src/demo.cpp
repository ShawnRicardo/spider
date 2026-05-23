#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include <cmath>
#include <vector>
#include <string>

class JointStatePublisher : public rclcpp::Node
{
public:
    JointStatePublisher() : Node("joint_state_publisher")
    {
        joint_state_pub_ = this->create_publisher<sensor_msgs::msg::JointState>(
            "/joint_states", 10);
        
        // 创建定时器，10Hz（0.1秒）触发回调
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(100),
            std::bind(&JointStatePublisher::timer_callback, this));
        
        init_joint_names();
        
        joint_positions_.resize(joint_names_.size(), 0.0);
        joint_velocities_.resize(joint_names_.size(), 0.0);
        joint_efforts_.resize(joint_names_.size(), 0.0);
        
        RCLCPP_INFO(this->get_logger(), "Joint State Publisher 已启动");
    }

private:
    void timer_callback()
    {
        auto joint_state_msg = sensor_msgs::msg::JointState();
        joint_state_msg.header.stamp = this->now();
        joint_state_msg.header.frame_id = "";
        joint_state_msg.name = joint_names_;
        joint_state_msg.position = joint_positions_;
        joint_state_msg.velocity = joint_velocities_;
        joint_state_msg.effort = joint_efforts_;
        
        joint_state_pub_->publish(joint_state_msg);
        
        update_joint_positions();
    }

    void init_joint_names()
    {
        joint_names_ = {
            "Joint1_R", "Joint2_R", "Joint3_R", "Joint4_R", "Joint5_R", "Joint6_R", "Joint7_R",
            "right_finger1_joint1", "right_finger1_joint2", "right_finger1_joint3", "right_finger1_joint4",
            "right_finger2_joint1", "right_finger2_joint2", "right_finger2_joint3", "right_finger2_joint4",
            "right_finger3_joint1", "right_finger3_joint2", "right_finger3_joint3", "right_finger3_joint4",
            "right_finger4_joint1", "right_finger4_joint2", "right_finger4_joint3", "right_finger4_joint4",
            "right_finger5_joint1", "right_finger5_joint2", "right_finger5_joint3", "right_finger5_joint4",
            "Joint1_L", "Joint2_L", "Joint3_L", "Joint4_L", "Joint5_L", "Joint6_L", "Joint7_L",
            "left_finger1_joint1", "left_finger1_joint2", "left_finger1_joint3", "left_finger1_joint4",
            "left_finger2_joint1", "left_finger2_joint2", "left_finger2_joint3", "left_finger2_joint4",
            "left_finger3_joint1", "left_finger3_joint2", "left_finger3_joint3", "left_finger3_joint4",
            "left_finger4_joint1", "left_finger4_joint2", "left_finger4_joint3", "left_finger4_joint4",
            "left_finger5_joint1", "left_finger5_joint2", "left_finger5_joint3", "left_finger5_joint4"
        };
    }

    void update_joint_positions()
    {
        double time_sec = this->now().seconds();
        
        for (size_t i = 0; i < joint_positions_.size(); ++i) {
            joint_positions_[i] = 0.5 * std::sin(time_sec + i * 0.1);
        }
    }

    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_state_pub_;
    rclcpp::TimerBase::SharedPtr timer_;
    std::vector<std::string> joint_names_;
    std::vector<double> joint_positions_;
    std::vector<double> joint_velocities_;
    std::vector<double> joint_efforts_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<JointStatePublisher>());
    rclcpp::shutdown();
    return 0;
}
