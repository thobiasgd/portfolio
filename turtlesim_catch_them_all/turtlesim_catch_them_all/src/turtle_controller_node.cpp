#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "my_robot_interfaces/msg/turtle_array.hpp"
#include "my_robot_interfaces/srv/catch_turtle.hpp"
#include "turtlesim/msg/pose.hpp"

#include <string>
#include <cmath>

using namespace std::placeholders;
using namespace std::chrono_literals;

class TurtleControllerNode : public rclcpp::Node{

    public:

        TurtleControllerNode(): Node("turtle_controller"){

            this->declare_parameter("linear_velocity", 2.0);
            linear_velocity_ = this->get_parameter("linear_velocity").as_double();
            this->declare_parameter("angular_velocity", 1.0);
            angular_velocity_ = this->get_parameter("angular_velocity").as_double();

            subscriber_ = this->create_subscription<my_robot_interfaces::msg::TurtleArray>(
                "alive_turtles", 10, std::bind(&TurtleControllerNode::GetClosestTurtle, this, _1));

            master_subscriber_ = this->create_subscription<turtlesim::msg::Pose>(
                "turtle1/pose", 10, std::bind(&TurtleControllerNode::CommandMaster, this, _1));

            publisher_ = this->create_publisher<geometry_msgs::msg::Twist>("turtle1/cmd_vel", 10);

            client_ = this->create_client<my_robot_interfaces::srv::CatchTurtle>("catch_turtle");

        }

    private:

        double linear_velocity_;
        double angular_velocity_;

        double master_x, master_y, master_delta_theta;
        double target_x, target_y, target_delta_theta;

        std::string closestTurtle;

        geometry_msgs::msg::Twist command_msg;

        rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr publisher_;
        rclcpp::Subscription<my_robot_interfaces::msg::TurtleArray>::SharedPtr subscriber_;
        rclcpp::Subscription<turtlesim::msg::Pose>::SharedPtr master_subscriber_;
        rclcpp::Client<my_robot_interfaces::srv::CatchTurtle>::SharedPtr client_;

        void GetClosestTurtle(const my_robot_interfaces::msg::TurtleArray::SharedPtr turtle_array){
            double aux = 10000;
            for (auto turtle : turtle_array->turtles){
                double absValue = std::sqrt(std::pow(turtle.x - master_x, 2) + std::pow(turtle.y - master_y, 2));
                if (absValue < aux){
                    aux = absValue;
                    closestTurtle = turtle.name;
                    target_x = turtle.x;
                    target_y = turtle.y;
                }
            }
            
            RCLCPP_INFO(this->get_logger(), closestTurtle.c_str());
        }

        void CommandMaster(const turtlesim::msg::Pose::SharedPtr master_pose){

            master_x = master_pose->x;  
            master_y = master_pose->y;

            double desired_theta = atan2(target_y - master_y, target_x - master_x);
            double diff_theta = desired_theta - master_pose->theta;
            double distance = std::sqrt(std::pow(target_x - master_x, 2) + std::pow(target_y - master_y, 2));

            // normaliza o ângulo
            if (diff_theta > M_PI) diff_theta -= 2 * M_PI;
            if (diff_theta < -M_PI) diff_theta += 2 * M_PI;

            command_msg = geometry_msgs::msg::Twist();
            command_msg.linear.x = std::min(distance, 1.0) * linear_velocity_;
            command_msg.angular.z = diff_theta * angular_velocity_; 

            if (target_x + target_y != 0) publisher_->publish(command_msg);

            if (distance <= 0.5){
                auto kill_msg = std::make_shared<my_robot_interfaces::srv::CatchTurtle::Request>();
                kill_msg->name = closestTurtle;
                client_->async_send_request(kill_msg);
            }
        }
};

int main(int argc, char **argv){
    rclcpp::init(argc, argv);
    auto node = std::make_shared<TurtleControllerNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
}


/* ros2 interface show geometry_msgs/msg/Twist
# This expresses velocity in free space broken into its linear and angular parts.

Vector3  linear
        float64 x
        float64 y
        float64 z
Vector3  angular
        float64 x
        float64 y
        float64 z 
        
---------------------------------------

ros2 interface show turtlesim/msg/Pose
float32 x
float32 y
float32 theta

float32 linear_velocity
float32 angular_velocity
        
*/

