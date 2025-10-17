#include "rclcpp/rclcpp.hpp"
#include "turtlesim/srv/spawn.hpp"
#include "turtlesim/srv/kill.hpp"
#include "my_robot_interfaces/msg/turtle_array.hpp"
#include "my_robot_interfaces/msg/turtle.hpp"
#include "my_robot_interfaces/srv/catch_turtle.hpp"
#include <random>

using namespace std::placeholders;

class TurtleSpawnerNode : public rclcpp::Node{
    public:
        TurtleSpawnerNode() : Node("turtle_spawner"), rd_(), gen_(rd_()), dis_pos_(1.0, 10.0), dis_ang_(0.0, 360.0){
            
            this->declare_parameter("spawn_time", 2.0);
            spawn_time_ = this->get_parameter("spawn_time").as_double();

            counter_turtle = 2;
            publisher_ = this->create_publisher<my_robot_interfaces::msg::TurtleArray>("alive_turtles", 10);
            client_ = this->create_client<turtlesim::srv::Spawn>("spawn");
            kill_client_ = this->create_client<turtlesim::srv::Kill>("kill");
            server_ = this->create_service<my_robot_interfaces::srv::CatchTurtle>(
                "catch_turtle", std::bind(&TurtleSpawnerNode::CallbackCaughtTurtle, this, _1, _2));
            timer_ = this->create_wall_timer(std::chrono::duration<double>(spawn_time_), 
                std::bind(&TurtleSpawnerNode::SendSpawnTurtleRequest, this));
            RCLCPP_INFO(this->get_logger(), "Initializing Turtle Spawner...");
        }

    private:
        rclcpp::Client<turtlesim::srv::Spawn>::SharedPtr client_;
        rclcpp::Client<turtlesim::srv::Kill>::SharedPtr kill_client_;
        rclcpp::TimerBase::SharedPtr timer_;
        rclcpp::Publisher<my_robot_interfaces::msg::TurtleArray>::SharedPtr publisher_;
        rclcpp::Service<my_robot_interfaces::srv::CatchTurtle>::SharedPtr server_;
        double spawn_time_;
        std::vector<my_robot_interfaces::msg::Turtle> alive_turtles_;
        int counter_turtle;

        std::random_device rd_;
        std::mt19937 gen_;
        std::uniform_real_distribution<double> dis_pos_;
        std::uniform_real_distribution<double> dis_ang_;

        void CallbackCaughtTurtle(const my_robot_interfaces::srv::CatchTurtle::Request::SharedPtr request,
                                  const my_robot_interfaces::srv::CatchTurtle::Response::SharedPtr response){
            response->success = true;     
            auto kill_request = std::make_shared<turtlesim::srv::Kill::Request>();
            kill_request->name = request->name;

            kill_client_->async_send_request(kill_request);

            for (size_t  i = 0; i < alive_turtles_.size(); i++){
                if (alive_turtles_[i].name == request->name){
                    alive_turtles_.erase(alive_turtles_.begin() + i);
                    break;
                }
            }

            my_robot_interfaces::msg::TurtleArray turtlesAlive;
            turtlesAlive.turtles = alive_turtles_;
            publisher_->publish(turtlesAlive);  
        }

        void SendSpawnTurtleRequest(){

            if (!client_->wait_for_service(std::chrono::duration<double>(spawn_time_))) {
                RCLCPP_WARN(this->get_logger(), "Spawn service not available yet");
                return;
            }

            double x = dis_pos_(gen_);
            double y = dis_pos_(gen_);
            double theta = dis_ang_(gen_) * M_PI / 180.0; // Convertendo para radianos
            
            auto request = std::make_shared<turtlesim::srv::Spawn::Request>();
            request->x = x;
            request->y = y;
            request->theta = theta;

            auto turtle = my_robot_interfaces::msg::Turtle();
            my_robot_interfaces::msg::TurtleArray turtle_array;
            turtle.name = "turtle"+std::to_string(counter_turtle);
            turtle.x = x;
            turtle.y = y;

            alive_turtles_.push_back(turtle);
            turtle_array.turtles = alive_turtles_;

            publisher_->publish(turtle_array);

            client_->async_send_request(
                request, std::bind(&TurtleSpawnerNode::CallbackResponse, this, _1)
            );
            counter_turtle++;
        }

        void CallbackResponse(rclcpp::Client<turtlesim::srv::Spawn>::SharedFuture future){
            auto response = future.get();
            RCLCPP_INFO(this->get_logger(), "Reponse Received...");
        }
};

int main(int argc, char **argv){
    rclcpp::init(argc, argv);
    auto node = std::make_shared<TurtleSpawnerNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
