#include <array>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/src/Core/util/Meta.h>
#include <limits>
#include <memory>
#include <rclcpp/logger.hpp>
#include <rclcpp/logging.hpp>
#include <rclcpp/node.hpp>
#include <rclcpp/publisher.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <std_msgs/msg/string.hpp>
#include <moveit/kinematic_constraints/utils.hpp>
#include <moveit/move_group_interface/move_group_interface.hpp>
#include <moveit/planning_interface/planning_interface.hpp>
#include <moveit/planning_scene_interface/planning_scene_interface.hpp>
#include <moveit/robot_state/robot_state.hpp>
#include <moveit_msgs/msg/move_it_error_codes.hpp>
#include <string>
#include <vector>
#include <rmcs_executor/component.hpp>
#include "utility/low_pass_filter.hpp"
#include <rmcs_msgs/arm_mode.hpp>
#include <rmcs_msgs/vtswitch.hpp>
#include "controller/arm/Action_planner/action_step.hpp"
#include "controller/arm/Action_planner/arm_action_machine.hpp"
#include "controller/arm/Action_planner/action_loader.hpp"

namespace rmcs_core::controller::arm {

class ArmController final
    : public rmcs_executor::Component
    , public rclcpp::Node {    
public:
        ArmController()
        : Node(
                get_component_name(),
                rclcpp::NodeOptions{}.automatically_declare_parameters_from_overrides(true)) 
        , custom_joint_filter_(0.2){
            register_input("/vt03/mode_switch", mode_switch_);
            register_input("/vt03/fn_1", fn_1_);
            register_input("/vt03/fn_2", fn_2_);
            register_input("/vt03/rotary_knob", rotary_knob_);
            register_input("/vt03/joystick/right", joystick_right_);
            register_input("/vt03/joystick/left", joystick_left_);
            register_output("/arm/mode", arm_mode_);
            for(std::size_t i = 0; i<6; ++i){
                const std::string joint_prefix = "/arm/joint_" + std::to_string(i + 1);
                register_input(joint_prefix + "/theta", theta[i]);
                register_input(joint_prefix + "/lower_limit", joint_lower_limit_[i]);
                register_input(joint_prefix + "/upper_limit", joint_upper_limit_[i]);
                register_output(joint_prefix + "/target_theta", target_theta[i], NAN);
            }
            register_output("/arm/enable_flag", is_arm_enable_, false);

            action_status_pub_ = this->create_publisher<std_msgs::msg::String>(
                "/arm/action/status", rclcpp::QoS{1});
            action_trigger_sub_ = this->create_subscription<std_msgs::msg::String>(
                "/arm/action/trigger", rclcpp::QoS{1},
                [this](const std_msgs::msg::String::ConstSharedPtr msg) {
                    start_action(msg->data);
                });

            tuning_target_sub_ = this->create_subscription<std_msgs::msg::Float64MultiArray>(
                "/arm/tuning/joint_target", rclcpp::QoS{1},
                [this](const std_msgs::msg::Float64MultiArray::ConstSharedPtr msg) {
                    if (msg->data.size() != 6) {
                        RCLCPP_WARN(
                            get_logger(), "Tuning: joint_target needs 6 values, got %zu",
                            msg->data.size());
                        return;
                    }
                    for (std::size_t i = 0; i < 6; ++i)
                        tuning_target_[i] = msg->data[i];
                });

            if (!this->has_parameter("actions_file"))
                this->declare_parameter("actions_file", std::string{});
            const auto actions_file = this->get_parameter("actions_file").as_string();
            if (actions_file.empty()) {
                RCLCPP_WARN(get_logger(), "Action: actions_file is not set, no action available");
            } else {
                try {
                    actions_ = ActionLoader::load(actions_file);
                    RCLCPP_INFO(
                        get_logger(), "Action: loaded %zu action(s) from %s", actions_.size(),
                        actions_file.c_str());
                    for (const auto& [name, steps] : actions_)
                        RCLCPP_INFO(get_logger(), "  - %s (%zu steps)", name.c_str(), steps.size());
                } catch (const std::exception& error) {
                    RCLCPP_ERROR(get_logger(), "Action: %s", error.what());
                }
            }
        }
        ~ArmController() override = default;

        void update() override{
            using namespace rmcs_msgs;
            auto sw = *mode_switch_;
            static bool initial_check_done{false};

            if (!initial_check_done) {
                *is_arm_enable_ = false;
                if (sw != VtSwitch::LEFT && sw != VtSwitch::UNKNOWN) {
                    RCLCPP_INFO(rclcpp::get_logger("arm_controller"),"initial!");
                    initial_check_done = true;
                }
                reset();
                return;
            }
            if ((sw == VtSwitch::LEFT || sw == VtSwitch::UNKNOWN)) {
                *is_arm_enable_ = false;
                *arm_mode_ = ArmMode::None;
                reset();
                last_arm_mode_ = ArmMode::None;
                return;
            }
            *is_arm_enable_ = true;
            if(*fn_1_ && !last_fn_1_){
                fn1_toggle_ = !fn1_toggle_;
            }
            if(*fn_2_ && !last_fn_2_){
                fn2_toggle_ = !fn2_toggle_;
            }
            mode_selection();
            if (last_arm_mode_ != *arm_mode_) {
                switch (*arm_mode_) {
                    case ArmMode::Custome: {
                        tuning_target_.fill(std::numeric_limits<double>::quiet_NaN());
                        for (std::size_t i = 0; i < 6; ++i) {
                            if (theta[i].ready() && !std::isnan(*theta[i])) {
                                *target_theta[i] = *theta[i];
                            } else if (std::isnan(*target_theta[i])) {
                                *target_theta[i] = 0.0;
                            }
                        }
                        break;
                    }
                    case ArmMode::execute_vt03_position:
                        for (std::size_t i = 0; i < 6; ++i) {
                            if (theta[i].ready() && !std::isnan(*theta[i])) {
                                *target_theta[i] = *theta[i];
                            } else if (std::isnan(*target_theta[i])) {
                            *target_theta[i] = 0.0;
                            }
                        }
                        break;
                    case ArmMode::execute_vt03_orientation:
                        for (std::size_t i = 0; i < 6; ++i) {
                            if (theta[i].ready() && !std::isnan(*theta[i])) {
                                *target_theta[i] = *theta[i];
                            } else if (std::isnan(*target_theta[i])) {
                            *target_theta[i] = 0.0;
                            }
                        }
                        break;
                    case ArmMode::Gripper:
                        break;
                    case ArmMode::None:
                        reset();
                        break;
                    default:
                        break;
                }
            }
            action_update();
            if (action_phase_ == ActionPhase::Idle)
                arm_control();
            last_fn_1_ = *fn_1_;
            last_fn_2_ = *fn_2_;
            last_arm_mode_ = *arm_mode_;
        }

private:
    void mode_selection() {
        auto sw = *mode_switch_;
        using namespace rmcs_msgs;
        if (sw == VtSwitch::MIDDLE && fn1_toggle_ == false) {        
                *arm_mode_ = ArmMode::execute_vt03_position;
        }else if(sw == VtSwitch::MIDDLE && fn1_toggle_ == true) {
                *arm_mode_ = ArmMode::execute_vt03_orientation;
        }else if(sw == VtSwitch::RIGHT && fn2_toggle_ == false) {
                *arm_mode_ = ArmMode::Gripper;
        }else if(sw == VtSwitch::RIGHT && fn2_toggle_ == true) {
                *arm_mode_ = ArmMode::Custome;
        }else{
            *arm_mode_ = ArmMode::None;
        }
    }
    void arm_control(){
        switch(*arm_mode_){
            using namespace rmcs_msgs;
            case ArmMode::Custome: {
                execute_custome();
                break;
            }
            case ArmMode::execute_vt03_position:{
                Vt03_Position_Control();
                break;
            }
            case ArmMode::execute_vt03_orientation:{
                Vt03_Orientation_Control();
                break;
            }
            case ArmMode::Gripper: {
                Gripper_Control();
                break;
        }
            default: {
                break;
            }
        }
    };
    void Gripper_Control(){
        // RCLCPP_INFO(rclcpp::get_logger("ArmController"),"Gripper Control Mode Active");
    }
    void execute_custome(){
        RCLCPP_INFO_THROTTLE(
            get_logger(), *this->get_clock(), 2000,
            "Tuning: Custome mode active, target comes from /arm/tuning/joint_target");
        for (std::size_t i = 0; i < 6; ++i) {
            if (std::isnan(tuning_target_[i]))
                continue;
            const double lower = joint_lower_limit_[i].ready() && !std::isnan(*joint_lower_limit_[i])
                                 ? *joint_lower_limit_[i] : -M_PI;
            const double upper = joint_upper_limit_[i].ready() && !std::isnan(*joint_upper_limit_[i])
                                 ? *joint_upper_limit_[i] : M_PI;
            *target_theta[i] = std::clamp(tuning_target_[i], lower, upper);
        }
    } 
    void Vt03_Position_Control(){
        constexpr double DEADZONE = 0.05; 
        constexpr double STEP = 0.003;
        for (std::size_t i = 0; i < 6; ++i) {
            if (std::isnan(*target_theta[i])) {
            *target_theta[i] = (theta[i].ready() && !std::isnan(*theta[i])) ? *theta[i] : 0.0;
            }
        }
        auto safe_clamp = [this](std::size_t idx, double target) {
            double lower = (joint_lower_limit_[idx].ready() && !std::isnan(*joint_lower_limit_[idx]))
                           ? *joint_lower_limit_[idx] : -M_PI;
            double upper = (joint_upper_limit_[idx].ready() && !std::isnan(*joint_upper_limit_[idx]))
                           ? *joint_upper_limit_[idx] : M_PI;
            return std::clamp(target, lower, upper);
        };
        if (std::fabs(joystick_left_->x()) > DEADZONE) {
            double next_target = *target_theta[2] - STEP * joystick_left_->x();
            *target_theta[2]   = safe_clamp(2, next_target);
        }
        if (std::fabs(joystick_right_->x()) > DEADZONE) {
            double next_target = *target_theta[1] + STEP * joystick_right_->x();
            *target_theta[1]   = safe_clamp(1, next_target);
        }
        if (std::fabs(joystick_left_->y()) > DEADZONE) {
            double next_target = *target_theta[0] + STEP * joystick_left_->y();
            *target_theta[0]   = safe_clamp(0, next_target);
        }
    }
    void Vt03_Orientation_Control(){
        // RCLCPP_INFO(rclcpp::get_logger("ArmController"),"Orientation Control Mode Active");
        constexpr double DEADZONE = 0.05; 
        constexpr double STEP = 0.003;
        for (std::size_t i = 0; i < 6; ++i) {
            if (std::isnan(*target_theta[i])) {
            *target_theta[i] = (theta[i].ready() && !std::isnan(*theta[i])) ? *theta[i] : 0.0;
            }
        }
        auto safe_clamp = [this](std::size_t idx, double target) {
            double lower = (joint_lower_limit_[idx].ready() && !std::isnan(*joint_lower_limit_[idx]))
                           ? *joint_lower_limit_[idx] : -M_PI;
            double upper = (joint_upper_limit_[idx].ready() && !std::isnan(*joint_upper_limit_[idx]))
                           ? *joint_upper_limit_[idx] : M_PI;
            return std::clamp(target, lower, upper);
        };
        if (std::fabs(joystick_left_->x()) > DEADZONE) {
            double next_target = *target_theta[5] - STEP * joystick_left_->x();
            *target_theta[5]   = safe_clamp(5, next_target);
        }
        if (std::fabs(joystick_right_->x()) > DEADZONE) {
            double next_target = *target_theta[4] + STEP * joystick_right_->x();
            *target_theta[4]   = safe_clamp(4, next_target);
        }
        if (std::fabs(joystick_left_->y()) > DEADZONE) {
            double next_target = *target_theta[3] - STEP * joystick_left_->y();
            *target_theta[3]   = safe_clamp(3, next_target);
        }
    }
    void reset() {
        Eigen::Matrix<double, 6, 1> filter_init_vec;
        for (std::size_t i = 0; i < 6; ++i) {
            *target_theta[i] = theta[i].ready() ? *theta[i] : 0.0;
        }
        custom_joint_filter_.reset();
    }


    void start_action(const std::string& name) {
        const auto action = actions_.find(name);
        if (action == actions_.end()) {
            RCLCPP_WARN(get_logger(), "Action: unknown name '%s'", name.c_str());
            return;
        }
        action_name_ = name;
        action_machine_.process(action->second);
        action_phase_ = ActionPhase::Planning;
        RCLCPP_INFO(
            get_logger(), "Action: '%s' requested (%zu steps)", name.c_str(),
            action->second.size());
    }

    void action_update() {
        switch (action_phase_) {
        case ActionPhase::Idle:
            break;

        case ActionPhase::Planning: {
            const auto trajectory = action_machine_.get_trajectory();
            if (!trajectory || trajectory->request_id == last_handled_request_id_)
                break;
            last_handled_request_id_ = trajectory->request_id;
            if (!trajectory->plan_success) {
                RCLCPP_ERROR(
                    get_logger(), "Action: '%s' planning failed - %s", action_name_.c_str(),
                    trajectory->error_message.c_str());
                action_phase_ = ActionPhase::Error;
                break;
            }
            // 规划结果不保证关节顺序,按名字建立映射
            joint_index_map_.fill(-1);
            for (std::size_t i = 0; i < 6; ++i) {
                const std::string wanted = "joint_" + std::to_string(i + 1);
                for (std::size_t j = 0; j < trajectory->joint_names.size(); ++j) {
                    if (trajectory->joint_names[j] == wanted)
                        joint_index_map_[i] = static_cast<int>(j);
                }
                if (joint_index_map_[i] < 0) {
                    RCLCPP_ERROR(get_logger(), "Action: '%s' missing in plan", wanted.c_str());
                    action_phase_ = ActionPhase::Error;
                    break;
                }
            }
            if (action_phase_ == ActionPhase::Error)
                break;
            trajectory_   = trajectory;
            point_index_  = 0;
            action_start_ = std::chrono::steady_clock::now();
            action_phase_ = ActionPhase::Executing;
            RCLCPP_INFO(
                get_logger(), "Action: '%s' planned - %zu points, %.2fs", action_name_.c_str(),
                trajectory->points.size(), trajectory->duration);
            break;
        }

        case ActionPhase::Executing: {
            // 按规划给出的时间戳推进,而不是按调用周期推点
            const double elapsed = std::chrono::duration<double>(
                                       std::chrono::steady_clock::now() - action_start_)
                                       .count();
            const auto& points = trajectory_->points;
            while (point_index_ + 1 < points.size()
                   && rclcpp::Duration(points[point_index_ + 1].time_from_start).seconds()
                          <= elapsed)
                ++point_index_;

            // 相邻轨迹点之间线性插值
            const auto& point = points[point_index_];
            const bool has_next = point_index_ + 1 < points.size();
            const auto& next    = has_next ? points[point_index_ + 1] : point;
            double ratio        = 0.0;
            if (has_next) {
                const double t0 =
                    rclcpp::Duration(point.time_from_start).seconds();
                const double t1 =
                    rclcpp::Duration(next.time_from_start).seconds();
                if (t1 > t0)
                    ratio = std::clamp((elapsed - t0) / (t1 - t0), 0.0, 1.0);
            }
            for (std::size_t i = 0; i < 6; ++i) {
                const auto source = static_cast<std::size_t>(joint_index_map_[i]);
                if (source >= point.positions.size() || source >= next.positions.size())
                    continue;
                const double position = point.positions[source];
                const double position_next = next.positions[source];
                *target_theta[i] = position + ratio * (position_next - position);
            }
            if (elapsed >= trajectory_->duration) {
                action_phase_ = ActionPhase::Done;
                RCLCPP_INFO(get_logger(), "Action: '%s' finished in %.2fs", action_name_.c_str(),
                            elapsed);
            }
            break;
        }

        case ActionPhase::Done: {
            if (trajectory_ && !trajectory_->points.empty())
                point_index_ = trajectory_->points.size() - 1;
            report_goal_error();
            report_action_status();
            action_phase_ = ActionPhase::Idle;
            return;
        }

        case ActionPhase::Error:
            report_action_status();
            action_phase_ = ActionPhase::Idle;
            return;
        }
        report_action_status();
    }

    void report_goal_error() const {
        if (!trajectory_ || trajectory_->points.empty())
            return;
        const auto& goal = trajectory_->points.back();
        double max_error = 0.0;
        for (std::size_t i = 0; i < 6; ++i) {
            const auto source = static_cast<std::size_t>(joint_index_map_[i]);
            if (source >= goal.positions.size())
                continue;
            const double target = goal.positions[source];
            const double actual =
                theta[i].ready() ? *theta[i] : std::numeric_limits<double>::quiet_NaN();
            const double error = target - actual;
            if (std::isfinite(error))
                max_error = std::max(max_error, std::fabs(error));
            RCLCPP_INFO(
                get_logger(), "Action: joint_%zu target=%+.4f actual=%+.4f error=%+.4f rad", i + 1,
                target, actual, error);
        }
        RCLCPP_INFO(get_logger(), "Action: max error = %.4f rad", max_error);
    }

    void report_action_status() {
        static const char* const phase_names[] = {"idle", "planning", "executing", "done", "error"};
        const auto now = this->now();
        const bool phase_changed = action_phase_ != last_reported_phase_;
        // 阶段变化立即发;同一阶段内部限流到 10Hz
        if (!phase_changed && last_status_time_.nanoseconds() != 0
            && (now - last_status_time_).seconds() < 0.1)
            return;
        last_status_time_    = now;
        last_reported_phase_ = action_phase_;

        std_msgs::msg::String msg;
        msg.data = "action=" + action_name_ + " phase="
                   + phase_names[static_cast<std::size_t>(action_phase_)];
        if (trajectory_ && action_phase_ == ActionPhase::Executing)
            msg.data += " point=" + std::to_string(point_index_ + 1) + "/"
                        + std::to_string(trajectory_->points.size());
        action_status_pub_->publish(msg);
    }

    enum class ActionPhase : std::uint8_t { Idle, Planning, Executing, Done, Error };
    ActionMachine action_machine_;
    ActionLoader::ActionMap actions_;
    ActionPhase action_phase_{ActionPhase::Idle};
    std::string action_name_;
    uint64_t last_handled_request_id_{0};
    std::shared_ptr<const ActionMachine::PlannedTrajectory> trajectory_;
    std::size_t point_index_{0};
    std::array<int, 6> joint_index_map_{{0, 1, 2, 3, 4, 5}};
    std::chrono::steady_clock::time_point action_start_{};
    rclcpp::Time last_status_time_{0, 0, RCL_ROS_TIME};
    ActionPhase last_reported_phase_{ActionPhase::Idle};
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr action_trigger_sub_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr action_status_pub_;
    rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr tuning_target_sub_;
    std::array<double, 6> tuning_target_{
        {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN(),
         std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN(),
         std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN()}};

    rmcs_msgs::ArmMode last_arm_mode_{rmcs_msgs::ArmMode::None};
    bool last_fn_1_{false};
    bool last_fn_2_{false};
    bool fn1_toggle_{false};
    bool fn2_toggle_{false};
    InputInterface<bool> fn_1_;
    InputInterface<bool> fn_2_;
    InputInterface<double> rotary_knob_;
    InputInterface<rmcs_msgs::VtSwitch> mode_switch_;
    InputInterface<Eigen::Vector2d> joystick_right_;
    InputInterface<Eigen::Vector2d> joystick_left_;
    OutputInterface<rmcs_msgs::ArmMode> arm_mode_;
    OutputInterface<bool> is_arm_enable_;
    std::array<InputInterface<double>,6> joint_lower_limit_;
    std::array<InputInterface<double>,6> joint_upper_limit_;
    InputInterface<double> theta[6];
    OutputInterface<double> target_theta[6];
    utility::LowPassFilter<6> custom_joint_filter_;
    };
} // namespace rmcs_core::controller::arm

#include <pluginlib/class_list_macros.hpp>
PLUGINLIB_EXPORT_CLASS(rmcs_core::controller::arm::ArmController, rmcs_executor::Component)
