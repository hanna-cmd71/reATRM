#include <cstddef>
#include <eigen3/Eigen/Dense>
#include <array>
#include <cmath>
#include <rclcpp/logger.hpp>
#include <rclcpp/logging.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rmcs_executor/component.hpp>
#include <string>
#include <string_view>
#include <unordered_set>
#include <tuple>
#include <vector>
#include "controller/pid/pid_calculator.hpp"

namespace rmcs_core::controller::arm {

class ArmSolver final
    : public rmcs_executor::Component
    , public rclcpp::Node {
    using TorqueVec = Eigen::Array<double, 6, 1>;
public:
        explicit ArmSolver()
        : Node(
                get_component_name(),
                rclcpp::NodeOptions{}.automatically_declare_parameters_from_overrides(true))
        , joint_angle_pid_controller{
                pid::PidCalculator(100.0, 0.0, 0.0),   
                pid::PidCalculator(270.0, 0.0, 0.0),   
                pid::PidCalculator(175.0, 0.0, 0.0),   
                pid::PidCalculator(50.0, 0.0, 0.0),    
                pid::PidCalculator(38.0, 0.0, 0.0),    
                pid::PidCalculator(10.0, 0.0, 0.0) }   
        , joint_vel_pid_controller{
                pid::PidCalculator(5.0, 0.0, 0.0),    
                pid::PidCalculator(3.0, 0.0, 0.0),     
                pid::PidCalculator(1.6, 0.0, 0.0),     
                pid::PidCalculator(2.0, 0.0, 0.0),     
                pid::PidCalculator(4.0, 0.0, 0.0),     
                pid::PidCalculator(0.5, 0.0, 0.0) }    
        {
            for(std::size_t i = 0; i < 6; ++i){
                const std::string joint_prefix = "/arm/joint_" + std::to_string(i+1);
                register_input(joint_prefix + "/theta", joint_theta[i]);
                register_input(joint_prefix + "/target_theta", joint_target_theta[i]);
                register_input(joint_prefix + "/lower_limit", joint_lower_limit_[i]);
                register_input(joint_prefix + "/upper_limit", joint_upper_limit_[i]);
                register_input(joint_prefix + "/velocity", joint_velocity_[i]);

                register_output(joint_prefix + "/motor/control_torque", target_torque_[i], NAN);
            }
            register_input("urdf_loaded", is_loaded);
            register_input("/arm/enable_flag", is_arm_enable);

            register_input("/arm/config/offsets_verified", offsets_verified_, false);

            last_target_theta_.fill(NAN);
            const auto list = this->get_parameter("controller_list").as_string_array();
            load_controller_list(list);
        }

        void update() override {
            //todo
            TorqueVec torque_cmd;
            torque_cmd.setZero();
            if(*is_arm_enable){
                if(!*is_loaded){
                    RCLCPP_WARN_THROTTLE(rclcpp::get_logger("[Arm_Solver]"), *this->get_clock(), 2000,"URDF not loaded! Arm torque output blocked.");
                    torque_cmd = zero_calculate();
                } else {
                    const bool verified = offsets_verified_.ready() && *offsets_verified_;
                    if (!verified) {
                        RCLCPP_WARN_THROTTLE(rclcpp::get_logger("Arm_Solver"), *this->get_clock(), 2000,"Arm offsets not verified! Torque output blocked. ""Calibrate then set offsets_verified=true.");
                        torque_cmd = zero_calculate();
                    } else {
                        if(!last_is_arm_enable){
                            for(auto& pid: joint_angle_pid_controller){
                                pid.reset();
                            }
                            for(auto& pid: joint_vel_pid_controller){
                                pid.reset();
                            }
                        }
                        for(auto fn : controller_list_){
                            torque_cmd += (this->*fn)();
                        }
                    }
                }
            } else {
                torque_cmd = zero_calculate();
            }
            last_is_arm_enable = *is_arm_enable;
            for(std::size_t i = 0; i < 6;++i ){
                *target_torque_[i] = torque_cmd[i];
            }
        }
private:
    using controller_type = TorqueVec (ArmSolver::*)();
    std::vector<controller_type> controller_list_;
    static double normalize_angle(double angle) {
        return std::remainder(angle, 2.0 * M_PI);
    }
    TorqueVec pid_calculate(){
        auto clamp_angle = [this](std::size_t idx, double target_theta) {
            const double lower_limit = *joint_lower_limit_[idx];
            const double upper_limit = *joint_upper_limit_[idx];
            if (target_theta < lower_limit) {
                return lower_limit;
            } else if (target_theta > upper_limit) {
                return upper_limit;
            }
            return target_theta;
        };
        TorqueVec torque_pid;
        torque_pid.setZero();
        for(std::size_t i = 0; i < 6; ++i){
            const double current_theta = *joint_theta[i];
            const double target_theta = clamp_angle(i, *joint_target_theta[i]);
            const double current_vel = *joint_velocity_[i];

            const double angle_error = normalize_angle(target_theta - current_theta);
            const double target_vel = joint_angle_pid_controller[i].update(angle_error);
            const double vel_error = target_vel - current_vel;
            torque_pid[i] = joint_vel_pid_controller[i].update(vel_error);
        }
        return torque_pid;
    }
    TorqueVec gravity_calculate(){
        //   G1 大臂     Th1 = q2
        //   G2 小臂     Th2 = q2 + q3
        //   G3 腕+末端  Th3 = q2 + q3 - q5   
        //   一阶矩 P*cos(Th) + Q*sin(Th)
        const double theta1 = *joint_theta[1];
        const double theta2 = theta1 + *joint_theta[2];
        const double theta3 = theta2 - *joint_theta[4];

        const double c1 = std::cos(theta1);
        const double s1 = std::sin(theta1);
        const double c2 = std::cos(theta2);
        const double s2 = std::sin(theta2);
        const double c3 = std::cos(theta3);
        const double s3 = std::sin(theta3);

        TorqueVec torque_gravity;
        torque_gravity.setZero();
        torque_gravity(1) = kGravityP1 * c1 + kGravityQ1 * s1 + kGravityP2 * c2 + kGravityQ2 * s2
                          + kGravityP3 * c3 + kGravityQ3 * s3 + kGravityE2;
        torque_gravity(2) = kGravityP2 * c2 + kGravityQ2 * s2 + kGravityP3 * c3 + kGravityQ3 * s3
                          + kGravityE3;
        torque_gravity(4) = kGravityP3 * c3 + kGravityQ3 * s3 + kGravityE5;
        return torque_gravity;
    }
    TorqueVec friction_calculate(){
        TorqueVec torque_friction;
        torque_friction.setZero();
        for (std::size_t i = 0; i < 6; ++i) {
            const double target = *joint_target_theta[i];
            const double target_velocity =
                (std::isfinite(target) && std::isfinite(last_target_theta_[i]))
                    ? (target - last_target_theta_[i]) * kUpdateRateHz
                    : 0.0;
            last_target_theta_[i] = target;

            if (kFrictionTorque[i] <= 0.0)
                continue;
            torque_friction(i) =
                kFrictionTorque[i] * std::tanh(target_velocity / kFrictionVelocity[i]);
        }
        return torque_friction;
    }
    TorqueVec zero_calculate(){
        TorqueVec torque_zero;
        torque_zero.fill(NAN);
        return torque_zero;
    }
    static constexpr std::array<std::tuple<std::string_view, controller_type>, 4> term_table_{
        {{"gravity", &ArmSolver::gravity_calculate},
         {"pid", &ArmSolver::pid_calculate},
         {"friction", &ArmSolver::friction_calculate},
         {"zero_torque", &ArmSolver::zero_calculate}}
    };

    void load_controller_list(const std::vector<std::string>& list){
        //todo Queue->gravity + friction + control_torque
        controller_list_.clear();
        controller_list_.reserve(list.size());
        std::unordered_set<std::string_view> record;
        record.reserve(list.size());
        for(const auto& name: list){
            const std::string_view name_view(name);
            if(!record.insert(name_view).second){
                RCLCPP_WARN(rclcpp::get_logger("ArmSolver"), "Duplicate controller name: %s", name.c_str());
                continue;
            }
            bool find = false;
            for(const auto& [term_name, fn]: term_table_){
                if(name_view == term_name){
                    controller_list_.push_back(fn);
                    find = true;
                    break;
                }
            }
            if(!find){
                RCLCPP_ERROR(rclcpp::get_logger("ArmSolver"), "Unknown controller name: %s", name.c_str());
            }
        }
    }
    std::array<pid::PidCalculator, 6> joint_angle_pid_controller;
    std::array<pid::PidCalculator, 6> joint_vel_pid_controller;
    //重力模型系数(静态辨识结果,单位 N*m)
    static constexpr double kGravityP1 = 20.284;
    static constexpr double kGravityQ1 = -60.824;
    static constexpr double kGravityP2 = -15.112;
    static constexpr double kGravityQ2 = 2.680;
    static constexpr double kGravityP3 = -0.579;
    static constexpr double kGravityQ3 = 0.380;
    static constexpr double kGravityE2 = -22.244;
    static constexpr double kGravityE3 = 1.534;
    static constexpr double kGravityE5 = 2.601;
    static constexpr std::array<double, 6> kFrictionTorque = {0.0, 11.6, 7.4, 0.0, 0.75, 0.0};
    static constexpr std::array<double, 6> kFrictionVelocity = {0.2, 0.2, 0.2, 0.2, 0.2, 0.2};

    std::array<double, 6> last_target_theta_{};  
    static constexpr double kUpdateRateHz = 1000.0;  

    std::array<InputInterface<double>, 6>  joint_theta;
    std::array<InputInterface<double>, 6>  joint_target_theta;
    std::array<InputInterface<double>, 6>  joint_lower_limit_;
    std::array<InputInterface<double>, 6>  joint_upper_limit_;
    std::array<InputInterface<double>, 6>  joint_velocity_;
    std::array<OutputInterface<double>, 6>  target_torque_;
    InputInterface<bool> is_loaded;
    InputInterface<bool> is_arm_enable;
    InputInterface<bool> offsets_verified_;      
    bool last_is_arm_enable{false};
    };
} // namespace rmcs_core::controller::arm

#include <pluginlib/class_list_macros.hpp>
PLUGINLIB_EXPORT_CLASS(rmcs_core::controller::arm::ArmSolver, rmcs_executor::Component)
