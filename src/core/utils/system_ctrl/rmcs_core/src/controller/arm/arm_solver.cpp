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
                register_input(joint_prefix + "/target_velocity", joint_target_velocity_[i]);
                register_input(joint_prefix + "/lower_limit", joint_lower_limit_[i]);
                register_input(joint_prefix + "/upper_limit", joint_upper_limit_[i]);
                register_input(joint_prefix + "/velocity", joint_velocity_[i]);

                register_output(joint_prefix + "/motor/control_torque", target_torque_[i], NAN);
            }
            register_input("urdf_loaded", is_loaded);
            register_input("/arm/enable_flag", is_arm_enable);

            register_input("/arm/config/offsets_verified", offsets_verified_, false);
            register_input("/arm/drag_enable", drag_enable_, false);

            register_input("/gripper/motor/angle", gripper_angle_);
            register_input("/gripper/motor/velocity", gripper_velocity_);
            register_input("/gripper/target_theta", gripper_target_);
            register_output("/gripper/motor/control_torque", gripper_torque_, NAN);

            last_target_theta_.fill(NAN);
            const auto list = this->get_parameter("controller_list").as_string_array();
            load_controller_list(list);
        }

        void update() override {
            //todo
            TorqueVec torque_cmd;
            torque_cmd.setZero();
            bool arm_ready = false;
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
                            angle_integral_.fill(0.0);
                        }
                        for(auto fn : controller_list_){
                            if (*drag_enable_ && fn == &ArmSolver::friction_calculate)
                                continue;
                            torque_cmd += (this->*fn)();
                        }
                        if (*drag_enable_) {
                            for (std::size_t i = 0; i < 6; ++i) {
                                drag_velocity_[i] +=
                                    (*joint_velocity_[i] - drag_velocity_[i]) * kDragVelocityLpf;
                                torque_cmd(i) -= kDragDamping[i] * drag_velocity_[i];
                            }
                        }
                        arm_ready = true;
                    }
                }
            } else {
                torque_cmd = zero_calculate();
            }
            last_is_arm_enable = *is_arm_enable;
            for(std::size_t i = 0; i < 6;++i ){
                *target_torque_[i] = torque_cmd[i];
            }
            *gripper_torque_ = arm_ready ? gripper_calculate() : NAN;
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
            // Velocity feedforward: the outer loop only corrects the residual. Without
            // it J5 trails J3 and the tool path bends.
            const double feed_forward =
                joint_target_velocity_[i].ready() && std::isfinite(*joint_target_velocity_[i])
                    ? *joint_target_velocity_[i]
                    : 0.0;
            const double target_vel =
                feed_forward + joint_angle_pid_controller[i].update(angle_error);
            const double vel_error = target_vel - current_vel;
            // Integral of the position error. Only helps a truly constant offset; here
            // the error is friction hysteresis, so it stays disabled.
            if (kAngleIntegralGain[i] > 0.0 && std::isfinite(angle_error)) {
                angle_integral_[i] = std::clamp(
                    angle_integral_[i] + kAngleIntegralGain[i] * angle_error / kUpdateRateHz,
                    -kAngleIntegralLimit[i], kAngleIntegralLimit[i]);
            } else {
                angle_integral_[i] = 0.0;
            }
            torque_pid[i] =
                joint_vel_pid_controller[i].update(vel_error) + angle_integral_[i];
        }
        return torque_pid;
    }
    TorqueVec gravity_calculate(){
        //   G1 upper arm  Th1 = q2
        //   G2 forearm    Th2 = q2 + q3
        //   G3 wrist+tool Th3 = q2 + q3 - q5
        //   first moment P*cos(Th) + Q*sin(Th)
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
        const double phi5 = kGravityJ5ConfigWeight * theta2 - *joint_theta[4];
        torque_gravity(4) =
            kGravityP5 * std::cos(phi5) + kGravityQ5 * std::sin(phi5) + kGravityE5;
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

            // Breakaway compensation. The velocity term above vanishes at rest, leaving
            // static friction and model error to the position loop (J3: 0.032 rad, about
            // 15mm at the tool). Driven by the position error and survives at zero
            // velocity, pulling the joint out of the dead band.
            const double current = *joint_theta[i];
            if (kBreakawayTorque[i] > 0.0 && std::isfinite(target) && std::isfinite(current)) {
                const double angle_error = normalize_angle(target - current);
                torque_friction(i) +=
                    kBreakawayTorque[i] * std::tanh(angle_error / kBreakawayError);
            }
        }
        return torque_friction;
    }
    TorqueVec zero_calculate(){
        TorqueVec torque_zero;
        torque_zero.fill(NAN);
        return torque_zero;
    }

    double gripper_calculate(){
        if (!gripper_target_.ready() || !gripper_angle_.ready())
            return NAN;
        const double target = *gripper_target_;
        const double angle = *gripper_angle_;
        if (!std::isfinite(target) || !std::isfinite(angle))
            return NAN;
        const double velocity = gripper_velocity_.ready() ? *gripper_velocity_ : 0.0;
        const double command =
            kGripperKp * (target - angle) - kGripperKv * velocity;
        return std::clamp(command, -kGripperMaxTorque, kGripperMaxTorque);
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
    // gravity model coefficients
    static constexpr double kGravityP1 = 20.284;
    static constexpr double kGravityQ1 = -60.824;
    static constexpr double kGravityP2 = -15.112;
    static constexpr double kGravityQ2 = 2.680;
    static constexpr double kGravityP3 = -0.579;
    static constexpr double kGravityQ3 = 0.380;
    static constexpr double kGravityJ5ConfigWeight = 0.35;
    static constexpr double kGravityP5 = 16.368;
    static constexpr double kGravityQ5 = -0.715;
    static constexpr double kGravityE2 = -22.244;
    static constexpr double kGravityE3 = 1.534;
    static constexpr double kGravityE5 = -13.608;
    static constexpr std::array<double, 6> kFrictionTorque = {0.0, 11.6, 7.4, 0.0, 0.75, 0.0};
    static constexpr std::array<double, 6> kFrictionVelocity = {0.2, 0.2, 0.2, 0.2, 0.2, 0.2};
    // Breakaway compensation, N*m per joint. Measured dead band: J3 +-3.7, J2 +-13.3.
    // A pure P loop or an integral cannot push the joint out of it. Memoryless and
    // bounded, so it does not wind up. Keep below the measured kinetic friction.
    static constexpr std::array<double, 6> kBreakawayTorque = {0.0, 5.0, 4.0, 0.0, 0.4, 0.0};
    static constexpr double kBreakawayError = 0.01;  // rad, error scale of the tanh

    // Integral of the position error, N*m/(rad*s), clamped in N*m. Useless against the
    // friction dead band (it pushes the error to the other side), so off by default.
    static constexpr std::array<double, 6> kAngleIntegralGain = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    static constexpr std::array<double, 6> kAngleIntegralLimit = {0.0, 5.0, 5.0, 0.0, 0.0, 0.0};
    std::array<double, 6> angle_integral_{};

    static constexpr std::array<double, 6> kDragDamping = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    static constexpr double kDragVelocityLpf = 0.05;
    std::array<double, 6> drag_velocity_{};

    // Gripper position loop. Keep kp low or the torque clamps and shocks the gears.
    // The open/close rate comes from kGripperSpeed in arm_controller; this only follows.
    static constexpr double kGripperKp = 8.0;         // N*m/rad
    static constexpr double kGripperKv = 0.5;         // N*m/(rad/s)
    static constexpr double kGripperMaxTorque = 1.5;  // N*m, grip force limit
    InputInterface<double> gripper_angle_;
    InputInterface<double> gripper_velocity_;
    InputInterface<double> gripper_target_;
    OutputInterface<double> gripper_torque_;

    std::array<double, 6> last_target_theta_{};  
    static constexpr double kUpdateRateHz = 1000.0;  

    std::array<InputInterface<double>, 6>  joint_theta;
    std::array<InputInterface<double>, 6>  joint_target_theta;
    std::array<InputInterface<double>, 6>  joint_target_velocity_;
    std::array<InputInterface<double>, 6>  joint_lower_limit_;
    std::array<InputInterface<double>, 6>  joint_upper_limit_;
    std::array<InputInterface<double>, 6>  joint_velocity_;
    std::array<OutputInterface<double>, 6>  target_torque_;
    InputInterface<bool> is_loaded;
    InputInterface<bool> is_arm_enable;
    InputInterface<bool> drag_enable_;
    InputInterface<bool> offsets_verified_;      
    bool last_is_arm_enable{false};
    };
} // namespace rmcs_core::controller::arm

#include <pluginlib/class_list_macros.hpp>
PLUGINLIB_EXPORT_CLASS(rmcs_core::controller::arm::ArmSolver, rmcs_executor::Component)
