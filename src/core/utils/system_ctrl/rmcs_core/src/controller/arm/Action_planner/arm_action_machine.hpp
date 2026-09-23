#pragma once
#include "controller/arm/Action_planner/action_step.hpp"
#include "controller/arm/Action_planner/ikfast_solver.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <eigen3/Eigen/Geometry>
#include <eigen3/Eigen/SVD>
#include <fstream>
#include <map>
#include <memory>
#include <moveit/move_group_interface/move_group_interface.hpp>
#include <moveit/robot_state/robot_state.hpp>
#include <moveit/robot_trajectory/robot_trajectory.hpp>
#include <moveit/trajectory_processing/time_optimal_trajectory_generation.hpp>
#include <moveit_msgs/msg/move_it_error_codes.hpp>
#include <rclcpp/duration.hpp>
#include <rclcpp/executors/multi_threaded_executor.hpp>
#include <rclcpp/logging.hpp>
#include <rclcpp/node.hpp>
#include <string>
#include <sstream>
#include <tf2/LinearMath/Matrix3x3.hpp>
#include <tf2/LinearMath/Quaternion.hpp>
#include <thread>
#include <trajectory_msgs/msg/joint_trajectory.hpp>
#include <trajectory_msgs/msg/joint_trajectory_point.hpp>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

namespace rmcs_core::controller::arm {

/// Tool frame. Currently link_6 itself; pose / linear / read_pose are relative to it.
inline constexpr const char* kTcpFrame = "link_6";

// Cartesian interpolation step for linear moves, in meters.
inline constexpr double kCartesianEefStep = 0.005;

// Time parameterization (TOTG). Consecutive motion steps are joined into a single path
// and timed once, so intermediate waypoints are blended through instead of stopping.
//   kPathTolerance : max deviation where a corner is blended, rad
//   kResampleDt    : output sample period, s (the controller interpolates between points)
inline constexpr double kPathTolerance = 0.05;
inline constexpr double kResampleDt = 0.01;
inline constexpr double kMinAngleChange = 0.001;
// Sample period for hold segments (delay / gripper).
inline constexpr double kHoldSampleInterval = 0.05;
inline constexpr double kPlanningTime = 5.0;

// TOTG needs velocity and acceleration limits per joint. The RobotModel in this process
// comes from the /robot_description topic, which carries the URDF only: velocities are
// there, accelerations are not. The defaults match config/joint_limits.yaml.
inline constexpr double kDefaultMaxJointVelocity = 5.0;
inline constexpr double kDefaultMaxJointAcceleration = 10.0;
// Waypoint rate (Hz) handed to TOTG for recorded trajectories.
inline constexpr double kTrajectoryOutputRate = 50.0;
// TOTG gives trapezoidal velocity, i.e. step changes in acceleration. A short moving
// average on the positions rounds those steps off; the path barely moves.
inline constexpr std::size_t kSmoothingWindow = 3;
// Tolerance for treating a recorded segment as continuous with the current pose.
inline constexpr double kTrajectoryJunctionTolerance = 0.05;

// min/max singular value of the Jacobian below this counts as near-singular: fewer than
// 6 effective DOF, so Cartesian motion and IK both fail. Here it happens when J4 and J6
// are parallel, i.e. J5 ~ 0.
inline constexpr double kSingularJacobianRatio = 1e-3;


class ActionMachine {
public:
    /// Timed event fired while the trajectory runs (currently gripper open/close).
    struct Event {
        double time{0.0};  // seconds from the start of the trajectory
        Action::MotionType type{Action::MotionType::Delay};
    };

    struct PlannedTrajectory {
        uint64_t request_id{0};
        bool plan_success{false};
        std::vector<trajectory_msgs::msg::JointTrajectoryPoint> points;
        std::vector<std::string> joint_names;
        std::vector<Event> events;
        // Parallel to joint_names: tool displacement in meters per radian of this joint.
        std::vector<double> joint_tcp_sensitivity;
        double duration{0.0};
        std::string error_message;
    };

    struct PlanRequest {
        uint64_t request_id{0};
        std::vector<Action::Step> steps;
    };

    ActionMachine()
        : node_(
              std::make_shared<rclcpp::Node>(
                  "arm_moveit_planner",
                  rclcpp::NodeOptions{}.automatically_declare_parameters_from_overrides(true)))
        , move_group_(
              std::make_unique<moveit::planning_interface::MoveGroupInterface>(
                  node_, "JG_arm")) {
        move_group_->startStateMonitor();

        exec_.add_node(node_);
        spin_thread_ = std::thread([this] { exec_.spin(); });

        running_.store(true, std::memory_order_release);
        moveit_thread_ = std::thread([this] {
            while (running_.load(std::memory_order_acquire)) {
                moveit_loop();
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        });
    }

    ~ActionMachine() {
        running_.store(false, std::memory_order_release);
        exec_.cancel();
        if (moveit_thread_.joinable())
            moveit_thread_.join();
        if (spin_thread_.joinable())
            spin_thread_.join();
    }

    ActionMachine(const ActionMachine&)            = delete;
    ActionMachine& operator=(const ActionMachine&) = delete;
    ActionMachine(ActionMachine&&)                 = delete;
    ActionMachine& operator=(ActionMachine&&)      = delete;

    void process(const std::vector<Action::Step>& steps_) {

        auto next          = std::make_shared<PlanRequest>();
        const auto current = plan_request_.load(std::memory_order_acquire);
        next->request_id   = current ? current->request_id + 1 : 1;
        next->steps        = steps_;
        plan_request_.store(next, std::memory_order_release);
    }

    std::shared_ptr<const PlannedTrajectory> get_trajectory() const {
        return planned_trajectory_.load(std::memory_order_acquire);
    }

private:
    /// A path is a list of joint configurations to pass through; the first one is the start.
    using Waypoints = std::vector<moveit::core::RobotStatePtr>;

    /// Joint limits from the robot model, fetched once (order joint_1..joint_6).
    const IkfastSolver::Limits& jointLimits() {
        if (joint_limits_ready_)
            return joint_limits_;
        for (auto& limit : joint_limits_)
            limit = {-M_PI, M_PI};
        const auto* group =
            move_group_->getRobotModel()->getJointModelGroup(move_group_->getName());
        if (group != nullptr) {
            for (const auto* joint : group->getActiveJointModels()) {
                for (const auto& name : joint->getVariableNames()) {
                    if (name.rfind("joint_", 0) != 0)
                        continue;
                    const int index = std::stoi(name.substr(6)) - 1;
                    if (index < 0 || index >= static_cast<int>(IkfastSolver::kJoints))
                        continue;
                    const auto& bounds = joint->getVariableBounds(name);
                    if (bounds.position_bounded_)
                        joint_limits_[static_cast<std::size_t>(index)] = {
                            bounds.min_position_, bounds.max_position_};
                }
            }
        }
        joint_limits_ready_ = true;
        return joint_limits_;
    }

    /// Joint values at the end of the chain, used to pick the nearest IK branch.
    IkfastSolver::Joints seedOf(const moveit::core::RobotState& state) const {
        IkfastSolver::Joints joints{};
        std::vector<double> positions;
        state.copyJointGroupPositions(move_group_->getName(), positions);
        for (std::size_t j = 0; j < joints.size() && j < positions.size(); ++j)
            joints[j] = positions[j];
        return joints;
    }

    /// RobotState from joint values, ready to append to the chain.
    moveit::core::RobotStatePtr stateOf(
        const moveit::core::RobotState& base, const IkfastSolver::Joints& joints) const {
        auto state = std::make_shared<moveit::core::RobotState>(base);
        state->setJointGroupPositions(
            move_group_->getName(), std::vector<double>(joints.begin(), joints.end()));
        state->update();
        return state;
    }

    static std::vector<std::string> splitCsv(const std::string& line) {
        std::vector<std::string> fields;
        std::stringstream stream(line);
        std::string field;
        while (std::getline(stream, field, ',')) {
            while (!field.empty() && (field.back() == '\r' || field.back() == ' '))
                field.pop_back();
            fields.push_back(field);
        }
        return fields;
    }

    /// Hold duration of a delay or gripper step.
    static double targetDuration(const Action::Step& step) {
        const auto& target = step.target();
        if (const auto* gripper = std::get_if<Action::GripperTarget>(&target))
            return gripper->duration;
        if (const auto* delay = std::get_if<Action::DelayTarget>(&target))
            return delay->duration;
        return 0.0;
    }

    /// JointTrajectory message -> waypoint sequence, start state prepended.
    Waypoints waypointsFromTrajectory(
        const trajectory_msgs::msg::JointTrajectory& msg,
        const moveit::core::RobotStatePtr& start_state) {
        const auto& names =
            msg.joint_names.empty() ? move_group_->getJointNames() : msg.joint_names;
        Waypoints waypoints;
        waypoints.reserve(msg.points.size() + 1);
        waypoints.push_back(start_state);
        std::vector<double> values(names.size(), 0.0);
        for (const auto& point : msg.points) {
            if (point.positions.size() < names.size())
                continue;
            std::copy_n(point.positions.begin(), names.size(), values.begin());
            auto state = std::make_shared<moveit::core::RobotState>(*start_state);
            state->setVariablePositions(names, values);
            state->update();
            waypoints.push_back(std::move(state));
        }
        return waypoints;
    }

    /// Plan one motion step (joint / pose / linear) into a waypoint sequence.
    bool planSingleStep(
        const Action::Step& step, const moveit::core::RobotStatePtr& start_state,
        Waypoints& out_waypoints) {
        out_waypoints.clear();

        bool ok = true;
        std::visit(
            [&](const auto& target) {
                using T = std::decay_t<decltype(target)>;
                if constexpr (std::is_same_v<T, Action::JointTarget>) {
                    const std::map<std::string, double> values{
                        {"joint_1", target.joint_1},
                        {"joint_2", target.joint_2},
                        {"joint_3", target.joint_3},
                        {"joint_4", target.joint_4},
                        {"joint_5", target.joint_5},
                        {"joint_6", target.joint_6},
                    };
                    if (!move_group_->setJointValueTarget(values)) {
                        RCLCPP_WARN(node_->get_logger(), "Joint target is out of bounds");
                        ok = false;
                        return;
                    }
                    moveit::planning_interface::MoveGroupInterface::Plan plan;
                    if (move_group_->plan(plan) != moveit::core::MoveItErrorCode::SUCCESS) {
                        ok = false;
                        return;
                    }
                    out_waypoints =
                        waypointsFromTrajectory(plan.trajectory.joint_trajectory, start_state);
                } else if constexpr (std::is_same_v<T, Action::PoseTarget>) {
                    // Closed-form IKfast instead of KDL, which diverges at the wrist
                    // singularity.
                    Eigen::Isometry3d goal = Eigen::Isometry3d::Identity();
                    goal.translation()     = Eigen::Vector3d(target.x, target.y, target.z);
                    goal.linear() =
                        (Eigen::AngleAxisd(target.yaw, Eigen::Vector3d::UnitZ())
                         * Eigen::AngleAxisd(target.pitch, Eigen::Vector3d::UnitY())
                         * Eigen::AngleAxisd(target.roll, Eigen::Vector3d::UnitX()))
                            .toRotationMatrix();
                    const auto seed = seedOf(*start_state);
                    IkfastSolver::Joints joints;
                    if (!ikfast_.solve(goal, seed, jointLimits(), joints)) {
                        warnIfSingular(*start_state);
                        RCLCPP_WARN(
                            node_->get_logger(),
                            "Pose target has no IK solution inside the joint limits");
                        ok = false;
                        return;
                    }
                    out_waypoints = {start_state, stateOf(*start_state, joints)};
                } else if constexpr (std::is_same_v<T, Action::LinearTarget>) {
                    const Eigen::Isometry3d& start = start_state->getGlobalLinkTransform(kTcpFrame);
                    Eigen::Vector3d dir(target.dir_x, target.dir_y, target.dir_z);
                    if (dir.norm() < 1e-9) {
                        RCLCPP_WARN(node_->get_logger(), "Linear step has a zero direction");
                        ok = false;
                        return;
                    }
                    dir.normalize();
                    Eigen::Isometry3d goal = start;
                    // frame=base: a fixed base_link axis (default).
                    // frame=tool: rotates with the end effector.
                    const Eigen::Vector3d offset = target.in_tool_frame
                                                       ? Eigen::Vector3d(start.linear() * dir)
                                                             * target.distance
                                                       : dir * target.distance;
                    goal.translation() += offset;
                    // Interpolate the line here and solve each point with IKfast instead
                    // of computeCartesianPath: its fraction is unreliable and it fails on
                    // the first step near the wrist singularity.
                    const double distance = offset.norm();
                    const int steps =
                        std::max(2, static_cast<int>(std::ceil(distance / kCartesianEefStep)));
                    out_waypoints.clear();
                    out_waypoints.push_back(start_state);
                    auto seed = seedOf(*start_state);
                    for (int i = 1; i <= steps; ++i) {
                        const double ratio = static_cast<double>(i) / static_cast<double>(steps);
                        Eigen::Isometry3d pose = start;
                        pose.translation()     = start.translation() + offset * ratio;
                        IkfastSolver::Joints joints;
                        if (!ikfast_.solve(pose, seed, jointLimits(), joints)) {
                            warnIfSingular(*start_state);
                            RCLCPP_WARN(
                                node_->get_logger(),
                                "Linear step: no IK solution %.1f mm along the straight path",
                                distance * ratio * 1000.0);
                            ok = false;
                            return;
                        }
                        seed = joints;
                        out_waypoints.push_back(stateOf(*start_state, joints));
                    }
                    RCLCPP_INFO(
                        node_->get_logger(), "Linear step (%s frame): (%.3f %.3f %.3f) -> (%.3f %.3f %.3f)",
                        target.in_tool_frame ? "tool" : "base", start.translation().x(),
                        start.translation().y(), start.translation().z(),
                        goal.translation().x(), goal.translation().y(),
                        goal.translation().z());
                } else {
                    ok = false;
                }
            },
            step.target());

        return ok && out_waypoints.size() >= 2;
    }

    /// read_pose / read_joint: print the current state, no motion (teaching aid).
    void reportCurrentState(Action::MotionType type) {
        const std::string group = move_group_->getName();
        if (type == Action::MotionType::ReadJoint) {
            std::vector<double> positions;
            move_group_->getCurrentState()->copyJointGroupPositions(group, positions);
            std::array<double, 6> joints{};
            const auto names = move_group_->getJointNames();
            for (std::size_t j = 0; j < names.size() && j < positions.size(); ++j) {
                if (names[j].rfind("joint_", 0) != 0)
                    continue;
                const int index = std::stoi(names[j].substr(6)) - 1;
                if (index >= 0 && index < 6)
                    joints[static_cast<std::size_t>(index)] = positions[j];
            }
            RCLCPP_INFO(
                node_->get_logger(), "Current joints: [%.4f, %.4f, %.4f, %.4f, %.4f, %.4f]",
                joints[0], joints[1], joints[2], joints[3], joints[4], joints[5]);
            return;
        }
        const auto pose = move_group_->getCurrentPose(kTcpFrame);
        double roll, pitch, yaw;
        tf2::Matrix3x3(
            tf2::Quaternion(
                pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z,
                pose.pose.orientation.w))
            .getRPY(roll, pitch, yaw);
        RCLCPP_INFO(
            node_->get_logger(),
            "Current pose: x=%.4f y=%.4f z=%.4f  roll=%.4f pitch=%.4f yaw=%.4f",
            pose.pose.position.x, pose.pose.position.y, pose.pose.position.z, roll, pitch, yaw);
    }

    /// min/max singular value of the Jacobian about the tool point; near 0 is singular.
    /// getJacobian takes the reference point in the tip link's own frame, hence zero.
    double jacobianRatio(const moveit::core::RobotState& state) const {
        const auto* joint_group =
            move_group_->getRobotModel()->getJointModelGroup(move_group_->getName());
        if (joint_group == nullptr)
            return 1.0;
        const Eigen::MatrixXd jacobian =
            state.getJacobian(joint_group, Eigen::Vector3d::Zero());
        if (jacobian.size() == 0)
            return 1.0;
        const Eigen::JacobiSVD<Eigen::MatrixXd> svd(jacobian);
        const auto& singular = svd.singularValues();
        if (singular.size() == 0 || singular(0) <= 0.0)
            return 1.0;
        return singular(singular.size() - 1) / singular(0);
    }

    /// Explain why a pose is stuck: singularity ratio plus the current joint values.
    void warnIfSingular(const moveit::core::RobotState& state) const {
        const double ratio = jacobianRatio(state);
        if (ratio >= kSingularJacobianRatio)
            return;
        std::string joints;
        std::vector<double> positions;
        state.copyJointGroupPositions(move_group_->getName(), positions);
        for (std::size_t i = 0; i < positions.size(); ++i) {
            char buffer[32];
            std::snprintf(buffer, sizeof(buffer), "%s%.3f", i == 0 ? "" : ", ", positions[i]);
            joints += buffer;
        }
        RCLCPP_WARN(
            node_->get_logger(),
            "Cartesian motion is blocked: the arm is near a wrist singularity "
            "(min/max singular value = %.1e, i.e. J4 and J6 axes are almost parallel, which "
            "happens when J5 is close to 0). Tilt J5 away from 0 (|J5| > 0.2) and re-teach this "
            "point. Current joints: [%s]",
            ratio, joints.c_str());
    }

    /// Per-joint velocity/acceleration limits for TOTG. Missing ones fall back to the
    /// defaults above, otherwise TOTG refuses to run at all.
    void fillSpeedLimits(
        const std::string& group, std::unordered_map<std::string, double>& velocity_limits,
        std::unordered_map<std::string, double>& acceleration_limits) const {
        velocity_limits.clear();
        acceleration_limits.clear();
        const auto* joint_group = move_group_->getRobotModel()->getJointModelGroup(group);
        if (joint_group == nullptr)
            return;
        bool fell_back = false;
        for (const auto* joint : joint_group->getActiveJointModels()) {
            for (const auto& name : joint->getVariableNames()) {
                const auto& bounds = joint->getVariableBounds(name);
                velocity_limits[name] =
                    bounds.velocity_bounded_ ? bounds.max_velocity_ : kDefaultMaxJointVelocity;
                if (bounds.acceleration_bounded_) {
                    acceleration_limits[name] = bounds.max_acceleration_;
                } else {
                    acceleration_limits[name] = kDefaultMaxJointAcceleration;
                    fell_back                 = true;
                }
            }
        }
        if (fell_back)
            RCLCPP_INFO(
                node_->get_logger(),
                "Joint acceleration limits missing from the robot model, using %.1f rad/s^2",
                kDefaultMaxJointAcceleration);
    }

    void moveit_loop() {
        const auto request = plan_request_.load(std::memory_order_acquire);
        if (!request || request->request_id == last_planned_id_)
            return;

        auto result          = std::make_shared<PlannedTrajectory>();
        result->request_id   = request->request_id;
        result->plan_success = true;
        result->joint_names  = move_group_->getJointNames();

        const std::string group = move_group_->getName();
        const auto model        = move_group_->getRobotModel();

        std::unordered_map<std::string, double> velocity_limits;
        std::unordered_map<std::string, double> acceleration_limits;
        fillSpeedLimits(group, velocity_limits, acceleration_limits);

        // Consecutive motion steps share one chain and are timed once, up to a step that
        // has to stop the arm (delay / gripper) or the end of the action.
        Waypoints chain{move_group_->getCurrentState()};
        double run_vel     = 1.0;
        double run_acc     = 1.0;
        bool run_active    = false;
        double time_offset = 0.0;
        bool failed        = false;

        // Time the chain into one continuous run and append it; the chain keeps its end.
        auto flush_run = [&]() -> bool {
            if (chain.size() >= 2) {
                // Drop repeated points, they would give TOTG a zero-length segment.
                Waypoints path;
                path.push_back(chain.front());
                std::vector<double> previous, current;
                for (std::size_t i = 1; i < chain.size(); ++i) {
                    path.back()->copyJointGroupPositions(group, previous);
                    chain[i]->copyJointGroupPositions(group, current);
                    double moved = 0.0;
                    for (std::size_t j = 0; j < previous.size() && j < current.size(); ++j)
                        moved += std::fabs(previous[j] - current[j]);
                    if (moved > 1e-9)
                        path.push_back(chain[i]);
                }
                if (path.size() < 2) {
                    auto last = chain.back();
                    chain.assign(1, last);
                    run_active = false;
                    return true;
                }
                robot_trajectory::RobotTrajectory trajectory(model, group);
                for (const auto& waypoint : path)
                    trajectory.addSuffixWayPoint(*waypoint, 0.0);
                trajectory_processing::TimeOptimalTrajectoryGeneration generator(
                    kPathTolerance, kResampleDt, kMinAngleChange);
                if (!generator.computeTimeStamps(
                        trajectory, velocity_limits, acceleration_limits, run_vel, run_acc)) {
                    result->error_message = "time parameterization failed";
                    return false;
                }
                const std::size_t count = trajectory.getWayPointCount();
                // The first run starts at the current pose; later runs repeat the previous end.
                const std::size_t first = result->points.empty() ? 0 : 1;
                std::vector<trajectory_msgs::msg::JointTrajectoryPoint> points;
                points.reserve(count - first);
                for (std::size_t i = first; i < count; ++i) {
                    const auto& waypoint = trajectory.getWayPoint(i);
                    trajectory_msgs::msg::JointTrajectoryPoint point;
                    waypoint.copyJointGroupPositions(group, point.positions);
                    waypoint.copyJointGroupVelocities(group, point.velocities);
                    point.time_from_start = rclcpp::Duration::from_seconds(
                        time_offset + trajectory.getWayPointDurationFromStart(i));
                    points.push_back(std::move(point));
                }
                // Round off the acceleration steps from TOTG and recompute velocities, so
                // starting and stopping do not feel abrupt. At a 10ms sample period the
                // positions barely move.
                if (points.size() > 2 * kSmoothingWindow) {
                    const auto original = points;
                    for (std::size_t i = kSmoothingWindow; i + kSmoothingWindow < points.size(); ++i) {
                        for (std::size_t j = 0; j < points[i].positions.size(); ++j) {
                            double sum = 0.0;
                            for (std::size_t k = i - kSmoothingWindow; k <= i + kSmoothingWindow; ++k)
                                sum += original[k].positions[j];
                            points[i].positions[j] =
                                sum / static_cast<double>(2 * kSmoothingWindow + 1);
                        }
                    }
                    for (std::size_t i = 0; i < points.size(); ++i) {
                        const std::size_t before = i > 0 ? i - 1 : i;
                        const std::size_t after  = i + 1 < points.size() ? i + 1 : i;
                        if (before == after)
                            break;
                        const double t0 = rclcpp::Duration(points[before].time_from_start).seconds();
                        const double t1 = rclcpp::Duration(points[after].time_from_start).seconds();
                        if (t1 <= t0)
                            continue;
                        points[i].velocities.resize(6, 0.0);
                        for (std::size_t j = 0; j < 6; ++j)
                            points[i].velocities[j] =
                                (points[after].positions[j] - points[before].positions[j]) / (t1 - t0);
                    }
                }
                for (auto& point : points)
                    result->points.push_back(std::move(point));
                time_offset += trajectory.getWayPointDurationFromStart(count - 1);
            }
            auto last = chain.back();
            chain.assign(1, last);
            run_vel = 1.0;
            run_acc = 1.0;
            run_active = false;
            return true;
        };

        // Hold in place (delay / gripper) at the end of the current chain.
        auto append_hold = [&](double duration) {
            std::vector<double> positions;
            chain.back()->copyJointGroupPositions(group, positions);
            const int segments =
                std::max(2, static_cast<int>(duration / kHoldSampleInterval) + 1);
            for (int k = 0; k < segments; ++k) {
                trajectory_msgs::msg::JointTrajectoryPoint point;
                point.positions       = positions;
                point.time_from_start = rclcpp::Duration::from_seconds(
                    time_offset
                    + duration * static_cast<double>(k) / static_cast<double>(segments - 1));
                result->points.push_back(std::move(point));
            }
            time_offset += duration;
        };

        // Recorded trajectory: read a (t, q1..q6) CSV and append the smoothed path to the
        // chain. No MoveIt, no IK - what was recorded is what is replayed.
        auto append_trajectory = [&](const Action::Step& step) -> bool {
            const auto* target = std::get_if<Action::TrajectoryTarget>(&step.target());
            if (target == nullptr)
                return false;
            std::ifstream file(target->file);
            if (!file) {
                result->error_message = "cannot open trajectory file '" + target->file + "'";
                return false;
            }
            std::string line;
            if (!std::getline(file, line)) {
                result->error_message = "trajectory file '" + target->file + "' is empty";
                return false;
            }
            const auto header       = splitCsv(line);
            int time_column         = -1;
            std::array<int, 6> joint_column{{-1, -1, -1, -1, -1, -1}};
            for (std::size_t i = 0; i < header.size(); ++i) {
                if (header[i] == "t")
                    time_column = static_cast<int>(i);
                for (std::size_t j = 0; j < 6; ++j) {
                    if (header[i] == "q" + std::to_string(j + 1))
                        joint_column[j] = static_cast<int>(i);
                }
            }
            const bool missing = time_column < 0
                                 || std::any_of(
                                     joint_column.begin(), joint_column.end(),
                                     [](int column) { return column < 0; });
            if (missing) {
                result->error_message =
                    "trajectory file '" + target->file + "' needs columns t and q1..q6";
                return false;
            }

            std::vector<double> times;
            std::vector<std::array<double, 6>> samples;
            while (std::getline(file, line)) {
                const auto fields = splitCsv(line);
                const std::size_t needed =
                    static_cast<std::size_t>(time_column) + 1;
                if (fields.size() < needed)
                    continue;
                try {
                    std::array<double, 6> sample{};
                    bool complete = true;
                    for (std::size_t j = 0; j < 6; ++j) {
                        const auto index = static_cast<std::size_t>(joint_column[j]);
                        if (index >= fields.size() || fields[index].empty()) {
                            complete = false;
                            break;
                        }
                        sample[j] = std::stod(fields[index]);
                    }
                    if (!complete)
                        continue;
                    times.push_back(std::stod(fields[static_cast<std::size_t>(time_column)]));
                    samples.push_back(sample);
                } catch (const std::exception&) {
                    continue;
                }
            }
            if (samples.size() < 2) {
                result->error_message =
                    "trajectory file '" + target->file + "' has no usable samples";
                return false;
            }

            const double duration = times.back() - times.front();
            if (duration < 1e-3) {
                result->error_message = "trajectory file '" + target->file + "' has zero length";
                return false;
            }
            const double speed = std::max(target->speed, 0.01);
            const double rate  = static_cast<double>(samples.size()) / duration;
            const std::size_t window =
                std::max<std::size_t>(1, static_cast<std::size_t>(target->smooth * rate));
            const std::size_t stride = std::max<std::size_t>(
                1, static_cast<std::size_t>(rate / kTrajectoryOutputRate));

            // Warn when the recording starts far from where the arm is now.
            std::vector<double> here;
            chain.back()->copyJointGroupPositions(group, here);
            double start_gap = 0.0;
            for (std::size_t j = 0; j < here.size() && j < 6; ++j)
                start_gap = std::max(start_gap, std::fabs(here[j] - samples.front()[j]));
            if (start_gap > kTrajectoryJunctionTolerance)
                RCLCPP_WARN(
                    node_->get_logger(),
                    "Trajectory '%s' starts %.3f rad away from where the arm is now; the arm has "
                    "to catch up first, which shows up as a jerk at the start. Re-record the "
                    "segments so they meet.",
                    target->file.c_str(), start_gap);

            // Path only: the smoothed waypoints join the chain and TOTG re-times them, so
            // slow or hesitant hand motion does not slow the replay down.
            const auto base = std::make_shared<moveit::core::RobotState>(*chain.back());
            std::size_t added = 0;
            for (std::size_t i = 0; i < samples.size(); i += stride) {
                std::array<double, 6> value{};
                const std::size_t begin = i > window ? i - window : 0;
                const std::size_t end   = std::min(samples.size() - 1, i + window);
                for (std::size_t j = 0; j < 6; ++j) {
                    double sum = 0.0;
                    for (std::size_t k = begin; k <= end; ++k)
                        sum += samples[k][j];
                    value[j] = sum / static_cast<double>(end - begin + 1);
                }
                chain.push_back(stateOf(*base, value));
                ++added;
            }
                RCLCPP_INFO(
                    node_->get_logger(),
                    "Trajectory '%s': %zu samples (%.1fs of hand motion) -> %zu path points, "
                    "retimed by TOTG",
                    target->file.c_str(), samples.size(), duration, added);
            return true;
        };

        // A change of vel/acc starts a new run. Steps sharing vel/acc are merged into one
        // trajectory that runs through without stopping.
        auto begin_run = [&](double step_vel, double step_acc) -> bool {
            step_vel = std::clamp(step_vel, 0.01, 1.0);
            step_acc = std::clamp(step_acc, 0.01, 1.0);
            if (run_active
                && (std::fabs(step_vel - run_vel) > 1e-9
                    || std::fabs(step_acc - run_acc) > 1e-9)) {
                if (!flush_run())
                    return false;
            }
            if (!run_active) {
                run_vel    = step_vel;
                run_acc    = step_acc;
                run_active = true;
            }
            return true;
        };

        for (std::size_t i = 0; i < request->steps.size() && !failed; ++i) {
            const auto& step = request->steps[i];
            const auto type  = step.type();

            if (type == Action::MotionType::OpenGripper
                || type == Action::MotionType::CloseGripper) {
                // Gripper steps are not part of the joint path, they only need time.
                if (!flush_run()) {
                    failed = true;
                    break;
                }
                const double trigger_time = time_offset;
                result->events.push_back(Event{trigger_time, type});
                append_hold(targetDuration(step));
                RCLCPP_INFO(
                    node_->get_logger(), "segment %zu: gripper action at %.2fs", i,
                    trigger_time);
                continue;
            }
            if (type == Action::MotionType::Delay) {
                if (!flush_run()) {
                    failed = true;
                    break;
                }
                append_hold(targetDuration(step));
                continue;
            }
            if (type == Action::MotionType::Trajectory) {
                // Replay speed comes from vel/acc, scaled by speed.
                const auto* target = std::get_if<Action::TrajectoryTarget>(&step.target());
                const double speed = target != nullptr ? std::max(target->speed, 0.01) : 1.0;
                if (!begin_run(step.params().vel * speed, step.params().acc * speed)) {
                    failed = true;
                    break;
                }
                if (!append_trajectory(step)) {
                    failed = true;
                    break;
                }
                continue;
            }
            if (type == Action::MotionType::ReadPose || type == Action::MotionType::ReadJoint) {
                reportCurrentState(type);
                continue;
            }

            move_group_->clearPoseTargets();
            move_group_->clearPathConstraints();
            move_group_->setStartState(*chain.back());
            move_group_->setPlanningTime(kPlanningTime);
            move_group_->setMaxVelocityScalingFactor(step.params().vel);
            move_group_->setMaxAccelerationScalingFactor(step.params().acc);
            move_group_->setGoalOrientationTolerance(step.params().tolerance_ori);
            move_group_->setGoalPositionTolerance(step.params().tolerance_pos);
            move_group_->setPlanningPipelineId(step.pipelineId());
            move_group_->setPlannerId(step.plannerId());

            // Same rule as above: a change of vel/acc breaks the run here.
            const auto& params = step.params();
            if (!begin_run(params.vel, params.acc)) {
                failed = true;
                break;
            }

            Waypoints waypoints;
            if (!planSingleStep(step, chain.back(), waypoints)) {
                result->error_message = "segment " + std::to_string(i) + " plan failed";
                failed                = true;
                break;
            }
            // First motion of a run replaces the chain, later ones append to it.
            if (chain.size() == 1) {
                chain = waypoints;
            } else {
                chain.insert(chain.end(), waypoints.begin() + 1, waypoints.end());
            }
            RCLCPP_INFO(
                node_->get_logger(), "segment %zu plan success (%zu waypoints)", i,
                waypoints.size());
        }

        if (!failed && !flush_run())
            failed = true;

        result->plan_success = !failed;
        if (failed)
            RCLCPP_WARN(node_->get_logger(), "%s", result->error_message.c_str());

        // Jacobian at the goal: per-joint tool displacement in m/rad, so the controller can
        // report a tracking error in millimeters instead of radians.
        result->joint_tcp_sensitivity.assign(result->joint_names.size(), 0.0);
        if (!result->points.empty()) {
            auto goal_state = std::make_shared<moveit::core::RobotState>(*move_group_->getCurrentState());
            goal_state->setVariablePositions(result->joint_names, result->points.back().positions);
            goal_state->update();
            const auto* joint_group = model->getJointModelGroup(group);
            // Reference point is in the tip link's own frame; the tip origin is zero.
            const Eigen::MatrixXd jacobian =
                goal_state->getJacobian(joint_group, Eigen::Vector3d::Zero());
            if (jacobian.rows() >= 3
                && jacobian.cols() == static_cast<int>(result->joint_names.size())) {
                for (std::size_t i = 0; i < result->joint_names.size(); ++i)
                    result->joint_tcp_sensitivity[i] = jacobian.topRows(3).col(i).norm();
            }
        }

        result->duration = time_offset;
        last_planned_id_ = request->request_id;
        planned_trajectory_.store(result, std::memory_order_release);
    }

    rclcpp::Node::SharedPtr node_;
    rclcpp::executors::MultiThreadedExecutor exec_;
    IkfastSolver ikfast_;
    IkfastSolver::Limits joint_limits_{};
    bool joint_limits_ready_{false};
    std::thread spin_thread_;
    std::unique_ptr<moveit::planning_interface::MoveGroupInterface> move_group_;
    std::atomic_bool running_{false};
    std::thread moveit_thread_;
    std::atomic<std::shared_ptr<const PlanRequest>> plan_request_{nullptr};
    std::atomic<std::shared_ptr<const PlannedTrajectory>> planned_trajectory_{nullptr};
    uint64_t last_planned_id_{0};
};

} // namespace rmcs_core::controller::arm
