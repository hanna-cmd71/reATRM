#pragma once
#include <string>
#include <variant>
#include <moveit/move_group_interface/move_group_interface.hpp>
#include <moveit/robot_state/robot_state.hpp>
#include <trajectory_msgs/msg/joint_trajectory_point.hpp>

namespace rmcs_core::controller::arm::Action {

enum class MotionType {
    Pose,
    Linear,
    Joint,
    Trajectory,
    OpenGripper,
    CloseGripper,
    Delay,
    ReadPose,
    ReadJoint,
};

struct PoseTarget {
    double x, y, z, roll, pitch, yaw;
};

struct JointTarget {
    double joint_1, joint_2, joint_3, joint_4, joint_5, joint_6;
};

struct LinearTarget {
    double dir_x, dir_y, dir_z, distance;
    /// false (default): dir is a fixed base_link axis.
    /// true: dir rotates with the tool frame (link_6).
    bool in_tool_frame{false};
};

/// Hold in place for a while (e.g. let the gripper finish).
struct DelayTarget {
    double duration;  // seconds
};

struct NoTarget {};

/// Recorded trajectory: replay a (t, q1..q6) CSV. No planning, no IK.
struct TrajectoryTarget {
    std::string file;      // CSV path: time in seconds, then q1..q6
    double speed{1.0};     // time scale, 0.5 = half speed
    double smooth{0.05};   // moving-average window, s, to remove hand tremor
};

/// Gripper open/close. Not part of the joint path, but it occupies time so the
/// controller can trigger it and the gripper has time to move.
struct GripperTarget {
    double duration;  // seconds
};

using Target =
    std::variant<
        NoTarget, PoseTarget, JointTarget, LinearTarget, TrajectoryTarget, DelayTarget,
        GripperTarget>;

struct MotionParams {
    double vel            = 0.05;
    double acc            = 0.03;
    double tolerance_pos  = 0.003;
    double tolerance_ori  = 0.2;
};

class Step {
public:
    // ---------- factories ----------
    static Step makeJoint(const JointTarget& target,
                          const MotionParams& params,
                          const std::string& pipeline = "ompl",
                          const std::string& planner  = " ")
    {
        return Step(MotionType::Joint, pipeline, planner, target, params);
    }

    static Step makePose(const PoseTarget& target,
                         const MotionParams& params,
                         const std::string& pipeline = "ompl",
                         const std::string& planner  = " ")
    {
        return Step(MotionType::Pose, pipeline, planner, target, params);
    }

    static Step makeLinear(const LinearTarget& target,
                           const MotionParams& params,
                           const std::string& pipeline = "pilz_industrial_motion_planner",
                           const std::string& planner  = "LIN")
    {
        return Step(MotionType::Linear, pipeline, planner, target, params);
    }

    static Step makeOpenGripper(double duration = 1.5) {
        return Step(MotionType::OpenGripper, "", "", GripperTarget{duration}, MotionParams{});
    }

    static Step makeCloseGripper(double duration = 1.5) {
        return Step(MotionType::CloseGripper, "", "", GripperTarget{duration}, MotionParams{});
    }

    static Step makeDelay(double duration) {
        return Step(MotionType::Delay, "", "", DelayTarget{duration}, MotionParams{});
    }

    /// Print the current tool pose, no motion (teaching aid).
    static Step makeReadPose() {
        return Step(MotionType::ReadPose, "", "", NoTarget{}, MotionParams{});
    }

    /// Print the current joint values, no motion (teaching aid).
    static Step makeReadJoint() {
        return Step(MotionType::ReadJoint, "", "", NoTarget{}, MotionParams{});
    }

    /// Recorded trajectory: replay a (t, q) file. Only its path is used; TOTG re-times
    /// it and params.vel/acc set the speed, same as joint/linear steps.
    static Step makeTrajectory(const TrajectoryTarget& target, const MotionParams& params) {
        return Step(MotionType::Trajectory, "", "", target, params);
    }

    // ---------- accessors ----------
    MotionType type() const { return motion_type_; }
    const std::string& pipelineId() const { return pipeline_id_; }
    const std::string& plannerId() const { return planner_id_; }
    const Target& target() const { return target_; }
    const MotionParams& params() const { return params_; }

private:
    Step(MotionType motion_type,
         std::string pipeline,
         std::string planner,
         Target target,
         MotionParams p)
        : motion_type_(motion_type)
        , pipeline_id_(std::move(pipeline))
        , planner_id_(std::move(planner))
        , target_(target)
        , params_(p)
    {}

    MotionType motion_type_;
    std::string pipeline_id_;
    std::string planner_id_;
    Target target_;
    MotionParams params_;
};

} // namespace rmcs_core::controller::arm::Action
