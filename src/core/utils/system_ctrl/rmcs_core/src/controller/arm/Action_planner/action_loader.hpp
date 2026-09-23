#pragma once

#include "controller/arm/Action_planner/action_step.hpp"

#include <yaml-cpp/yaml.h>

#include <cstddef>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace rmcs_core::controller::arm {
///   joint           joint[6] (rad)
///   pose            xyz[3] (m) + rpy[3] (rad)
///   linear          dir[3] + distance (m) + frame(base|tool)
///   trajectory      file (CSV: t,q1..q6) + speed + smooth
///   open_gripper / close_gripper
///   delay           duration (s)
///   read_pose / read_joint
/// optional: vel, acc, tol_pos, tol_ori
class ActionLoader {
public:
    using ActionMap = std::map<std::string, std::vector<Action::Step>>;

    static ActionMap load(const std::string& path) {
        YAML::Node root;
        try {
            root = YAML::LoadFile(path);
        } catch (const std::exception& error) {
            throw std::runtime_error("cannot load action file '" + path + "': " + error.what());
        }
        if (!root.IsMap())
            throw std::runtime_error("action file '" + path + "' must be a map of action names");

        ActionMap actions;
        for (const auto& entry : root) {
            const std::string name = entry.first.as<std::string>();
            const YAML::Node& body  = entry.second;
            if (!body.IsSequence())
                throw std::runtime_error("action '" + name + "' must be a list of steps");
            auto& steps = actions[name];
            for (const auto& node : body)
                steps.push_back(parse_step(name, node));
            if (steps.empty())
                throw std::runtime_error("action '" + name + "' has no steps");
        }
        return actions;
    }

private:
    static double require_double(
        const YAML::Node& node, const std::string& field, const std::string& where) {
        if (!node[field])
            throw std::runtime_error(where + ": missing field '" + field + "'");
        return node[field].as<double>();
    }

    static std::vector<double> require_vector(
        const YAML::Node& node, const std::string& field, std::size_t size,
        const std::string& where) {
        if (!node[field])
            throw std::runtime_error(where + ": missing field '" + field + "'");
        const auto values = node[field].as<std::vector<double>>();
        if (values.size() != size)
            throw std::runtime_error(
                where + ": field '" + field + "' expects " + std::to_string(size)
                + " values, got " + std::to_string(values.size()));
        return values;
    }

    static Action::MotionParams parse_params(const YAML::Node& node) {
        Action::MotionParams params;  
        if (node["vel"])
            params.vel = node["vel"].as<double>();
        if (node["acc"])
            params.acc = node["acc"].as<double>();
        if (node["tol_pos"])
            params.tolerance_pos = node["tol_pos"].as<double>();
        if (node["tol_ori"])
            params.tolerance_ori = node["tol_ori"].as<double>();
        return params;
    }

    static Action::Step parse_step(const std::string& action, const YAML::Node& node) {
        if (!node["type"])
            throw std::runtime_error("action '" + action + "': a step has no 'type'");
        const std::string type = node["type"].as<std::string>();
        const std::string where = "action '" + action + "'";

        if (type == "joint") {
            const auto values = require_vector(node, "joint", 6, where);
            return Action::Step::makeJoint(
                Action::JointTarget{
                    .joint_1 = values[0], .joint_2 = values[1], .joint_3 = values[2],
                    .joint_4 = values[3], .joint_5 = values[4], .joint_6 = values[5]},
                parse_params(node));
        }
        if (type == "pose") {
            const auto xyz = require_vector(node, "xyz", 3, where);
            const auto rpy = require_vector(node, "rpy", 3, where);
            return Action::Step::makePose(
                Action::PoseTarget{
                    .x = xyz[0], .y = xyz[1], .z = xyz[2],
                    .roll = rpy[0], .pitch = rpy[1], .yaw = rpy[2]},
                parse_params(node));
        }
        if (type == "linear") {
            const auto dir = require_vector(node, "dir", 3, where);
            const std::string frame = node["frame"] ? node["frame"].as<std::string>() : "base";
            if (frame != "base" && frame != "tool")
                throw std::runtime_error(
                    where + ": linear 'frame' must be 'base' or 'tool', got '" + frame + "'");
            return Action::Step::makeLinear(
                Action::LinearTarget{
                    .dir_x = dir[0], .dir_y = dir[1], .dir_z = dir[2],
                    .distance = require_double(node, "distance", where),
                    .in_tool_frame = (frame == "tool")},
                parse_params(node));
        }
        if (type == "open_gripper")
            return Action::Step::makeOpenGripper(
                node["duration"] ? node["duration"].as<double>() : 1.5);
        if (type == "trajectory") {
            if (!node["file"])
                throw std::runtime_error(where + ": trajectory needs 'file'");
            Action::TrajectoryTarget target;
            target.file = node["file"].as<std::string>();
            if (node["speed"])
                target.speed = node["speed"].as<double>();
            if (node["smooth"])
                target.smooth = node["smooth"].as<double>();
            // Path only, re-timed by TOTG; default to half the joint limits.
            auto params = parse_params(node);
            if (!node["vel"])
                params.vel = 0.5;
            if (!node["acc"])
                params.acc = 0.5;
            return Action::Step::makeTrajectory(target, params);
        }
        if (type == "close_gripper")
            return Action::Step::makeCloseGripper(
                node["duration"] ? node["duration"].as<double>() : 1.5);
        if (type == "delay")
            return Action::Step::makeDelay(node["duration"] ? node["duration"].as<double>() : 0.5);
        if (type == "read_pose")
            return Action::Step::makeReadPose();
        if (type == "read_joint")
            return Action::Step::makeReadJoint();

        throw std::runtime_error("action '" + action + "': unknown step type '" + type + "'");
    }
};

} // namespace rmcs_core::controller::arm
