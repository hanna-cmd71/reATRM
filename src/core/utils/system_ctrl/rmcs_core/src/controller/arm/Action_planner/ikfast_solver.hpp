#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include <eigen3/Eigen/Geometry>

#define IKFAST_HAS_LIBRARY
#include "controller/arm/IKfast/ikfast.h"

namespace rmcs_core::controller::arm {

/// Wrapper around the generated closed-form IK for JGarm (ikfast, 2026-07-28).
/// KDL, which MoveIt uses, diverges near the wrist singularity; this does not.
///
/// The solver was generated when joint_5 had rpy="1.5708 0.022902 0", the URDF now
/// has "1.5708 0 0". The difference is exactly a constant offset on q5, so adding it
/// back aligns the solver with the current model (residual 0.008mm / 0.001deg).
class IkfastSolver {
public:
    static constexpr std::size_t kJoints = 6;
    static constexpr double kJoint5Offset = 0.022902;

    using Joints = std::array<double, kJoints>;
    /// [lower, upper] per joint, same order as Joints.
    using Limits = std::array<std::array<double, 2>, kJoints>;

    /// Solve for a tool pose in the base frame. seed selects the nearest branch.
    /// Returns true only if a solution inside the joint limits exists.
    bool solve(const Eigen::Isometry3d& target, const Joints& seed, const Limits& limits,
               Joints& solution) const {
        // ikfast expects the 3x3 rotation matrix in row-major order.
        std::array<double, 9> rotation{};
        for (std::size_t row = 0; row < 3; ++row) {
            for (std::size_t column = 0; column < 3; ++column)
                rotation[row * 3 + column] = target.rotation()(row, column);
        }
        const std::array<double, 3> translation{
            target.translation().x(), target.translation().y(), target.translation().z()};

        ikfast::IkSolutionList<IkReal> solutions;
        if (!ComputeIk(translation.data(), rotation.data(), nullptr, solutions))
            return false;

        std::vector<IkReal> values(static_cast<std::size_t>(GetNumJoints()));
        bool found            = false;
        double best_distance  = std::numeric_limits<double>::max();
        for (std::size_t i = 0; i < solutions.GetNumSolutions(); ++i) {
            const auto& candidate = solutions.GetSolution(i);
            std::vector<IkReal> free(candidate.GetFree().size());
            candidate.GetSolution(values.data(), free.empty() ? nullptr : free.data());
            if (values.size() < kJoints)
                continue;

            Joints joints{};
            bool inside  = true;
            double distance = 0.0;
            for (std::size_t j = 0; j < kJoints; ++j) {
                // Shift q5 back to the current URDF convention.
                joints[j] = wrap(values[j] + (j == 4 ? kJoint5Offset : 0.0));
                if (joints[j] < limits[j][0] - kJointTolerance
                    || joints[j] > limits[j][1] + kJointTolerance)
                    inside = false;
                distance += squared(wrap(joints[j] - seed[j]));
            }
            if (inside && distance < best_distance) {
                best_distance = distance;
                solution      = joints;
                found         = true;
            }
        }
        return found;
    }

private:
    static constexpr double kJointTolerance = 1e-6;

    static double wrap(double angle) {
        return std::remainder(angle, 2.0 * M_PI);
    }
    static double squared(double value) {
        return value * value;
    }
};

} // namespace rmcs_core::controller::arm
