#pragma once

#include <functional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "wavefront/api/solver.hpp"

namespace wavefront {

struct SweepParameter {
  std::string name;
  std::vector<double> values;
};

struct SweepEvaluation {
  std::unordered_map<std::string, double> assignments;
  double objective = 0.0;
};

inline void apply_sweep_parameter(ProblemSpec& problem, SolverConfig& config, std::string_view name, double value) {
  const std::string text = std::to_string(value);
  if (name == "medium.density") {
    problem.medium.density.text = text;
  } else if (name == "medium.stiffness") {
    problem.medium.stiffness.text = text;
  } else if (name == "medium.damping") {
    problem.medium.damping.text = text;
  } else if (name == "medium.dispersion") {
    problem.medium.dispersion.text = text;
  } else if (name == "config.cfl") {
    config.cfl = value;
  } else if (name == "config.center_frequency") {
    config.center_frequency = value;
  } else {
    throw std::invalid_argument("unsupported sweep parameter");
  }
}

template <typename Objective>
std::vector<SweepEvaluation> run_parameter_sweep(
    const ProblemSpec& base_problem,
    const SolverConfig& base_config,
    const std::vector<SweepParameter>& parameters,
    Objective&& objective) {
  std::vector<SweepEvaluation> results;
  if (parameters.empty()) {
    results.push_back({{}, objective(base_problem, base_config)});
    return results;
  }

  SweepEvaluation current;
  std::function<void(std::size_t, ProblemSpec, SolverConfig)> recurse =
      [&](std::size_t index, ProblemSpec problem, SolverConfig config) {
        if (index == parameters.size()) {
          current.objective = objective(problem, config);
          results.push_back(current);
          return;
        }

        const auto& parameter = parameters[index];
        for (double value : parameter.values) {
          apply_sweep_parameter(problem, config, parameter.name, value);
          current.assignments[parameter.name] = value;
          recurse(index + 1, problem, config);
        }
        current.assignments.erase(parameter.name);
      };

  recurse(0, base_problem, base_config);
  return results;
}

template <typename Objective>
double finite_difference_gradient(
    const ProblemSpec& base_problem,
    const SolverConfig& base_config,
    std::string_view parameter_name,
    double base_value,
    double epsilon,
    Objective&& objective) {
  if (epsilon <= 0.0) {
    throw std::invalid_argument("epsilon must be positive");
  }

  ProblemSpec lower_problem = base_problem;
  ProblemSpec upper_problem = base_problem;
  SolverConfig lower_config = base_config;
  SolverConfig upper_config = base_config;
  apply_sweep_parameter(lower_problem, lower_config, parameter_name, base_value - epsilon);
  apply_sweep_parameter(upper_problem, upper_config, parameter_name, base_value + epsilon);

  const double lower = objective(lower_problem, lower_config);
  const double upper = objective(upper_problem, upper_config);
  return (upper - lower) / (2.0 * epsilon);
}

}  // namespace wavefront
