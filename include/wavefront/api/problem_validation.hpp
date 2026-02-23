#pragma once

#include <string>
#include <vector>

#include "wavefront/api/solver.hpp"

namespace wavefront {

struct ValidationIssue {
  std::string message;
  bool fatal = true;
};

std::vector<ValidationIssue> validate_problem(const ProblemSpec& problem, const SolverConfig& config);
void throw_if_invalid(const ProblemSpec& problem, const SolverConfig& config);

}  // namespace wavefront
