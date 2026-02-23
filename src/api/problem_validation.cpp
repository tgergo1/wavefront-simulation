#include "wavefront/api/problem_validation.hpp"

#include <sstream>
#include <stdexcept>

namespace wavefront {

std::vector<ValidationIssue> validate_problem(const ProblemSpec& problem, const SolverConfig& config) {
  std::vector<ValidationIssue> issues;

  if (problem.grid.dims == 0) {
    issues.push_back({"grid.dims must be positive", true});
    return issues;
  }

  if (problem.grid.shape.size() != problem.grid.dims) {
    issues.push_back({"grid.shape rank must equal grid.dims", true});
  }
  if (problem.grid.spacing.size() != problem.grid.dims) {
    issues.push_back({"grid.spacing rank must equal grid.dims", true});
  }
  if (!problem.grid.origin.empty() && problem.grid.origin.size() != problem.grid.dims) {
    issues.push_back({"grid.origin must be empty or have rank grid.dims", true});
  }

  for (std::size_t axis = 0; axis < problem.grid.shape.size(); ++axis) {
    if (problem.grid.shape[axis] < 3) {
      issues.push_back({"grid.shape entries must be at least 3 to support stencils", true});
      break;
    }
  }

  for (std::size_t axis = 0; axis < problem.grid.spacing.size(); ++axis) {
    if (problem.grid.spacing[axis] <= 0.0) {
      issues.push_back({"grid.spacing entries must be positive", true});
      break;
    }
  }

  if (problem.field_components == 0) {
    issues.push_back({"field_components must be positive", true});
  }

  for (const auto& boundary : problem.boundaries) {
    if (boundary.axis >= problem.grid.dims) {
      issues.push_back({"boundary.axis must be in [0, grid.dims)", true});
      break;
    }
  }

  if (config.cfl <= 0.0 || config.cfl > 1.0) {
    issues.push_back({"config.cfl must be in (0, 1]", true});
  }

  if (config.threads == 0) {
    issues.push_back({"config.threads must be >= 1", true});
  }

  if (config.spatial_order != 2 && config.spatial_order != 4) {
    issues.push_back({"config.spatial_order must be 2 or 4", true});
  }

  if (config.precision == PrecisionMode::ExactReference) {
#if !WAVEFRONT_HAS_LIMITLESS
    issues.push_back({"ExactReference mode requires WAVEFRONT_HAS_LIMITLESS=1", true});
#endif
    if (config.reference_window == 0) {
      issues.push_back({"config.reference_window must be >= 1 in ExactReference mode", true});
    }
  }

  return issues;
}

void throw_if_invalid(const ProblemSpec& problem, const SolverConfig& config) {
  const auto issues = validate_problem(problem, config);
  if (issues.empty()) {
    return;
  }

  std::ostringstream out;
  out << "Invalid wavefront problem/config:";
  for (const auto& issue : issues) {
    if (issue.fatal) {
      out << "\n - " << issue.message;
    }
  }
  throw std::invalid_argument(out.str());
}

}  // namespace wavefront
