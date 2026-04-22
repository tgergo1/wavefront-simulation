#include "wavefront/api/problem_validation.hpp"

#include <algorithm>
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

  if (config.center_frequency <= 0.0) {
    issues.push_back({"config.center_frequency must be positive", true});
  }

  if (config.threads == 0) {
    issues.push_back({"config.threads must be >= 1", true});
  }

  if (config.spatial_order != 2 && config.spatial_order != 4) {
    issues.push_back({"config.spatial_order must be 2 or 4", true});
  }

  if (config.far_field_samples == 0) {
    issues.push_back({"config.far_field_samples must be >= 1", true});
  }

  if (config.precision == PrecisionMode::ExactReference) {
#if !WAVEFRONT_HAS_LIMITLESS
    issues.push_back({"ExactReference mode requires WAVEFRONT_HAS_LIMITLESS=1", true});
#endif
    if (config.reference_window == 0) {
      issues.push_back({"config.reference_window must be >= 1 in ExactReference mode", true});
    }
  }

  for (const auto& probe : problem.monitors.probes) {
    if (probe.index.size() != problem.grid.dims) {
      issues.push_back({"probe monitor indices must match grid.dims", true});
      break;
    }
    if (probe.component >= problem.field_components) {
      issues.push_back({"probe monitor component must be in [0, field_components)", true});
      break;
    }
  }

  for (const auto& surface : problem.monitors.surfaces) {
    if (surface.geometry_region.empty() && surface.axis >= problem.grid.dims) {
      issues.push_back({"surface monitor axis must be in [0, grid.dims)", true});
      break;
    }
    if (surface.component >= problem.field_components) {
      issues.push_back({"surface monitor component must be in [0, field_components)", true});
      break;
    }
    if (!surface.geometry_region.empty() &&
        std::none_of(problem.geometry.begin(), problem.geometry.end(), [&](const GeometryRegion& region) {
          return region.name == surface.geometry_region;
        })) {
      issues.push_back({"surface monitor geometry_region must reference an existing geometry region", true});
      break;
    }
    if (surface.shell_thickness < 0.0) {
      issues.push_back({"surface monitor shell_thickness must be non-negative", true});
      break;
    }
  }

  for (const auto& region : problem.geometry) {
    if (region.shape == GeometryShape::Box &&
        (region.min_corner.size() != problem.grid.dims || region.max_corner.size() != problem.grid.dims)) {
      issues.push_back({"box geometry regions must provide min_corner/max_corner for each dimension", true});
      break;
    }
    if (region.shape == GeometryShape::Sphere &&
        (region.center.size() != problem.grid.dims || region.radius <= 0.0)) {
      issues.push_back({"sphere geometry regions must provide center and positive radius", true});
      break;
    }
    if (region.shape == GeometryShape::Layer &&
        (region.axis >= problem.grid.dims || region.upper <= region.lower)) {
      issues.push_back({"layer geometry regions must have valid axis and upper > lower", true});
      break;
    }
    if (region.shape == GeometryShape::Polygon &&
        (problem.grid.dims != 2 || region.vertices.size() < 6 || region.vertices.size() % 2 != 0)) {
      issues.push_back({"polygon geometry regions require dims=2 and an even vertex list with at least 3 points", true});
      break;
    }
    if (region.shape == GeometryShape::SignedDistanceField && region.signed_distance.text.empty()) {
      issues.push_back({"signed-distance geometry regions must provide a signed_distance expression", true});
      break;
    }
    if (region.shape == GeometryShape::Fractal) {
      if (problem.grid.dims != 2) {
        issues.push_back({"fractal geometry regions currently require dims=2", true});
        break;
      }
      if (region.fractal_generator != "koch_snowflake") {
        issues.push_back({"fractal geometry regions currently support fractal_generator='koch_snowflake' only", true});
        break;
      }
      if (region.center.size() != 2 || region.radius <= 0.0) {
        issues.push_back({"fractal geometry regions require 2D center and positive radius", true});
        break;
      }
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
