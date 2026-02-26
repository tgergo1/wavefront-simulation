#include "wavefront/physics/boundary.hpp"

#include <cmath>
#include <mutex>
#include <string>

#include "wavefront/core/threading.hpp"

namespace wavefront {
namespace {

double parse_boundary_parameter(const SymbolicExpr& expr, double fallback) {
  try {
    std::size_t pos = 0;
    const double value = std::stod(expr.text, &pos);
    if (pos == expr.text.size()) {
      return value;
    }
  } catch (...) {
  }
  return fallback;
}

}  // namespace

BoundaryMetrics apply_boundary_conditions(
    const GridLayout& grid,
    const std::vector<BoundarySpec>& boundaries,
    std::vector<double>& pml_memory,
    FieldBuffer<double>& next_field,
    const FieldBuffer<double>& current_field,
    const FieldBuffer<double>& previous_field,
    std::size_t component,
    bool split_pml,
    double dt,
    double pml_sigma_default,
    std::size_t threads,
    bool deterministic) {
  BoundaryMetrics metrics;

  if (boundaries.empty()) {
    return metrics;
  }

  const std::size_t points = grid.total_points();
  const std::size_t components = next_field.components();

  if (pml_memory.size() != points * components) {
    pml_memory.assign(points * components, 0.0);
  }

  for (const auto& boundary : boundaries) {
    const double parameter = parse_boundary_parameter(boundary.parameter, pml_sigma_default);

    std::mutex mtx;
    deterministic_parallel_for(
        points,
        threads,
        deterministic,
        [&](std::size_t begin, std::size_t end) {
          BoundaryMetrics local_metrics;
          for (std::size_t flat = begin; flat < end; ++flat) {
            const auto index = grid.unravel_index(flat);
            if (!grid.is_boundary_cell(index, boundary.axis, boundary.upper_face)) {
              continue;
            }

            std::vector<std::size_t> interior = index;
            interior[boundary.axis] = boundary.upper_face ? (grid.shape()[boundary.axis] - 2) : 1;
            const std::size_t interior_flat = grid.flatten_index(interior);

            const std::size_t offset = flat * components + component;
            const double current = current_field.at_flat(flat, component);
            const double previous = previous_field.at_flat(flat, component);
            double next = next_field.at_flat(flat, component);

            switch (boundary.type) {
              case BoundaryType::Dirichlet:
                next = parameter;
                local_metrics.reflected_energy += current * current;
                break;
              case BoundaryType::Neumann: {
                const double interior_value = current_field.at_flat(interior_flat, component);
                const double h = grid.spacing()[boundary.axis];
                const double sign = boundary.upper_face ? 1.0 : -1.0;
                next = interior_value + sign * parameter * h;
                local_metrics.reflected_energy += (next - interior_value) * (next - interior_value);
                break;
              }
              case BoundaryType::Robin: {
                const double interior_value = current_field.at_flat(interior_flat, component);
                const double h = grid.spacing()[boundary.axis];
                const double gradient = (interior_value - current) / h;
                next = parameter - gradient;
                local_metrics.reflected_energy += gradient * gradient;
                break;
              }
              case BoundaryType::Periodic: {
                // Periodicity is handled directly by wrapped neighbor stencils in the core solver.
                // Avoid overriding `next` here with lagged values from `current`, which introduces
                // non-physical phase artifacts at the boundary.
                break;
              }
              case BoundaryType::Impedance: {
                const double velocity = (current - previous) / dt;
                next = current - parameter * dt * velocity;
                local_metrics.absorbed_energy += std::fabs(parameter * velocity) * dt;
                break;
              }
              case BoundaryType::PML: {
                const double sigma = parameter > 0.0 ? parameter : pml_sigma_default;
                const double decay = std::exp(-sigma * dt);
                if (split_pml) {
                  const double velocity = (current - previous) / dt;
                  pml_memory[offset] = decay * pml_memory[offset] + (1.0 - decay) * velocity;
                  next = decay * next - (1.0 - decay) * dt * pml_memory[offset];
                } else {
                  next = decay * next;
                }
                local_metrics.absorbed_energy += std::fabs(current - next);
                break;
              }
            }

            next_field.at_flat(flat, component) = next;
          }

          std::lock_guard<std::mutex> lock(mtx);
          metrics.reflected_energy += local_metrics.reflected_energy;
          metrics.absorbed_energy += local_metrics.absorbed_energy;
        });
  }

  return metrics;
}

}  // namespace wavefront
