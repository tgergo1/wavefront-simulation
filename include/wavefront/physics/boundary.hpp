#pragma once

#include <cstddef>
#include <vector>

#include "wavefront/api/solver.hpp"
#include "wavefront/core/field.hpp"
#include "wavefront/core/grid.hpp"

namespace wavefront {

struct BoundaryMetrics {
  double reflected_energy = 0.0;
  double absorbed_energy = 0.0;
};

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
    std::size_t threads = 0,
    bool deterministic = true);

}  // namespace wavefront
