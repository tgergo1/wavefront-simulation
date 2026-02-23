#include <doctest/doctest.h>

#include <vector>

#include "../test_common.hpp"
#include "wavefront/core/field.hpp"
#include "wavefront/core/grid.hpp"
#include "wavefront/physics/boundary.hpp"
#include "wavefront/physics/interface.hpp"

TEST_CASE("interface reflection and transmission obey impedance relationships") {
  const double rho_1 = 1.0;
  const double k_1 = 1.0;
  const double rho_2 = 4.0;
  const double k_2 = 1.0;

  const auto flux = wavefront::compute_interface_flux(1.0, 0.2, rho_1, k_1, rho_2, k_2);
  CHECK(std::isfinite(flux.reflected));
  CHECK(std::isfinite(flux.transmitted));
  CHECK(std::isfinite(flux.mode_conversion));

  const double z1 = wavefront::impedance(rho_1, wavefront::phase_velocity(k_1, rho_1));
  const double z2 = wavefront::impedance(rho_2, wavefront::phase_velocity(k_2, rho_2));
  CHECK(flux.reflected == doctest::Approx(wavefront::reflection_coefficient(z1, z2)));
}

TEST_CASE("Dirichlet boundary enforces fixed value at boundary nodes") {
  wavefront::GridSpec spec;
  spec.dims = 1;
  spec.shape = {8};
  spec.spacing = {1.0};
  spec.origin = {0.0};

  wavefront::GridLayout grid(spec);
  wavefront::FieldBuffer<double> prev(grid, 1);
  wavefront::FieldBuffer<double> curr(grid, 1);
  wavefront::FieldBuffer<double> next(grid, 1);
  prev.fill(0.0);
  curr.fill(0.5);
  next.fill(0.5);

  std::vector<wavefront::BoundarySpec> boundaries = {
      wavefront::BoundarySpec{wavefront::BoundaryType::Dirichlet, 0, false, wavefront::SymbolicExpr{"2.0"}},
      wavefront::BoundarySpec{wavefront::BoundaryType::Dirichlet, 0, true, wavefront::SymbolicExpr{"2.0"}},
  };

  std::vector<double> pml_memory;
  wavefront::apply_boundary_conditions(grid, boundaries, pml_memory, next, curr, prev, 0, true, 0.1, 0.0);

  CHECK(next.at_index({0}, 0) == doctest::Approx(2.0));
  CHECK(next.at_index({7}, 0) == doctest::Approx(2.0));
}
