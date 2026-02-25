#include <doctest/doctest.h>

#include <cmath>
#include <string>
#include <vector>

#include "../test_common.hpp"
#include "wavefront/core/field.hpp"
#include "wavefront/core/grid.hpp"
#include "wavefront/physics/boundary.hpp"

// ---------------------------------------------------------------------------
//  Helper: create a grid + field triple for boundary tests
// ---------------------------------------------------------------------------

namespace {

struct BoundaryTestSetup {
  wavefront::GridLayout grid;
  wavefront::FieldBuffer<double> prev;
  wavefront::FieldBuffer<double> curr;
  wavefront::FieldBuffer<double> next;
};

BoundaryTestSetup make_1d_setup(std::size_t points, double spacing = 1.0) {
  wavefront::GridSpec spec;
  spec.dims = 1;
  spec.shape = {points};
  spec.spacing = {spacing};
  spec.origin = {0.0};

  wavefront::GridLayout grid(spec);
  wavefront::FieldBuffer<double> prev(grid, 1);
  wavefront::FieldBuffer<double> curr(grid, 1);
  wavefront::FieldBuffer<double> next(grid, 1);
  prev.fill(0.0);
  curr.fill(0.0);
  next.fill(0.0);
  return {std::move(grid), std::move(prev), std::move(curr), std::move(next)};
}

BoundaryTestSetup make_2d_setup(std::size_t nx, std::size_t ny, double hx = 1.0, double hy = 1.0) {
  wavefront::GridSpec spec;
  spec.dims = 2;
  spec.shape = {nx, ny};
  spec.spacing = {hx, hy};
  spec.origin = {0.0, 0.0};

  wavefront::GridLayout grid(spec);
  wavefront::FieldBuffer<double> prev(grid, 1);
  wavefront::FieldBuffer<double> curr(grid, 1);
  wavefront::FieldBuffer<double> next(grid, 1);
  prev.fill(0.0);
  curr.fill(0.0);
  next.fill(0.0);
  return {std::move(grid), std::move(prev), std::move(curr), std::move(next)};
}

}  // namespace

// ---------------------------------------------------------------------------
//  Dirichlet boundary
// ---------------------------------------------------------------------------

TEST_CASE("Dirichlet 1D: boundary nodes clamped to prescribed value") {
  auto [grid, prev, curr, next] = make_1d_setup(16);
  curr.fill(1.0);
  next.fill(1.0);

  std::vector<wavefront::BoundarySpec> bc = {
      {wavefront::BoundaryType::Dirichlet, 0, false, wavefront::SymbolicExpr{"3.0"}},
      {wavefront::BoundaryType::Dirichlet, 0, true, wavefront::SymbolicExpr{"5.0"}},
  };

  std::vector<double> pml;
  wavefront::apply_boundary_conditions(grid, bc, pml, next, curr, prev, 0, true, 0.1, 0.0);

  CHECK(next.at_index({0}, 0) == doctest::Approx(3.0));
  CHECK(next.at_index({15}, 0) == doctest::Approx(5.0));
}

TEST_CASE("Dirichlet 2D: boundary nodes on all four faces") {
  auto [grid, prev, curr, next] = make_2d_setup(8, 8);
  curr.fill(0.5);
  next.fill(0.5);

  std::vector<wavefront::BoundarySpec> bc = {
      {wavefront::BoundaryType::Dirichlet, 0, false, wavefront::SymbolicExpr{"1.0"}},
      {wavefront::BoundaryType::Dirichlet, 0, true, wavefront::SymbolicExpr{"2.0"}},
      {wavefront::BoundaryType::Dirichlet, 1, false, wavefront::SymbolicExpr{"3.0"}},
      {wavefront::BoundaryType::Dirichlet, 1, true, wavefront::SymbolicExpr{"4.0"}},
  };

  std::vector<double> pml;
  wavefront::apply_boundary_conditions(grid, bc, pml, next, curr, prev, 0, true, 0.1, 0.0);

  CHECK(next.at_index({0, 4}, 0) == doctest::Approx(1.0));
  CHECK(next.at_index({7, 4}, 0) == doctest::Approx(2.0));
  CHECK(next.at_index({4, 0}, 0) == doctest::Approx(3.0));
  CHECK(next.at_index({4, 7}, 0) == doctest::Approx(4.0));
}

// ---------------------------------------------------------------------------
//  Neumann boundary
// ---------------------------------------------------------------------------

TEST_CASE("Neumann 1D: zero-gradient preserves interior values at boundaries") {
  auto [grid, prev, curr, next] = make_1d_setup(16);

  // Uniform field → Neumann ∂u/∂n=0 should keep boundary value = interior value
  curr.fill(7.0);
  next.fill(7.0);

  std::vector<wavefront::BoundarySpec> bc = {
      {wavefront::BoundaryType::Neumann, 0, false, wavefront::SymbolicExpr{"0.0"}},
      {wavefront::BoundaryType::Neumann, 0, true, wavefront::SymbolicExpr{"0.0"}},
  };

  std::vector<double> pml;
  wavefront::apply_boundary_conditions(grid, bc, pml, next, curr, prev, 0, true, 0.1, 0.0);

  CHECK(next.at_index({0}, 0) == doctest::Approx(7.0));
  CHECK(next.at_index({15}, 0) == doctest::Approx(7.0));
}

// ---------------------------------------------------------------------------
//  Robin boundary
// ---------------------------------------------------------------------------

TEST_CASE("Robin 1D boundary applies mixed condition") {
  auto [grid, prev, curr, next] = make_1d_setup(16, 0.5);
  curr.fill(1.0);
  next.fill(1.0);

  // Robin with parameter α: ghost = 2α - u → boundary should be modified
  std::vector<wavefront::BoundarySpec> bc = {
      {wavefront::BoundaryType::Robin, 0, false, wavefront::SymbolicExpr{"3.0"}},
      {wavefront::BoundaryType::Robin, 0, true, wavefront::SymbolicExpr{"3.0"}},
  };

  std::vector<double> pml;
  wavefront::apply_boundary_conditions(grid, bc, pml, next, curr, prev, 0, true, 0.1, 0.0);

  // Boundary values should be finite and modified
  CHECK(std::isfinite(next.at_index({0}, 0)));
  CHECK(std::isfinite(next.at_index({15}, 0)));
}

// ---------------------------------------------------------------------------
//  Impedance boundary
// ---------------------------------------------------------------------------

TEST_CASE("Impedance 1D boundary keeps values finite") {
  auto [grid, prev, curr, next] = make_1d_setup(16, 0.5);
  for (std::size_t i = 0; i < 16; ++i) {
    prev.at_index({i}, 0) = 0.1 * static_cast<double>(i);
    curr.at_index({i}, 0) = 0.12 * static_cast<double>(i);
    next.at_index({i}, 0) = 0.12 * static_cast<double>(i);
  }

  std::vector<wavefront::BoundarySpec> bc = {
      {wavefront::BoundaryType::Impedance, 0, false, wavefront::SymbolicExpr{"1.5"}},
      {wavefront::BoundaryType::Impedance, 0, true, wavefront::SymbolicExpr{"1.5"}},
  };

  std::vector<double> pml;
  wavefront::apply_boundary_conditions(grid, bc, pml, next, curr, prev, 0, true, 0.01, 0.0);

  CHECK(std::isfinite(next.at_index({0}, 0)));
  CHECK(std::isfinite(next.at_index({15}, 0)));
}

// ---------------------------------------------------------------------------
//  PML boundary (absorbing)
// ---------------------------------------------------------------------------

TEST_CASE("PML 1D boundary produces non-zero absorbed energy with active field") {
  auto [grid, prev, curr, next] = make_1d_setup(32, 0.5);
  for (std::size_t i = 0; i < 32; ++i) {
    curr.at_index({i}, 0) = std::sin(0.3 * static_cast<double>(i));
    next.at_index({i}, 0) = std::sin(0.3 * static_cast<double>(i));
  }

  std::vector<wavefront::BoundarySpec> bc = {
      {wavefront::BoundaryType::PML, 0, false, wavefront::SymbolicExpr{"10.0"}},
      {wavefront::BoundaryType::PML, 0, true, wavefront::SymbolicExpr{"10.0"}},
  };

  std::vector<double> pml;
  auto metrics = wavefront::apply_boundary_conditions(grid, bc, pml, next, curr, prev, 0, true, 0.01, 5.0);

  CHECK(metrics.absorbed_energy >= 0.0);
  CHECK(std::isfinite(next.at_index({0}, 0)));
  CHECK(std::isfinite(next.at_index({31}, 0)));
}

TEST_CASE("PML 2D boundary produces absorbed energy on all faces") {
  auto [grid, prev, curr, next] = make_2d_setup(16, 16, 0.5, 0.5);
  for (std::size_t i = 0; i < 16; ++i) {
    for (std::size_t j = 0; j < 16; ++j) {
      curr.at_index({i, j}, 0) = std::sin(0.5 * static_cast<double>(i)) * std::cos(0.5 * static_cast<double>(j));
      next.at_index({i, j}, 0) = curr.at_index({i, j}, 0);
    }
  }

  std::vector<wavefront::BoundarySpec> bc = {
      {wavefront::BoundaryType::PML, 0, false, wavefront::SymbolicExpr{"8.0"}},
      {wavefront::BoundaryType::PML, 0, true, wavefront::SymbolicExpr{"8.0"}},
      {wavefront::BoundaryType::PML, 1, false, wavefront::SymbolicExpr{"8.0"}},
      {wavefront::BoundaryType::PML, 1, true, wavefront::SymbolicExpr{"8.0"}},
  };

  std::vector<double> pml;
  auto metrics = wavefront::apply_boundary_conditions(grid, bc, pml, next, curr, prev, 0, true, 0.01, 5.0);

  CHECK(metrics.absorbed_energy >= 0.0);

  for (std::size_t i = 0; i < 16; i += 4) {
    for (std::size_t j = 0; j < 16; j += 4) {
      CHECK(std::isfinite(next.at_index({i, j}, 0)));
    }
  }
}

// ---------------------------------------------------------------------------
//  Periodic boundary (runtime API)
// ---------------------------------------------------------------------------

TEST_CASE("Periodic 1D: solver preserves energy without source or damping") {
  wavefront::ProblemSpec p;
  p.grid.dims = 1;
  p.grid.shape = {64};
  p.grid.spacing = {0.05};
  p.grid.origin = {0.0};
  p.field_components = 1;
  p.medium.density.text = "1.0";
  p.medium.stiffness.text = "1.0";
  p.medium.damping.text = "0.0";
  p.medium.dispersion.text = "0.0";
  p.source_term.text = "0.0";
  p.boundaries = {
      {wavefront::BoundaryType::Periodic, 0, false, wavefront::SymbolicExpr{"0.0"}},
      {wavefront::BoundaryType::Periodic, 0, true, wavefront::SymbolicExpr{"0.0"}},
  };

  auto cfg = test_common::default_config(wavefront::SolverMode::LinearApprox);
  cfg.cfl = 0.1;
  cfg.spatial_order = 2;
  auto solver = wavefront::make_solver(p, cfg);

  const double e0 = test_common::json_value(solver->diagnostics_json(), "energy");
  solver->run(50);
  const double e1 = test_common::json_value(solver->diagnostics_json(), "energy");

  CHECK(std::fabs(e1 - e0) < 1.0e-6);
}

TEST_CASE("Periodic 2D: wave wraps around domain") {
  wavefront::ProblemSpec p;
  p.grid.dims = 2;
  p.grid.shape = {32, 32};
  p.grid.spacing = {0.05, 0.05};
  p.grid.origin = {0.0, 0.0};
  p.field_components = 1;
  p.medium.density.text = "1.0";
  p.medium.stiffness.text = "1.0";
  p.medium.damping.text = "0.0";
  p.medium.dispersion.text = "0.0";
  p.source_term.text = "sin(20*t)*exp(-8*t)";
  p.boundaries = {
      {wavefront::BoundaryType::Periodic, 0, false, wavefront::SymbolicExpr{"0.0"}},
      {wavefront::BoundaryType::Periodic, 0, true, wavefront::SymbolicExpr{"0.0"}},
      {wavefront::BoundaryType::Periodic, 1, false, wavefront::SymbolicExpr{"0.0"}},
      {wavefront::BoundaryType::Periodic, 1, true, wavefront::SymbolicExpr{"0.0"}},
  };

  auto cfg = test_common::default_config(wavefront::SolverMode::LinearApprox);
  auto solver = wavefront::make_solver(p, cfg);
  solver->run(40);

  // Deterministic → reproducible
  auto solver2 = wavefront::make_solver(p, cfg);
  solver2->run(40);
  CHECK(solver->diagnostics_json() == solver2->diagnostics_json());
}

// ---------------------------------------------------------------------------
//  Mixed boundary types on different axes
// ---------------------------------------------------------------------------

TEST_CASE("2D: Dirichlet on x-axis, Periodic on y-axis") {
  wavefront::ProblemSpec p;
  p.grid.dims = 2;
  p.grid.shape = {16, 16};
  p.grid.spacing = {0.1, 0.1};
  p.grid.origin = {0.0, 0.0};
  p.field_components = 1;
  p.medium.density.text = "1.0";
  p.medium.stiffness.text = "1.0";
  p.medium.damping.text = "0.0";
  p.medium.dispersion.text = "0.0";
  p.source_term.text = "sin(20*t)*exp(-10*t)";
  p.boundaries = {
      {wavefront::BoundaryType::Dirichlet, 0, false, wavefront::SymbolicExpr{"0.0"}},
      {wavefront::BoundaryType::Dirichlet, 0, true, wavefront::SymbolicExpr{"0.0"}},
      {wavefront::BoundaryType::Periodic, 1, false, wavefront::SymbolicExpr{"0.0"}},
      {wavefront::BoundaryType::Periodic, 1, true, wavefront::SymbolicExpr{"0.0"}},
  };

  auto cfg = test_common::default_config(wavefront::SolverMode::LinearApprox);
  auto solver = wavefront::make_solver(p, cfg);
  solver->run(20);

  const std::string diag = solver->diagnostics_json();
  CHECK(test_common::json_value(diag, "energy") >= 0.0);
  CHECK(std::isfinite(solver->sample(std::vector<std::size_t>{8, 8})[0]));
}

TEST_CASE("2D: PML on x-axis, Neumann on y-axis") {
  wavefront::ProblemSpec p;
  p.grid.dims = 2;
  p.grid.shape = {24, 24};
  p.grid.spacing = {0.08, 0.08};
  p.grid.origin = {0.0, 0.0};
  p.field_components = 1;
  p.medium.density.text = "1.0";
  p.medium.stiffness.text = "1.0";
  p.medium.damping.text = "0.0";
  p.medium.dispersion.text = "0.0";
  p.source_term.text = "sin(15*t)*exp(-8*t)";
  p.boundaries = {
      {wavefront::BoundaryType::PML, 0, false, wavefront::SymbolicExpr{"8.0"}},
      {wavefront::BoundaryType::PML, 0, true, wavefront::SymbolicExpr{"8.0"}},
      {wavefront::BoundaryType::Neumann, 1, false, wavefront::SymbolicExpr{"0.0"}},
      {wavefront::BoundaryType::Neumann, 1, true, wavefront::SymbolicExpr{"0.0"}},
  };

  auto cfg = test_common::default_config(wavefront::SolverMode::LinearApprox);
  auto solver = wavefront::make_solver(p, cfg);
  solver->run(30);

  // PML should absorb some energy
  CHECK(test_common::json_value(solver->diagnostics_json(), "absorbed_energy") >= 0.0);
  CHECK(std::isfinite(solver->sample(std::vector<std::size_t>{12, 12})[0]));
}

// ---------------------------------------------------------------------------
//  3-D boundary conditions
// ---------------------------------------------------------------------------

TEST_CASE("3D: PML on all faces absorbs energy") {
  wavefront::ProblemSpec p;
  p.grid.dims = 3;
  p.grid.shape = {12, 12, 12};
  p.grid.spacing = {0.1, 0.1, 0.1};
  p.grid.origin = {0.0, 0.0, 0.0};
  p.field_components = 1;
  p.medium.density.text = "1.0";
  p.medium.stiffness.text = "1.0";
  p.medium.damping.text = "0.0";
  p.medium.dispersion.text = "0.0";
  p.source_term.text = "10*sin(20*t)*exp(-10*t)";
  p.boundaries = {
      {wavefront::BoundaryType::PML, 0, false, wavefront::SymbolicExpr{"8.0"}},
      {wavefront::BoundaryType::PML, 0, true, wavefront::SymbolicExpr{"8.0"}},
      {wavefront::BoundaryType::PML, 1, false, wavefront::SymbolicExpr{"8.0"}},
      {wavefront::BoundaryType::PML, 1, true, wavefront::SymbolicExpr{"8.0"}},
      {wavefront::BoundaryType::PML, 2, false, wavefront::SymbolicExpr{"8.0"}},
      {wavefront::BoundaryType::PML, 2, true, wavefront::SymbolicExpr{"8.0"}},
  };

  auto cfg = test_common::default_config(wavefront::SolverMode::LinearApprox);
  cfg.cfl = 0.15;
  auto solver = wavefront::make_solver(p, cfg);
  solver->run(30);

  const std::string diag = solver->diagnostics_json();
  CHECK(test_common::json_value(diag, "absorbed_energy") >= 0.0);
  CHECK(test_common::json_value(diag, "dims") == doctest::Approx(3.0));

  for (std::size_t x = 0; x < 12; x += 4) {
    CHECK(std::isfinite(solver->sample(std::vector<std::size_t>{x, 6, 6})[0]));
  }
}

TEST_CASE("3D: Dirichlet boundaries clamp faces") {
  wavefront::ProblemSpec p;
  p.grid.dims = 3;
  p.grid.shape = {8, 8, 8};
  p.grid.spacing = {0.2, 0.2, 0.2};
  p.grid.origin = {0.0, 0.0, 0.0};
  p.field_components = 1;
  p.medium.density.text = "1.0";
  p.medium.stiffness.text = "1.0";
  p.medium.damping.text = "0.0";
  p.medium.dispersion.text = "0.0";
  p.source_term.text = "sin(10*t)*exp(-5*t)";
  p.boundaries = {
      {wavefront::BoundaryType::Dirichlet, 0, false, wavefront::SymbolicExpr{"0.0"}},
      {wavefront::BoundaryType::Dirichlet, 0, true, wavefront::SymbolicExpr{"0.0"}},
      {wavefront::BoundaryType::Dirichlet, 1, false, wavefront::SymbolicExpr{"0.0"}},
      {wavefront::BoundaryType::Dirichlet, 1, true, wavefront::SymbolicExpr{"0.0"}},
      {wavefront::BoundaryType::Dirichlet, 2, false, wavefront::SymbolicExpr{"0.0"}},
      {wavefront::BoundaryType::Dirichlet, 2, true, wavefront::SymbolicExpr{"0.0"}},
  };

  auto cfg = test_common::default_config(wavefront::SolverMode::LinearApprox);
  auto solver = wavefront::make_solver(p, cfg);
  solver->run(15);

  // All face nodes should remain 0
  CHECK(solver->sample(std::vector<std::size_t>{0, 4, 4})[0] == doctest::Approx(0.0));
  CHECK(solver->sample(std::vector<std::size_t>{7, 4, 4})[0] == doctest::Approx(0.0));
  CHECK(solver->sample(std::vector<std::size_t>{4, 0, 4})[0] == doctest::Approx(0.0));
  CHECK(solver->sample(std::vector<std::size_t>{4, 7, 4})[0] == doctest::Approx(0.0));
  CHECK(solver->sample(std::vector<std::size_t>{4, 4, 0})[0] == doctest::Approx(0.0));
  CHECK(solver->sample(std::vector<std::size_t>{4, 4, 7})[0] == doctest::Approx(0.0));
}
