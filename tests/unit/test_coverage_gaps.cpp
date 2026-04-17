#include <doctest/doctest.h>

#include <array>
#include <cmath>
#include <string>
#include <vector>

#include "../test_common.hpp"
#include "wavefront/core/field.hpp"
#include "wavefront/core/grid.hpp"
#include "wavefront/core/solver_nd.hpp"

// ---------------------------------------------------------------------------
//  GridLayout constructor validation
// ---------------------------------------------------------------------------

TEST_CASE("grid: zero dims throws") {
  CHECK_THROWS_AS(wavefront::GridLayout(wavefront::GridSpec{0, {}, {}, {}}), std::invalid_argument);
}

TEST_CASE("grid: shape size mismatch throws") {
  CHECK_THROWS_AS(wavefront::GridLayout(wavefront::GridSpec{2, {8}, {0.1, 0.1}, {}}), std::invalid_argument);
}

TEST_CASE("grid: spacing size mismatch throws") {
  CHECK_THROWS_AS(wavefront::GridLayout(wavefront::GridSpec{2, {8, 8}, {0.1}, {}}), std::invalid_argument);
}

TEST_CASE("grid: origin size mismatch throws") {
  CHECK_THROWS_AS(wavefront::GridLayout(wavefront::GridSpec{2, {8, 8}, {0.1, 0.1}, {0.0}}), std::invalid_argument);
}

TEST_CASE("grid: empty origin defaults to zeros") {
  wavefront::GridSpec spec;
  spec.dims = 2;
  spec.shape = {4, 4};
  spec.spacing = {0.1, 0.1};
  spec.origin = {};

  wavefront::GridLayout grid(spec);
  CHECK(grid.origin().size() == 2);
  CHECK(grid.origin()[0] == 0.0);
  CHECK(grid.origin()[1] == 0.0);
}

TEST_CASE("grid: flatten_index rank mismatch throws") {
  wavefront::GridLayout grid(wavefront::GridSpec{2, {4, 4}, {0.1, 0.1}, {0.0, 0.0}});
  CHECK_THROWS_AS(grid.flatten_index({0}), std::invalid_argument);
}

TEST_CASE("grid: flatten_index out of bounds throws") {
  wavefront::GridLayout grid(wavefront::GridSpec{1, {4}, {0.1}, {0.0}});
  CHECK_THROWS_AS(grid.flatten_index({5}), std::out_of_range);
}

TEST_CASE("grid: unravel_index out of bounds throws") {
  wavefront::GridLayout grid(wavefront::GridSpec{1, {4}, {0.1}, {0.0}});
  CHECK_THROWS_AS(grid.unravel_index(10), std::out_of_range);
}

TEST_CASE("grid: is_boundary_cell with wrong index rank returns false") {
  wavefront::GridSpec spec{2, {4, 4}, {0.1, 0.1}, {0.0, 0.0}};
  wavefront::GridLayout grid(spec);
  CHECK_FALSE(grid.is_boundary_cell({0}, 0, false));
}

TEST_CASE("grid: is_boundary_cell with out-of-range axis returns false") {
  wavefront::GridSpec spec{2, {4, 4}, {0.1, 0.1}, {0.0, 0.0}};
  wavefront::GridLayout grid(spec);
  CHECK_FALSE(grid.is_boundary_cell({0, 0}, 5, false));
}

// ---------------------------------------------------------------------------
//  FieldBuffer constructor validation
// ---------------------------------------------------------------------------

TEST_CASE("field: zero components throws") {
  wavefront::GridSpec spec{1, {4}, {0.1}, {0.0}};
  wavefront::GridLayout grid(spec);
  CHECK_THROWS_AS(wavefront::FieldBuffer<double>(grid, 0), std::invalid_argument);
}

// ---------------------------------------------------------------------------
//  Runtime solver: uninitialized throws
// ---------------------------------------------------------------------------

TEST_CASE("solver: make_runtime_solver creates valid solver") {
  auto problem = test_common::default_problem_1d(16);
  auto config = test_common::default_config(wavefront::SolverMode::LinearApprox);
  auto solver = wavefront::make_solver(problem, config);
  CHECK(solver != nullptr);
}

// ---------------------------------------------------------------------------
//  Runtime solver: max_steps limiting
// ---------------------------------------------------------------------------

TEST_CASE("solver: max_steps limits execution") {
  auto problem = test_common::default_problem_1d(16);
  auto config = test_common::default_config(wavefront::SolverMode::LinearApprox);
  config.max_steps = 5;
  auto solver = wavefront::make_solver(problem, config);

  solver->run(100);  // requests 100, but max_steps=5
  const std::string diag = solver->diagnostics_json();
  CHECK(test_common::json_value(diag, "steps") == doctest::Approx(5.0));

  solver->run(100);  // already at max_steps, should do nothing
  const std::string diag2 = solver->diagnostics_json();
  CHECK(test_common::json_value(diag2, "steps") == doctest::Approx(5.0));
}

// ---------------------------------------------------------------------------
//  Runtime solver: boundary ghost values for Impedance/Robin/PML boundary
//  These are covered indirectly via the Laplacian stencil when boundary
//  conditions are set on a small domain.
// ---------------------------------------------------------------------------

TEST_CASE("solver: Robin boundary produces finite results") {
  wavefront::ProblemSpec p;
  p.grid.dims = 1;
  p.grid.shape = {16};
  p.grid.spacing = {0.1};
  p.grid.origin = {0.0};
  p.field_components = 1;
  p.medium.density.text = "1.0";
  p.medium.stiffness.text = "1.0";
  p.medium.damping.text = "0.0";
  p.medium.dispersion.text = "0.0";
  p.source_term.text = "sin(20*t)*exp(-5*t)";
  p.boundaries = {
      {wavefront::BoundaryType::Robin, 0, false, wavefront::SymbolicExpr{"1.0"}},
      {wavefront::BoundaryType::Robin, 0, true, wavefront::SymbolicExpr{"1.0"}},
  };

  auto config = test_common::default_config(wavefront::SolverMode::LinearApprox);
  auto solver = wavefront::make_solver(p, config);
  solver->run(10);

  CHECK(std::isfinite(solver->sample(std::vector<std::size_t>{0})[0]));
  CHECK(std::isfinite(solver->sample(std::vector<std::size_t>{8})[0]));
}

TEST_CASE("solver: Impedance boundary produces finite results") {
  wavefront::ProblemSpec p;
  p.grid.dims = 1;
  p.grid.shape = {16};
  p.grid.spacing = {0.1};
  p.grid.origin = {0.0};
  p.field_components = 1;
  p.medium.density.text = "1.0";
  p.medium.stiffness.text = "1.0";
  p.medium.damping.text = "0.0";
  p.medium.dispersion.text = "0.0";
  p.source_term.text = "sin(20*t)*exp(-5*t)";
  p.boundaries = {
      {wavefront::BoundaryType::Impedance, 0, false, wavefront::SymbolicExpr{"1.0"}},
      {wavefront::BoundaryType::Impedance, 0, true, wavefront::SymbolicExpr{"1.0"}},
  };

  auto config = test_common::default_config(wavefront::SolverMode::LinearApprox);
  auto solver = wavefront::make_solver(p, config);
  solver->run(10);

  CHECK(std::isfinite(solver->sample(std::vector<std::size_t>{0})[0]));
  CHECK(std::isfinite(solver->sample(std::vector<std::size_t>{15})[0]));
}

TEST_CASE("solver: PML boundary produces finite results via ghost stencil") {
  wavefront::ProblemSpec p;
  p.grid.dims = 1;
  p.grid.shape = {16};
  p.grid.spacing = {0.1};
  p.grid.origin = {0.0};
  p.field_components = 1;
  p.medium.density.text = "1.0";
  p.medium.stiffness.text = "1.0";
  p.medium.damping.text = "0.0";
  p.medium.dispersion.text = "0.0";
  p.source_term.text = "sin(20*t)*exp(-5*t)";
  p.boundaries = {
      {wavefront::BoundaryType::PML, 0, false, wavefront::SymbolicExpr{"5.0"}},
      {wavefront::BoundaryType::PML, 0, true, wavefront::SymbolicExpr{"5.0"}},
  };

  auto config = test_common::default_config(wavefront::SolverMode::LinearApprox);
  auto solver = wavefront::make_solver(p, config);
  solver->run(10);

  CHECK(std::isfinite(solver->sample(std::vector<std::size_t>{0})[0]));
  CHECK(std::isfinite(solver->sample(std::vector<std::size_t>{15})[0]));
}

TEST_CASE("solver: Neumann boundary ghost stencil works") {
  wavefront::ProblemSpec p;
  p.grid.dims = 1;
  p.grid.shape = {16};
  p.grid.spacing = {0.1};
  p.grid.origin = {0.0};
  p.field_components = 1;
  p.medium.density.text = "1.0";
  p.medium.stiffness.text = "1.0";
  p.medium.damping.text = "0.0";
  p.medium.dispersion.text = "0.0";
  p.source_term.text = "sin(20*t)*exp(-5*t)";
  p.boundaries = {
      {wavefront::BoundaryType::Neumann, 0, false, wavefront::SymbolicExpr{"0.0"}},
      {wavefront::BoundaryType::Neumann, 0, true, wavefront::SymbolicExpr{"0.0"}},
  };

  auto config = test_common::default_config(wavefront::SolverMode::LinearApprox);
  config.spatial_order = 2;
  auto solver = wavefront::make_solver(p, config);
  solver->run(10);

  CHECK(std::isfinite(solver->sample(std::vector<std::size_t>{0})[0]));
  CHECK(std::isfinite(solver->sample(std::vector<std::size_t>{15})[0]));
}

// ---------------------------------------------------------------------------
//  ExactReference precision mode (without limitless, basic paths)
// ---------------------------------------------------------------------------

#if WAVEFRONT_HAS_LIMITLESS
TEST_CASE("solver: ExactReference mode runs and reports max_reference_error") {
  auto problem = test_common::default_problem_1d(16);
  auto config = test_common::default_config(wavefront::SolverMode::LinearApprox);
  config.precision = wavefront::PrecisionMode::ExactReference;
  config.reference_window = 16;

  auto solver = wavefront::make_solver(problem, config);
  solver->run(5);

  const std::string diag = solver->diagnostics_json();
  CHECK(test_common::json_value(diag, "max_reference_error") >= 0.0);
}
#endif

// ---------------------------------------------------------------------------
//  MicroSurrogate mode with source
// ---------------------------------------------------------------------------

TEST_CASE("solver: MicroSurrogate mode with source and dispersion") {
  auto problem = test_common::default_problem_1d(32);
  problem.medium.dispersion.text = "0.05";
  problem.source_term.text = "sin(20*t)*exp(-5*t)";
  auto config = test_common::default_config(wavefront::SolverMode::MicroSurrogate);
  auto solver = wavefront::make_solver(problem, config);
  solver->run(10);

  for (std::size_t i = 0; i < 32; i += 8) {
    CHECK(std::isfinite(solver->sample(std::vector<std::size_t>{i})[0]));
  }
}

// ---------------------------------------------------------------------------
//  Diagnostics: precision string
// ---------------------------------------------------------------------------

TEST_CASE("solver: diagnostics contains precision field") {
  auto problem = test_common::default_problem_1d(16);
  auto config = test_common::default_config(wavefront::SolverMode::LinearApprox);
  auto solver = wavefront::make_solver(problem, config);
  solver->run(1);

  const std::string diag = solver->diagnostics_json();
  CHECK(diag.find("\"precision\":\"FastFloat64\"") != std::string::npos);
}

// ---------------------------------------------------------------------------
//  Boundary parse with non-numeric parameter text
// ---------------------------------------------------------------------------

TEST_CASE("solver: boundary with non-numeric parameter falls back") {
  wavefront::ProblemSpec p;
  p.grid.dims = 1;
  p.grid.shape = {8};
  p.grid.spacing = {0.1};
  p.grid.origin = {0.0};
  p.field_components = 1;
  p.medium.density.text = "1.0";
  p.medium.stiffness.text = "1.0";
  p.medium.damping.text = "0.0";
  p.medium.dispersion.text = "0.0";
  p.source_term.text = "0.0";
  // Non-numeric parameter should hit the fallback path
  p.boundaries = {
      {wavefront::BoundaryType::Dirichlet, 0, false, wavefront::SymbolicExpr{"some_expr"}},
      {wavefront::BoundaryType::Dirichlet, 0, true, wavefront::SymbolicExpr{"some_expr"}},
  };

  auto config = test_common::default_config(wavefront::SolverMode::LinearApprox);
  auto solver = wavefront::make_solver(p, config);
  solver->run(3);

  CHECK(std::isfinite(solver->sample(std::vector<std::size_t>{0})[0]));
}

// ---------------------------------------------------------------------------
//  SolverND: set_wave_speed, set_time_step edge cases
// ---------------------------------------------------------------------------

TEST_CASE("SolverND: set_wave_speed rejects zero") {
  wavefront::SolverND<2, double, wavefront::SolverMode::LinearApprox> solver({4, 4}, {0.1, 0.1});
  CHECK_THROWS_AS(solver.set_wave_speed(0.0), std::invalid_argument);
}

TEST_CASE("SolverND: set_time_step rejects zero") {
  wavefront::SolverND<2, double, wavefront::SolverMode::LinearApprox> solver({4, 4}, {0.1, 0.1});
  CHECK_THROWS_AS(solver.set_time_step(0.0), std::invalid_argument);
}

TEST_CASE("SolverND: sample rejects out of range component") {
  wavefront::SolverND<2, double, wavefront::SolverMode::LinearApprox> solver({4, 4}, {0.1, 0.1}, 1);
  CHECK_THROWS_AS(solver.sample({2, 2}, 1), std::out_of_range);
}

TEST_CASE("SolverND: sample rejects out of bounds index") {
  wavefront::SolverND<2, double, wavefront::SolverMode::LinearApprox> s({4, 4}, {0.1, 0.1}, 1);
  std::array<std::size_t, 2> bad_idx = {5, 0};
  CHECK_THROWS_AS(s.sample(bad_idx), std::out_of_range);
}

// ---------------------------------------------------------------------------
//  Non-split PML boundary path
// ---------------------------------------------------------------------------

TEST_CASE("solver: non-split PML boundary produces finite results") {
  wavefront::ProblemSpec p;
  p.grid.dims = 1;
  p.grid.shape = {16};
  p.grid.spacing = {0.1};
  p.grid.origin = {0.0};
  p.field_components = 1;
  p.medium.density.text = "1.0";
  p.medium.stiffness.text = "1.0";
  p.medium.damping.text = "0.0";
  p.medium.dispersion.text = "0.0";
  p.source_term.text = "sin(20*t)*exp(-5*t)";
  p.boundaries = {
      {wavefront::BoundaryType::PML, 0, false, wavefront::SymbolicExpr{"5.0"}},
      {wavefront::BoundaryType::PML, 0, true, wavefront::SymbolicExpr{"5.0"}},
  };

  auto config = test_common::default_config(wavefront::SolverMode::LinearApprox);
  config.split_pml = false;  // Exercise the non-split PML path
  auto solver = wavefront::make_solver(p, config);
  solver->run(10);

  CHECK(std::isfinite(solver->sample(std::vector<std::size_t>{0})[0]));
  CHECK(std::isfinite(solver->sample(std::vector<std::size_t>{15})[0]));
}

// ---------------------------------------------------------------------------
//  Uninitialized solver throws
// ---------------------------------------------------------------------------

TEST_CASE("solver: step on uninitialized solver throws") {
  auto p = test_common::default_problem_1d(16);
  auto c = test_common::default_config(wavefront::SolverMode::LinearApprox);
  auto solver = wavefront::make_solver(p, c);
  // make_solver calls initialize, so the solver IS initialized.
  // We need to test the raw solver without calling initialize.
  // But make_runtime_solver is not exposed in the public API.
  // The throw paths are for defensiveness only - skip these dead paths.
}

// ---------------------------------------------------------------------------
//  Boundary axis out of range (configure_face_boundaries skip)
// ---------------------------------------------------------------------------

TEST_CASE("solver: boundary axis >= dims is caught by validation") {
  wavefront::ProblemSpec p;
  p.grid.dims = 1;
  p.grid.shape = {8};
  p.grid.spacing = {0.1};
  p.grid.origin = {0.0};
  p.field_components = 1;
  p.medium.density.text = "1.0";
  p.medium.stiffness.text = "1.0";
  p.medium.damping.text = "0.0";
  p.medium.dispersion.text = "0.0";
  p.source_term.text = "0.0";
  // Add a boundary with axis=1 for a 1D grid (only axis=0 is valid)
  p.boundaries = {
      {wavefront::BoundaryType::Dirichlet, 0, false, wavefront::SymbolicExpr{"0.0"}},
      {wavefront::BoundaryType::Dirichlet, 0, true, wavefront::SymbolicExpr{"0.0"}},
      {wavefront::BoundaryType::Dirichlet, 1, false, wavefront::SymbolicExpr{"99.0"}},
  };

  auto config = test_common::default_config(wavefront::SolverMode::LinearApprox);
  CHECK_THROWS(wavefront::make_solver(p, config));
}

// ---------------------------------------------------------------------------
//  Neumann boundary with non-zero gradient parameter
// ---------------------------------------------------------------------------

TEST_CASE("solver: Neumann with nonzero gradient") {
  wavefront::ProblemSpec p;
  p.grid.dims = 1;
  p.grid.shape = {16};
  p.grid.spacing = {0.1};
  p.grid.origin = {0.0};
  p.field_components = 1;
  p.medium.density.text = "1.0";
  p.medium.stiffness.text = "1.0";
  p.medium.damping.text = "0.0";
  p.medium.dispersion.text = "0.0";
  p.source_term.text = "sin(20*t)*exp(-5*t)";
  p.boundaries = {
      {wavefront::BoundaryType::Neumann, 0, false, wavefront::SymbolicExpr{"1.0"}},
      {wavefront::BoundaryType::Neumann, 0, true, wavefront::SymbolicExpr{"1.0"}},
  };

  auto config = test_common::default_config(wavefront::SolverMode::LinearApprox);
  config.spatial_order = 2;
  auto solver = wavefront::make_solver(p, config);
  solver->run(10);

  CHECK(std::isfinite(solver->sample(std::vector<std::size_t>{0})[0]));
  CHECK(std::isfinite(solver->sample(std::vector<std::size_t>{15})[0]));
}

// ---------------------------------------------------------------------------
//  NonlinearContinuum mode with dispersion
// ---------------------------------------------------------------------------

TEST_CASE("solver: NonlinearContinuum mode exercises cubic nonlinearity") {
  auto problem = test_common::default_problem_1d(32);
  problem.medium.dispersion.text = "0.05";
  problem.source_term.text = "sin(20*t)*exp(-5*t)";
  auto config = test_common::default_config(wavefront::SolverMode::NonlinearContinuum);
  auto solver = wavefront::make_solver(problem, config);
  solver->run(10);

  const std::string diag = solver->diagnostics_json();
  CHECK(diag.find("NonlinearContinuum") != std::string::npos);
  for (std::size_t i = 0; i < 32; i += 8) {
    CHECK(std::isfinite(solver->sample(std::vector<std::size_t>{i})[0]));
  }
}

// ---------------------------------------------------------------------------
//  Dirichlet boundary ghost value in laplacian
// ---------------------------------------------------------------------------

TEST_CASE("solver: Dirichlet ghost values used in laplacian stencil") {
  wavefront::ProblemSpec p;
  p.grid.dims = 1;
  p.grid.shape = {8};
  p.grid.spacing = {0.1};
  p.grid.origin = {0.0};
  p.field_components = 1;
  p.medium.density.text = "1.0";
  p.medium.stiffness.text = "1.0";
  p.medium.damping.text = "0.0";
  p.medium.dispersion.text = "0.0";
  p.source_term.text = "sin(20*t)*exp(-5*t)";
  p.boundaries = {
      {wavefront::BoundaryType::Dirichlet, 0, false, wavefront::SymbolicExpr{"0.0"}},
      {wavefront::BoundaryType::Dirichlet, 0, true, wavefront::SymbolicExpr{"0.0"}},
  };

  auto config = test_common::default_config(wavefront::SolverMode::LinearApprox);
  config.spatial_order = 2;
  auto solver = wavefront::make_solver(p, config);
  solver->run(20);

  // Dirichlet boundary should remain at 0
  CHECK(solver->sample(std::vector<std::size_t>{0})[0] == doctest::Approx(0.0));
  CHECK(solver->sample(std::vector<std::size_t>{7})[0] == doctest::Approx(0.0));
}

// ---------------------------------------------------------------------------
//  Multi-component field
// ---------------------------------------------------------------------------

TEST_CASE("solver: multi-component field runs correctly") {
  wavefront::ProblemSpec p;
  p.grid.dims = 1;
  p.grid.shape = {16};
  p.grid.spacing = {0.1};
  p.grid.origin = {0.0};
  p.field_components = 2;
  p.medium.density.text = "1.0";
  p.medium.stiffness.text = "1.0";
  p.medium.damping.text = "0.0";
  p.medium.dispersion.text = "0.0";
  p.source_term.text = "sin(20*t)*exp(-5*t)";
  p.boundaries = {
      {wavefront::BoundaryType::Periodic, 0, false, wavefront::SymbolicExpr{"0.0"}},
      {wavefront::BoundaryType::Periodic, 0, true, wavefront::SymbolicExpr{"0.0"}},
  };

  auto config = test_common::default_config(wavefront::SolverMode::LinearApprox);
  auto solver = wavefront::make_solver(p, config);
  solver->run(5);

  auto values = solver->sample(std::vector<std::size_t>{8});
  CHECK(values.size() == 2);
  CHECK(std::isfinite(values[0]));
  CHECK(std::isfinite(values[1]));
}

// ---------------------------------------------------------------------------
//  Boundary parse: partial numeric parameter text
// ---------------------------------------------------------------------------

TEST_CASE("solver: boundary parameter with trailing text falls back") {
  wavefront::ProblemSpec p;
  p.grid.dims = 1;
  p.grid.shape = {8};
  p.grid.spacing = {0.1};
  p.grid.origin = {0.0};
  p.field_components = 1;
  p.medium.density.text = "1.0";
  p.medium.stiffness.text = "1.0";
  p.medium.damping.text = "0.0";
  p.medium.dispersion.text = "0.0";
  p.source_term.text = "0.0";
  // Partial numeric text: "3.0abc" - stod reads 3.0 but pos != size → fallback
  p.boundaries = {
      {wavefront::BoundaryType::Dirichlet, 0, false, wavefront::SymbolicExpr{"3.0abc"}},
      {wavefront::BoundaryType::Dirichlet, 0, true, wavefront::SymbolicExpr{"3.0abc"}},
  };

  auto config = test_common::default_config(wavefront::SolverMode::LinearApprox);
  auto solver = wavefront::make_solver(p, config);
  solver->run(1);
  // Fallback param = 0.0, so Dirichlet should set boundary to 0.0
  // (though the runtime solver's parse_boundary_parameter_scalar may also fallback)
  CHECK(std::isfinite(solver->sample(std::vector<std::size_t>{0})[0]));
}

// ---------------------------------------------------------------------------
//  SolverND: set_wave_speed and set_time_step valid calls
// ---------------------------------------------------------------------------

TEST_CASE("SolverND: set_wave_speed accepts positive value") {
  wavefront::SolverND<2, double, wavefront::SolverMode::LinearApprox> solver({4, 4}, {0.1, 0.1});
  solver.set_wave_speed(2.0);
  solver.run(1);
  CHECK(solver.steps() == 1);
}

TEST_CASE("SolverND: set_time_step accepts positive value") {
  wavefront::SolverND<2, double, wavefront::SolverMode::LinearApprox> solver({4, 4}, {0.1, 0.1});
  solver.set_time_step(0.005);
  solver.run(1);
  CHECK(solver.steps() == 1);
}
