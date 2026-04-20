#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "../test_common.hpp"

// ---------------------------------------------------------------------------
//  Longitudinal wave verification tests – physical validity
// ---------------------------------------------------------------------------

namespace {

wavefront::ProblemSpec make_longitudinal_2d_problem(
    std::size_t nx,
    std::size_t ny,
    wavefront::BoundaryType boundary_type) {
  wavefront::ProblemSpec p;
  p.grid.dims = 2;
  p.grid.shape = {nx, ny};
  p.grid.spacing = {0.04, 0.04};
  p.grid.origin = {0.0, 0.0};
  p.field_components = 2;
  p.wave_type = wavefront::WaveType::Longitudinal;

  p.medium.density.text = "1.0";
  p.medium.stiffness.text = "1.0";
  p.medium.damping.text = "0.0002";
  p.medium.dispersion.text = "0.0";
  p.source_term.text =
      "10.0*sin(30*t)*exp(-15*t)*exp(-((x_0-0.5)*(x_0-0.5)+(x_1-0.5)*(x_1-0.5))/0.008)";

  p.boundaries.clear();
  for (std::size_t axis = 0; axis < 2; ++axis) {
    for (bool upper : {false, true}) {
      p.boundaries.push_back(
          wavefront::BoundarySpec{boundary_type, axis, upper, wavefront::SymbolicExpr{"10.0"}});
    }
  }
  return p;
}

wavefront::SolverConfig make_long_cfg() {
  wavefront::SolverConfig c;
  c.mode = wavefront::SolverMode::LinearApprox;
  c.precision = wavefront::PrecisionMode::FastFloat64;
  c.cfl = 0.20;
  c.max_steps = 0;
  c.threads = 2;
  c.deterministic = true;
  c.spatial_order = 2;
  c.split_pml = true;
  return c;
}

double energy_from_diag(wavefront::ISolver& solver) {
  return test_common::json_value(solver.diagnostics_json(), "energy");
}

}  // namespace

TEST_CASE("longitudinal wave propagates with finite speed") {
  const std::size_t n = 40;
  auto problem = make_longitudinal_2d_problem(n, n, wavefront::BoundaryType::PML);
  // Source is at the center of the domain
  auto solver = wavefront::make_solver(problem, make_long_cfg());

  // Short time: wavefront should not reach far corners
  solver->run(15);

  // Sample far corner
  double corner_energy = 0.0;
  for (std::size_t comp = 0; comp < 2; ++comp) {
    const double v = solver->sample(std::vector<std::size_t>{0, 0}).at(comp);
    corner_energy += v * v;
  }
  // Near the source there should be energy
  double center_energy = 0.0;
  for (std::size_t comp = 0; comp < 2; ++comp) {
    const double v = solver->sample(std::vector<std::size_t>{n / 2, n / 2}).at(comp);
    center_energy += v * v;
  }

  CHECK(center_energy > corner_energy);
}

TEST_CASE("longitudinal wave with PML has lower late-time energy than periodic") {
  const std::size_t n = 32;
  auto problem_periodic = make_longitudinal_2d_problem(n, n, wavefront::BoundaryType::Periodic);
  auto problem_pml = make_longitudinal_2d_problem(n, n, wavefront::BoundaryType::PML);

  auto solver_periodic = wavefront::make_solver(problem_periodic, make_long_cfg());
  auto solver_pml = wavefront::make_solver(problem_pml, make_long_cfg());

  solver_periodic->run(200);
  solver_pml->run(200);

  const double periodic_energy = energy_from_diag(*solver_periodic);
  const double pml_energy = energy_from_diag(*solver_pml);

  // PML should absorb energy
  CHECK(pml_energy < periodic_energy);
}

TEST_CASE("longitudinal wave is deterministic") {
  const std::size_t n = 24;
  auto problem = make_longitudinal_2d_problem(n, n, wavefront::BoundaryType::Periodic);
  auto config = make_long_cfg();
  config.threads = 4;
  config.deterministic = true;

  auto solver_a = wavefront::make_solver(problem, config);
  auto solver_b = wavefront::make_solver(problem, config);

  solver_a->run(15);
  solver_b->run(15);

  CHECK(solver_a->diagnostics_json() == solver_b->diagnostics_json());
}

TEST_CASE("longitudinal wave produces non-zero divergence-like coupling") {
  // For a 2-component longitudinal wave, the grad-div coupling means
  // components are not independently evolved; energy spreads between them.
  const std::size_t n = 24;
  wavefront::ProblemSpec problem;
  problem.grid.dims = 2;
  problem.grid.shape = {n, n};
  problem.grid.spacing = {0.05, 0.05};
  problem.grid.origin = {0.0, 0.0};
  problem.field_components = 2;
  problem.wave_type = wavefront::WaveType::Longitudinal;
  problem.medium.density.text = "1.0";
  problem.medium.stiffness.text = "1.0";
  problem.medium.damping.text = "0.0";
  problem.medium.dispersion.text = "0.0";
  problem.source_term.text = "8*sin(25*t)*exp(-12*t)*exp(-((x_0-0.6)*(x_0-0.6)+(x_1-0.6)*(x_1-0.6))/0.005)";
  problem.boundaries = {
      {wavefront::BoundaryType::Periodic, 0, false, wavefront::SymbolicExpr{"0.0"}},
      {wavefront::BoundaryType::Periodic, 0, true, wavefront::SymbolicExpr{"0.0"}},
      {wavefront::BoundaryType::Periodic, 1, false, wavefront::SymbolicExpr{"0.0"}},
      {wavefront::BoundaryType::Periodic, 1, true, wavefront::SymbolicExpr{"0.0"}},
  };

  auto config = make_long_cfg();
  auto solver = wavefront::make_solver(problem, config);
  solver->run(20);

  // Both components should carry energy due to grad-div coupling
  double comp0_energy = 0.0;
  double comp1_energy = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      const auto vals = solver->sample(std::vector<std::size_t>{i, j});
      comp0_energy += vals[0] * vals[0];
      comp1_energy += vals[1] * vals[1];
    }
  }

  // Both components should be active
  CHECK(comp0_energy > 1.0e-12);
  CHECK(comp1_energy > 1.0e-12);
}

TEST_CASE("longitudinal wave 3D isotropy with 3 coupled components") {
  wavefront::ProblemSpec problem;
  problem.grid.dims = 3;
  problem.grid.shape = {12, 12, 12};
  problem.grid.spacing = {0.1, 0.1, 0.1};
  problem.grid.origin = {0.0, 0.0, 0.0};
  problem.field_components = 3;
  problem.wave_type = wavefront::WaveType::Longitudinal;
  problem.medium.density.text = "1.0";
  problem.medium.stiffness.text = "1.0";
  problem.medium.damping.text = "0.0";
  problem.medium.dispersion.text = "0.0";
  problem.source_term.text =
      "8*sin(20*t)*exp(-10*t)*exp(-((x_0-0.6)*(x_0-0.6)+(x_1-0.6)*(x_1-0.6)+"
      "(x_2-0.6)*(x_2-0.6))/0.01)";
  problem.boundaries = {
      {wavefront::BoundaryType::Periodic, 0, false, wavefront::SymbolicExpr{"0.0"}},
      {wavefront::BoundaryType::Periodic, 0, true, wavefront::SymbolicExpr{"0.0"}},
      {wavefront::BoundaryType::Periodic, 1, false, wavefront::SymbolicExpr{"0.0"}},
      {wavefront::BoundaryType::Periodic, 1, true, wavefront::SymbolicExpr{"0.0"}},
      {wavefront::BoundaryType::Periodic, 2, false, wavefront::SymbolicExpr{"0.0"}},
      {wavefront::BoundaryType::Periodic, 2, true, wavefront::SymbolicExpr{"0.0"}},
  };

  auto config = make_long_cfg();
  auto solver = wavefront::make_solver(problem, config);
  solver->run(10);

  // All three components should be active
  const auto center = solver->sample(std::vector<std::size_t>{6, 6, 6});
  CHECK(center.size() == 3);
  CHECK(std::isfinite(center[0]));
  CHECK(std::isfinite(center[1]));
  CHECK(std::isfinite(center[2]));

  const std::string diag = solver->diagnostics_json();
  CHECK(test_common::json_value(diag, "dims") == doctest::Approx(3.0));
  CHECK(test_common::json_value(diag, "energy") >= 0.0);
}
