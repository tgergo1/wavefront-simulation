#include <doctest/doctest.h>

#include <array>
#include <cmath>
#include <string>
#include <vector>

#include "../test_common.hpp"
#include "wavefront/core/solver_nd.hpp"

// ---------------------------------------------------------------------------
//  Longitudinal wave unit tests – SolverND (compile-time N-D solver)
// ---------------------------------------------------------------------------

TEST_CASE("SolverND 2D longitudinal wave with 2 components produces finite results") {
  wavefront::SolverND<2, double, wavefront::SolverMode::LinearApprox> solver(
      {20, 20}, {0.1, 0.1}, 2);
  solver.set_wave_type(wavefront::WaveType::Longitudinal);
  solver.set_source_amplitude(1.0e-3);
  solver.run(8);

  CHECK(solver.steps() == 8);
  CHECK(std::isfinite(solver.sample({10, 10}, 0)));
  CHECK(std::isfinite(solver.sample({10, 10}, 1)));
}

TEST_CASE("SolverND 3D longitudinal wave with 3 components produces finite results") {
  wavefront::SolverND<3, double, wavefront::SolverMode::LinearApprox> solver(
      {8, 8, 8}, {0.2, 0.2, 0.2}, 3);
  solver.set_wave_type(wavefront::WaveType::Longitudinal);
  solver.set_source_amplitude(1.0e-3);
  solver.run(5);

  CHECK(solver.steps() == 5);
  CHECK(std::isfinite(solver.sample({4, 4, 4}, 0)));
  CHECK(std::isfinite(solver.sample({4, 4, 4}, 1)));
  CHECK(std::isfinite(solver.sample({4, 4, 4}, 2)));
}

TEST_CASE("SolverND 2D longitudinal NonlinearContinuum produces finite results") {
  wavefront::SolverND<2, double, wavefront::SolverMode::NonlinearContinuum> solver(
      {16, 16}, {0.1, 0.1}, 2);
  solver.set_wave_type(wavefront::WaveType::Longitudinal);
  solver.set_source_amplitude(5.0e-4);
  solver.set_nonlinear_coefficient(0.02);
  solver.run(6);

  CHECK(solver.steps() == 6);
  CHECK(std::isfinite(solver.sample({8, 8}, 0)));
  CHECK(std::isfinite(solver.sample({8, 8}, 1)));
}

TEST_CASE("SolverND 2D longitudinal MicroSurrogate produces finite results") {
  wavefront::SolverND<2, double, wavefront::SolverMode::MicroSurrogate> solver(
      {16, 16}, {0.1, 0.1}, 2);
  solver.set_wave_type(wavefront::WaveType::Longitudinal);
  solver.set_source_amplitude(5.0e-4);
  solver.set_micro_gradient_coefficient(0.01);
  solver.set_memory_attenuation(0.005);
  solver.run(6);

  CHECK(solver.steps() == 6);
  CHECK(std::isfinite(solver.sample({8, 8}, 0)));
  CHECK(std::isfinite(solver.sample({8, 8}, 1)));
}

TEST_CASE("SolverND longitudinal with 1 component behaves like scalar wave") {
  // With 1 component, longitudinal = transverse (scalar case)
  wavefront::SolverND<2, double, wavefront::SolverMode::LinearApprox> solver_t(
      {16, 16}, {0.1, 0.1}, 1);
  solver_t.set_wave_type(wavefront::WaveType::Transverse);
  solver_t.set_source_amplitude(1.0e-3);
  solver_t.run(8);

  wavefront::SolverND<2, double, wavefront::SolverMode::LinearApprox> solver_l(
      {16, 16}, {0.1, 0.1}, 1);
  solver_l.set_wave_type(wavefront::WaveType::Longitudinal);
  solver_l.set_source_amplitude(1.0e-3);
  solver_l.run(8);

  // Both should produce identical results for scalar fields
  CHECK(solver_t.sample({8, 8}) == doctest::Approx(solver_l.sample({8, 8})));
  CHECK(solver_t.sample({4, 12}) == doctest::Approx(solver_l.sample({4, 12})));
}

TEST_CASE("SolverND longitudinal with 2 components differs from transverse with spatial variation") {
  // SolverND uses spatially-uniform sources, so the difference between
  // transverse and longitudinal only manifests when spatial gradients exist.
  // We test this via the runtime solver instead.
  wavefront::ProblemSpec problem;
  problem.grid.dims = 2;
  problem.grid.shape = {20, 20};
  problem.grid.spacing = {0.06, 0.06};
  problem.grid.origin = {0.0, 0.0};
  problem.field_components = 2;
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

  auto config = test_common::default_config(wavefront::SolverMode::LinearApprox);

  // Transverse
  auto problem_t = problem;
  problem_t.wave_type = wavefront::WaveType::Transverse;
  auto solver_t = wavefront::make_solver(problem_t, config);
  solver_t->run(15);

  // Longitudinal
  auto problem_l = problem;
  problem_l.wave_type = wavefront::WaveType::Longitudinal;
  auto solver_l = wavefront::make_solver(problem_l, config);
  solver_l->run(15);

  // With coupled components and a localised source, solutions should diverge
  double max_diff = 0.0;
  for (std::size_t i = 0; i < 20; ++i) {
    for (std::size_t j = 0; j < 20; ++j) {
      const auto vt = solver_t->sample(std::vector<std::size_t>{i, j});
      const auto vl = solver_l->sample(std::vector<std::size_t>{i, j});
      for (std::size_t c = 0; c < 2; ++c) {
        max_diff = std::max(max_diff, std::fabs(vt[c] - vl[c]));
      }
    }
  }
  CHECK(max_diff > 1.0e-10);
}

// ---------------------------------------------------------------------------
//  Longitudinal wave unit tests – runtime solver (ISolver)
// ---------------------------------------------------------------------------

TEST_CASE("runtime solver longitudinal 1D scalar produces finite results") {
  wavefront::ProblemSpec problem;
  problem.grid.dims = 1;
  problem.grid.shape = {64};
  problem.grid.spacing = {0.02};
  problem.grid.origin = {0.0};
  problem.field_components = 1;
  problem.wave_type = wavefront::WaveType::Longitudinal;
  problem.medium.density.text = "1.0";
  problem.medium.stiffness.text = "1.0";
  problem.medium.damping.text = "0.0";
  problem.medium.dispersion.text = "0.0";
  problem.source_term.text = "10*sin(30*t)*exp(-15*t)*exp(-((x_0-0.6)*(x_0-0.6))/0.005)";
  problem.boundaries = {
      {wavefront::BoundaryType::Periodic, 0, false, wavefront::SymbolicExpr{"0.0"}},
      {wavefront::BoundaryType::Periodic, 0, true, wavefront::SymbolicExpr{"0.0"}},
  };

  auto config = test_common::default_config(wavefront::SolverMode::LinearApprox);
  auto solver = wavefront::make_solver(problem, config);
  solver->run(20);

  const auto sample = solver->sample(std::vector<std::size_t>{32});
  CHECK(sample.size() == 1);
  CHECK(std::isfinite(sample[0]));

  const std::string diag = solver->diagnostics_json();
  CHECK(diag.find("\"wave_type\":\"Longitudinal\"") != std::string::npos);
}

TEST_CASE("runtime solver longitudinal 2D with 2 components produces finite coupled results") {
  wavefront::ProblemSpec problem;
  problem.grid.dims = 2;
  problem.grid.shape = {24, 24};
  problem.grid.spacing = {0.05, 0.05};
  problem.grid.origin = {0.0, 0.0};
  problem.field_components = 2;
  problem.wave_type = wavefront::WaveType::Longitudinal;
  problem.medium.density.text = "1.0";
  problem.medium.stiffness.text = "1.0";
  problem.medium.damping.text = "0.0";
  problem.medium.dispersion.text = "0.0";
  problem.source_term.text = "8*sin(30*t)*exp(-15*t)*exp(-((x_0-0.6)*(x_0-0.6)+(x_1-0.6)*(x_1-0.6))/0.005)";
  problem.boundaries = {
      {wavefront::BoundaryType::Periodic, 0, false, wavefront::SymbolicExpr{"0.0"}},
      {wavefront::BoundaryType::Periodic, 0, true, wavefront::SymbolicExpr{"0.0"}},
      {wavefront::BoundaryType::Periodic, 1, false, wavefront::SymbolicExpr{"0.0"}},
      {wavefront::BoundaryType::Periodic, 1, true, wavefront::SymbolicExpr{"0.0"}},
  };

  auto config = test_common::default_config(wavefront::SolverMode::LinearApprox);
  auto solver = wavefront::make_solver(problem, config);
  solver->run(15);

  const auto center = solver->sample(std::vector<std::size_t>{12, 12});
  CHECK(center.size() == 2);
  CHECK(std::isfinite(center[0]));
  CHECK(std::isfinite(center[1]));

  const std::string diag = solver->diagnostics_json();
  CHECK(test_common::json_value(diag, "energy") >= 0.0);
  CHECK(diag.find("\"wave_type\":\"Longitudinal\"") != std::string::npos);
}

TEST_CASE("runtime solver longitudinal 2D with PML boundaries") {
  wavefront::ProblemSpec problem;
  problem.grid.dims = 2;
  problem.grid.shape = {32, 32};
  problem.grid.spacing = {0.04, 0.04};
  problem.grid.origin = {0.0, 0.0};
  problem.field_components = 2;
  problem.wave_type = wavefront::WaveType::Longitudinal;
  problem.medium.density.text = "1.0";
  problem.medium.stiffness.text = "1.2";
  problem.medium.damping.text = "0.0005";
  problem.medium.dispersion.text = "0.0";
  problem.source_term.text = "12*sin(25*t)*exp(-12*t)*exp(-((x_0-0.6)*(x_0-0.6)+(x_1-0.6)*(x_1-0.6))/0.008)";
  problem.boundaries = {
      {wavefront::BoundaryType::PML, 0, false, wavefront::SymbolicExpr{"10.0"}},
      {wavefront::BoundaryType::PML, 0, true, wavefront::SymbolicExpr{"10.0"}},
      {wavefront::BoundaryType::PML, 1, false, wavefront::SymbolicExpr{"10.0"}},
      {wavefront::BoundaryType::PML, 1, true, wavefront::SymbolicExpr{"10.0"}},
  };

  auto config = test_common::default_config(wavefront::SolverMode::LinearApprox);
  auto solver = wavefront::make_solver(problem, config);
  solver->run(30);

  const auto sample = solver->sample(std::vector<std::size_t>{16, 16});
  CHECK(sample.size() == 2);
  CHECK(std::isfinite(sample[0]));
  CHECK(std::isfinite(sample[1]));

  const std::string diag = solver->diagnostics_json();
  CHECK(test_common::json_value(diag, "absorbed_energy") >= 0.0);
}

TEST_CASE("runtime solver transverse diagnostics report wave_type") {
  auto problem = test_common::default_problem_1d(64);
  auto config = test_common::default_config(wavefront::SolverMode::LinearApprox);
  auto solver = wavefront::make_solver(problem, config);
  solver->run(5);

  const std::string diag = solver->diagnostics_json();
  CHECK(diag.find("\"wave_type\":\"Transverse\"") != std::string::npos);
}

TEST_CASE("runtime solver longitudinal NonlinearContinuum 2D with coupled components") {
  wavefront::ProblemSpec problem;
  problem.grid.dims = 2;
  problem.grid.shape = {20, 20};
  problem.grid.spacing = {0.06, 0.06};
  problem.grid.origin = {0.0, 0.0};
  problem.field_components = 2;
  problem.wave_type = wavefront::WaveType::Longitudinal;
  problem.medium.density.text = "1.0";
  problem.medium.stiffness.text = "1.0";
  problem.medium.damping.text = "0.0";
  problem.medium.dispersion.text = "5.0";
  problem.source_term.text = "5*sin(20*t)*exp(-10*t)";
  problem.boundaries = {
      {wavefront::BoundaryType::Periodic, 0, false, wavefront::SymbolicExpr{"0.0"}},
      {wavefront::BoundaryType::Periodic, 0, true, wavefront::SymbolicExpr{"0.0"}},
      {wavefront::BoundaryType::Periodic, 1, false, wavefront::SymbolicExpr{"0.0"}},
      {wavefront::BoundaryType::Periodic, 1, true, wavefront::SymbolicExpr{"0.0"}},
  };

  auto config = test_common::default_config(wavefront::SolverMode::NonlinearContinuum);
  auto solver = wavefront::make_solver(problem, config);
  solver->run(10);

  const auto center = solver->sample(std::vector<std::size_t>{10, 10});
  CHECK(center.size() == 2);
  CHECK(std::isfinite(center[0]));
  CHECK(std::isfinite(center[1]));
}
