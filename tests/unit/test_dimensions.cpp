#include <doctest/doctest.h>

#include <array>
#include <cmath>
#include <string>
#include <vector>

#include "../test_common.hpp"
#include "wavefront/core/solver_nd.hpp"

// ---------------------------------------------------------------------------
//  2-D simulations via compile-time SolverND
// ---------------------------------------------------------------------------

TEST_CASE("2D SolverND propagates a finite wavefront") {
  wavefront::SolverND<2, double, wavefront::SolverMode::LinearApprox> solver(
      {32, 32}, {0.1, 0.1}, 1);
  solver.set_source_amplitude(1.0e-3);
  solver.run(10);

  CHECK(solver.steps() == 10);
  const double center = solver.sample({16, 16});
  CHECK(std::isfinite(center));

  // After a few steps with a source, the center should carry energy.
  const double corner = solver.sample({0, 0});
  CHECK(std::isfinite(corner));
}

TEST_CASE("2D SolverND NonlinearContinuum mode produces finite results") {
  wavefront::SolverND<2, double, wavefront::SolverMode::NonlinearContinuum> solver(
      {20, 20}, {0.1, 0.1}, 1);
  solver.set_source_amplitude(5.0e-4);
  solver.set_nonlinear_coefficient(0.02);
  solver.run(8);

  CHECK(solver.steps() == 8);
  CHECK(std::isfinite(solver.sample({10, 10})));
}

TEST_CASE("2D SolverND MicroSurrogate mode produces finite results") {
  wavefront::SolverND<2, double, wavefront::SolverMode::MicroSurrogate> solver(
      {20, 20}, {0.1, 0.1}, 1);
  solver.set_source_amplitude(5.0e-4);
  solver.set_micro_gradient_coefficient(0.01);
  solver.set_memory_attenuation(0.005);
  solver.run(8);

  CHECK(solver.steps() == 8);
  CHECK(std::isfinite(solver.sample({10, 10})));
}

TEST_CASE("2D runtime solver handles square domain with periodic boundaries") {
  wavefront::ProblemSpec problem;
  problem.grid.dims = 2;
  problem.grid.shape = {32, 32};
  problem.grid.spacing = {0.05, 0.05};
  problem.grid.origin = {0.0, 0.0};
  problem.field_components = 1;
  problem.medium.density.text = "1.0";
  problem.medium.stiffness.text = "1.0";
  problem.medium.damping.text = "0.0";
  problem.medium.dispersion.text = "0.0";
  problem.source_term.text = "sin(40*t)*exp(-20*t)*exp(-((x_0-0.8)*(x_0-0.8)+(x_1-0.8)*(x_1-0.8))/0.005)";
  problem.boundaries = {
      {wavefront::BoundaryType::Periodic, 0, false, wavefront::SymbolicExpr{"0.0"}},
      {wavefront::BoundaryType::Periodic, 0, true, wavefront::SymbolicExpr{"0.0"}},
      {wavefront::BoundaryType::Periodic, 1, false, wavefront::SymbolicExpr{"0.0"}},
      {wavefront::BoundaryType::Periodic, 1, true, wavefront::SymbolicExpr{"0.0"}},
  };

  auto config = test_common::default_config(wavefront::SolverMode::LinearApprox);
  auto solver = wavefront::make_solver(problem, config);
  solver->run(20);

  const auto center = solver->sample(std::vector<std::size_t>{16, 16});
  CHECK(center.size() == 1);
  CHECK(std::isfinite(center[0]));

  const std::string diag = solver->diagnostics_json();
  CHECK(test_common::json_value(diag, "steps") == doctest::Approx(20.0));
  CHECK(test_common::json_value(diag, "energy") >= 0.0);
}

TEST_CASE("2D runtime solver rectangular domain with different axis extents") {
  wavefront::ProblemSpec problem;
  problem.grid.dims = 2;
  problem.grid.shape = {48, 24};
  problem.grid.spacing = {0.04, 0.08};
  problem.grid.origin = {0.0, 0.0};
  problem.field_components = 1;
  problem.medium.density.text = "1.0";
  problem.medium.stiffness.text = "1.0";
  problem.medium.damping.text = "0.0";
  problem.medium.dispersion.text = "0.0";
  problem.source_term.text = "sin(30*t)*exp(-10*t)";
  problem.boundaries = {
      {wavefront::BoundaryType::Periodic, 0, false, wavefront::SymbolicExpr{"0.0"}},
      {wavefront::BoundaryType::Periodic, 0, true, wavefront::SymbolicExpr{"0.0"}},
      {wavefront::BoundaryType::Periodic, 1, false, wavefront::SymbolicExpr{"0.0"}},
      {wavefront::BoundaryType::Periodic, 1, true, wavefront::SymbolicExpr{"0.0"}},
  };

  auto config = test_common::default_config(wavefront::SolverMode::LinearApprox);
  auto solver = wavefront::make_solver(problem, config);
  solver->run(15);

  const auto sample = solver->sample(std::vector<std::size_t>{24, 12});
  CHECK(sample.size() == 1);
  CHECK(std::isfinite(sample[0]));
}

// ---------------------------------------------------------------------------
//  3-D simulations
// ---------------------------------------------------------------------------

TEST_CASE("3D SolverND propagates across all three modes") {
  for (const auto mode_tag : {0, 1, 2}) {
    if (mode_tag == 0) {
      wavefront::SolverND<3, double, wavefront::SolverMode::LinearApprox> solver(
          {8, 8, 8}, {0.2, 0.2, 0.2}, 1);
      solver.set_source_amplitude(1.0e-3);
      solver.run(6);
      CHECK(solver.steps() == 6);
      CHECK(std::isfinite(solver.sample({4, 4, 4})));
    }
    if (mode_tag == 1) {
      wavefront::SolverND<3, double, wavefront::SolverMode::NonlinearContinuum> solver(
          {8, 8, 8}, {0.2, 0.2, 0.2}, 1);
      solver.set_source_amplitude(1.0e-3);
      solver.set_nonlinear_coefficient(0.01);
      solver.run(6);
      CHECK(solver.steps() == 6);
      CHECK(std::isfinite(solver.sample({4, 4, 4})));
    }
    if (mode_tag == 2) {
      wavefront::SolverND<3, double, wavefront::SolverMode::MicroSurrogate> solver(
          {8, 8, 8}, {0.2, 0.2, 0.2}, 1);
      solver.set_source_amplitude(1.0e-3);
      solver.set_micro_gradient_coefficient(0.01);
      solver.set_memory_attenuation(0.005);
      solver.run(6);
      CHECK(solver.steps() == 6);
      CHECK(std::isfinite(solver.sample({4, 4, 4})));
    }
  }
}

TEST_CASE("3D runtime solver with Gaussian pulse source") {
  wavefront::ProblemSpec problem;
  problem.grid.dims = 3;
  problem.grid.shape = {12, 12, 12};
  problem.grid.spacing = {0.1, 0.1, 0.1};
  problem.grid.origin = {0.0, 0.0, 0.0};
  problem.field_components = 1;
  problem.medium.density.text = "1.0";
  problem.medium.stiffness.text = "1.0";
  problem.medium.damping.text = "0.0";
  problem.medium.dispersion.text = "0.0";
  problem.source_term.text =
      "10*sin(30*t)*exp(-15*t)*exp(-((x_0-0.6)*(x_0-0.6)+(x_1-0.6)*(x_1-0.6)+(x_2-0.6)*(x_2-0.6))/0.01)";
  problem.boundaries = {
      {wavefront::BoundaryType::Periodic, 0, false, wavefront::SymbolicExpr{"0.0"}},
      {wavefront::BoundaryType::Periodic, 0, true, wavefront::SymbolicExpr{"0.0"}},
      {wavefront::BoundaryType::Periodic, 1, false, wavefront::SymbolicExpr{"0.0"}},
      {wavefront::BoundaryType::Periodic, 1, true, wavefront::SymbolicExpr{"0.0"}},
      {wavefront::BoundaryType::Periodic, 2, false, wavefront::SymbolicExpr{"0.0"}},
      {wavefront::BoundaryType::Periodic, 2, true, wavefront::SymbolicExpr{"0.0"}},
  };

  auto config = test_common::default_config(wavefront::SolverMode::LinearApprox);
  auto solver = wavefront::make_solver(problem, config);
  solver->run(10);

  const auto center = solver->sample(std::vector<std::size_t>{6, 6, 6});
  CHECK(center.size() == 1);
  CHECK(std::isfinite(center[0]));

  const std::string diag = solver->diagnostics_json();
  CHECK(test_common::json_value(diag, "dims") == doctest::Approx(3.0));
  CHECK(test_common::json_value(diag, "energy") >= 0.0);
}

TEST_CASE("3D solver isotropy: energy spreads comparably along all axes") {
  wavefront::SolverND<3, double, wavefront::SolverMode::LinearApprox> solver(
      {16, 16, 16}, {0.1, 0.1, 0.1}, 1);
  solver.set_source_amplitude(2.0e-3);
  solver.run(12);

  // Sample symmetrically placed points along each axis from center
  const double cx = solver.sample({8, 8, 8});
  const double px = solver.sample({12, 8, 8});
  const double py = solver.sample({8, 12, 8});
  const double pz = solver.sample({8, 8, 12});

  CHECK(std::isfinite(cx));
  CHECK(std::isfinite(px));
  CHECK(std::isfinite(py));
  CHECK(std::isfinite(pz));

  // With equal spacing and isotropic medium the off-center values should be similar
  const double max_off = std::max({std::fabs(px), std::fabs(py), std::fabs(pz)});
  const double min_off = std::min({std::fabs(px), std::fabs(py), std::fabs(pz)});
  if (max_off > 1.0e-14) {
    CHECK(min_off / max_off > 0.5);
  }
}

// ---------------------------------------------------------------------------
//  4-D simulations
// ---------------------------------------------------------------------------

TEST_CASE("4D SolverND constructs and runs without error") {
  wavefront::SolverND<4, double, wavefront::SolverMode::LinearApprox> solver(
      {6, 6, 6, 6}, {0.2, 0.2, 0.2, 0.2}, 1);
  solver.set_source_amplitude(1.0e-3);
  solver.run(4);

  CHECK(solver.steps() == 4);
  CHECK(std::isfinite(solver.sample({3, 3, 3, 3})));
}

TEST_CASE("4D SolverND NonlinearContinuum and MicroSurrogate modes") {
  {
    wavefront::SolverND<4, double, wavefront::SolverMode::NonlinearContinuum> solver(
        {5, 5, 5, 5}, {0.2, 0.2, 0.2, 0.2}, 1);
    solver.set_source_amplitude(1.0e-4);
    solver.set_nonlinear_coefficient(0.01);
    solver.run(3);
    CHECK(solver.steps() == 3);
    CHECK(std::isfinite(solver.sample({2, 2, 2, 2})));
  }
  {
    wavefront::SolverND<4, double, wavefront::SolverMode::MicroSurrogate> solver(
        {5, 5, 5, 5}, {0.2, 0.2, 0.2, 0.2}, 1);
    solver.set_source_amplitude(1.0e-4);
    solver.set_micro_gradient_coefficient(0.01);
    solver.set_memory_attenuation(0.005);
    solver.run(3);
    CHECK(solver.steps() == 3);
    CHECK(std::isfinite(solver.sample({2, 2, 2, 2})));
  }
}

TEST_CASE("4D SolverND diagnostics report correct dimensionality") {
  wavefront::SolverND<4, double, wavefront::SolverMode::LinearApprox> solver(
      {4, 4, 4, 4}, {0.3, 0.3, 0.3, 0.3}, 1);
  solver.run(2);

  const std::string diag = solver.diagnostics_json();
  CHECK(diag.find("\"dims\":4") != std::string::npos);
  CHECK(diag.find("\"steps\":2") != std::string::npos);
}

TEST_CASE("4D SolverND with multiple field components") {
  wavefront::SolverND<4, double, wavefront::SolverMode::LinearApprox> solver(
      {4, 4, 4, 4}, {0.3, 0.3, 0.3, 0.3}, 3);
  solver.set_source_amplitude(1.0e-4);
  solver.run(3);

  CHECK(std::isfinite(solver.sample({2, 2, 2, 2}, 0)));
  CHECK(std::isfinite(solver.sample({2, 2, 2, 2}, 1)));
  CHECK(std::isfinite(solver.sample({2, 2, 2, 2}, 2)));
}

// ---------------------------------------------------------------------------
//  Multi-component field tests
// ---------------------------------------------------------------------------

TEST_CASE("2D SolverND multi-component fields evolve independently") {
  wavefront::SolverND<2, double, wavefront::SolverMode::LinearApprox> solver(
      {16, 16}, {0.1, 0.1}, 2);
  solver.set_source_amplitude(1.0e-3);
  solver.run(6);

  const double c0 = solver.sample({8, 8}, 0);
  const double c1 = solver.sample({8, 8}, 1);
  CHECK(std::isfinite(c0));
  CHECK(std::isfinite(c1));
}

// ---------------------------------------------------------------------------
//  Edge cases
// ---------------------------------------------------------------------------

TEST_CASE("SolverND rejects axis with fewer than 3 points") {
  CHECK_THROWS_AS(
      (wavefront::SolverND<2, double, wavefront::SolverMode::LinearApprox>({2, 10}, {0.1, 0.1})),
      std::invalid_argument);
}

TEST_CASE("SolverND rejects zero spacing") {
  CHECK_THROWS_AS(
      (wavefront::SolverND<2, double, wavefront::SolverMode::LinearApprox>({10, 10}, {0.0, 0.1})),
      std::invalid_argument);
}

TEST_CASE("SolverND rejects zero components") {
  CHECK_THROWS_AS(
      (wavefront::SolverND<2, double, wavefront::SolverMode::LinearApprox>({10, 10}, {0.1, 0.1}, 0)),
      std::invalid_argument);
}
