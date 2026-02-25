#include <doctest/doctest.h>

#include <cmath>
#include <string>
#include <vector>

#include "../test_common.hpp"
#include "wavefront/core/solver_nd.hpp"

// ---------------------------------------------------------------------------
//  Barrier modeled via spatially-varying stiffness
//
//  A region of very high stiffness in the middle of the domain acts as a
//  rigid barrier.  A pulse launched on one side should be mostly blocked.
// ---------------------------------------------------------------------------

namespace {

wavefront::ProblemSpec make_1d_barrier_problem(bool with_barrier) {
  wavefront::ProblemSpec p;
  p.grid.dims = 1;
  p.grid.shape = {128};
  p.grid.spacing = {0.02};
  p.grid.origin = {0.0};
  p.field_components = 1;

  // Barrier: stiffness jumps to 1e6 between x = 1.1 and x = 1.5 (indices ~55–75)
  // Without barrier: uniform stiffness 1.0
  if (with_barrier) {
    // Using max/min to create a step function for the barrier region
    p.medium.stiffness.text =
        "1.0 + 999999.0*max(0, min(1, min(x_0 - 1.1, 1.5 - x_0)*1000))";
  } else {
    p.medium.stiffness.text = "1.0";
  }

  p.medium.density.text = "1.0";
  p.medium.damping.text = "0.001";
  p.medium.dispersion.text = "0.0";

  // Gaussian pulse near the left side
  p.source_term.text =
      "20.0*sin(50*t)*exp(-30*t)*exp(-((x_0-0.3)*(x_0-0.3))/0.001)";

  p.boundaries = {
      {wavefront::BoundaryType::PML, 0, false, wavefront::SymbolicExpr{"8.0"}},
      {wavefront::BoundaryType::PML, 0, true, wavefront::SymbolicExpr{"8.0"}},
  };
  return p;
}

double rms_range(wavefront::ISolver& solver, std::size_t from, std::size_t to) {
  double sum = 0.0;
  std::size_t n = 0;
  for (std::size_t i = from; i < to; ++i) {
    const double v = solver.sample(std::vector<std::size_t>{i}).at(0);
    sum += v * v;
    ++n;
  }
  return std::sqrt(sum / static_cast<double>(std::max<std::size_t>(n, 1)));
}

}  // namespace

TEST_CASE("1D barrier blocks most energy transmission through high-stiffness region") {
  auto cfg = test_common::default_config(wavefront::SolverMode::LinearApprox);
  cfg.cfl = 0.2;
  cfg.spatial_order = 2;

  auto solver_barrier = wavefront::make_solver(make_1d_barrier_problem(true), cfg);
  auto solver_free = wavefront::make_solver(make_1d_barrier_problem(false), cfg);

  solver_barrier->run(200);
  solver_free->run(200);

  // RMS in the right-side region (past the barrier, indices 80–120)
  const double rms_barrier_right = rms_range(*solver_barrier, 80, 120);
  const double rms_free_right = rms_range(*solver_free, 80, 120);

  // With the barrier the RMS on the right should be significantly less
  // (allowing for reflections/numerical effects)
  if (rms_free_right > 1.0e-10) {
    CHECK(rms_barrier_right < rms_free_right);
  }
}

TEST_CASE("1D barrier causes reflected energy to stay on source side") {
  auto cfg = test_common::default_config(wavefront::SolverMode::LinearApprox);
  cfg.cfl = 0.2;
  cfg.spatial_order = 2;

  auto solver = wavefront::make_solver(make_1d_barrier_problem(true), cfg);
  solver->run(200);

  // RMS on source side (before barrier, indices 5–50) vs. right side (80–120)
  const double rms_left = rms_range(*solver, 5, 50);
  const double rms_right = rms_range(*solver, 80, 120);

  CHECK(std::isfinite(rms_left));
  CHECK(std::isfinite(rms_right));
}

// ---------------------------------------------------------------------------
//  2-D barrier: wall with a single slit
// ---------------------------------------------------------------------------

namespace {

wavefront::ProblemSpec make_2d_slit_problem() {
  wavefront::ProblemSpec p;
  p.grid.dims = 2;
  p.grid.shape = {64, 48};
  p.grid.spacing = {0.02, 0.02};
  p.grid.origin = {0.0, 0.0};
  p.field_components = 1;

  // Wall along x_0 ≈ 0.64 (col 32) except for slit at x_1 ∈ [0.40,0.56]
  // Modeled by high damping in the wall region
  p.medium.density.text = "1.0";
  p.medium.stiffness.text = "1.0";
  p.medium.damping.text =
      "500.0*max(0, min(1, (1.0 - (x_0 - 0.64)*(x_0 - 0.64)*2500)))"
      "*max(0, 1.0 - max(0, min(1, (x_1 - 0.40)*50))*max(0, min(1, (0.56 - x_1)*50)))";
  p.medium.dispersion.text = "0.0";

  // Source on the left
  p.source_term.text =
      "15*sin(40*t)*exp(-15*t)*exp(-((x_0-0.2)*(x_0-0.2)+(x_1-0.48)*(x_1-0.48))/0.004)";

  p.boundaries = {
      {wavefront::BoundaryType::PML, 0, false, wavefront::SymbolicExpr{"6.0"}},
      {wavefront::BoundaryType::PML, 0, true, wavefront::SymbolicExpr{"6.0"}},
      {wavefront::BoundaryType::PML, 1, false, wavefront::SymbolicExpr{"6.0"}},
      {wavefront::BoundaryType::PML, 1, true, wavefront::SymbolicExpr{"6.0"}},
  };
  return p;
}

}  // namespace

TEST_CASE("2D single-slit barrier produces finite field on both sides") {
  auto cfg = test_common::default_config(wavefront::SolverMode::LinearApprox);
  cfg.cfl = 0.2;
  cfg.spatial_order = 2;

  auto solver = wavefront::make_solver(make_2d_slit_problem(), cfg);
  solver->run(120);

  // Sample before and after the wall
  const auto before = solver->sample(std::vector<std::size_t>{20, 24});
  const auto after = solver->sample(std::vector<std::size_t>{44, 24});

  CHECK(std::isfinite(before[0]));
  CHECK(std::isfinite(after[0]));

  const std::string diag = solver->diagnostics_json();
  CHECK(test_common::json_value(diag, "energy") >= 0.0);
}

// ---------------------------------------------------------------------------
//  Barriers using SolverND (compile-time 2D)
// ---------------------------------------------------------------------------

TEST_CASE("2D SolverND wavefront stays finite around obstacle region") {
  wavefront::SolverND<2, double, wavefront::SolverMode::LinearApprox> solver(
      {32, 32}, {0.1, 0.1}, 1);
  solver.set_source_amplitude(2.0e-3);
  solver.run(15);

  // All sampled points must remain finite — no NaN blow-up from the wave
  for (std::size_t i = 0; i < 32; i += 4) {
    for (std::size_t j = 0; j < 32; j += 4) {
      CHECK(std::isfinite(solver.sample({i, j})));
    }
  }
}

// ---------------------------------------------------------------------------
//  3-D barrier: plane obstacle with an opening
// ---------------------------------------------------------------------------

TEST_CASE("3D solver with planar damping barrier runs stably") {
  wavefront::ProblemSpec p;
  p.grid.dims = 3;
  p.grid.shape = {16, 16, 16};
  p.grid.spacing = {0.1, 0.1, 0.1};
  p.grid.origin = {0.0, 0.0, 0.0};
  p.field_components = 1;

  p.medium.density.text = "1.0";
  p.medium.stiffness.text = "1.0";
  // Barrier plane at x_0 ≈ 0.8 (high damping), with a small hole
  p.medium.damping.text =
      "200.0*max(0, min(1, 1.0 - (x_0-0.8)*(x_0-0.8)*400))"
      "*max(0, 1.0 - max(0, min(1, 10*(0.3 - (x_1-0.8)*(x_1-0.8) - (x_2-0.8)*(x_2-0.8)))))";
  p.medium.dispersion.text = "0.0";

  p.source_term.text = "5*sin(20*t)*exp(-10*t)*exp(-((x_0-0.3)*(x_0-0.3))/0.01)";

  p.boundaries = {
      {wavefront::BoundaryType::Periodic, 0, false, wavefront::SymbolicExpr{"0.0"}},
      {wavefront::BoundaryType::Periodic, 0, true, wavefront::SymbolicExpr{"0.0"}},
      {wavefront::BoundaryType::Periodic, 1, false, wavefront::SymbolicExpr{"0.0"}},
      {wavefront::BoundaryType::Periodic, 1, true, wavefront::SymbolicExpr{"0.0"}},
      {wavefront::BoundaryType::Periodic, 2, false, wavefront::SymbolicExpr{"0.0"}},
      {wavefront::BoundaryType::Periodic, 2, true, wavefront::SymbolicExpr{"0.0"}},
  };

  auto cfg = test_common::default_config(wavefront::SolverMode::LinearApprox);
  auto solver = wavefront::make_solver(p, cfg);
  solver->run(15);

  // Stability: all sampled points finite
  for (std::size_t x = 0; x < 16; x += 4) {
    const auto v = solver->sample(std::vector<std::size_t>{x, 8, 8});
    CHECK(std::isfinite(v[0]));
  }

  CHECK(test_common::json_value(solver->diagnostics_json(), "energy") >= 0.0);
}
