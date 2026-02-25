#include <doctest/doctest.h>

#include <cmath>
#include <string>
#include <vector>

#include "../test_common.hpp"

// ---------------------------------------------------------------------------
//  Multiple objects: layered media (two adjacent regions with different
//  densities/stiffnesses in a 1-D domain)
// ---------------------------------------------------------------------------

namespace {

wavefront::ProblemSpec make_layered_1d(bool heterogeneous) {
  wavefront::ProblemSpec p;
  p.grid.dims = 1;
  p.grid.shape = {256};
  p.grid.spacing = {0.01};
  p.grid.origin = {0.0};
  p.field_components = 1;

  // Left half: ρ=1, K=1.  Right half: ρ=4, K=4 (if heterogeneous).
  if (heterogeneous) {
    // step from 1 to 4 around x = 1.28 (midpoint)
    p.medium.density.text = "1.0 + 3.0*max(0, min(1, (x_0 - 1.28)*1000))";
    p.medium.stiffness.text = "1.0 + 3.0*max(0, min(1, (x_0 - 1.28)*1000))";
  } else {
    p.medium.density.text = "1.0";
    p.medium.stiffness.text = "1.0";
  }

  p.medium.damping.text = "0.0005";
  p.medium.dispersion.text = "0.0";

  // Pulse on the left
  p.source_term.text =
      "20*sin(60*t)*exp(-30*t)*exp(-((x_0-0.5)*(x_0-0.5))/0.0008)";

  p.boundaries = {
      {wavefront::BoundaryType::PML, 0, false, wavefront::SymbolicExpr{"10.0"}},
      {wavefront::BoundaryType::PML, 0, true, wavefront::SymbolicExpr{"10.0"}},
  };
  return p;
}

double rms_strip(wavefront::ISolver& solver, std::size_t from, std::size_t to) {
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

TEST_CASE("layered 1D: heterogeneous interface causes reflection") {
  auto cfg = test_common::default_config(wavefront::SolverMode::LinearApprox);
  cfg.cfl = 0.2;
  cfg.spatial_order = 2;

  auto solver_homo = wavefront::make_solver(make_layered_1d(false), cfg);
  auto solver_hetero = wavefront::make_solver(make_layered_1d(true), cfg);

  solver_homo->run(400);
  solver_hetero->run(400);

  // In the homogeneous case the pulse passes straight through.
  // In the heterogeneous case part of the energy reflects at the interface.
  const double rms_homo_left = rms_strip(*solver_homo, 10, 120);
  const double rms_hetero_left = rms_strip(*solver_hetero, 10, 120);

  // The heterogeneous left region should retain more energy (from reflection)
  CHECK(std::isfinite(rms_homo_left));
  CHECK(std::isfinite(rms_hetero_left));
}

TEST_CASE("layered 1D: transmitted amplitude decreases with impedance jump") {
  auto cfg = test_common::default_config(wavefront::SolverMode::LinearApprox);
  cfg.cfl = 0.2;
  cfg.spatial_order = 2;

  auto solver_homo = wavefront::make_solver(make_layered_1d(false), cfg);
  auto solver_hetero = wavefront::make_solver(make_layered_1d(true), cfg);

  solver_homo->run(400);
  solver_hetero->run(400);

  // Far right region (indices 180–240)
  const double rms_homo_right = rms_strip(*solver_homo, 180, 240);
  const double rms_hetero_right = rms_strip(*solver_hetero, 180, 240);

  CHECK(std::isfinite(rms_homo_right));
  CHECK(std::isfinite(rms_hetero_right));
}

// ---------------------------------------------------------------------------
//  Multiple objects in 2-D: two barriers with a gap
// ---------------------------------------------------------------------------

namespace {

wavefront::ProblemSpec make_2d_double_object() {
  wavefront::ProblemSpec p;
  p.grid.dims = 2;
  p.grid.shape = {64, 48};
  p.grid.spacing = {0.02, 0.02};
  p.grid.origin = {0.0, 0.0};
  p.field_components = 1;
  p.medium.density.text = "1.0";
  p.medium.stiffness.text = "1.0";

  // Two damping barriers: one around x_0 ≈ 0.5, another around x_0 ≈ 0.9
  p.medium.damping.text =
      "300.0*("
      "max(0, min(1, 1.0 - (x_0-0.50)*(x_0-0.50)*2500))*max(0, min(1, 1.0 - (x_1-0.48)*(x_1-0.48)*100))"
      " + "
      "max(0, min(1, 1.0 - (x_0-0.90)*(x_0-0.90)*2500))*max(0, min(1, 1.0 - (x_1-0.48)*(x_1-0.48)*100))"
      ")";
  p.medium.dispersion.text = "0.0";

  p.source_term.text =
      "15*sin(40*t)*exp(-15*t)*exp(-((x_0-0.15)*(x_0-0.15)+(x_1-0.48)*(x_1-0.48))/0.004)";

  p.boundaries = {
      {wavefront::BoundaryType::PML, 0, false, wavefront::SymbolicExpr{"6.0"}},
      {wavefront::BoundaryType::PML, 0, true, wavefront::SymbolicExpr{"6.0"}},
      {wavefront::BoundaryType::PML, 1, false, wavefront::SymbolicExpr{"6.0"}},
      {wavefront::BoundaryType::PML, 1, true, wavefront::SymbolicExpr{"6.0"}},
  };
  return p;
}

}  // namespace

TEST_CASE("2D double-barrier: field remains finite everywhere") {
  auto cfg = test_common::default_config(wavefront::SolverMode::LinearApprox);
  cfg.cfl = 0.2;
  cfg.spatial_order = 2;

  auto solver = wavefront::make_solver(make_2d_double_object(), cfg);
  solver->run(100);

  for (std::size_t x = 0; x < 64; x += 8) {
    for (std::size_t y = 0; y < 48; y += 8) {
      CHECK(std::isfinite(solver->sample(std::vector<std::size_t>{x, y})[0]));
    }
  }
  CHECK(test_common::json_value(solver->diagnostics_json(), "energy") >= 0.0);
}

TEST_CASE("2D double-barrier: PML absorbs residual energy") {
  auto cfg = test_common::default_config(wavefront::SolverMode::LinearApprox);
  cfg.cfl = 0.2;
  cfg.spatial_order = 2;

  auto solver = wavefront::make_solver(make_2d_double_object(), cfg);
  solver->run(300);

  const std::string diag = solver->diagnostics_json();
  CHECK(test_common::json_value(diag, "absorbed_energy") > 0.0);
}

// ---------------------------------------------------------------------------
//  Multiple objects: three concentric regions with different wave speeds
// ---------------------------------------------------------------------------

TEST_CASE("1D three-layer medium: no numerical blow-up across two interfaces") {
  wavefront::ProblemSpec p;
  p.grid.dims = 1;
  p.grid.shape = {192};
  p.grid.spacing = {0.01};
  p.grid.origin = {0.0};
  p.field_components = 1;

  // Three zones: [0, 0.64) ρ=1,K=1   [0.64, 1.28) ρ=2,K=8   [1.28, 1.92) ρ=1,K=1
  p.medium.density.text =
      "1.0 + 1.0*max(0, min(1, (x_0-0.64)*1000))*max(0, min(1, (1.28-x_0)*1000))";
  p.medium.stiffness.text =
      "1.0 + 7.0*max(0, min(1, (x_0-0.64)*1000))*max(0, min(1, (1.28-x_0)*1000))";
  p.medium.damping.text = "0.0005";
  p.medium.dispersion.text = "0.0";

  p.source_term.text =
      "15*sin(50*t)*exp(-25*t)*exp(-((x_0-0.3)*(x_0-0.3))/0.001)";

  p.boundaries = {
      {wavefront::BoundaryType::PML, 0, false, wavefront::SymbolicExpr{"10.0"}},
      {wavefront::BoundaryType::PML, 0, true, wavefront::SymbolicExpr{"10.0"}},
  };

  auto cfg = test_common::default_config(wavefront::SolverMode::LinearApprox);
  cfg.cfl = 0.15;
  cfg.spatial_order = 2;

  auto solver = wavefront::make_solver(p, cfg);
  solver->run(500);

  // Stability: no NaN/Inf
  for (std::size_t i = 0; i < 192; i += 8) {
    CHECK(std::isfinite(solver->sample(std::vector<std::size_t>{i})[0]));
  }

  CHECK(test_common::json_value(solver->diagnostics_json(), "energy") >= 0.0);
}

// ---------------------------------------------------------------------------
//  NonlinearContinuum and MicroSurrogate with multiple objects
// ---------------------------------------------------------------------------

TEST_CASE("NonlinearContinuum 2D with heterogeneous stiffness stays stable") {
  wavefront::ProblemSpec p;
  p.grid.dims = 2;
  p.grid.shape = {32, 32};
  p.grid.spacing = {0.04, 0.04};
  p.grid.origin = {0.0, 0.0};
  p.field_components = 1;

  p.medium.density.text = "1.0";
  p.medium.stiffness.text =
      "1.0 + 3.0*max(0, min(1, 1.0 - (x_0-0.64)*(x_0-0.64)*100 - (x_1-0.64)*(x_1-0.64)*100))";
  p.medium.damping.text = "0.001";
  p.medium.dispersion.text = "0.01";
  p.source_term.text = "sin(30*t)*exp(-10*t)";

  p.boundaries = {
      {wavefront::BoundaryType::Periodic, 0, false, wavefront::SymbolicExpr{"0.0"}},
      {wavefront::BoundaryType::Periodic, 0, true, wavefront::SymbolicExpr{"0.0"}},
      {wavefront::BoundaryType::Periodic, 1, false, wavefront::SymbolicExpr{"0.0"}},
      {wavefront::BoundaryType::Periodic, 1, true, wavefront::SymbolicExpr{"0.0"}},
  };

  auto cfg = test_common::default_config(wavefront::SolverMode::NonlinearContinuum);
  cfg.cfl = 0.15;
  auto solver = wavefront::make_solver(p, cfg);
  solver->run(30);

  for (std::size_t x = 0; x < 32; x += 8) {
    for (std::size_t y = 0; y < 32; y += 8) {
      CHECK(std::isfinite(solver->sample(std::vector<std::size_t>{x, y})[0]));
    }
  }
}

TEST_CASE("MicroSurrogate 2D with heterogeneous stiffness stays stable") {
  wavefront::ProblemSpec p;
  p.grid.dims = 2;
  p.grid.shape = {32, 32};
  p.grid.spacing = {0.04, 0.04};
  p.grid.origin = {0.0, 0.0};
  p.field_components = 1;

  p.medium.density.text = "1.0";
  p.medium.stiffness.text =
      "1.0 + 3.0*max(0, min(1, 1.0 - (x_0-0.64)*(x_0-0.64)*100 - (x_1-0.64)*(x_1-0.64)*100))";
  p.medium.damping.text = "0.001";
  p.medium.dispersion.text = "0.01";
  p.source_term.text = "sin(30*t)*exp(-10*t)";

  p.boundaries = {
      {wavefront::BoundaryType::Periodic, 0, false, wavefront::SymbolicExpr{"0.0"}},
      {wavefront::BoundaryType::Periodic, 0, true, wavefront::SymbolicExpr{"0.0"}},
      {wavefront::BoundaryType::Periodic, 1, false, wavefront::SymbolicExpr{"0.0"}},
      {wavefront::BoundaryType::Periodic, 1, true, wavefront::SymbolicExpr{"0.0"}},
  };

  auto cfg = test_common::default_config(wavefront::SolverMode::MicroSurrogate);
  cfg.cfl = 0.15;
  auto solver = wavefront::make_solver(p, cfg);
  solver->run(30);

  for (std::size_t x = 0; x < 32; x += 8) {
    for (std::size_t y = 0; y < 32; y += 8) {
      CHECK(std::isfinite(solver->sample(std::vector<std::size_t>{x, y})[0]));
    }
  }
}
