#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "../test_common.hpp"

namespace {

wavefront::ProblemSpec make_1d_pulse_problem(std::size_t points, wavefront::BoundaryType boundary_type) {
  wavefront::ProblemSpec p;
  p.grid.dims = 1;
  p.grid.shape = {points};
  p.grid.spacing = {0.02};
  p.grid.origin = {0.0};
  p.field_components = 1;

  p.medium.density.text = "1.0";
  p.medium.stiffness.text = "1.2";
  p.medium.damping.text = "0.0003";
  p.medium.dispersion.text = "0.0";
  p.source_term.text = "14.0*sin(40*t)*exp(-20*t)*exp(-((x_0-0.16)*(x_0-0.16))/0.0010)";

  p.boundaries = {
      wavefront::BoundarySpec{boundary_type, 0, false, wavefront::SymbolicExpr{"9.0"}},
      wavefront::BoundarySpec{boundary_type, 0, true, wavefront::SymbolicExpr{"9.0"}},
  };
  return p;
}

wavefront::SolverConfig make_cfg() {
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

TEST_CASE("finite-speed propagation prevents phantom opposite-side signal in early time") {
  auto solver = wavefront::make_solver(make_1d_pulse_problem(256, wavefront::BoundaryType::PML), make_cfg());

  // Short time horizon: wavefront should not physically reach the far-right strip.
  solver->run(55);

  const double right_rms = rms_strip(*solver, 236, 256);
  CHECK(right_rms < 5e-3);
}

TEST_CASE("PML suppresses late-time wrap-around relative to periodic boundaries") {
  auto cfg = make_cfg();

  auto solver_periodic = wavefront::make_solver(make_1d_pulse_problem(128, wavefront::BoundaryType::Periodic), cfg);
  auto solver_pml = wavefront::make_solver(make_1d_pulse_problem(128, wavefront::BoundaryType::PML), cfg);

  // Long enough for boundary interactions to dominate late-time behavior.
  solver_periodic->run(900);
  solver_pml->run(900);

  const std::string periodic_diag = solver_periodic->diagnostics_json();
  const std::string pml_diag = solver_pml->diagnostics_json();

  const double periodic_energy = test_common::json_value(periodic_diag, "energy");
  const double pml_energy = test_common::json_value(pml_diag, "energy");
  const double periodic_absorbed = test_common::json_value(periodic_diag, "absorbed_energy");
  const double pml_absorbed = test_common::json_value(pml_diag, "absorbed_energy");

  CHECK(pml_energy < periodic_energy * 0.9);
  CHECK(periodic_absorbed < 1e-12);
  CHECK(pml_absorbed > 1e-3);
}
