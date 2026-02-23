#include <doctest/doctest.h>

#include <chrono>
#include <cmath>
#include <string>

#include "../test_common.hpp"
#include "wavefront/core/solver_nd.hpp"
#include "wavefront/physics/interface.hpp"

TEST_CASE("Yee1966 staggered-grid style propagation remains CFL-stable") {
  wavefront::SolverND<3, double, wavefront::SolverMode::LinearApprox> solver({16, 16, 16}, {0.1, 0.1, 0.1}, 1);
  solver.set_source_amplitude(1.0e-3);
  solver.run(12);

  CHECK(std::isfinite(solver.sample({8, 8, 8})));
  CHECK(std::isfinite(solver.sample({9, 8, 8})));
}

TEST_CASE("Berenger1994-style PML boundary shows nonzero absorbed energy") {
  auto problem = test_common::default_problem_1d(256);
  problem.source_term.text = "sin(t)";
  problem.boundaries = {
      wavefront::BoundarySpec{wavefront::BoundaryType::PML, 0, false, wavefront::SymbolicExpr{"12.0"}},
      wavefront::BoundarySpec{wavefront::BoundaryType::PML, 0, true, wavefront::SymbolicExpr{"12.0"}},
  };

  auto solver = wavefront::make_solver(problem, test_common::default_config(wavefront::SolverMode::LinearApprox));
  solver->run(40);

  const std::string diagnostics = solver->diagnostics_json();
  CHECK(test_common::json_value(diagnostics, "absorbed_energy") > 0.0);
}

TEST_CASE("Virieux1986 heterogeneous interface coefficients remain bounded") {
  const auto flux = wavefront::compute_interface_flux(1.0, 0.35, 2.1, 6.0, 2.6, 8.2);

  CHECK(std::isfinite(flux.reflected));
  CHECK(std::isfinite(flux.transmitted));
  CHECK(std::fabs(flux.reflected) <= 1.5);
  CHECK(std::fabs(flux.transmitted) <= 3.0);
  CHECK(flux.mode_conversion >= 0.0);
}

TEST_CASE("performance gate for float-mode throughput") {
  auto problem = test_common::default_problem_1d(384);
  auto config = test_common::default_config(wavefront::SolverMode::LinearApprox);

  auto solver = wavefront::make_solver(problem, config);

  const auto begin = std::chrono::steady_clock::now();
  solver->run(120);
  const auto end = std::chrono::steady_clock::now();

  const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
  CHECK(elapsed_ms < 3000);
}
