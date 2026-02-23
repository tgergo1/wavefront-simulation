#include <doctest/doctest.h>

#include <string>

#include "../test_common.hpp"

TEST_CASE("linear conservative case keeps bounded energy drift") {
  auto problem = test_common::default_problem_1d(160);
  problem.medium.damping.text = "0.0";
  problem.source_term.text = "0.0";
  problem.boundaries.clear();

  auto config = test_common::default_config(wavefront::SolverMode::LinearApprox);
  config.cfl = 0.1;
  config.spatial_order = 2;
  auto solver = wavefront::make_solver(problem, config);

  const std::string initial_diag = solver->diagnostics_json();
  const double initial_energy = test_common::json_value(initial_diag, "energy");

  solver->run(40);

  const std::string final_diag = solver->diagnostics_json();
  const double final_energy = test_common::json_value(final_diag, "energy");

  const double absolute_drift = std::fabs(final_energy - initial_energy);
  CHECK(absolute_drift < 1e-3);
}

TEST_CASE("deterministic mode yields reproducible diagnostics") {
  auto problem = test_common::default_problem_1d(128);
  auto config = test_common::default_config(wavefront::SolverMode::LinearApprox);
  config.threads = 4;
  config.deterministic = true;

  auto solver_a = wavefront::make_solver(problem, config);
  auto solver_b = wavefront::make_solver(problem, config);

  solver_a->run(20);
  solver_b->run(20);

  CHECK(solver_a->diagnostics_json() == solver_b->diagnostics_json());
}
