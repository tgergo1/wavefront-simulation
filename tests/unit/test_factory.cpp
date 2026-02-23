#include <doctest/doctest.h>

#include <array>
#include <string>

#include "../test_common.hpp"
#include "wavefront/core/solver_nd.hpp"

TEST_CASE("runtime solver factory supports all interchangeable modes") {
  const auto problem = test_common::default_problem_1d(96);

  for (const auto mode : {wavefront::SolverMode::LinearApprox,
                          wavefront::SolverMode::NonlinearContinuum,
                          wavefront::SolverMode::MicroSurrogate}) {
    auto solver = wavefront::make_solver(problem, test_common::default_config(mode));
    solver->run(6);
    const auto sample = solver->sample(std::vector<std::size_t>{problem.grid.shape[0] / 2});
    CHECK(sample.size() == 1);
    CHECK(std::isfinite(sample[0]));

    const std::string diagnostics = solver->diagnostics_json();
    CHECK(diagnostics.find("\"mode\"") != std::string::npos);
    CHECK(test_common::json_value(diagnostics, "steps") == doctest::Approx(6.0));
  }
}

TEST_CASE("compile-time SolverND API runs and reports diagnostics") {
  wavefront::SolverND<2, double, wavefront::SolverMode::LinearApprox> solver({64, 64}, {0.1, 0.1}, 1);
  solver.set_source_amplitude(1.0e-3);
  solver.run(4);

  CHECK(solver.steps() == 4);
  CHECK(std::isfinite(solver.sample({32, 32})));
  CHECK(solver.diagnostics_json().find("\"steps\":4") != std::string::npos);
}
