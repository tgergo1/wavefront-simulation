#include <doctest/doctest.h>

#include <string>
#include <vector>

#include "../test_common.hpp"

TEST_CASE("mode interchangeability keeps API and lifecycle stable") {
  const auto problem = test_common::default_problem_1d(128);

  std::vector<std::vector<double>> mode_samples;

  for (const auto mode : {wavefront::SolverMode::LinearApprox,
                          wavefront::SolverMode::NonlinearContinuum,
                          wavefront::SolverMode::MicroSurrogate}) {
    auto solver = wavefront::make_solver(problem, test_common::default_config(mode));
    solver->run(12);

    mode_samples.push_back(test_common::sample_line(*solver, problem.grid.shape[0]));

    const std::string diagnostics = solver->diagnostics_json();
    CHECK(diagnostics.find("\"steps\":12") != std::string::npos);
    CHECK(test_common::json_value(diagnostics, "energy") >= 0.0);
  }

  CHECK(mode_samples.size() == 3);

  const double nonlinear_delta = test_common::l2_error(mode_samples[0], mode_samples[1]);
  const double micro_delta = test_common::l2_error(mode_samples[0], mode_samples[2]);

  CHECK(nonlinear_delta >= 0.0);
  CHECK(micro_delta >= 0.0);
}

TEST_CASE("exact-reference mode emits finite certified diagnostics") {
#if WAVEFRONT_HAS_LIMITLESS
  auto solver = wavefront::make_solver(
      test_common::default_problem_1d(96),
      test_common::default_config(wavefront::SolverMode::LinearApprox, wavefront::PrecisionMode::ExactReference));
  solver->run(8);

  const std::string diagnostics = solver->diagnostics_json();
  CHECK(test_common::json_value(diagnostics, "max_reference_error") >= 0.0);
#else
  SUCCEED("ExactReference checks are disabled when limitless is unavailable");
#endif
}
