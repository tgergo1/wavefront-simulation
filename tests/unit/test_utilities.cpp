#include <doctest/doctest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

#include "../test_common.hpp"
#include "wavefront/material/library.hpp"
#include "wavefront/optimization/sweep.hpp"
#include "wavefront/utils/plotting.hpp"

TEST_CASE("material library sweep helpers and plotting utilities support workflows") {
  const auto water = wavefront::builtin_material("water");
  CHECK(water.name == "water");
  CHECK(water.medium.density.text == "1000.0");

  auto problem = test_common::default_problem_1d(32);
  problem.medium = water.medium;
  problem.source_term.text = "sin(t)";

  auto config = test_common::default_config(wavefront::SolverMode::MicroSurrogate);
  config.family = wavefront::SolverFamily::AngularSpectrum;
  config.center_frequency = 1.5;

  const auto objective = [](const wavefront::ProblemSpec& candidate_problem, const wavefront::SolverConfig& candidate_config) {
    auto solver = wavefront::make_solver(candidate_problem, candidate_config);
    solver->run(4);
    return solver->field_snapshot().values.at(candidate_problem.grid.shape[0] / 2);
  };

  const auto sweep = wavefront::run_parameter_sweep(
      problem,
      config,
      {
          {"config.center_frequency", {1.0, 2.0}},
          {"medium.damping", {0.0001, 0.01}},
      },
      objective);
  CHECK(sweep.size() == 4);
  CHECK(sweep.front().assignments.count("config.center_frequency") == 1);

  const double gradient =
      wavefront::finite_difference_gradient(problem, config, "config.center_frequency", config.center_frequency, 0.1, objective);
  CHECK(std::isfinite(gradient));

  auto solver = wavefront::make_solver(problem, config);
  solver->run(3);
  const auto snapshot = solver->field_snapshot();
  const auto series = wavefront::snapshot_component_series(snapshot, 0);
  const auto [min_value, max_value] = wavefront::snapshot_minmax(snapshot);
  const auto normalized = wavefront::normalize_series(series);

  CHECK(series.size() == problem.grid.shape[0]);
  CHECK(max_value >= min_value);
  CHECK(normalized.size() == series.size());
  CHECK(*std::max_element(normalized.begin(), normalized.end()) <= doctest::Approx(1.0));
}

TEST_CASE("material library includes exotic presets") {
  const auto presets = wavefront::builtin_materials();
  CHECK(presets.size() >= 13);

  const std::array<std::string_view, 9> exotic_names = {
      "honey",
      "hyperhoney",
      "oobleck",
      "aerogel",
      "ferrofluid",
      "plasma",
      "metamaterial",
      "neutron_star_crust",
      "strange_matter",
  };

  for (const auto name : exotic_names) {
    const auto preset = wavefront::builtin_material(name);
    CHECK(preset.name == name);
    CHECK_FALSE(preset.medium.density.text.empty());
    CHECK_FALSE(preset.medium.stiffness.text.empty());
  }

  CHECK_THROWS_AS(wavefront::builtin_material("unknown_exotic_goo"), std::out_of_range);
}

TEST_CASE("exotic material presets remain solver-compatible") {
  const std::array<std::string_view, 8> material_names = {
      "honey",
      "hyperhoney",
      "oobleck",
      "aerogel",
      "ferrofluid",
      "plasma",
      "metamaterial",
      "strange_matter",
  };

  auto config = test_common::default_config(wavefront::SolverMode::MicroSurrogate);
  config.family = wavefront::SolverFamily::AngularSpectrum;

  for (const auto name : material_names) {
    auto problem = test_common::default_problem_1d(40);
    problem.medium = wavefront::builtin_material(name).medium;
    problem.source_term.text = "sin(8*t)*exp(-4*t)";

    auto solver = wavefront::make_solver(problem, config);
    solver->run(5);

    const auto sample = solver->sample(std::vector<std::size_t>{20});
    CHECK(sample.size() == 1);
    CHECK(std::isfinite(sample[0]));
    CHECK(std::isfinite(solver->field_snapshot().values.at(20)));
  }
}
