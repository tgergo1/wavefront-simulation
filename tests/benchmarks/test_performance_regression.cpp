#include <doctest/doctest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <string>
#include <vector>

#include "performance_test_utils.hpp"

namespace {

wavefront::ProblemSpec make_mode_comparison_problem() {
  wavefront::ProblemSpec problem;
  problem.grid.dims = 2;
  problem.grid.shape = {40, 40};
  problem.grid.spacing = {0.03, 0.03};
  problem.grid.origin = {0.0, 0.0};
  problem.field_components = 1;
  problem.medium.density.text = "1.0";
  problem.medium.stiffness.text =
      "1.0 + 0.30*exp(-8.0*((x_0-0.60)*(x_0-0.60)+(x_1-0.60)*(x_1-0.60)))";
  problem.medium.damping.text = "0.0005";
  problem.medium.dispersion.text = "0.01";
  problem.source_term.text =
      "6.0*sin(18.0*t)*exp(-7.0*t)*exp(-((x_0-0.45)*(x_0-0.45)+(x_1-0.45)*(x_1-0.45))/0.02)";
  problem.boundaries = {
      {wavefront::BoundaryType::PML, 0, false, wavefront::SymbolicExpr{"8.0"}},
      {wavefront::BoundaryType::PML, 0, true, wavefront::SymbolicExpr{"8.0"}},
      {wavefront::BoundaryType::PML, 1, false, wavefront::SymbolicExpr{"8.0"}},
      {wavefront::BoundaryType::PML, 1, true, wavefront::SymbolicExpr{"8.0"}},
  };
  return problem;
}

wavefront::ProblemSpec make_monitor_heavy_longitudinal_problem() {
  wavefront::ProblemSpec problem;
  problem.grid.dims = 2;
  problem.grid.shape = {36, 36};
  problem.grid.spacing = {0.04, 0.04};
  problem.grid.origin = {0.0, 0.0};
  problem.field_components = 2;
  problem.wave_type = wavefront::WaveType::Longitudinal;
  problem.medium.density.text = "1.0";
  problem.medium.stiffness.text = "1.0 + 0.2*x_0";
  problem.medium.damping.text = "0.0008";
  problem.medium.dispersion.text = "0.0";
  problem.source_term.text =
      "8.0*sin(24.0*t)*exp(-12.0*t)*exp(-((x_0-0.50)*(x_0-0.50)+(x_1-0.50)*(x_1-0.50))/0.01)";
  problem.boundaries = {
      {wavefront::BoundaryType::PML, 0, false, wavefront::SymbolicExpr{"10.0"}},
      {wavefront::BoundaryType::PML, 0, true, wavefront::SymbolicExpr{"10.0"}},
      {wavefront::BoundaryType::PML, 1, false, wavefront::SymbolicExpr{"10.0"}},
      {wavefront::BoundaryType::PML, 1, true, wavefront::SymbolicExpr{"10.0"}},
  };
  problem.monitors.probes = {
      {"center_probe", {18, 18}, 0, false},
      {"off_axis_probe", {24, 18}, 1, false},
  };
  problem.monitors.snapshot_interval = 8;
  problem.monitors.spectrum_bins = 8;
  problem.monitors.enable_far_field = true;
  return problem;
}

bool has_utc_timestamp_shape(const std::string& value) {
  return value.size() == 24 && value[4] == '-' && value[7] == '-' && value[10] == 'T' && value[13] == ':' &&
         value[16] == ':' && value[19] == '.' && value.back() == 'Z';
}

}  // namespace

TEST_CASE("performance telemetry captures UTC timestamps and duration metrics") {
  auto problem = test_common::default_problem_1d(256);
  problem.source_term.text = "sin(14.0*t)*exp(-6.0*t)";

  auto config = test_common::default_config(wavefront::SolverMode::LinearApprox);
  config.cfl = 0.25;
  config.threads = 2;

  auto result = performance_test::benchmark_solver_run("telemetry-1d-baseline", std::move(problem), config, 96);
  performance_test::log_measurement(result.measurement);

  CHECK(has_utc_timestamp_shape(result.measurement.started_at_utc));
  CHECK(has_utc_timestamp_shape(result.measurement.finished_at_utc));
  CHECK(result.measurement.started_at_utc <= result.measurement.finished_at_utc);
  CHECK(result.measurement.wall_time > std::chrono::nanoseconds::zero());
  CHECK(result.measurement.elapsed_milliseconds > 0.0);
  CHECK(result.measurement.steps_per_second > 0.0);
  CHECK(result.measurement.cell_updates_per_second > 0.0);
  CHECK(result.measurement.nanoseconds_per_cell_update > 0.0);
  CHECK(std::isfinite(result.measurement.energy));
  CHECK(result.solver->field_snapshot().step == 96);
}

TEST_CASE("performance regression compares representative solver modes") {
  auto linear_cfg = test_common::default_config(wavefront::SolverMode::LinearApprox);
  linear_cfg.cfl = 0.18;

  auto nonlinear_cfg = test_common::default_config(wavefront::SolverMode::NonlinearContinuum);
  nonlinear_cfg.cfl = 0.14;

  auto surrogate_cfg = test_common::default_config(wavefront::SolverMode::MicroSurrogate);
  surrogate_cfg.cfl = 0.14;

  auto linear = performance_test::benchmark_solver_run("mode-comparison", make_mode_comparison_problem(), linear_cfg, 48);
  auto nonlinear =
      performance_test::benchmark_solver_run("mode-comparison", make_mode_comparison_problem(), nonlinear_cfg, 48);
  auto surrogate =
      performance_test::benchmark_solver_run("mode-comparison", make_mode_comparison_problem(), surrogate_cfg, 48);

  performance_test::log_measurement(linear.measurement);
  performance_test::log_measurement(nonlinear.measurement);
  performance_test::log_measurement(surrogate.measurement);

  for (const auto* run : {&linear, &nonlinear, &surrogate}) {
    CHECK(run->measurement.wall_time > std::chrono::nanoseconds::zero());
    CHECK(run->measurement.elapsed_milliseconds < 5000.0);
    CHECK(run->measurement.steps_per_second > 0.0);
    CHECK(run->measurement.cell_updates_per_second > 0.0);
    CHECK(run->measurement.nanoseconds_per_cell_update > 0.0);
    CHECK(std::isfinite(run->measurement.energy));
    CHECK(run->solver->field_snapshot().step == 48);
  }

  const double slowest_elapsed_ms =
      std::max({linear.measurement.elapsed_milliseconds,
                nonlinear.measurement.elapsed_milliseconds,
                surrogate.measurement.elapsed_milliseconds});
  const double fastest_elapsed_ms =
      std::min({linear.measurement.elapsed_milliseconds,
                nonlinear.measurement.elapsed_milliseconds,
                surrogate.measurement.elapsed_milliseconds});

  CHECK(slowest_elapsed_ms >= fastest_elapsed_ms);
  CHECK(slowest_elapsed_ms / fastest_elapsed_ms < 50.0);
}

TEST_CASE("performance regression captures monitor-heavy longitudinal workloads") {
  auto config = test_common::default_config(wavefront::SolverMode::LinearApprox);
  config.cfl = 0.18;
  config.spatial_order = 2;
  config.far_field_samples = 12;

  auto result = performance_test::benchmark_solver_run(
      "longitudinal-monitoring", make_monitor_heavy_longitudinal_problem(), config, 72);
  performance_test::log_measurement(result.measurement);

  CHECK(result.measurement.elapsed_milliseconds < 5000.0);
  CHECK(result.measurement.steps_per_second > 0.0);
  CHECK(result.measurement.cell_updates_per_second > 0.0);
  CHECK(result.measurement.nanoseconds_per_cell_update > 0.0);
  CHECK(std::isfinite(result.measurement.energy));
  CHECK(result.measurement.absorbed_energy >= 0.0);
  CHECK(result.solver->probe_history("center_probe").size() == 73);
  CHECK(result.solver->probe_history("off_axis_probe").size() == 73);
  CHECK(result.solver->field_snapshot().components == 2);
  CHECK(result.solver->far_field_pattern().amplitudes.size() == 12);
}
