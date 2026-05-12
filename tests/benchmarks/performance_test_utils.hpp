#pragma once

#include <chrono>
#include <ctime>
#include <iomanip>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>

#include <doctest/doctest.h>

#include "../test_common.hpp"
#include "wavefront/api/solver.hpp"

namespace performance_test {

struct Measurement {
  std::string scenario;
  std::string mode;
  std::string wave_type;
  std::string started_at_utc;
  std::string finished_at_utc;
  std::chrono::nanoseconds wall_time{0};
  double elapsed_milliseconds = 0.0;
  double steps_per_second = 0.0;
  double cell_updates_per_second = 0.0;
  double nanoseconds_per_cell_update = 0.0;
  double energy = 0.0;
  double absorbed_energy = 0.0;
  std::string diagnostics_json;
};

struct RunResult {
  Measurement measurement;
  std::unique_ptr<wavefront::ISolver> solver;
};

inline std::string solver_mode_name(wavefront::SolverMode mode) {
  switch (mode) {
    case wavefront::SolverMode::LinearApprox:
      return "LinearApprox";
    case wavefront::SolverMode::NonlinearContinuum:
      return "NonlinearContinuum";
    case wavefront::SolverMode::MicroSurrogate:
      return "MicroSurrogate";
  }

  return "Unknown";
}

inline std::string wave_type_name(wavefront::WaveType wave_type) {
  switch (wave_type) {
    case wavefront::WaveType::Transverse:
      return "Transverse";
    case wavefront::WaveType::Longitudinal:
      return "Longitudinal";
  }

  return "Unknown";
}

inline std::tm utc_tm(std::time_t time_value) {
  std::tm out{};
#if defined(_WIN32)
  gmtime_s(&out, &time_value);
#else
  gmtime_r(&time_value, &out);
#endif
  return out;
}

inline std::string format_utc_timestamp(std::chrono::system_clock::time_point time_point) {
  using namespace std::chrono;

  const auto truncated = floor<milliseconds>(time_point);
  const auto whole_seconds = time_point_cast<seconds>(truncated);
  const auto milliseconds_part = duration_cast<milliseconds>(truncated - whole_seconds).count();
  const auto calendar_time = utc_tm(system_clock::to_time_t(whole_seconds));

  std::ostringstream out;
  out << std::put_time(&calendar_time, "%Y-%m-%dT%H:%M:%S") << '.' << std::setw(3) << std::setfill('0')
      << milliseconds_part << 'Z';
  return out.str();
}

inline std::size_t grid_points(const wavefront::ProblemSpec& problem) {
  return std::accumulate(problem.grid.shape.begin(),
                         problem.grid.shape.end(),
                         std::size_t{1},
                         [](std::size_t lhs, std::size_t rhs) { return lhs * rhs; });
}

inline void log_measurement(const Measurement& measurement) {
  INFO("scenario=" << measurement.scenario << ", mode=" << measurement.mode << ", wave_type=" << measurement.wave_type
                   << ", started_at_utc=" << measurement.started_at_utc
                   << ", finished_at_utc=" << measurement.finished_at_utc
                   << ", elapsed_ms=" << measurement.elapsed_milliseconds
                   << ", steps_per_second=" << measurement.steps_per_second
                   << ", cell_updates_per_second=" << measurement.cell_updates_per_second
                   << ", nanoseconds_per_cell_update=" << measurement.nanoseconds_per_cell_update
                   << ", energy=" << measurement.energy << ", absorbed_energy=" << measurement.absorbed_energy
                   << ", diagnostics=" << measurement.diagnostics_json);
}

inline RunResult benchmark_solver_run(
    std::string_view scenario,
    wavefront::ProblemSpec problem,
    wavefront::SolverConfig config,
    std::size_t steps) {
  using namespace std::chrono;

  auto solver = wavefront::make_solver(problem, config);

  const auto started_at_wall = system_clock::now();
  const auto started_at_steady = steady_clock::now();
  solver->run(steps);
  const auto finished_at_steady = steady_clock::now();
  const auto finished_at_wall = system_clock::now();

  const auto wall_time = duration_cast<nanoseconds>(finished_at_steady - started_at_steady);
  const double elapsed_seconds = duration<double>(wall_time).count();
  const double cell_updates =
      static_cast<double>(grid_points(problem)) * static_cast<double>(problem.field_components) * static_cast<double>(steps);
  const std::string diagnostics = solver->diagnostics_json();

  Measurement measurement;
  measurement.scenario = std::string{scenario};
  measurement.mode = solver_mode_name(config.mode);
  measurement.wave_type = wave_type_name(problem.wave_type);
  measurement.started_at_utc = format_utc_timestamp(started_at_wall);
  measurement.finished_at_utc = format_utc_timestamp(finished_at_wall);
  measurement.wall_time = wall_time;
  measurement.elapsed_milliseconds = duration<double, std::milli>(wall_time).count();
  measurement.steps_per_second = elapsed_seconds > 0.0 ? static_cast<double>(steps) / elapsed_seconds : 0.0;
  measurement.cell_updates_per_second = elapsed_seconds > 0.0 ? cell_updates / elapsed_seconds : 0.0;
  measurement.nanoseconds_per_cell_update =
      cell_updates > 0.0 ? duration<double, std::nano>(wall_time).count() / cell_updates : 0.0;
  measurement.energy = test_common::json_value(diagnostics, "energy");
  measurement.absorbed_energy = test_common::json_value(diagnostics, "absorbed_energy");
  measurement.diagnostics_json = diagnostics;

  return {std::move(measurement), std::move(solver)};
}

}  // namespace performance_test
