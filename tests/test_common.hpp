#pragma once

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "wavefront/api/solver.hpp"

namespace test_common {

inline wavefront::ProblemSpec default_problem_1d(std::size_t points = 128) {
  wavefront::ProblemSpec problem;
  problem.grid.dims = 1;
  problem.grid.shape = {points};
  problem.grid.spacing = {0.02};
  problem.grid.origin = {0.0};
  problem.field_components = 1;
  problem.medium.density.text = "1.0";
  problem.medium.stiffness.text = "1.0";
  problem.medium.damping.text = "0.0";
  problem.medium.dispersion.text = "0.01";
  problem.source_term.text = "0.0";
  problem.boundaries = {
      wavefront::BoundarySpec{wavefront::BoundaryType::Periodic, 0, false, wavefront::SymbolicExpr{"0.0"}},
      wavefront::BoundarySpec{wavefront::BoundaryType::Periodic, 0, true, wavefront::SymbolicExpr{"0.0"}},
  };
  return problem;
}

inline wavefront::SolverConfig default_config(
    wavefront::SolverMode mode,
    wavefront::PrecisionMode precision = wavefront::PrecisionMode::FastFloat64) {
  wavefront::SolverConfig config;
  config.mode = mode;
  config.precision = precision;
  config.cfl = 0.3;
  config.max_steps = 0;
  config.threads = 2;
  config.deterministic = true;
  config.spatial_order = 4;
  config.split_pml = true;
  config.reference_window = 32;
  return config;
}

inline double json_value(const std::string& json, const std::string& key) {
  const std::string needle = "\"" + key + "\":";
  const std::size_t begin = json.find(needle);
  if (begin == std::string::npos) {
    throw std::runtime_error("key not found in diagnostics: " + key);
  }

  std::size_t value_begin = begin + needle.size();
  std::size_t value_end = value_begin;
  while (value_end < json.size() &&
         (std::isdigit(static_cast<unsigned char>(json[value_end])) != 0 || json[value_end] == '.' ||
          json[value_end] == 'e' || json[value_end] == 'E' || json[value_end] == '+' || json[value_end] == '-')) {
    ++value_end;
  }

  return std::stod(json.substr(value_begin, value_end - value_begin));
}

inline std::vector<double> sample_line(wavefront::ISolver& solver, std::size_t points) {
  std::vector<double> out(points, 0.0);
  for (std::size_t i = 0; i < points; ++i) {
    out[i] = solver.sample(std::vector<std::size_t>{i}).at(0);
  }
  return out;
}

inline double l2_norm(const std::vector<double>& values) {
  double sum = 0.0;
  for (double value : values) {
    sum += value * value;
  }
  return std::sqrt(sum / static_cast<double>(std::max<std::size_t>(1, values.size())));
}

inline double l2_error(const std::vector<double>& lhs, const std::vector<double>& rhs) {
  if (lhs.size() != rhs.size()) {
    throw std::invalid_argument("L2 error vectors must have the same shape");
  }

  std::vector<double> diff(lhs.size(), 0.0);
  for (std::size_t i = 0; i < lhs.size(); ++i) {
    diff[i] = lhs[i] - rhs[i];
  }
  return l2_norm(diff);
}

}  // namespace test_common
