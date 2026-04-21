#pragma once

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

#include "wavefront/api/solver.hpp"

namespace wavefront {

inline std::vector<double> snapshot_component_series(const FieldSnapshot& snapshot, std::size_t component) {
  if (component >= snapshot.components) {
    throw std::out_of_range("component out of range");
  }

  std::vector<double> series;
  if (!snapshot.values.empty()) {
    series.reserve(snapshot.values.size() / std::max<std::size_t>(1, snapshot.components));
    for (std::size_t i = component; i < snapshot.values.size(); i += snapshot.components) {
      series.push_back(snapshot.values[i]);
    }
    return series;
  }

  series.reserve(snapshot.complex_values.size() / std::max<std::size_t>(1, snapshot.components));
  for (std::size_t i = component; i < snapshot.complex_values.size(); i += snapshot.components) {
    series.push_back(snapshot.complex_values[i].magnitude());
  }
  return series;
}

inline std::pair<double, double> snapshot_minmax(const FieldSnapshot& snapshot) {
  const auto series = snapshot_component_series(snapshot, 0);
  if (series.empty()) {
    return {0.0, 0.0};
  }
  const auto [min_it, max_it] = std::minmax_element(series.begin(), series.end());
  return {*min_it, *max_it};
}

inline std::vector<double> normalize_series(const std::vector<double>& values) {
  if (values.empty()) {
    return {};
  }
  double max_abs = 0.0;
  for (double value : values) {
    max_abs = std::max(max_abs, std::fabs(value));
  }
  if (max_abs == 0.0) {
    return values;
  }
  std::vector<double> normalized(values.size(), 0.0);
  for (std::size_t i = 0; i < values.size(); ++i) {
    normalized[i] = values[i] / max_abs;
  }
  return normalized;
}

}  // namespace wavefront
