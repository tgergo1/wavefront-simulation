#pragma once

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

#include "wavefront/core/grid.hpp"

namespace wavefront {

template <typename Scalar>
class FieldBuffer {
 public:
  FieldBuffer() = default;

  FieldBuffer(GridLayout grid, std::size_t components)
      : grid_(std::move(grid)), components_(components), values_(grid_.total_points() * components_, Scalar{}) {
    if (components_ == 0) {
      throw std::invalid_argument("FieldBuffer components must be positive");
    }
  }

  void fill(Scalar value) { std::fill(values_.begin(), values_.end(), value); }

  std::size_t components() const { return components_; }
  std::size_t points() const { return grid_.total_points(); }
  const GridLayout& grid() const { return grid_; }

  Scalar& at_flat(std::size_t flat, std::size_t component) {
    return values_.at(flat * components_ + component);
  }

  const Scalar& at_flat(std::size_t flat, std::size_t component) const {
    return values_.at(flat * components_ + component);
  }

  Scalar& at_index(const std::vector<std::size_t>& index, std::size_t component) {
    return at_flat(grid_.flatten_index(index), component);
  }

  const Scalar& at_index(const std::vector<std::size_t>& index, std::size_t component) const {
    return at_flat(grid_.flatten_index(index), component);
  }

  std::vector<Scalar>& data() { return values_; }
  const std::vector<Scalar>& data() const { return values_; }

 private:
  GridLayout grid_ = GridLayout(GridSpec{1, {1}, {1.0}, {0.0}});
  std::size_t components_ = 1;
  std::vector<Scalar> values_;
};

}  // namespace wavefront
