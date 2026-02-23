#pragma once

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "wavefront/api/solver.hpp"

namespace wavefront {

class GridLayout {
 public:
  explicit GridLayout(const GridSpec& spec) : dims_(spec.dims), shape_(spec.shape), spacing_(spec.spacing), origin_(spec.origin) {
    if (dims_ == 0) {
      throw std::invalid_argument("Grid dims must be positive");
    }
    if (shape_.size() != dims_) {
      throw std::invalid_argument("Grid shape size must match dims");
    }
    if (spacing_.size() != dims_) {
      throw std::invalid_argument("Grid spacing size must match dims");
    }
    if (origin_.empty()) {
      origin_.assign(dims_, 0.0);
    }
    if (origin_.size() != dims_) {
      throw std::invalid_argument("Grid origin size must match dims");
    }

    strides_.assign(dims_, 1);
    for (std::size_t i = dims_; i-- > 1;) {
      strides_[i - 1] = strides_[i] * shape_[i];
    }
  }

  std::size_t dims() const { return dims_; }
  const std::vector<std::size_t>& shape() const { return shape_; }
  const std::vector<double>& spacing() const { return spacing_; }
  const std::vector<double>& origin() const { return origin_; }
  const std::vector<std::size_t>& strides() const { return strides_; }

  std::size_t total_points() const {
    std::size_t product = 1;
    for (std::size_t extent : shape_) {
      product *= extent;
    }
    return product;
  }

  double min_spacing() const { return *std::min_element(spacing_.begin(), spacing_.end()); }

  std::size_t flatten_index(const std::vector<std::size_t>& index) const {
    if (index.size() != dims_) {
      throw std::invalid_argument("Index rank mismatch for flatten_index");
    }
    std::size_t flat = 0;
    for (std::size_t axis = 0; axis < dims_; ++axis) {
      if (index[axis] >= shape_[axis]) {
        throw std::out_of_range("Index out of bounds for axis " + std::to_string(axis));
      }
      flat += index[axis] * strides_[axis];
    }
    return flat;
  }

  std::vector<std::size_t> unravel_index(std::size_t flat) const {
    if (flat >= total_points()) {
      throw std::out_of_range("Flat index out of bounds");
    }
    std::vector<std::size_t> index(dims_, 0);
    for (std::size_t axis = 0; axis < dims_; ++axis) {
      index[axis] = (flat / strides_[axis]) % shape_[axis];
    }
    return index;
  }

  bool is_boundary_cell(const std::vector<std::size_t>& index, std::size_t axis, bool upper_face) const {
    if (index.size() != dims_ || axis >= dims_) {
      return false;
    }
    return upper_face ? (index[axis] + 1 == shape_[axis]) : (index[axis] == 0);
  }

 private:
  std::size_t dims_;
  std::vector<std::size_t> shape_;
  std::vector<double> spacing_;
  std::vector<double> origin_;
  std::vector<std::size_t> strides_;
};

}  // namespace wavefront
