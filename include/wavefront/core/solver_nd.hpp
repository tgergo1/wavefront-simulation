#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "wavefront/api/solver.hpp"

namespace wavefront {

template <std::size_t N, typename Scalar, SolverMode Mode>
class SolverND {
 public:
  using Index = std::array<std::size_t, N>;

  SolverND(const std::array<std::size_t, N>& shape, const std::array<Scalar, N>& spacing, std::size_t components = 1)
      : shape_(shape), spacing_(spacing), components_(components) {
    if (components_ == 0) {
      throw std::invalid_argument("components must be positive");
    }
    for (std::size_t axis = 0; axis < N; ++axis) {
      if (shape_[axis] < 3) {
        throw std::invalid_argument("each axis requires at least 3 points");
      }
      if (spacing_[axis] <= Scalar{0}) {
        throw std::invalid_argument("spacing must be positive");
      }
    }

    strides_[N - 1] = 1;
    for (std::size_t axis = N - 1; axis > 0; --axis) {
      strides_[axis - 1] = strides_[axis] * shape_[axis];
    }

    total_points_ = 1;
    for (std::size_t axis = 0; axis < N; ++axis) {
      total_points_ *= shape_[axis];
    }

    u_prev_.assign(total_points_ * components_, Scalar{0});
    u_curr_.assign(total_points_ * components_, Scalar{0});
    u_next_.assign(total_points_ * components_, Scalar{0});

    Scalar min_spacing = spacing_[0];
    for (std::size_t axis = 1; axis < N; ++axis) {
      min_spacing = std::min(min_spacing, spacing_[axis]);
    }
    dt_ = Scalar{0.4} * min_spacing / wave_speed_;
  }

  void set_wave_speed(Scalar value) {
    if (value <= Scalar{0}) {
      throw std::invalid_argument("wave speed must be positive");
    }
    wave_speed_ = value;
  }

  void set_time_step(Scalar value) {
    if (value <= Scalar{0}) {
      throw std::invalid_argument("time step must be positive");
    }
    dt_ = value;
  }

  void set_nonlinear_coefficient(Scalar value) { nonlinear_coeff_ = value; }
  void set_micro_gradient_coefficient(Scalar value) { micro_grad_coeff_ = value; }
  void set_memory_attenuation(Scalar value) { memory_coeff_ = value; }
  void set_source_amplitude(Scalar value) { source_amplitude_ = value; }

  void step() {
    const Scalar c2dt2 = wave_speed_ * wave_speed_ * dt_ * dt_;

    for (std::size_t flat = 0; flat < total_points_; ++flat) {
      for (std::size_t component = 0; component < components_; ++component) {
        const std::size_t offset = flat * components_ + component;
        const Scalar u = u_curr_[offset];
        const Scalar lap = laplacian(flat, component);

        Scalar forcing = source_amplitude_ * std::sin(static_cast<Scalar>(steps_) * dt_);

        Scalar correction = Scalar{0};
        if constexpr (Mode == SolverMode::NonlinearContinuum) {
          correction += nonlinear_coeff_ * u * u * u;
        }
        if constexpr (Mode == SolverMode::MicroSurrogate) {
          correction += micro_grad_coeff_ * lap;
          const Scalar memory = (u_curr_[offset] - u_prev_[offset]) / dt_;
          correction -= memory_coeff_ * memory;
        }

        u_next_[offset] = Scalar{2} * u_curr_[offset] - u_prev_[offset] + c2dt2 * lap + dt_ * dt_ * (forcing + correction);
      }
    }

    u_prev_.swap(u_curr_);
    u_curr_.swap(u_next_);
    ++steps_;
  }

  void run(std::size_t count) {
    for (std::size_t i = 0; i < count; ++i) {
      step();
    }
  }

  Scalar sample(const Index& index, std::size_t component = 0) const {
    if (component >= components_) {
      throw std::out_of_range("component out of range");
    }
    return u_curr_[flatten(index) * components_ + component];
  }

  std::size_t steps() const { return steps_; }

  std::string diagnostics_json() const {
    std::ostringstream out;
    out << "{"
        << "\"dims\":" << N << ","
        << "\"mode\":" << static_cast<int>(Mode) << ","
        << "\"steps\":" << steps_ << ","
        << "\"dt\":" << static_cast<long double>(dt_) << "}";
    return out.str();
  }

 private:
  std::size_t flatten(const Index& index) const {
    std::size_t flat = 0;
    for (std::size_t axis = 0; axis < N; ++axis) {
      if (index[axis] >= shape_[axis]) {
        throw std::out_of_range("index out of bounds");
      }
      flat += index[axis] * strides_[axis];
    }
    return flat;
  }

  Index unravel(std::size_t flat) const {
    Index index{};
    for (std::size_t axis = 0; axis < N; ++axis) {
      index[axis] = (flat / strides_[axis]) % shape_[axis];
    }
    return index;
  }

  Scalar laplacian(std::size_t flat, std::size_t component) const {
    const Index center = unravel(flat);

    Scalar sum = Scalar{0};
    for (std::size_t axis = 0; axis < N; ++axis) {
      Index lower = center;
      Index upper = center;

      if (center[axis] == 0) {
        lower[axis] = shape_[axis] - 1;
      } else {
        lower[axis] -= 1;
      }

      if (center[axis] + 1 == shape_[axis]) {
        upper[axis] = 0;
      } else {
        upper[axis] += 1;
      }

      const Scalar u_center = u_curr_[flat * components_ + component];
      const Scalar u_lower = u_curr_[flatten(lower) * components_ + component];
      const Scalar u_upper = u_curr_[flatten(upper) * components_ + component];

      const Scalar inv_h2 = Scalar{1} / (spacing_[axis] * spacing_[axis]);
      sum += (u_upper - Scalar{2} * u_center + u_lower) * inv_h2;
    }

    return sum;
  }

  std::array<std::size_t, N> shape_{};
  std::array<std::size_t, N> strides_{};
  std::array<Scalar, N> spacing_{};
  std::size_t components_ = 1;
  std::size_t total_points_ = 0;
  std::size_t steps_ = 0;

  Scalar wave_speed_ = Scalar{1};
  Scalar dt_ = Scalar{0.01};
  Scalar nonlinear_coeff_ = Scalar{0.05};
  Scalar micro_grad_coeff_ = Scalar{0.02};
  Scalar memory_coeff_ = Scalar{0.01};
  Scalar source_amplitude_ = Scalar{0};

  std::vector<Scalar> u_prev_;
  std::vector<Scalar> u_curr_;
  std::vector<Scalar> u_next_;
};

}  // namespace wavefront
