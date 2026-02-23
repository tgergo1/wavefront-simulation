#include "runtime_solver.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "wavefront/api/problem_validation.hpp"
#include "wavefront/core/field.hpp"
#include "wavefront/core/grid.hpp"
#include "wavefront/core/threading.hpp"
#include "wavefront/exact/exact_number.hpp"
#include "wavefront/physics/boundary.hpp"
#include "wavefront/physics/interface.hpp"
#include "wavefront/symbolic/expression.hpp"

namespace wavefront {
namespace {

CompiledExpression compile_or_default(const SymbolicExpr& expression, const char* fallback) {
  return CompiledExpression::compile(expression.text.empty() ? fallback : expression.text);
}

double positive_or_floor(double value, double floor) {
  return value > floor ? value : floor;
}

std::string mode_name(SolverMode mode) {
  switch (mode) {
    case SolverMode::LinearApprox:
      return "LinearApprox";
    case SolverMode::NonlinearContinuum:
      return "NonlinearContinuum";
    case SolverMode::MicroSurrogate:
      return "MicroSurrogate";
  }
  return "Unknown";
}

std::string precision_name(PrecisionMode mode) {
  switch (mode) {
    case PrecisionMode::FastFloat64:
      return "FastFloat64";
    case PrecisionMode::ExactReference:
      return "ExactReference";
  }
  return "Unknown";
}

class RuntimeSolver final : public ISolver {
 public:
  void initialize(const ProblemSpec& problem, const SolverConfig& config) override {
    throw_if_invalid(problem, config);

    problem_ = problem;
    config_ = config;
    grid_ = GridLayout(problem_.grid);
    previous_ = FieldBuffer<double>(grid_, problem_.field_components);
    current_ = FieldBuffer<double>(grid_, problem_.field_components);
    next_ = FieldBuffer<double>(grid_, problem_.field_components);

    previous_.fill(0.0);
    current_.fill(0.0);
    next_.fill(0.0);

    density_expr_ = compile_or_default(problem_.medium.density, "1.0");
    stiffness_expr_ = compile_or_default(problem_.medium.stiffness, "1.0");
    damping_expr_ = compile_or_default(problem_.medium.damping, "0.0");
    dispersion_expr_ = compile_or_default(problem_.medium.dispersion, "0.0");
    source_expr_ = compile_or_default(problem_.source_term, "0.0");

    EvaluationContext center_context;
    center_context.x.resize(grid_.dims(), 0.0L);
    center_context.u.resize(problem_.field_components, 0.0L);

    for (std::size_t axis = 0; axis < grid_.dims(); ++axis) {
      center_context.x[axis] = static_cast<long double>(grid_.origin()[axis] +
                                                        (grid_.shape()[axis] / 2) * grid_.spacing()[axis]);
    }

    const double density = positive_or_floor(density_expr_.evaluate_double(center_context), 1.0e-12);
    const double stiffness = positive_or_floor(stiffness_expr_.evaluate_double(center_context), 1.0e-12);
    const double speed = phase_velocity(stiffness, density);
    dt_ = (config_.cfl * grid_.min_spacing()) / (speed * std::sqrt(static_cast<double>(grid_.dims())));

    const std::size_t center_flat = center_flat_index();
    for (std::size_t component = 0; component < problem_.field_components; ++component) {
      current_.at_flat(center_flat, component) = 1.0e-6;
      previous_.at_flat(center_flat, component) = 1.0e-6;
    }

    step_count_ = 0;
    last_energy_ = compute_energy();
    max_reference_error_ = 0.0;
    total_reflected_energy_ = 0.0;
    total_absorbed_energy_ = 0.0;
    initialized_ = true;
  }

  void step() override {
    if (!initialized_) {
      throw std::logic_error("RuntimeSolver is not initialized");
    }

    const std::size_t points = grid_.total_points();
    const std::size_t components = problem_.field_components;

    deterministic_parallel_for(
        points,
        config_.threads,
        config_.deterministic,
        [&](std::size_t begin, std::size_t end) {
          for (std::size_t flat = begin; flat < end; ++flat) {
            for (std::size_t component = 0; component < components; ++component) {
              update_cell(flat, component);
            }
          }
        });

    for (std::size_t component = 0; component < components; ++component) {
      const BoundaryMetrics metrics = apply_boundary_conditions(
          grid_,
          problem_.boundaries,
          pml_memory_,
          next_,
          current_,
          previous_,
          component,
          config_.split_pml,
          dt_,
          8.0 / (grid_.min_spacing() * static_cast<double>(grid_.dims())));
      total_reflected_energy_ += metrics.reflected_energy;
      total_absorbed_energy_ += metrics.absorbed_energy;
    }

    previous_.data().swap(current_.data());
    current_.data().swap(next_.data());

    ++step_count_;
    last_energy_ = compute_energy();
  }

  void run(std::size_t steps) override {
    if (!initialized_) {
      throw std::logic_error("RuntimeSolver is not initialized");
    }

    std::size_t allowed_steps = steps;
    if (config_.max_steps > 0) {
      if (step_count_ >= config_.max_steps) {
        return;
      }
      allowed_steps = std::min(steps, config_.max_steps - step_count_);
    }

    for (std::size_t i = 0; i < allowed_steps; ++i) {
      step();
    }
  }

  std::vector<double> sample(std::span<const std::size_t> index) const override {
    if (!initialized_) {
      throw std::logic_error("RuntimeSolver is not initialized");
    }

    std::vector<std::size_t> index_vec(index.begin(), index.end());
    const std::size_t flat = grid_.flatten_index(index_vec);

    std::vector<double> values(problem_.field_components, 0.0);
    for (std::size_t component = 0; component < problem_.field_components; ++component) {
      values[component] = current_.at_flat(flat, component);
    }
    return values;
  }

  std::string diagnostics_json() const override {
    std::ostringstream out;
    out << "{"
        << "\"dims\":" << grid_.dims() << ","
        << "\"steps\":" << step_count_ << ","
        << "\"dt\":" << dt_ << ","
        << "\"mode\":\"" << mode_name(config_.mode) << "\"," 
        << "\"precision\":\"" << precision_name(config_.precision) << "\"," 
        << "\"energy\":" << last_energy_ << ","
        << "\"max_reference_error\":" << max_reference_error_ << ","
        << "\"reflected_energy\":" << total_reflected_energy_ << ","
        << "\"absorbed_energy\":" << total_absorbed_energy_ << "}";
    return out.str();
  }

 private:
  std::size_t center_flat_index() const {
    std::vector<std::size_t> center(grid_.dims(), 0);
    for (std::size_t axis = 0; axis < grid_.dims(); ++axis) {
      center[axis] = grid_.shape()[axis] / 2;
    }
    return grid_.flatten_index(center);
  }

  std::vector<std::size_t> wrapped_neighbor(const std::vector<std::size_t>& index, std::size_t axis, int offset) const {
    std::vector<std::size_t> result = index;
    const std::size_t extent = grid_.shape()[axis];
    const long long base = static_cast<long long>(index[axis]);
    const long long shifted = (base + offset) % static_cast<long long>(extent);
    result[axis] = static_cast<std::size_t>(shifted < 0 ? shifted + static_cast<long long>(extent) : shifted);
    return result;
  }

  double laplacian_at(const FieldBuffer<double>& field, std::size_t flat, std::size_t component) const {
    const std::vector<std::size_t> center = grid_.unravel_index(flat);

    double lap = 0.0;
    const double u0 = field.at_flat(flat, component);
    for (std::size_t axis = 0; axis < grid_.dims(); ++axis) {
      const double h = grid_.spacing()[axis];
      const double inv_h2 = 1.0 / (h * h);

      if (config_.spatial_order == 4 && grid_.shape()[axis] >= 5) {
        const auto i_m2 = wrapped_neighbor(center, axis, -2);
        const auto i_m1 = wrapped_neighbor(center, axis, -1);
        const auto i_p1 = wrapped_neighbor(center, axis, +1);
        const auto i_p2 = wrapped_neighbor(center, axis, +2);

        const double u_m2 = field.at_index(i_m2, component);
        const double u_m1 = field.at_index(i_m1, component);
        const double u_p1 = field.at_index(i_p1, component);
        const double u_p2 = field.at_index(i_p2, component);

        lap += (-u_p2 + 16.0 * u_p1 - 30.0 * u0 + 16.0 * u_m1 - u_m2) * (inv_h2 / 12.0);
      } else {
        const auto i_m1 = wrapped_neighbor(center, axis, -1);
        const auto i_p1 = wrapped_neighbor(center, axis, +1);

        const double u_m1 = field.at_index(i_m1, component);
        const double u_p1 = field.at_index(i_p1, component);
        lap += (u_p1 - 2.0 * u0 + u_m1) * inv_h2;
      }
    }

    return lap;
  }

  double directional_gradient(const FieldBuffer<double>& field, std::size_t flat, std::size_t component, std::size_t axis) const {
    const std::vector<std::size_t> center = grid_.unravel_index(flat);
    const auto i_m1 = wrapped_neighbor(center, axis, -1);
    const auto i_p1 = wrapped_neighbor(center, axis, +1);

    const double u_m1 = field.at_index(i_m1, component);
    const double u_p1 = field.at_index(i_p1, component);
    return (u_p1 - u_m1) / (2.0 * grid_.spacing()[axis]);
  }

  EvaluationContext build_context(std::size_t flat, std::size_t component, long double t) const {
    const std::vector<std::size_t> index = grid_.unravel_index(flat);

    EvaluationContext context;
    context.x.resize(grid_.dims(), 0.0L);
    context.t = t;
    context.u.resize(problem_.field_components, 0.0L);

    for (std::size_t axis = 0; axis < grid_.dims(); ++axis) {
      context.x[axis] = static_cast<long double>(grid_.origin()[axis] + index[axis] * grid_.spacing()[axis]);
    }

    for (std::size_t comp = 0; comp < problem_.field_components; ++comp) {
      context.u[comp] = current_.at_flat(flat, comp);
      for (std::size_t axis = 0; axis < grid_.dims(); ++axis) {
        const double derivative = directional_gradient(current_, flat, comp, axis);
        context.derivatives.emplace("du" + std::to_string(comp) + "_dx" + std::to_string(axis), derivative);
      }
    }

    context.extra.emplace("component", static_cast<long double>(component));
    return context;
  }

  void update_cell(std::size_t flat, std::size_t component) {
    const double current = current_.at_flat(flat, component);
    const double previous = previous_.at_flat(flat, component);

    const EvaluationContext context = build_context(flat, component, static_cast<long double>(step_count_) * dt_);

    const double density = positive_or_floor(density_expr_.evaluate_double(context), 1.0e-12);
    const double stiffness = positive_or_floor(stiffness_expr_.evaluate_double(context), 1.0e-12);
    const double damping = std::max(0.0, damping_expr_.evaluate_double(context));
    const double dispersion = dispersion_expr_.evaluate_double(context);
    const double source = source_expr_.evaluate_double(context);

    const double c2 = stiffness / density;
    const double lap = laplacian_at(current_, flat, component);
    const double velocity = (current - previous) / dt_;
    const double dt2 = dt_ * dt_;

    double rhs = c2 * lap + source - damping * velocity;

    if (config_.mode == SolverMode::NonlinearContinuum) {
      rhs += dispersion * current * current * current;
    } else if (config_.mode == SolverMode::MicroSurrogate) {
      const double grad0 = directional_gradient(current_, flat, component, 0);
      const double grad1 = grid_.dims() > 1 ? directional_gradient(current_, flat, component, 1) : 0.0;
      const double anisotropy = grad0 * grad0 - grad1 * grad1;
      const double memory_kernel = -0.04 * std::tanh(velocity);
      const double gradient_correction = 0.1 * dispersion * lap;
      rhs += gradient_correction + memory_kernel + 0.05 * anisotropy;
    }

    const double next = 2.0 * current - previous + dt2 * rhs;
    next_.at_flat(flat, component) = next;

    if (config_.precision == PrecisionMode::ExactReference) {
      const std::size_t checked = flat % grid_.total_points();
      if (checked < config_.reference_window) {
        const exact::ExactNumber ex_prev(static_cast<long double>(previous));
        const exact::ExactNumber ex_curr(static_cast<long double>(current));
        const exact::ExactNumber ex_two(static_cast<std::int64_t>(2));
        const exact::ExactNumber ex_dt2(static_cast<long double>(dt2));
        const exact::ExactNumber ex_rhs(static_cast<long double>(rhs));

        const exact::ExactNumber ex_next = ex_two * ex_curr - ex_prev + ex_dt2 * ex_rhs;
        const auto interval = exact::certify_nonlinear_operation(ex_next, next, 1.0e-10L, 1.0e-12L);
        const double abs_error = std::fabs(static_cast<double>(ex_next.to_long_double() - next));
        max_reference_error_ = std::max(max_reference_error_, abs_error);
        if (!interval.contains(next)) {
          max_reference_error_ = std::max(max_reference_error_, static_cast<double>(interval.width()));
        }
      }
    }
  }

  double compute_energy() const {
    double energy = 0.0;
    for (std::size_t flat = 0; flat < grid_.total_points(); ++flat) {
      for (std::size_t component = 0; component < problem_.field_components; ++component) {
        const double u = current_.at_flat(flat, component);
        const double velocity = (current_.at_flat(flat, component) - previous_.at_flat(flat, component)) / dt_;
        energy += 0.5 * (u * u + velocity * velocity);
      }
    }
    return energy;
  }

  ProblemSpec problem_;
  SolverConfig config_;
  GridLayout grid_ = GridLayout(GridSpec{1, {3}, {1.0}, {0.0}});

  FieldBuffer<double> previous_;
  FieldBuffer<double> current_;
  FieldBuffer<double> next_;
  std::vector<double> pml_memory_;

  CompiledExpression density_expr_;
  CompiledExpression stiffness_expr_;
  CompiledExpression damping_expr_;
  CompiledExpression dispersion_expr_;
  CompiledExpression source_expr_;

  std::size_t step_count_ = 0;
  double dt_ = 1.0e-3;
  double last_energy_ = 0.0;
  double max_reference_error_ = 0.0;
  double total_reflected_energy_ = 0.0;
  double total_absorbed_energy_ = 0.0;
  bool initialized_ = false;
};

}  // namespace

std::unique_ptr<ISolver> make_runtime_solver() {
  return std::make_unique<RuntimeSolver>();
}

}  // namespace wavefront
