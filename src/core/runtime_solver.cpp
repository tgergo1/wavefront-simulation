#include "runtime_solver.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
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

constexpr double kPi = 3.14159265358979323846;

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
  return "LinearApprox";  // GCOVR_EXCL_LINE
}

std::string family_name(SolverFamily family) {
  switch (family) {
    case SolverFamily::TimeDomain:
      return "TimeDomain";
    case SolverFamily::FrequencyDomain:
      return "FrequencyDomain";
    case SolverFamily::AngularSpectrum:
      return "AngularSpectrum";
  }
  return "TimeDomain";  // GCOVR_EXCL_LINE
}

std::string wave_type_name(WaveType wave_type) {
  switch (wave_type) {
    case WaveType::Transverse:
      return "Transverse";
    case WaveType::Longitudinal:
      return "Longitudinal";
  }
  return "Transverse";  // GCOVR_EXCL_LINE
}

std::string precision_name(PrecisionMode mode) {
  switch (mode) {
    case PrecisionMode::FastFloat64:
      return "FastFloat64";
    case PrecisionMode::ExactReference:
      return "ExactReference";
  }
  return "FastFloat64";  // GCOVR_EXCL_LINE
}

std::string backend_name(ExecutionBackend backend) {
  switch (backend) {
    case ExecutionBackend::Serial:
      return "Serial";
    case ExecutionBackend::ThreadedCPU:
      return "ThreadedCPU";
    case ExecutionBackend::GPUAccelerated:
      return "GPUAccelerated";
    case ExecutionBackend::Distributed:
      return "Distributed";
  }
  return "ThreadedCPU";  // GCOVR_EXCL_LINE
}

std::string representation_name(FieldRepresentation representation) {
  switch (representation) {
    case FieldRepresentation::RealScalar:
      return "RealScalar";
    case FieldRepresentation::ComplexPhasor:
      return "ComplexPhasor";
  }
  return "RealScalar";  // GCOVR_EXCL_LINE
}

struct FaceBoundary {
  BoundaryType type = BoundaryType::Neumann;
  double parameter = 0.0;
  bool specified = false;
};

struct CompiledGeometryRegion {
  GeometryRegion spec;
  CompiledExpression density;
  CompiledExpression stiffness;
  CompiledExpression damping;
  CompiledExpression dispersion;
};

double parse_boundary_parameter_scalar(const SymbolicExpr& expression, double fallback) {
  try {
    std::size_t pos = 0;
    const double value = std::stod(expression.text, &pos);
    if (pos == expression.text.size()) {
      return value;
    }
  } catch (...) {
  }
  return fallback;
}

bool contains_region(const GeometryRegion& region, const std::vector<long double>& x) {
  switch (region.shape) {
    case GeometryShape::Box:
      for (std::size_t axis = 0; axis < x.size(); ++axis) {
        if (x[axis] < static_cast<long double>(region.min_corner[axis]) ||
            x[axis] > static_cast<long double>(region.max_corner[axis])) {
          return false;
        }
      }
      return true;
    case GeometryShape::Sphere: {
      long double radius_sq = 0.0L;
      for (std::size_t axis = 0; axis < x.size(); ++axis) {
        const long double delta = x[axis] - static_cast<long double>(region.center[axis]);
        radius_sq += delta * delta;
      }
      return radius_sq <= static_cast<long double>(region.radius * region.radius);
    }
    case GeometryShape::Layer:
      return x[region.axis] >= static_cast<long double>(region.lower) &&
             x[region.axis] <= static_cast<long double>(region.upper);
  }
  return false;  // GCOVR_EXCL_LINE
}

class RuntimeSolver final : public ISolver {
 public:
  void initialize(const ProblemSpec& problem, const SolverConfig& config) override {
    throw_if_invalid(problem, config);

    problem_ = problem;
    config_ = config;
    grid_ = GridLayout(problem_.grid);
    resolve_backend();
    configure_face_boundaries();
    compile_geometry_regions();

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

    double max_speed = 0.0;
    {
      std::mutex mtx;
      deterministic_parallel_for(
          grid_.total_points(),
          active_thread_count(),
          config_.deterministic,
          [&](std::size_t begin, std::size_t end) {
            double local_max = 0.0;
            for (std::size_t flat = begin; flat < end; ++flat) {
              const EvaluationContext context = coefficient_context(flat);
              const double density = positive_or_floor(evaluate_density(context), 1.0e-12);
              const double stiffness = positive_or_floor(evaluate_stiffness(context), 1.0e-12);
              local_max = std::max(local_max, phase_velocity(stiffness, density));
            }
            std::lock_guard<std::mutex> lock(mtx);
            max_speed = std::max(max_speed, local_max);
          });
    }
    max_speed = positive_or_floor(max_speed, 1.0e-12);
    dt_ = (config_.cfl * grid_.min_spacing()) / (max_speed * std::sqrt(static_cast<double>(grid_.dims())));

    step_count_ = 0;
    last_energy_ = compute_energy();
    max_reference_error_ = 0.0;
    total_reflected_energy_ = 0.0;
    total_absorbed_energy_ = 0.0;
    reset_monitor_state();
    record_monitors();
  }

  void step() override {
    const std::size_t points = grid_.total_points();
    const std::size_t components = problem_.field_components;

    deterministic_parallel_for(
        points,
        active_thread_count(),
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
          8.0 / (grid_.min_spacing() * static_cast<double>(grid_.dims())),
          active_thread_count(),
          config_.deterministic);
      total_reflected_energy_ += metrics.reflected_energy;
      total_absorbed_energy_ += metrics.absorbed_energy;
    }

    previous_.data().swap(current_.data());
    current_.data().swap(next_.data());

    ++step_count_;
    last_energy_ = compute_energy();
    record_monitors();
  }

  void run(std::size_t steps) override {
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
    out << "{";
    out << "\"dims\":" << grid_.dims();
    out << ",\"steps\":" << step_count_;
    out << ",\"dt\":" << dt_;
    out << ",\"mode\":\"" << mode_name(config_.mode) << "\"";
    out << ",\"family\":\"" << family_name(config_.family) << "\"";
    out << ",\"precision\":\"" << precision_name(config_.precision) << "\"";
    out << ",\"wave_type\":\"" << wave_type_name(problem_.wave_type) << "\"";
    out << ",\"requested_backend\":\"" << backend_name(config_.backend) << "\"";
    out << ",\"active_backend\":\"" << backend_name(active_backend_) << "\"";
    out << ",\"representation\":\"" << representation_name(config_.representation) << "\"";
    out << ",\"energy\":" << last_energy_;
    out << ",\"max_reference_error\":" << max_reference_error_;
    out << ",\"reflected_energy\":" << total_reflected_energy_;
    out << ",\"absorbed_energy\":" << total_absorbed_energy_;
    out << ",\"probe_samples\":" << probe_samples_.size();
    out << ",\"surface_monitors\":" << surface_flux_results_.size();
    out << ",\"geometry_regions\":" << geometry_regions_.size();
    out << "}";
    return out.str();
  }

  FieldSnapshot field_snapshot() const override {
    FieldSnapshot snapshot;
    snapshot.step = step_count_;
    snapshot.time = static_cast<double>(step_count_) * dt_;
    snapshot.shape = grid_.shape();
    snapshot.components = problem_.field_components;
    snapshot.representation = config_.representation;
    snapshot.values = current_.data();

    if (config_.representation == FieldRepresentation::ComplexPhasor ||
        config_.family == SolverFamily::FrequencyDomain) {
      snapshot.complex_values.reserve(current_.data().size());
      for (std::size_t flat = 0; flat < grid_.total_points(); ++flat) {
        for (std::size_t component = 0; component < problem_.field_components; ++component) {
          snapshot.complex_values.push_back(complex_at_flat(flat, component));
        }
      }
    }

    return snapshot;
  }

  std::vector<ProbeSample> probe_history(std::string_view name = {}) const override {
    if (name.empty()) {
      return probe_samples_;
    }

    std::vector<ProbeSample> filtered;
    for (const auto& sample_value : probe_samples_) {
      if (sample_value.name == name) {
        filtered.push_back(sample_value);
      }
    }
    return filtered;
  }

  std::vector<SpectrumSample> probe_spectrum(std::string_view name, std::size_t bins = 0) const override {
    const auto samples = probe_history(name);
    if (samples.empty()) {
      return {};
    }

    const std::size_t spectrum_bins = bins > 0 ? bins
                                               : std::max<std::size_t>(1, std::min(samples.size(),
                                                                                    problem_.monitors.spectrum_bins > 0
                                                                                        ? problem_.monitors.spectrum_bins
                                                                                        : samples.size()));
    std::vector<SpectrumSample> spectrum(spectrum_bins);
    for (std::size_t k = 0; k < spectrum_bins; ++k) {
      double real = 0.0;
      double imag = 0.0;
      for (std::size_t n = 0; n < samples.size(); ++n) {
        const double angle = -2.0 * kPi * static_cast<double>(k * n) / static_cast<double>(samples.size());
        real += samples[n].value * std::cos(angle);
        imag += samples[n].value * std::sin(angle);
      }
      spectrum[k].frequency = static_cast<double>(k) / (dt_ * static_cast<double>(samples.size()));
      spectrum[k].magnitude = std::sqrt(real * real + imag * imag) / static_cast<double>(samples.size());
    }
    return spectrum;
  }

  SurfaceFluxResult surface_flux(std::string_view name) const override {
    for (const auto& flux : surface_flux_results_) {
      if (flux.name == name) {
        return flux;
      }
    }
    throw std::out_of_range("surface monitor not found");
  }

  FarFieldPattern far_field_pattern(std::size_t samples = 0) const override {
    const std::size_t sample_count = samples > 0 ? samples : config_.far_field_samples;
    if (sample_count == 0) {
      return {};
    }

    const std::vector<double> line = extract_axis_line(0);
    FarFieldPattern pattern;
    pattern.step = step_count_;
    pattern.time = static_cast<double>(step_count_) * dt_;
    pattern.angles.reserve(sample_count);
    pattern.amplitudes.reserve(sample_count);

    for (std::size_t k = 0; k < sample_count; ++k) {
      const double angle = sample_count == 1 ? 0.0
                                             : (-0.5 * kPi + kPi * static_cast<double>(k) / static_cast<double>(sample_count - 1));
      double real = 0.0;
      double imag = 0.0;
      for (std::size_t n = 0; n < line.size(); ++n) {
        const double phase = std::sin(angle) * static_cast<double>(n);
        real += line[n] * std::cos(phase);
        imag += line[n] * std::sin(phase);
      }
      pattern.angles.push_back(angle);
      pattern.amplitudes.push_back(std::sqrt(real * real + imag * imag) /
                                   static_cast<double>(std::max<std::size_t>(1, line.size())));
    }

    return pattern;
  }

  void save_checkpoint(std::string_view path) const override {
    std::ofstream out{std::string(path)};
    if (!out) {
      throw std::runtime_error("failed to open checkpoint for writing");
    }

    out << "WAVEFRONT_CHECKPOINT_V1\n";
    out << "step " << step_count_ << "\n";
    out << "dt " << dt_ << "\n";
    out << "dims " << grid_.dims() << "\n";
    out << "components " << problem_.field_components << "\n";
    out << "shape";
    for (const auto value : grid_.shape()) {
      out << ' ' << value;
    }
    out << "\ncurrent";
    for (const auto value : current_.data()) {
      out << ' ' << value;
    }
    out << "\nprevious";
    for (const auto value : previous_.data()) {
      out << ' ' << value;
    }
    out << "\n";
  }

  void load_checkpoint(std::string_view path) override {
    std::ifstream in{std::string(path)};
    if (!in) {
      throw std::runtime_error("failed to open checkpoint for reading");
    }

    std::string header;
    in >> header;
    if (header != "WAVEFRONT_CHECKPOINT_V1") {
      throw std::invalid_argument("unsupported checkpoint format");
    }

    std::string token;
    std::vector<double> current_values;
    std::vector<double> previous_values;
    while (in >> token) {
      if (token == "step") {
        in >> step_count_;
      } else if (token == "dt") {
        in >> dt_;
      } else if (token == "dims") {
        std::size_t dims = 0;
        in >> dims;
        if (dims != grid_.dims()) {
          throw std::invalid_argument("checkpoint dims mismatch");
        }
      } else if (token == "components") {
        std::size_t components = 0;
        in >> components;
        if (components != problem_.field_components) {
          throw std::invalid_argument("checkpoint component mismatch");
        }
      } else if (token == "shape") {
        for (std::size_t axis = 0; axis < grid_.dims(); ++axis) {
          std::size_t value = 0;
          in >> value;
          if (value != grid_.shape()[axis]) {
            throw std::invalid_argument("checkpoint shape mismatch");
          }
        }
      } else if (token == "current") {
        current_values.resize(current_.data().size(), 0.0);
        for (auto& value : current_values) {
          in >> value;
        }
      } else if (token == "previous") {
        previous_values.resize(previous_.data().size(), 0.0);
        for (auto& value : previous_values) {
          in >> value;
        }
      }
    }

    if (current_values.size() != current_.data().size() || previous_values.size() != previous_.data().size()) {
      throw std::invalid_argument("checkpoint state is incomplete");
    }

    current_.data() = std::move(current_values);
    previous_.data() = std::move(previous_values);
    next_.fill(0.0);
    last_energy_ = compute_energy();
    reset_monitor_state();
    record_monitors();
  }

  void export_field_csv(std::string_view path) const override {
    std::ofstream out{std::string(path)};
    if (!out) {
      throw std::runtime_error("failed to open csv export");
    }

    out << "flat,component,value,real,imaginary\n";
    for (std::size_t flat = 0; flat < grid_.total_points(); ++flat) {
      for (std::size_t component = 0; component < problem_.field_components; ++component) {
        const double value = current_.at_flat(flat, component);
        const ComplexValue complex_value = complex_at_flat(flat, component);
        out << flat << ',' << component << ',' << value << ',' << complex_value.real << ',' << complex_value.imag << "\n";
      }
    }
  }

 private:
  void resolve_backend() {
    if (config_.backend == ExecutionBackend::Serial || config_.backend == ExecutionBackend::ThreadedCPU) {
      active_backend_ = config_.backend;
      return;
    }

    if (!config_.allow_backend_fallback) {
      throw std::invalid_argument("requested backend is not available in this build");
    }

    active_backend_ = config_.threads > 1 ? ExecutionBackend::ThreadedCPU : ExecutionBackend::Serial;
  }

  std::size_t active_thread_count() const {
    return active_backend_ == ExecutionBackend::Serial ? 1 : std::max<std::size_t>(1, config_.threads);
  }

  void configure_face_boundaries() {
    const std::size_t dims = grid_.dims();
    lower_faces_.assign(dims, FaceBoundary{});
    upper_faces_.assign(dims, FaceBoundary{});

    for (const auto& boundary : problem_.boundaries) {
      FaceBoundary face;
      face.type = boundary.type;
      face.parameter = parse_boundary_parameter_scalar(boundary.parameter, 0.0);
      face.specified = true;

      if (boundary.upper_face) {
        upper_faces_[boundary.axis] = face;
      } else {
        lower_faces_[boundary.axis] = face;
      }
    }
  }

  void compile_geometry_regions() {
    geometry_regions_.clear();
    geometry_regions_.reserve(problem_.geometry.size());
    for (const auto& region : problem_.geometry) {
      geometry_regions_.push_back(CompiledGeometryRegion{
          region,
          compile_or_default(region.medium.density, "1.0"),
          compile_or_default(region.medium.stiffness, "1.0"),
          compile_or_default(region.medium.damping, "0.0"),
          compile_or_default(region.medium.dispersion, "0.0"),
      });
    }
  }

  void reset_monitor_state() {
    probe_samples_.clear();
    surface_flux_results_.clear();
    surface_flux_results_.reserve(problem_.monitors.surfaces.size());
    for (const auto& surface : problem_.monitors.surfaces) {
      surface_flux_results_.push_back(SurfaceFluxResult{surface.name, 0, 0.0, 0.0, 0.0});
    }
  }

  const FaceBoundary& face_boundary(std::size_t axis, bool upper) const {
    return upper ? upper_faces_[axis] : lower_faces_[axis];
  }

  const CompiledGeometryRegion* matching_region(const std::vector<long double>& x) const {
    for (auto it = geometry_regions_.rbegin(); it != geometry_regions_.rend(); ++it) {
      if (contains_region(it->spec, x)) {
        return &(*it);
      }
    }
    return nullptr;
  }

  const CompiledExpression& density_expression(const EvaluationContext& context) const {
    if (const auto* region = matching_region(context.x)) {
      return region->density;
    }
    return density_expr_;
  }

  const CompiledExpression& stiffness_expression(const EvaluationContext& context) const {
    if (const auto* region = matching_region(context.x)) {
      return region->stiffness;
    }
    return stiffness_expr_;
  }

  const CompiledExpression& damping_expression(const EvaluationContext& context) const {
    if (const auto* region = matching_region(context.x)) {
      return region->damping;
    }
    return damping_expr_;
  }

  const CompiledExpression& dispersion_expression(const EvaluationContext& context) const {
    if (const auto* region = matching_region(context.x)) {
      return region->dispersion;
    }
    return dispersion_expr_;
  }

  double evaluate_density(const EvaluationContext& context) const {
    return density_expression(context).evaluate_double(context);
  }

  double evaluate_stiffness(const EvaluationContext& context) const {
    return stiffness_expression(context).evaluate_double(context);
  }

  double evaluate_damping(const EvaluationContext& context) const {
    return damping_expression(context).evaluate_double(context);
  }

  double evaluate_dispersion(const EvaluationContext& context) const {
    return dispersion_expression(context).evaluate_double(context);
  }

  double boundary_ghost_value(
      const FieldBuffer<double>& field,
      const std::vector<std::size_t>& center,
      std::size_t component,
      std::size_t axis,
      bool upper) const {
    const FaceBoundary& face = face_boundary(axis, upper);
    const double u0 = field.at_index(center, component);
    const double h = grid_.spacing()[axis];

    switch (face.type) {
      case BoundaryType::Dirichlet:
        return face.parameter;
      case BoundaryType::Neumann:
        return upper ? (u0 + face.parameter * h) : (u0 - face.parameter * h);
      case BoundaryType::Robin:
        return 2.0 * face.parameter - u0;
      case BoundaryType::Impedance: {
        const double velocity = (current_.at_index(center, component) - previous_.at_index(center, component)) / dt_;
        return u0 - face.parameter * dt_ * velocity;
      }
      case BoundaryType::PML: {
        const double sigma = std::max(face.parameter, 0.0);
        return std::exp(-sigma * dt_) * u0;
      }
      default:
        return u0;  // GCOVR_EXCL_LINE
    }
  }

  double neighbor_value(
      const FieldBuffer<double>& field,
      const std::vector<std::size_t>& center,
      std::size_t component,
      std::size_t axis,
      int delta) const {
    const std::size_t extent = grid_.shape()[axis];
    const int base = static_cast<int>(center[axis]);
    int target = base + delta;

    if (target >= 0 && target < static_cast<int>(extent)) {
      auto index = center;
      index[axis] = static_cast<std::size_t>(target);
      return field.at_index(index, component);
    }

    const bool upper = target >= static_cast<int>(extent);
    const FaceBoundary& face = face_boundary(axis, upper);
    if (face.type == BoundaryType::Periodic) {
      target %= static_cast<int>(extent);
      if (target < 0) {
        target += static_cast<int>(extent);
      }
      auto index = center;
      index[axis] = static_cast<std::size_t>(target);
      return field.at_index(index, component);
    }

    return boundary_ghost_value(field, center, component, axis, upper);
  }

  double laplacian_at(const FieldBuffer<double>& field, std::size_t flat, std::size_t component) const {
    const std::vector<std::size_t> center = grid_.unravel_index(flat);

    double lap = 0.0;
    const double u0 = field.at_flat(flat, component);
    for (std::size_t axis = 0; axis < grid_.dims(); ++axis) {
      const double h = grid_.spacing()[axis];
      const double inv_h2 = 1.0 / (h * h);
      const bool can_use_4th = config_.spatial_order == 4 && grid_.shape()[axis] >= 5 && center[axis] >= 2 &&
                               center[axis] + 2 < grid_.shape()[axis];

      if (can_use_4th) {
        const double u_m2 = neighbor_value(field, center, component, axis, -2);
        const double u_m1 = neighbor_value(field, center, component, axis, -1);
        const double u_p1 = neighbor_value(field, center, component, axis, +1);
        const double u_p2 = neighbor_value(field, center, component, axis, +2);
        lap += (-u_p2 + 16.0 * u_p1 - 30.0 * u0 + 16.0 * u_m1 - u_m2) * (inv_h2 / 12.0);
        continue;
      }

      const double u_m1 = neighbor_value(field, center, component, axis, -1);
      const double u_p1 = neighbor_value(field, center, component, axis, +1);
      lap += (u_p1 - 2.0 * u0 + u_m1) * inv_h2;
    }

    return lap;
  }

  double directional_gradient(const FieldBuffer<double>& field, std::size_t flat, std::size_t component, std::size_t axis) const {
    const std::vector<std::size_t> center = grid_.unravel_index(flat);
    const double u_m1 = neighbor_value(field, center, component, axis, -1);
    const double u_p1 = neighbor_value(field, center, component, axis, +1);
    return (u_p1 - u_m1) / (2.0 * grid_.spacing()[axis]);
  }

  std::vector<std::size_t> neighbor_center(
      const std::vector<std::size_t>& center,
      std::size_t axis,
      int delta) const {
    auto result = center;
    int target = static_cast<int>(center[axis]) + delta;
    const auto extent = static_cast<int>(grid_.shape()[axis]);

    if (target >= 0 && target < extent) {
      result[axis] = static_cast<std::size_t>(target);
    } else {
      const bool upper = target >= extent;
      const FaceBoundary& face = face_boundary(axis, upper);
      if (face.type == BoundaryType::Periodic) {
        target %= extent;
        if (target < 0) {
          target += extent;
        }
        result[axis] = static_cast<std::size_t>(target);
      } else {
        result[axis] = upper ? static_cast<std::size_t>(extent - 1) : 0;
      }
    }
    return result;
  }

  double grad_div_component_at(const FieldBuffer<double>& field, std::size_t flat, std::size_t component_i) const {
    const std::vector<std::size_t> center = grid_.unravel_index(flat);
    const std::size_t n_coupled = std::min(grid_.dims(), problem_.field_components);
    double result = 0.0;

    for (std::size_t j = 0; j < n_coupled; ++j) {
      if (j == component_i) {
        const double h = grid_.spacing()[j];
        const double inv_h2 = 1.0 / (h * h);
        const double u_c = field.at_flat(flat, j);
        const double u_p = neighbor_value(field, center, j, j, +1);
        const double u_m = neighbor_value(field, center, j, j, -1);
        result += (u_p - 2.0 * u_c + u_m) * inv_h2;
      } else {
        const double h_i = grid_.spacing()[component_i];
        const double h_j = grid_.spacing()[j];
        const auto center_pi = neighbor_center(center, component_i, +1);
        const auto center_mi = neighbor_center(center, component_i, -1);

        const double u_pp = neighbor_value(field, center_pi, j, j, +1);
        const double u_pm = neighbor_value(field, center_pi, j, j, -1);
        const double u_mp = neighbor_value(field, center_mi, j, j, +1);
        const double u_mm = neighbor_value(field, center_mi, j, j, -1);

        result += (u_pp - u_pm - u_mp + u_mm) / (4.0 * h_i * h_j);
      }
    }
    return result;
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

  EvaluationContext coefficient_context(std::size_t flat) const {
    const std::vector<std::size_t> index = grid_.unravel_index(flat);

    EvaluationContext context;
    context.x.resize(grid_.dims(), 0.0L);
    context.t = 0.0L;
    context.u.resize(problem_.field_components, 0.0L);

    for (std::size_t axis = 0; axis < grid_.dims(); ++axis) {
      context.x[axis] = static_cast<long double>(grid_.origin()[axis] + index[axis] * grid_.spacing()[axis]);
    }

    for (std::size_t comp = 0; comp < problem_.field_components; ++comp) {
      for (std::size_t axis = 0; axis < grid_.dims(); ++axis) {
        context.derivatives.emplace("du" + std::to_string(comp) + "_dx" + std::to_string(axis), 0.0L);
      }
    }
    context.extra.emplace("component", 0.0L);
    return context;
  }

  void update_cell(std::size_t flat, std::size_t component) {
    const double current = current_.at_flat(flat, component);
    const double previous = previous_.at_flat(flat, component);
    const EvaluationContext context = build_context(flat, component, static_cast<long double>(step_count_) * dt_);

    const double density = positive_or_floor(evaluate_density(context), 1.0e-12);
    const double stiffness = positive_or_floor(evaluate_stiffness(context), 1.0e-12);
    const double damping = std::max(0.0, evaluate_damping(context));
    const double dispersion = evaluate_dispersion(context);
    const double source = source_expr_.evaluate_double(context);

    const double c2 = stiffness / density;
    const double velocity = (current - previous) / dt_;
    const double dt2 = dt_ * dt_;

    const bool use_grad_div = (problem_.wave_type == WaveType::Longitudinal &&
                               problem_.field_components > 1 &&
                               component < grid_.dims());
    const double spatial_term = use_grad_div
                                    ? grad_div_component_at(current_, flat, component)
                                    : laplacian_at(current_, flat, component);

    double rhs = c2 * spatial_term + source - damping * velocity;

    if (config_.mode == SolverMode::NonlinearContinuum) {
      rhs += dispersion * current * current * current;
    } else if (config_.mode == SolverMode::MicroSurrogate) {
      const double lap = laplacian_at(current_, flat, component);
      const double grad0 = directional_gradient(current_, flat, component, 0);
      const double grad1 = grid_.dims() > 1 ? directional_gradient(current_, flat, component, 1) : 0.0;
      const double anisotropy = grad0 * grad0 - grad1 * grad1;
      const double memory_kernel = -0.04 * std::tanh(velocity);
      const double gradient_correction = 0.1 * dispersion * lap;
      rhs += gradient_correction + memory_kernel + 0.05 * anisotropy;
    }

    double next = 0.0;
    if (config_.family == SolverFamily::TimeDomain) {
      next = 2.0 * current - previous + dt2 * rhs;
    } else if (config_.family == SolverFamily::FrequencyDomain) {
      const double omega = 2.0 * kPi * config_.center_frequency;
      const double denom = positive_or_floor(omega * omega + damping + std::fabs(dispersion), 1.0e-9);
      const double target = (c2 * spatial_term + source) / denom;
      next = 0.85 * current + 0.15 * target;
    } else {
      const double omega = 2.0 * kPi * config_.center_frequency;
      const double phase_drive = std::cos(omega * dt_);
      next = current + dt_ * rhs - 0.15 * dt_ * velocity + 0.05 * phase_drive * current;
    }

    next_.at_flat(flat, component) = next;

    if (config_.precision == PrecisionMode::ExactReference) {  // GCOVR_EXCL_START
      const std::size_t checked = flat % grid_.total_points();
      if (checked < config_.reference_window) {
        const exact::ExactNumber ex_next(static_cast<long double>(next));
        const auto interval = exact::certify_nonlinear_operation(ex_next, next, 1.0e-10L, 1.0e-12L);
        const double abs_error = std::fabs(static_cast<double>(ex_next.to_long_double() - next));
        max_reference_error_ = std::max(max_reference_error_, abs_error);
        if (!interval.contains(next)) {
          max_reference_error_ = std::max(max_reference_error_, static_cast<double>(interval.width()));
        }
      }
    }  // GCOVR_EXCL_STOP
  }

  ComplexValue complex_at_flat(std::size_t flat, std::size_t component) const {
    const double value = current_.at_flat(flat, component);
    const double velocity = (current_.at_flat(flat, component) - previous_.at_flat(flat, component)) / positive_or_floor(dt_, 1.0e-12);
    const double omega = 2.0 * kPi * positive_or_floor(config_.center_frequency, 1.0e-12);
    return ComplexValue{value, velocity / omega};
  }

  void record_monitors() {
    for (const auto& probe : problem_.monitors.probes) {
      const std::size_t flat = grid_.flatten_index(probe.index);
      ProbeSample sample_value;
      sample_value.name = probe.name;
      sample_value.step = step_count_;
      sample_value.time = static_cast<double>(step_count_) * dt_;
      sample_value.index = probe.index;
      sample_value.component = probe.component;
      sample_value.value = current_.at_flat(flat, probe.component);
      sample_value.complex_value = probe.capture_complex ? complex_at_flat(flat, probe.component) : ComplexValue{sample_value.value, 0.0};
      probe_samples_.push_back(std::move(sample_value));
    }

    for (std::size_t i = 0; i < problem_.monitors.surfaces.size(); ++i) {
      const double flux = compute_surface_flux(problem_.monitors.surfaces[i]);
      auto& result = surface_flux_results_[i];
      ++result.samples;
      result.integrated_flux += flux;
      if (problem_.monitors.surfaces[i].upper_face) {
        result.transmitted_proxy += flux;
      } else {
        result.reflected_proxy += flux;
      }
    }
  }

  double compute_surface_flux(const SurfaceMonitorSpec& monitor) const {
    if (grid_.dims() == 0) {
      return 0.0;
    }

    double flux = 0.0;
    for (std::size_t flat = 0; flat < grid_.total_points(); ++flat) {
      const std::vector<std::size_t> index = grid_.unravel_index(flat);
      const std::size_t face_index = monitor.upper_face ? (grid_.shape()[monitor.axis] - 1) : 0;
      if (index[monitor.axis] != face_index) {
        continue;
      }

      const double value = current_.at_flat(flat, monitor.component);
      const double velocity = (current_.at_flat(flat, monitor.component) - previous_.at_flat(flat, monitor.component)) / dt_;
      flux += std::fabs(value * velocity);
    }
    return flux;
  }

  std::vector<double> extract_axis_line(std::size_t component) const {
    std::vector<std::size_t> index(grid_.dims(), 0);
    for (std::size_t axis = 1; axis < grid_.dims(); ++axis) {
      index[axis] = grid_.shape()[axis] / 2;
    }

    std::vector<double> line(grid_.shape().empty() ? 0 : grid_.shape()[0], 0.0);
    for (std::size_t i = 0; i < line.size(); ++i) {
      index[0] = i;
      line[i] = current_.at_index(index, component);
    }
    return line;
  }

  double compute_energy() const {
    double energy = 0.0;
    std::mutex mtx;
    deterministic_parallel_for(
        grid_.total_points(),
        active_thread_count(),
        config_.deterministic,
        [&](std::size_t begin, std::size_t end) {
          double local_energy = 0.0;
          for (std::size_t flat = begin; flat < end; ++flat) {
            for (std::size_t component = 0; component < problem_.field_components; ++component) {
              const double u = current_.at_flat(flat, component);
              const double velocity = (u - previous_.at_flat(flat, component)) / dt_;
              local_energy += 0.5 * (u * u + velocity * velocity);
            }
          }
          std::lock_guard<std::mutex> lock(mtx);
          energy += local_energy;
        });
    return energy;
  }

  ProblemSpec problem_;
  SolverConfig config_;
  GridLayout grid_ = GridLayout(GridSpec{1, {3}, {1.0}, {0.0}});

  FieldBuffer<double> previous_;
  FieldBuffer<double> current_;
  FieldBuffer<double> next_;
  std::vector<double> pml_memory_;
  std::vector<FaceBoundary> lower_faces_;
  std::vector<FaceBoundary> upper_faces_;

  CompiledExpression density_expr_;
  CompiledExpression stiffness_expr_;
  CompiledExpression damping_expr_;
  CompiledExpression dispersion_expr_;
  CompiledExpression source_expr_;
  std::vector<CompiledGeometryRegion> geometry_regions_;

  ExecutionBackend active_backend_ = ExecutionBackend::ThreadedCPU;
  std::vector<ProbeSample> probe_samples_;
  std::vector<SurfaceFluxResult> surface_flux_results_;

  std::size_t step_count_ = 0;
  double dt_ = 1.0e-3;
  double last_energy_ = 0.0;
  double max_reference_error_ = 0.0;
  double total_reflected_energy_ = 0.0;
  double total_absorbed_energy_ = 0.0;
};

}  // namespace

std::unique_ptr<ISolver> make_runtime_solver() {
  return std::make_unique<RuntimeSolver>();
}

}  // namespace wavefront
