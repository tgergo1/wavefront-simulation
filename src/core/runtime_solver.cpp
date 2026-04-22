#include "runtime_solver.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <limits>
#include <mutex>
#include <sstream>
#include <span>
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
constexpr long double kPolygonRayCastEpsilon = 1.0e-18L;
constexpr std::size_t kMaxFractalIterations = 6;

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
  CompiledExpression signed_distance;
  std::vector<double> generated_vertices;
};

struct ExpressionRequirements {
  bool needs_position = false;
  bool needs_time = false;
  bool needs_field = false;
  bool needs_derivatives = false;
  bool needs_component = false;

  [[nodiscard]] bool spatial_only() const {
    return !needs_time && !needs_field && !needs_derivatives && !needs_component;
  }
};

bool is_prefixed_index_variable(std::string_view name, std::string_view prefix) {
  if (!name.starts_with(prefix) || name.size() <= prefix.size()) {
    return false;
  }
  return std::all_of(name.begin() + static_cast<std::ptrdiff_t>(prefix.size()), name.end(), [](char ch) {
    return std::isdigit(static_cast<unsigned char>(ch));
  });
}

ExpressionRequirements analyze_expression_requirements(const CompiledExpression& expression) {
  ExpressionRequirements requirements;
  for (const auto& instruction : expression.bytecode()) {
    if (instruction.op != OpCode::PushVariable) {
      continue;
    }
    const std::string_view symbol = instruction.symbol;
    if (symbol == "t") {
      requirements.needs_time = true;
      continue;
    }
    if (symbol == "component") {
      requirements.needs_component = true;
      continue;
    }
    if (is_prefixed_index_variable(symbol, "x_")) {
      requirements.needs_position = true;
      continue;
    }
    if (is_prefixed_index_variable(symbol, "u_")) {
      requirements.needs_field = true;
      continue;
    }
    if (symbol.starts_with("du")) {
      requirements.needs_derivatives = true;
      continue;
    }
    // Keep unknown variables on the conservative path so they still receive
    // spatial context instead of being misclassified as constants.
    requirements.needs_position = true;
  }
  return requirements;
}

ExpressionRequirements merge_requirements(ExpressionRequirements lhs, const ExpressionRequirements& rhs) {
  lhs.needs_position = lhs.needs_position || rhs.needs_position;
  lhs.needs_time = lhs.needs_time || rhs.needs_time;
  lhs.needs_field = lhs.needs_field || rhs.needs_field;
  lhs.needs_derivatives = lhs.needs_derivatives || rhs.needs_derivatives;
  lhs.needs_component = lhs.needs_component || rhs.needs_component;
  return lhs;
}

struct CompiledWaveSource {
  WaveSourceSpec spec;
  CompiledExpression expression;
  ExpressionRequirements requirements;
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

bool point_in_polygon_2d(std::span<const long double> x, const std::vector<double>& vertices) {
  if (x.size() != 2 || vertices.size() < 6 || vertices.size() % 2 != 0) {
    return false;
  }

  const long double px = x[0];
  const long double py = x[1];
  bool inside = false;
  const std::size_t count = vertices.size() / 2;
  for (std::size_t i = 0, j = count - 1; i < count; j = i++) {
    const long double xi = static_cast<long double>(vertices[2 * i]);
    const long double yi = static_cast<long double>(vertices[2 * i + 1]);
    const long double xj = static_cast<long double>(vertices[2 * j]);
    const long double yj = static_cast<long double>(vertices[2 * j + 1]);
    const bool crosses = ((yi > py) != (yj > py)) &&
                         (px < (xj - xi) * (py - yi) / ((yj - yi) + kPolygonRayCastEpsilon) + xi);
    if (crosses) {
      inside = !inside;
    }
  }
  return inside;
}

std::vector<double> build_koch_snowflake_vertices(const GeometryRegion& region) {
  if (region.fractal_generator != "koch_snowflake") {
    return region.vertices;
  }

  const long double cx = region.center.size() >= 1 ? static_cast<long double>(region.center[0]) : 0.0L;
  const long double cy = region.center.size() >= 2 ? static_cast<long double>(region.center[1]) : 0.0L;
  const long double radius = static_cast<long double>(region.radius > 0.0 ? region.radius : 0.25);
  const long double scale = static_cast<long double>(region.fractal_scale > 0.0 ? region.fractal_scale : 1.0);
  const long double effective_radius = radius * scale;
  const long double sqrt3_over_2 = std::sqrt(3.0L) * 0.5L;

  std::vector<std::array<long double, 2>> polygon = {
      {cx, cy + effective_radius},
      {cx - sqrt3_over_2 * effective_radius, cy - 0.5L * effective_radius},
      {cx + sqrt3_over_2 * effective_radius, cy - 0.5L * effective_radius},
  };

  const std::size_t iterations = std::min<std::size_t>(region.fractal_iterations, kMaxFractalIterations);
  for (std::size_t iteration = 0; iteration < iterations; ++iteration) {
    std::vector<std::array<long double, 2>> refined;
    refined.reserve(polygon.size() * 4);
    for (std::size_t i = 0; i < polygon.size(); ++i) {
      const auto& a = polygon[i];
      const auto& b = polygon[(i + 1) % polygon.size()];
      const long double dx = b[0] - a[0];
      const long double dy = b[1] - a[1];
      const std::array<long double, 2> p1{a[0] + dx / 3.0L, a[1] + dy / 3.0L};
      const std::array<long double, 2> p3{a[0] + 2.0L * dx / 3.0L, a[1] + 2.0L * dy / 3.0L};
      const long double cos60 = 0.5L;
      const long double sin60 = std::sqrt(3.0L) * 0.5L;
      const std::array<long double, 2> peak{
          p1[0] + cos60 * (p3[0] - p1[0]) - sin60 * (p3[1] - p1[1]),
          p1[1] + sin60 * (p3[0] - p1[0]) + cos60 * (p3[1] - p1[1]),
      };
      refined.push_back(a);
      refined.push_back(p1);
      refined.push_back(peak);
      refined.push_back(p3);
    }
    polygon.swap(refined);
  }

  std::vector<double> vertices;
  vertices.reserve(polygon.size() * 2);
  for (const auto& point : polygon) {
    vertices.push_back(static_cast<double>(point[0]));
    vertices.push_back(static_cast<double>(point[1]));
  }
  return vertices;
}

std::size_t compute_shell_cells_from_thickness(double shell_thickness, double min_spacing) {
  if (min_spacing <= 0.0) {
    return 1;
  }
  return std::max<std::size_t>(1, static_cast<std::size_t>(std::ceil(shell_thickness / min_spacing)));
}

bool contains_region(const CompiledGeometryRegion& region, std::span<const long double> x) {
  switch (region.spec.shape) {
    case GeometryShape::Box:
      for (std::size_t axis = 0; axis < x.size(); ++axis) {
        if (x[axis] < static_cast<long double>(region.spec.min_corner[axis]) ||
            x[axis] > static_cast<long double>(region.spec.max_corner[axis])) {
          return false;
        }
      }
      return true;
    case GeometryShape::Sphere: {
      long double radius_sq = 0.0L;
      for (std::size_t axis = 0; axis < x.size(); ++axis) {
        const long double delta = x[axis] - static_cast<long double>(region.spec.center[axis]);
        radius_sq += delta * delta;
      }
      return radius_sq <= static_cast<long double>(region.spec.radius * region.spec.radius);
    }
    case GeometryShape::Layer:
      return x[region.spec.axis] >= static_cast<long double>(region.spec.lower) &&
             x[region.spec.axis] <= static_cast<long double>(region.spec.upper);
    case GeometryShape::Polygon:
      return point_in_polygon_2d(x, region.spec.vertices);
    case GeometryShape::SignedDistanceField: {
      EvaluationContext context;
      context.x.assign(x.begin(), x.end());
      return region.signed_distance.evaluate_double(context) <= 0.0;
    }
    case GeometryShape::Fractal:
      return point_in_polygon_2d(x, region.generated_vertices);
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
    compile_wave_sources();
    previous_wave_fields_.assign(compiled_sources_.size(), FieldBuffer<double>(grid_, problem_.field_components));
    current_wave_fields_.assign(compiled_sources_.size(), FieldBuffer<double>(grid_, problem_.field_components));
    next_wave_fields_.assign(compiled_sources_.size(), FieldBuffer<double>(grid_, problem_.field_components));
    for (auto& field : previous_wave_fields_) {
      field.fill(0.0);
    }
    for (auto& field : current_wave_fields_) {
      field.fill(0.0);
    }
    for (auto& field : next_wave_fields_) {
      field.fill(0.0);
    }
    analyze_expression_usage();
    prepare_spatial_metadata();
    build_static_coefficient_cache();

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
              EvaluationContext context;
              if (cached_density_.empty() || cached_stiffness_.empty()) {
                context.x.assign(coordinates_for_flat(flat).begin(), coordinates_for_flat(flat).end());
              }
              const double density = positive_or_floor(evaluate_density(flat, context), 1.0e-12);
              const double stiffness = positive_or_floor(evaluate_stiffness(flat, context), 1.0e-12);
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

    for (std::size_t wave_index = 0; wave_index < compiled_sources_.size(); ++wave_index) {
      deterministic_parallel_for(
          points,
          active_thread_count(),
          config_.deterministic,
          [&](std::size_t begin, std::size_t end) {
            for (std::size_t flat = begin; flat < end; ++flat) {
              for (std::size_t component = 0; component < components; ++component) {
                update_wave_cell(wave_index, flat, component);
              }
            }
          });
    }

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

    for (std::size_t wave_index = 0; wave_index < compiled_sources_.size(); ++wave_index) {
      for (std::size_t component = 0; component < components; ++component) {
        apply_boundary_conditions(
            grid_,
            problem_.boundaries,
            wave_pml_memories_[wave_index],
            next_wave_fields_[wave_index],
            current_wave_fields_[wave_index],
            previous_wave_fields_[wave_index],
            component,
            config_.split_pml,
            dt_,
            8.0 / (grid_.min_spacing() * static_cast<double>(grid_.dims())),
            active_thread_count(),
            config_.deterministic);
      }
    }

    previous_.data().swap(current_.data());
    current_.data().swap(next_.data());
    for (std::size_t wave_index = 0; wave_index < compiled_sources_.size(); ++wave_index) {
      previous_wave_fields_[wave_index].data().swap(current_wave_fields_[wave_index].data());
      current_wave_fields_[wave_index].data().swap(next_wave_fields_[wave_index].data());
    }

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
    out << ",\"wave_sources\":" << compiled_sources_.size();
    out << ",\"collision_monitors\":" << collision_surface_results_.size();
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

  CollisionSurfaceResult collision_surface(std::string_view name) const override {
    for (const auto& collision : collision_surface_results_) {
      if (collision.name == name) {
        return collision;
      }
    }
    throw std::out_of_range("collision monitor not found");
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

    if (!has_extended_checkpoint_data()) {
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
      return;
    }

    out << "WAVEFRONT_CHECKPOINT_V2\n";
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
    out << "\nwaves " << compiled_sources_.size() << "\n";
    for (std::size_t i = 0; i < compiled_sources_.size(); ++i) {
      const auto& source = compiled_sources_[i].spec;
      out << "wave_meta " << i << ' ' << std::quoted(source.name) << ' ' << std::quoted(source.wave_id) << ' '
          << std::quoted(source.wave_class) << ' ' << std::quoted(source.term.text) << "\n";
      out << "wave_current " << i;
      for (const auto value : current_wave_fields_[i].data()) {
        out << ' ' << value;
      }
      out << "\nwave_previous " << i;
      for (const auto value : previous_wave_fields_[i].data()) {
        out << ' ' << value;
      }
      out << "\n";
    }
    out << "surface_results " << surface_flux_results_.size() << "\n";
    for (const auto& result : surface_flux_results_) {
      out << "surface_result " << std::quoted(result.name) << ' ' << result.samples << ' ' << result.integrated_flux << ' '
          << result.reflected_proxy << ' ' << result.transmitted_proxy << ' ' << result.peak_flux << ' ' << result.phase_proxy
          << "\n";
    }
    out << "collision_results " << collision_surface_results_.size() << "\n";
    for (const auto& result : collision_surface_results_) {
      out << "collision_result " << std::quoted(result.name) << ' ' << result.samples << ' ' << result.integrated_collision << ' '
          << result.peak_collision << ' ' << result.self_activity << ' ' << result.reflected_collision << ' '
          << result.transmitted_collision << ' ' << result.wave_pairs.size() << ' ' << result.class_pairs.size() << "\n";
      for (const auto& pair : result.wave_pairs) {
        out << "collision_wave_pair " << std::quoted(pair.left_label) << ' ' << std::quoted(pair.right_label) << ' '
            << pair.samples << ' ' << pair.integrated_collision << ' ' << pair.peak_collision << "\n";
      }
      for (const auto& pair : result.class_pairs) {
        out << "collision_class_pair " << std::quoted(pair.left_label) << ' ' << std::quoted(pair.right_label) << ' '
            << pair.samples << ' ' << pair.integrated_collision << ' ' << pair.peak_collision << "\n";
      }
      out << "collision_result_end\n";
    }
  }

  void load_checkpoint(std::string_view path) override {
    std::ifstream in{std::string(path)};
    if (!in) {
      throw std::runtime_error("failed to open checkpoint for reading");
    }

    std::string header;
    in >> header;
    if (header != "WAVEFRONT_CHECKPOINT_V1" && header != "WAVEFRONT_CHECKPOINT_V2") {
      throw std::invalid_argument("unsupported checkpoint format");
    }

    std::string token;
    std::vector<double> current_values;
    std::vector<double> previous_values;
    std::vector<std::vector<double>> wave_current_values(compiled_sources_.size());
    std::vector<std::vector<double>> wave_previous_values(compiled_sources_.size());
    std::vector<SurfaceFluxResult> loaded_surface_results;
    std::vector<CollisionSurfaceResult> loaded_collision_results;
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
      } else if (token == "waves") {
        std::size_t wave_count = 0;
        in >> wave_count;
        if (wave_count != compiled_sources_.size()) {
          throw std::invalid_argument("checkpoint wave count mismatch");
        }
      } else if (token == "wave_meta") {
        std::size_t wave_index = 0;
        std::string name;
        std::string wave_id;
        std::string wave_class;
        std::string term;
        in >> wave_index >> std::quoted(name) >> std::quoted(wave_id) >> std::quoted(wave_class) >> std::quoted(term);
        if (wave_index >= compiled_sources_.size()) {
          throw std::invalid_argument("checkpoint wave index mismatch");
        }
      } else if (token == "wave_current") {
        std::size_t wave_index = 0;
        in >> wave_index;
        if (wave_index >= compiled_sources_.size()) {
          throw std::invalid_argument("checkpoint wave index mismatch");
        }
        wave_current_values[wave_index].resize(current_.data().size(), 0.0);
        for (auto& value : wave_current_values[wave_index]) {
          in >> value;
        }
      } else if (token == "wave_previous") {
        std::size_t wave_index = 0;
        in >> wave_index;
        if (wave_index >= compiled_sources_.size()) {
          throw std::invalid_argument("checkpoint wave index mismatch");
        }
        wave_previous_values[wave_index].resize(previous_.data().size(), 0.0);
        for (auto& value : wave_previous_values[wave_index]) {
          in >> value;
        }
      } else if (token == "surface_results") {
        std::size_t count = 0;
        in >> count;
        loaded_surface_results.reserve(count);
      } else if (token == "surface_result") {
        SurfaceFluxResult result;
        in >> std::quoted(result.name) >> result.samples >> result.integrated_flux >> result.reflected_proxy >>
            result.transmitted_proxy >> result.peak_flux >> result.phase_proxy;
        loaded_surface_results.push_back(std::move(result));
      } else if (token == "collision_results") {
        std::size_t count = 0;
        in >> count;
        loaded_collision_results.reserve(count);
      } else if (token == "collision_result") {
        CollisionSurfaceResult result;
        std::size_t wave_pair_count = 0;
        std::size_t class_pair_count = 0;
        in >> std::quoted(result.name) >> result.samples >> result.integrated_collision >> result.peak_collision >>
            result.self_activity >> result.reflected_collision >> result.transmitted_collision >> wave_pair_count >>
            class_pair_count;
        result.wave_pairs.reserve(wave_pair_count);
        result.class_pairs.reserve(class_pair_count);
        while (in >> token) {
          if (token == "collision_result_end") {
            break;
          }
          CollisionPairResult pair;
          in >> std::quoted(pair.left_label) >> std::quoted(pair.right_label) >> pair.samples >> pair.integrated_collision >>
              pair.peak_collision;
          if (token == "collision_wave_pair") {
            result.wave_pairs.push_back(std::move(pair));
          } else if (token == "collision_class_pair") {
            result.class_pairs.push_back(std::move(pair));
          } else {
            throw std::invalid_argument("unsupported collision pair format");
          }
        }
        loaded_collision_results.push_back(std::move(result));
      }
    }

    if (current_values.size() != current_.data().size() || previous_values.size() != previous_.data().size()) {
      throw std::invalid_argument("checkpoint state is incomplete");
    }

    current_.data() = std::move(current_values);
    previous_.data() = std::move(previous_values);
    next_.fill(0.0);
    pml_memory_.clear();
    for (std::size_t wave_index = 0; wave_index < compiled_sources_.size(); ++wave_index) {
      if (header == "WAVEFRONT_CHECKPOINT_V2") {
        if (wave_current_values[wave_index].size() != current_.data().size() ||
            wave_previous_values[wave_index].size() != previous_.data().size()) {
          throw std::invalid_argument("checkpoint wave state is incomplete");
        }
        current_wave_fields_[wave_index].data() = std::move(wave_current_values[wave_index]);
        previous_wave_fields_[wave_index].data() = std::move(wave_previous_values[wave_index]);
      } else {
        current_wave_fields_[wave_index].fill(0.0);
        previous_wave_fields_[wave_index].fill(0.0);
      }
      next_wave_fields_[wave_index].fill(0.0);
      wave_pml_memories_[wave_index].clear();
    }
    last_energy_ = compute_energy();
    if (header == "WAVEFRONT_CHECKPOINT_V2") {
      probe_samples_.clear();
      surface_flux_results_ = std::move(loaded_surface_results);
      collision_surface_results_ = std::move(loaded_collision_results);
    } else {
      reset_monitor_state();
      record_monitors();
    }
  }

  void export_field_csv(std::string_view path) const override {
    std::ofstream out{std::string(path)};
    if (!out) {
      throw std::runtime_error("failed to open csv export");
    }

    const bool include_collision_columns = compiled_sources_.size() >= 2;
    if (include_collision_columns) {
      out << "flat,component,value,real,imaginary,collision_activity,self_activity,dominant_wave_pair,dominant_class_pair\n";
    } else {
      out << "flat,component,value,real,imaginary\n";
    }
    for (std::size_t flat = 0; flat < grid_.total_points(); ++flat) {
      for (std::size_t component = 0; component < problem_.field_components; ++component) {
        const double value = current_.at_flat(flat, component);
        const ComplexValue complex_value = complex_at_flat(flat, component);
        out << flat << ',' << component << ',' << value << ',' << complex_value.real << ',' << complex_value.imag;
        if (include_collision_columns) {
          double self_activity = 0.0;
          double collision_activity = 0.0;
          double dominant_wave_value = -1.0;
          double dominant_class_value = -1.0;
          std::string dominant_wave_pair;
          std::string dominant_class_pair;
          const double threshold = default_collision_threshold();
          std::vector<double> class_activity(source_classes_.size(), 0.0);
          for (std::size_t wave_index = 0; wave_index < compiled_sources_.size(); ++wave_index) {
            const double wave_value = current_wave_fields_[wave_index].at_flat(flat, component);
            const double velocity =
                (current_wave_fields_[wave_index].at_flat(flat, component) -
                 previous_wave_fields_[wave_index].at_flat(flat, component)) /
                positive_or_floor(dt_, 1.0e-12);
            const double activity = std::fabs(wave_value * velocity);
            self_activity += activity;
            if (activity >= threshold) {
              const std::size_t class_index = source_class_index(compiled_sources_[wave_index].spec.wave_class);
              if (class_index < class_activity.size()) {
                class_activity[class_index] += activity;
              }
            }
          }
          for (std::size_t i = 0; i < compiled_sources_.size(); ++i) {
            const double left_value = current_wave_fields_[i].at_flat(flat, component);
            const double left_velocity =
                (current_wave_fields_[i].at_flat(flat, component) - previous_wave_fields_[i].at_flat(flat, component)) /
                positive_or_floor(dt_, 1.0e-12);
            const double left_activity = std::fabs(left_value * left_velocity);
            if (left_activity < threshold) {
              continue;
            }
            for (std::size_t j = i + 1; j < compiled_sources_.size(); ++j) {
              const double right_value = current_wave_fields_[j].at_flat(flat, component);
              const double right_velocity =
                  (current_wave_fields_[j].at_flat(flat, component) -
                   previous_wave_fields_[j].at_flat(flat, component)) /
                  positive_or_floor(dt_, 1.0e-12);
              const double right_activity = std::fabs(right_value * right_velocity);
              if (right_activity < threshold) {
                continue;
              }
              const double pair_collision = std::sqrt(left_activity * right_activity);
              collision_activity += pair_collision;
              if (pair_collision > dominant_wave_value) {
                dominant_wave_value = pair_collision;
                dominant_wave_pair = compiled_sources_[i].spec.wave_id + "|" + compiled_sources_[j].spec.wave_id;
              }
            }
          }
          for (std::size_t i = 0; i < class_activity.size(); ++i) {
            for (std::size_t j = i + 1; j < class_activity.size(); ++j) {
              const double pair_collision = std::sqrt(class_activity[i] * class_activity[j]);
              if (pair_collision > dominant_class_value) {
                dominant_class_value = pair_collision;
                dominant_class_pair = source_classes_[i] + "|" + source_classes_[j];
              }
            }
          }
          out << ',' << collision_activity << ',' << self_activity << ',' << std::quoted(dominant_wave_pair) << ','
              << std::quoted(dominant_class_pair);
        }
        out << "\n";
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
          compile_or_default(region.signed_distance, "1.0"),
          region.shape == GeometryShape::Fractal ? build_koch_snowflake_vertices(region) : std::vector<double>{},
        });
    }
  }

  void prepare_spatial_metadata() {
    const std::size_t points = grid_.total_points();
    const std::size_t dims = grid_.dims();
    coordinate_cache_.assign(points * dims, 0.0L);
    region_index_cache_.assign(points, -1);

    for (std::size_t flat = 0; flat < points; ++flat) {
      const auto index = grid_.unravel_index(flat);
      auto coordinates = coordinates_for_flat(flat);
      for (std::size_t axis = 0; axis < dims; ++axis) {
        coordinates[axis] = static_cast<long double>(grid_.origin()[axis] + index[axis] * grid_.spacing()[axis]);
      }

      for (std::size_t offset = 0; offset < geometry_regions_.size(); ++offset) {
        const std::size_t region_index = geometry_regions_.size() - 1 - offset;
        if (contains_region(geometry_regions_[region_index], coordinates)) {
          region_index_cache_[flat] = static_cast<int>(region_index);
          break;
        }
      }
    }
  }

  void compile_wave_sources() {
    compiled_sources_.clear();
    source_classes_.clear();

    if (problem_.sources.empty()) {
      WaveSourceSpec source;
      source.name = "default";
      source.wave_id = "default";
      source.wave_class = "default";
      source.term = problem_.source_term;
      compiled_sources_.push_back(CompiledWaveSource{
          source,
          compile_or_default(source.term, "0.0"),
          {},
      });
    } else {
      compiled_sources_.reserve(problem_.sources.size());
      for (std::size_t i = 0; i < problem_.sources.size(); ++i) {
        WaveSourceSpec source = problem_.sources[i];
        if (source.name.empty()) {
          source.name = "source-" + std::to_string(i);
        }
        if (source.wave_id.empty()) {
          source.wave_id = source.name;
        }
        if (source.wave_class.empty()) {
          source.wave_class = source.wave_id;
        }
        compiled_sources_.push_back(CompiledWaveSource{
            source,
            compile_or_default(source.term, "0.0"),
            {},
        });
      }
    }

    for (const auto& source : compiled_sources_) {
      if (std::find(source_classes_.begin(), source_classes_.end(), source.spec.wave_class) == source_classes_.end()) {
        source_classes_.push_back(source.spec.wave_class);
      }
    }

    wave_pml_memories_.assign(compiled_sources_.size(), {});
  }

  void analyze_expression_usage() {
    source_requirements_ = {};
    for (auto& source : compiled_sources_) {
      source.requirements = analyze_expression_requirements(source.expression);
      source_requirements_ = merge_requirements(source_requirements_, source.requirements);
    }
    dynamic_coefficient_requirements_ = {};

    density_static_ = analyze_expression_requirements(density_expr_).spatial_only();
    stiffness_static_ = analyze_expression_requirements(stiffness_expr_).spatial_only();
    damping_static_ = analyze_expression_requirements(damping_expr_).spatial_only();
    dispersion_static_ = analyze_expression_requirements(dispersion_expr_).spatial_only();

    if (!density_static_) {
      dynamic_coefficient_requirements_ =
          merge_requirements(dynamic_coefficient_requirements_, analyze_expression_requirements(density_expr_));
    }
    if (!stiffness_static_) {
      dynamic_coefficient_requirements_ =
          merge_requirements(dynamic_coefficient_requirements_, analyze_expression_requirements(stiffness_expr_));
    }
    if (!damping_static_) {
      dynamic_coefficient_requirements_ =
          merge_requirements(dynamic_coefficient_requirements_, analyze_expression_requirements(damping_expr_));
    }
    if (!dispersion_static_) {
      dynamic_coefficient_requirements_ =
          merge_requirements(dynamic_coefficient_requirements_, analyze_expression_requirements(dispersion_expr_));
    }

    for (const auto& region : geometry_regions_) {
      const ExpressionRequirements density_requirements = analyze_expression_requirements(region.density);
      const ExpressionRequirements stiffness_requirements = analyze_expression_requirements(region.stiffness);
      const ExpressionRequirements damping_requirements = analyze_expression_requirements(region.damping);
      const ExpressionRequirements dispersion_requirements = analyze_expression_requirements(region.dispersion);

      density_static_ = density_static_ && density_requirements.spatial_only();
      stiffness_static_ = stiffness_static_ && stiffness_requirements.spatial_only();
      damping_static_ = damping_static_ && damping_requirements.spatial_only();
      dispersion_static_ = dispersion_static_ && dispersion_requirements.spatial_only();

      if (!density_requirements.spatial_only()) {
        dynamic_coefficient_requirements_ = merge_requirements(dynamic_coefficient_requirements_, density_requirements);
      }
      if (!stiffness_requirements.spatial_only()) {
        dynamic_coefficient_requirements_ = merge_requirements(dynamic_coefficient_requirements_, stiffness_requirements);
      }
      if (!damping_requirements.spatial_only()) {
        dynamic_coefficient_requirements_ = merge_requirements(dynamic_coefficient_requirements_, damping_requirements);
      }
      if (!dispersion_requirements.spatial_only()) {
        dynamic_coefficient_requirements_ = merge_requirements(dynamic_coefficient_requirements_, dispersion_requirements);
      }
    }
  }

  void build_static_coefficient_cache() {
    const std::size_t points = grid_.total_points();
    cached_density_.assign(density_static_ ? points : 0, 0.0);
    cached_stiffness_.assign(stiffness_static_ ? points : 0, 0.0);
    cached_damping_.assign(damping_static_ ? points : 0, 0.0);
    cached_dispersion_.assign(dispersion_static_ ? points : 0, 0.0);

    if (!density_static_ && !stiffness_static_ && !damping_static_ && !dispersion_static_) {
      return;
    }

    for (std::size_t flat = 0; flat < points; ++flat) {
      EvaluationContext context;
      context.x.assign(coordinates_for_flat(flat).begin(), coordinates_for_flat(flat).end());
      if (density_static_) {
        cached_density_[flat] = evaluate_density_uncached(flat, context);
      }
      if (stiffness_static_) {
        cached_stiffness_[flat] = evaluate_stiffness_uncached(flat, context);
      }
      if (damping_static_) {
        cached_damping_[flat] = evaluate_damping_uncached(flat, context);
      }
      if (dispersion_static_) {
        cached_dispersion_[flat] = evaluate_dispersion_uncached(flat, context);
      }
    }
  }

  void reset_monitor_state() {
    probe_samples_.clear();
    surface_flux_results_.clear();
    surface_flux_results_.reserve(problem_.monitors.surfaces.size());
    for (const auto& surface : problem_.monitors.surfaces) {
      surface_flux_results_.push_back(SurfaceFluxResult{surface.name, 0, 0.0, 0.0, 0.0, 0.0, 0.0});
    }
    collision_surface_results_.clear();
    collision_surface_results_.reserve(problem_.monitors.collisions.size());
    for (const auto& collision : problem_.monitors.collisions) {
      CollisionSurfaceResult result;
      result.name = collision.name;
      for (std::size_t i = 0; i < compiled_sources_.size(); ++i) {
        for (std::size_t j = i + 1; j < compiled_sources_.size(); ++j) {
          result.wave_pairs.push_back(CollisionPairResult{
              compiled_sources_[i].spec.wave_id,
              compiled_sources_[j].spec.wave_id,
              0,
              0.0,
              0.0,
          });
        }
      }
      for (std::size_t i = 0; i < source_classes_.size(); ++i) {
        for (std::size_t j = i + 1; j < source_classes_.size(); ++j) {
          result.class_pairs.push_back(CollisionPairResult{source_classes_[i], source_classes_[j], 0, 0.0, 0.0});
        }
      }
      collision_surface_results_.push_back(std::move(result));
    }
  }

  struct SurfaceFluxSample {
    double flux = 0.0;
    double peak = 0.0;
    double phase = 0.0;
  };

  struct CollisionSurfaceSample {
    double collision = 0.0;
    double peak = 0.0;
    double self_activity = 0.0;
    std::vector<double> wave_pairs;
    std::vector<double> class_pairs;
  };

  int geometry_region_index_by_name(std::string_view name) const {
    for (std::size_t i = 0; i < geometry_regions_.size(); ++i) {
      if (geometry_regions_[i].spec.name == name) {
        return static_cast<int>(i);
      }
    }
    return -1;
  }

  bool is_geometry_surface_cell(std::size_t flat, int target_region_index, std::size_t shell_cells) const {
    if (flat >= region_index_cache_.size() || region_index_cache_[flat] != target_region_index) {
      return false;
    }

    const std::vector<std::size_t> index = grid_.unravel_index(flat);
    for (std::size_t axis = 0; axis < grid_.dims(); ++axis) {
      for (std::size_t step = 1; step <= shell_cells; ++step) {
        if (index[axis] < step || index[axis] + step >= grid_.shape()[axis]) {
          return true;
        }

        auto lower = index;
        lower[axis] -= step;
        auto upper = index;
        upper[axis] += step;
        if (region_index_cache_[grid_.flatten_index(lower)] != target_region_index ||
            region_index_cache_[grid_.flatten_index(upper)] != target_region_index) {
          return true;
        }
      }
    }
    return false;
  }

  bool monitor_includes_flat(
      std::size_t flat,
      std::size_t axis,
      bool upper_face,
      std::string_view geometry_region,
      double shell_thickness) const {
    if (!geometry_region.empty()) {
      const int target_region = geometry_region_index_by_name(geometry_region);
      if (target_region < 0) {
        return false;
      }
      const std::size_t shell_cells = compute_shell_cells_from_thickness(shell_thickness, grid_.min_spacing());
      return is_geometry_surface_cell(flat, target_region, shell_cells);
    }

    const std::vector<std::size_t> index = grid_.unravel_index(flat);
    const std::size_t face_index = upper_face ? (grid_.shape()[axis] - 1) : 0;
    return index[axis] == face_index;
  }

  std::size_t source_class_index(std::string_view source_class) const {
    const auto it = std::find(source_classes_.begin(), source_classes_.end(), source_class);
    return it == source_classes_.end() ? source_classes_.size()
                                       : static_cast<std::size_t>(std::distance(source_classes_.begin(), it));
  }

  const FaceBoundary& face_boundary(std::size_t axis, bool upper) const {
    return upper ? upper_faces_[axis] : lower_faces_[axis];
  }

  std::span<long double> coordinates_for_flat(std::size_t flat) {
    assert(flat < grid_.total_points());
    return {coordinate_cache_.data() + flat * grid_.dims(), grid_.dims()};
  }

  std::span<const long double> coordinates_for_flat(std::size_t flat) const {
    assert(flat < grid_.total_points());
    return {coordinate_cache_.data() + flat * grid_.dims(), grid_.dims()};
  }

  const CompiledGeometryRegion* region_for_flat(std::size_t flat) const {
    if (flat >= region_index_cache_.size()) {
      return nullptr;
    }
    const int region_index = region_index_cache_[flat];
    return region_index >= 0 ? &geometry_regions_[static_cast<std::size_t>(region_index)] : nullptr;
  }

  const CompiledExpression& density_expression(std::size_t flat) const {
    if (const auto* region = region_for_flat(flat)) {
      return region->density;
    }
    return density_expr_;
  }

  const CompiledExpression& stiffness_expression(std::size_t flat) const {
    if (const auto* region = region_for_flat(flat)) {
      return region->stiffness;
    }
    return stiffness_expr_;
  }

  const CompiledExpression& damping_expression(std::size_t flat) const {
    if (const auto* region = region_for_flat(flat)) {
      return region->damping;
    }
    return damping_expr_;
  }

  const CompiledExpression& dispersion_expression(std::size_t flat) const {
    if (const auto* region = region_for_flat(flat)) {
      return region->dispersion;
    }
    return dispersion_expr_;
  }

  double evaluate_density_uncached(std::size_t flat, const EvaluationContext& context) const {
    return density_expression(flat).evaluate_double(context);
  }

  double evaluate_stiffness_uncached(std::size_t flat, const EvaluationContext& context) const {
    return stiffness_expression(flat).evaluate_double(context);
  }

  double evaluate_damping_uncached(std::size_t flat, const EvaluationContext& context) const {
    return damping_expression(flat).evaluate_double(context);
  }

  double evaluate_dispersion_uncached(std::size_t flat, const EvaluationContext& context) const {
    return dispersion_expression(flat).evaluate_double(context);
  }

  double evaluate_density(std::size_t flat, const EvaluationContext& context) const {
    return cached_density_.empty() ? evaluate_density_uncached(flat, context) : cached_density_[flat];
  }

  double evaluate_stiffness(std::size_t flat, const EvaluationContext& context) const {
    return cached_stiffness_.empty() ? evaluate_stiffness_uncached(flat, context) : cached_stiffness_[flat];
  }

  double evaluate_damping(std::size_t flat, const EvaluationContext& context) const {
    return cached_damping_.empty() ? evaluate_damping_uncached(flat, context) : cached_damping_[flat];
  }

  double evaluate_dispersion(std::size_t flat, const EvaluationContext& context) const {
    return cached_dispersion_.empty() ? evaluate_dispersion_uncached(flat, context) : cached_dispersion_[flat];
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

  EvaluationContext build_context(
      std::size_t flat,
      std::size_t component,
      long double t,
      const FieldBuffer<double>& field,
      const ExpressionRequirements& requirements) const {
    EvaluationContext context;
    if (requirements.needs_position) {
      context.x.assign(coordinates_for_flat(flat).begin(), coordinates_for_flat(flat).end());
    }
    if (requirements.needs_time) {
      context.t = t;
    }
    if (requirements.needs_field || requirements.needs_derivatives) {
      context.u.resize(problem_.field_components, 0.0L);
      for (std::size_t comp = 0; comp < problem_.field_components; ++comp) {
        context.u[comp] = field.at_flat(flat, comp);
      }
    }
    if (requirements.needs_derivatives) {
      context.derivatives.reserve(problem_.field_components * grid_.dims());
      for (std::size_t comp = 0; comp < problem_.field_components; ++comp) {
        for (std::size_t axis = 0; axis < grid_.dims(); ++axis) {
          const double derivative = directional_gradient(field, flat, comp, axis);
          context.derivatives.emplace("du" + std::to_string(comp) + "_dx" + std::to_string(axis), derivative);
        }
      }
    }
    if (requirements.needs_component) {
      context.extra.reserve(1);
      context.extra.emplace("component", static_cast<long double>(component));
    }
    return context;
  }

  EvaluationContext build_context(
      std::size_t flat,
      std::size_t component,
      long double t,
      const ExpressionRequirements& requirements) const {
    return build_context(flat, component, t, current_, requirements);
  }

  double evaluate_source(std::size_t flat, std::size_t component, long double t) const {
    double source_sum = 0.0;
    for (const auto& source : compiled_sources_) {
      if (source.expression.bytecode().empty()) {
        continue;
      }
      const EvaluationContext context = build_context(flat, component, t, current_, source.requirements);
      source_sum += source.expression.evaluate_double(context);
    }
    return source_sum;
  }

  double evaluate_wave_source(std::size_t wave_index, std::size_t flat, std::size_t component, long double t) const {
    if (wave_index >= compiled_sources_.size()) {
      return 0.0;
    }
    const auto& source = compiled_sources_[wave_index];
    if (source.expression.bytecode().empty()) {
      return 0.0;
    }
    const EvaluationContext context =
        build_context(flat, component, t, current_wave_fields_[wave_index], source.requirements);
    return source.expression.evaluate_double(context);
  }

  bool all_coefficients_cached() const {
    return !cached_density_.empty() && !cached_stiffness_.empty() && !cached_damping_.empty() &&
           !cached_dispersion_.empty();
  }

  void populate_dynamic_coefficients(
      std::size_t flat,
      std::size_t component,
      long double t,
      double& density,
      double& stiffness,
      double& damping,
      double& dispersion) const {
    density = cached_density_.empty() ? 0.0 : cached_density_[flat];
    stiffness = cached_stiffness_.empty() ? 0.0 : cached_stiffness_[flat];
    damping = cached_damping_.empty() ? 0.0 : cached_damping_[flat];
    dispersion = cached_dispersion_.empty() ? 0.0 : cached_dispersion_[flat];

    if (all_coefficients_cached()) {
      return;
    }

    const EvaluationContext context = build_context(flat, component, t, dynamic_coefficient_requirements_);
    if (cached_density_.empty()) {
      density = evaluate_density_uncached(flat, context);
    }
    if (cached_stiffness_.empty()) {
      stiffness = evaluate_stiffness_uncached(flat, context);
    }
    if (cached_damping_.empty()) {
      damping = evaluate_damping_uncached(flat, context);
    }
    if (cached_dispersion_.empty()) {
      dispersion = evaluate_dispersion_uncached(flat, context);
    }
  }

  void update_cell(std::size_t flat, std::size_t component) {
    const double current = current_.at_flat(flat, component);
    const double previous = previous_.at_flat(flat, component);
    const long double time = static_cast<long double>(step_count_) * dt_;
    double density = 0.0;
    double stiffness = 0.0;
    double damping = 0.0;
    double dispersion = 0.0;
    populate_dynamic_coefficients(flat, component, time, density, stiffness, damping, dispersion);

    density = positive_or_floor(density, 1.0e-12);
    stiffness = positive_or_floor(stiffness, 1.0e-12);
    damping = std::max(0.0, damping);
    const double source = evaluate_source(flat, component, time);

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

  void update_wave_cell(std::size_t wave_index, std::size_t flat, std::size_t component) {
    const auto& wave_current = current_wave_fields_[wave_index];
    const auto& wave_previous = previous_wave_fields_[wave_index];
    const double current = wave_current.at_flat(flat, component);
    const double previous = wave_previous.at_flat(flat, component);
    const long double time = static_cast<long double>(step_count_) * dt_;
    double density = 0.0;
    double stiffness = 0.0;
    double damping = 0.0;
    double dispersion = 0.0;
    populate_dynamic_coefficients(flat, component, time, density, stiffness, damping, dispersion);

    density = positive_or_floor(density, 1.0e-12);
    stiffness = positive_or_floor(stiffness, 1.0e-12);
    damping = std::max(0.0, damping);
    const double source = evaluate_wave_source(wave_index, flat, component, time);

    const double c2 = stiffness / density;
    const double velocity = (current - previous) / dt_;
    const double dt2 = dt_ * dt_;

    const bool use_grad_div = (problem_.wave_type == WaveType::Longitudinal &&
                               problem_.field_components > 1 &&
                               component < grid_.dims());
    const double spatial_term = use_grad_div
                                    ? grad_div_component_at(wave_current, flat, component)
                                    : laplacian_at(wave_current, flat, component);

    double rhs = c2 * spatial_term + source - damping * velocity;

    if (config_.mode == SolverMode::NonlinearContinuum) {
      rhs += dispersion * current * current * current;
    } else if (config_.mode == SolverMode::MicroSurrogate) {
      const double lap = laplacian_at(wave_current, flat, component);
      const double grad0 = directional_gradient(wave_current, flat, component, 0);
      const double grad1 = grid_.dims() > 1 ? directional_gradient(wave_current, flat, component, 1) : 0.0;
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

    next_wave_fields_[wave_index].at_flat(flat, component) = next;
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
      const SurfaceFluxSample sample = compute_surface_flux(problem_.monitors.surfaces[i]);
      auto& result = surface_flux_results_[i];
      ++result.samples;
      result.integrated_flux += sample.flux;
      result.peak_flux = std::max(result.peak_flux, sample.peak);
      result.phase_proxy += (sample.phase - result.phase_proxy) / static_cast<double>(result.samples);
      if (problem_.monitors.surfaces[i].geometry_region.empty()) {
        if (problem_.monitors.surfaces[i].upper_face) {
          result.transmitted_proxy += sample.flux;
        } else {
          result.reflected_proxy += sample.flux;
        }
      }
    }

    for (std::size_t i = 0; i < problem_.monitors.collisions.size(); ++i) {
      const CollisionSurfaceSample sample = compute_collision_surface(problem_.monitors.collisions[i]);
      auto& result = collision_surface_results_[i];
      ++result.samples;
      result.integrated_collision += sample.collision;
      result.peak_collision = std::max(result.peak_collision, sample.peak);
      result.self_activity += sample.self_activity;
      if (problem_.monitors.collisions[i].geometry_region.empty()) {
        if (problem_.monitors.collisions[i].upper_face) {
          result.transmitted_collision += sample.collision;
        } else {
          result.reflected_collision += sample.collision;
        }
      }
      for (std::size_t pair_index = 0; pair_index < result.wave_pairs.size() && pair_index < sample.wave_pairs.size(); ++pair_index) {
        auto& pair = result.wave_pairs[pair_index];
        ++pair.samples;
        pair.integrated_collision += sample.wave_pairs[pair_index];
        pair.peak_collision = std::max(pair.peak_collision, sample.wave_pairs[pair_index]);
      }
      for (std::size_t pair_index = 0; pair_index < result.class_pairs.size() && pair_index < sample.class_pairs.size();
           ++pair_index) {
        auto& pair = result.class_pairs[pair_index];
        ++pair.samples;
        pair.integrated_collision += sample.class_pairs[pair_index];
        pair.peak_collision = std::max(pair.peak_collision, sample.class_pairs[pair_index]);
      }
    }
  }

  SurfaceFluxSample compute_surface_flux(const SurfaceMonitorSpec& monitor) const {
    if (grid_.dims() == 0) {
      return {};
    }

    SurfaceFluxSample sample;
    double phase_sum = 0.0;
    std::size_t phase_count = 0;

    if (!monitor.geometry_region.empty()) {
      const int target_region = geometry_region_index_by_name(monitor.geometry_region);
      if (target_region < 0) {
        return {};
      }

      const std::size_t shell_cells = compute_shell_cells_from_thickness(monitor.shell_thickness, grid_.min_spacing());
      for (std::size_t flat = 0; flat < grid_.total_points(); ++flat) {
        if (!is_geometry_surface_cell(flat, target_region, shell_cells)) {
          continue;
        }

        const double value = current_.at_flat(flat, monitor.component);
        const double velocity = (current_.at_flat(flat, monitor.component) - previous_.at_flat(flat, monitor.component)) / dt_;
        const double local_flux = std::fabs(value * velocity);
        sample.flux += local_flux;
        sample.peak = std::max(sample.peak, local_flux);
        if (std::fabs(value) > 1.0e-15 || std::fabs(velocity) > 1.0e-15) {
          phase_sum += std::atan2(velocity, value);
          ++phase_count;
        }
      }
      sample.phase = phase_count > 0 ? phase_sum / static_cast<double>(phase_count) : 0.0;
      return sample;
    }

    for (std::size_t flat = 0; flat < grid_.total_points(); ++flat) {
      const std::vector<std::size_t> index = grid_.unravel_index(flat);
      const std::size_t face_index = monitor.upper_face ? (grid_.shape()[monitor.axis] - 1) : 0;
      if (index[monitor.axis] != face_index) {
        continue;
      }

      const double value = current_.at_flat(flat, monitor.component);
      const double velocity = (current_.at_flat(flat, monitor.component) - previous_.at_flat(flat, monitor.component)) / dt_;
      const double local_flux = std::fabs(value * velocity);
      sample.flux += local_flux;
      sample.peak = std::max(sample.peak, local_flux);
      if (std::fabs(value) > 1.0e-15 || std::fabs(velocity) > 1.0e-15) {
        phase_sum += std::atan2(velocity, value);
        ++phase_count;
      }
    }
    sample.phase = phase_count > 0 ? phase_sum / static_cast<double>(phase_count) : 0.0;
    return sample;
  }

  CollisionSurfaceSample compute_collision_surface(const CollisionMonitorSpec& monitor) const {
    CollisionSurfaceSample sample;
    sample.wave_pairs.assign(compiled_sources_.size() > 1 ? (compiled_sources_.size() * (compiled_sources_.size() - 1)) / 2 : 0, 0.0);
    sample.class_pairs.assign(source_classes_.size() > 1 ? (source_classes_.size() * (source_classes_.size() - 1)) / 2 : 0, 0.0);
    if (compiled_sources_.size() < 2 || grid_.dims() == 0) {
      return sample;
    }

    std::vector<double> source_activity(compiled_sources_.size(), 0.0);
    std::vector<double> class_activity(source_classes_.size(), 0.0);
    for (std::size_t flat = 0; flat < grid_.total_points(); ++flat) {
      if (!monitor_includes_flat(flat, monitor.axis, monitor.upper_face, monitor.geometry_region, monitor.shell_thickness)) {
        continue;
      }

      std::fill(source_activity.begin(), source_activity.end(), 0.0);
      std::fill(class_activity.begin(), class_activity.end(), 0.0);
      for (std::size_t wave_index = 0; wave_index < compiled_sources_.size(); ++wave_index) {
        const double value = current_wave_fields_[wave_index].at_flat(flat, monitor.component);
        const double velocity =
            (current_wave_fields_[wave_index].at_flat(flat, monitor.component) -
             previous_wave_fields_[wave_index].at_flat(flat, monitor.component)) /
            positive_or_floor(dt_, 1.0e-12);
        const double activity = std::fabs(value * velocity);
        sample.self_activity += activity;
        if (activity < monitor.threshold) {
          continue;
        }
        source_activity[wave_index] = activity;
        const std::size_t class_index = source_class_index(compiled_sources_[wave_index].spec.wave_class);
        if (class_index < class_activity.size()) {
          class_activity[class_index] += activity;
        }
      }

      std::size_t pair_offset = 0;
      for (std::size_t i = 0; i < source_activity.size(); ++i) {
        for (std::size_t j = i + 1; j < source_activity.size(); ++j) {
          const double pair_collision = std::sqrt(source_activity[i] * source_activity[j]);
          sample.wave_pairs[pair_offset++] += pair_collision;
          sample.collision += pair_collision;
          sample.peak = std::max(sample.peak, pair_collision);
        }
      }

      pair_offset = 0;
      for (std::size_t i = 0; i < class_activity.size(); ++i) {
        for (std::size_t j = i + 1; j < class_activity.size(); ++j) {
          const double pair_collision = std::sqrt(class_activity[i] * class_activity[j]);
          sample.class_pairs[pair_offset++] += pair_collision;
        }
      }
    }
    return sample;
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

  bool has_extended_checkpoint_data() const {
    return compiled_sources_.size() > 1 || !problem_.monitors.collisions.empty();
  }

  double default_collision_threshold() const {
    double threshold = std::numeric_limits<double>::infinity();
    for (const auto& monitor : problem_.monitors.collisions) {
      threshold = std::min(threshold, monitor.threshold);
    }
    return std::isfinite(threshold) ? threshold : 0.0;
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
  std::vector<FieldBuffer<double>> previous_wave_fields_;
  std::vector<FieldBuffer<double>> current_wave_fields_;
  std::vector<FieldBuffer<double>> next_wave_fields_;
  std::vector<double> pml_memory_;
  std::vector<std::vector<double>> wave_pml_memories_;
  std::vector<FaceBoundary> lower_faces_;
  std::vector<FaceBoundary> upper_faces_;

  CompiledExpression density_expr_;
  CompiledExpression stiffness_expr_;
  CompiledExpression damping_expr_;
  CompiledExpression dispersion_expr_;
  std::vector<CompiledWaveSource> compiled_sources_;
  std::vector<std::string> source_classes_;
  std::vector<CompiledGeometryRegion> geometry_regions_;
  std::vector<long double> coordinate_cache_;
  std::vector<int> region_index_cache_;
  std::vector<double> cached_density_;
  std::vector<double> cached_stiffness_;
  std::vector<double> cached_damping_;
  std::vector<double> cached_dispersion_;
  ExpressionRequirements source_requirements_;
  ExpressionRequirements dynamic_coefficient_requirements_;
  bool density_static_ = false;
  bool stiffness_static_ = false;
  bool damping_static_ = false;
  bool dispersion_static_ = false;

  ExecutionBackend active_backend_ = ExecutionBackend::ThreadedCPU;
  std::vector<ProbeSample> probe_samples_;
  std::vector<SurfaceFluxResult> surface_flux_results_;
  std::vector<CollisionSurfaceResult> collision_surface_results_;

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
