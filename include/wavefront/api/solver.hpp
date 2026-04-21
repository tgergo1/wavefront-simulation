#pragma once

#include <cmath>
#include <cstddef>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace wavefront {

enum class SolverMode {
  LinearApprox,
  NonlinearContinuum,
  MicroSurrogate,
};

enum class SolverFamily {
  TimeDomain,
  FrequencyDomain,
  AngularSpectrum,
};

enum class WaveType {
  Transverse,
  Longitudinal,
};

enum class PrecisionMode {
  FastFloat64,
  ExactReference,
};

enum class BoundaryType {
  Dirichlet,
  Neumann,
  Robin,
  Periodic,
  Impedance,
  PML,
};

enum class ExecutionBackend {
  Serial,
  ThreadedCPU,
  GPUAccelerated,
  Distributed,
};

enum class FieldRepresentation {
  RealScalar,
  ComplexPhasor,
};

enum class GeometryShape {
  Box,
  Sphere,
  Layer,
};

struct GridSpec {
  std::size_t dims = 1;
  std::vector<std::size_t> shape;
  std::vector<double> spacing;
  std::vector<double> origin;
};

struct SymbolicExpr {
  std::string text = "0";
};

struct MediumLaw {
  SymbolicExpr density{"1.0"};
  SymbolicExpr stiffness{"1.0"};
  SymbolicExpr damping{"0.0"};
  SymbolicExpr dispersion{"0.0"};
};

struct GeometryRegion {
  std::string name;
  GeometryShape shape = GeometryShape::Box;
  std::vector<double> min_corner;
  std::vector<double> max_corner;
  std::vector<double> center;
  double radius = 0.0;
  std::size_t axis = 0;
  double lower = 0.0;
  double upper = 0.0;
  MediumLaw medium;
};

struct ProbeMonitorSpec {
  std::string name;
  std::vector<std::size_t> index;
  std::size_t component = 0;
  bool capture_complex = false;
};

struct SurfaceMonitorSpec {
  std::string name;
  std::size_t axis = 0;
  bool upper_face = false;
  std::size_t component = 0;
};

struct MonitorSpec {
  std::vector<ProbeMonitorSpec> probes;
  std::vector<SurfaceMonitorSpec> surfaces;
  std::size_t snapshot_interval = 0;
  std::size_t spectrum_bins = 0;
  bool enable_far_field = false;
};

struct BoundarySpec {
  BoundaryType type = BoundaryType::Dirichlet;
  std::size_t axis = 0;
  bool upper_face = false;
  SymbolicExpr parameter{"0.0"};
};

struct ProblemSpec {
  GridSpec grid;
  MediumLaw medium;
  std::vector<GeometryRegion> geometry;
  std::vector<BoundarySpec> boundaries;
  MonitorSpec monitors;
  SymbolicExpr source_term{"0.0"};
  std::size_t field_components = 1;
  WaveType wave_type = WaveType::Transverse;
};

struct SolverConfig {
  SolverMode mode = SolverMode::LinearApprox;
  SolverFamily family = SolverFamily::TimeDomain;
  PrecisionMode precision = PrecisionMode::FastFloat64;
  ExecutionBackend backend = ExecutionBackend::ThreadedCPU;
  FieldRepresentation representation = FieldRepresentation::RealScalar;
  double cfl = 0.4;
  std::size_t max_steps = 0;
  std::size_t threads = 1;
  bool deterministic = true;
  std::size_t spatial_order = 2;
  bool split_pml = true;
  std::size_t reference_window = 32;
  double center_frequency = 1.0;
  bool allow_backend_fallback = true;
  std::size_t far_field_samples = 16;
};

struct ComplexValue {
  double real = 0.0;
  double imag = 0.0;

  [[nodiscard]] double magnitude() const { return std::sqrt(real * real + imag * imag); }
  [[nodiscard]] double phase() const { return std::atan2(imag, real); }
};

struct FieldSnapshot {
  std::size_t step = 0;
  double time = 0.0;
  std::vector<std::size_t> shape;
  std::size_t components = 0;
  FieldRepresentation representation = FieldRepresentation::RealScalar;
  std::vector<double> values;
  std::vector<ComplexValue> complex_values;
};

struct ProbeSample {
  std::string name;
  std::size_t step = 0;
  double time = 0.0;
  std::vector<std::size_t> index;
  std::size_t component = 0;
  double value = 0.0;
  ComplexValue complex_value;
};

struct SpectrumSample {
  double frequency = 0.0;
  double magnitude = 0.0;
};

struct SurfaceFluxResult {
  std::string name;
  std::size_t samples = 0;
  double integrated_flux = 0.0;
  double reflected_proxy = 0.0;
  double transmitted_proxy = 0.0;
};

struct FarFieldPattern {
  std::size_t step = 0;
  double time = 0.0;
  std::vector<double> angles;
  std::vector<double> amplitudes;
};

class ISolver {
 public:
  virtual ~ISolver() = default;

  virtual void initialize(const ProblemSpec& problem, const SolverConfig& config) = 0;
  virtual void step() = 0;
  virtual void run(std::size_t steps) = 0;
  virtual std::vector<double> sample(std::span<const std::size_t> index) const = 0;
  virtual std::string diagnostics_json() const = 0;
  virtual FieldSnapshot field_snapshot() const = 0;
  virtual std::vector<ProbeSample> probe_history(std::string_view name = {}) const = 0;
  virtual std::vector<SpectrumSample> probe_spectrum(std::string_view name, std::size_t bins = 0) const = 0;
  virtual SurfaceFluxResult surface_flux(std::string_view name) const = 0;
  virtual FarFieldPattern far_field_pattern(std::size_t samples = 0) const = 0;
  virtual void save_checkpoint(std::string_view path) const = 0;
  virtual void load_checkpoint(std::string_view path) = 0;
  virtual void export_field_csv(std::string_view path) const = 0;
};

std::unique_ptr<ISolver> make_solver(const ProblemSpec& problem, const SolverConfig& config);

}  // namespace wavefront
