#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "wavefront/api/solver.hpp"

namespace py = pybind11;

namespace {

class PySolver {
 public:
  PySolver(const wavefront::ProblemSpec& problem, const wavefront::SolverConfig& config)
      : impl_(wavefront::make_solver(problem, config)) {}

  void step() { impl_->step(); }

  void run(std::size_t steps) { impl_->run(steps); }

  std::vector<double> sample(const std::vector<std::size_t>& index) const {
    if (index.empty()) {
      throw std::invalid_argument("index must not be empty");
    }
    return impl_->sample(std::span<const std::size_t>(index.data(), index.size()));
  }

  std::string diagnostics_json() const { return impl_->diagnostics_json(); }
  wavefront::FieldSnapshot field_snapshot() const { return impl_->field_snapshot(); }
  std::vector<wavefront::ProbeSample> probe_history(const std::string& name = "") const { return impl_->probe_history(name); }
  std::vector<wavefront::SpectrumSample> probe_spectrum(const std::string& name, std::size_t bins = 0) const {
    return impl_->probe_spectrum(name, bins);
  }
  wavefront::SurfaceFluxResult surface_flux(const std::string& name) const { return impl_->surface_flux(name); }
  wavefront::FarFieldPattern far_field_pattern(std::size_t samples = 0) const { return impl_->far_field_pattern(samples); }
  void save_checkpoint(const std::string& path) const { impl_->save_checkpoint(path); }
  void load_checkpoint(const std::string& path) { impl_->load_checkpoint(path); }
  void export_field_csv(const std::string& path) const { impl_->export_field_csv(path); }

 private:
  std::unique_ptr<wavefront::ISolver> impl_;
};

}  // namespace

PYBIND11_MODULE(_wavefront, m) {
  m.doc() = "Wavefront N-D wave simulation bindings";

  py::enum_<wavefront::SolverMode>(m, "SolverMode")
      .value("LinearApprox", wavefront::SolverMode::LinearApprox)
      .value("NonlinearContinuum", wavefront::SolverMode::NonlinearContinuum)
      .value("MicroSurrogate", wavefront::SolverMode::MicroSurrogate);

  py::enum_<wavefront::SolverFamily>(m, "SolverFamily")
      .value("TimeDomain", wavefront::SolverFamily::TimeDomain)
      .value("FrequencyDomain", wavefront::SolverFamily::FrequencyDomain)
      .value("AngularSpectrum", wavefront::SolverFamily::AngularSpectrum);

  py::enum_<wavefront::WaveType>(m, "WaveType")
      .value("Transverse", wavefront::WaveType::Transverse)
      .value("Longitudinal", wavefront::WaveType::Longitudinal);

  py::enum_<wavefront::PrecisionMode>(m, "PrecisionMode")
      .value("FastFloat64", wavefront::PrecisionMode::FastFloat64)
      .value("ExactReference", wavefront::PrecisionMode::ExactReference);

  py::enum_<wavefront::BoundaryType>(m, "BoundaryType")
      .value("Dirichlet", wavefront::BoundaryType::Dirichlet)
      .value("Neumann", wavefront::BoundaryType::Neumann)
      .value("Robin", wavefront::BoundaryType::Robin)
      .value("Periodic", wavefront::BoundaryType::Periodic)
      .value("Impedance", wavefront::BoundaryType::Impedance)
      .value("PML", wavefront::BoundaryType::PML);

  py::enum_<wavefront::ExecutionBackend>(m, "ExecutionBackend")
      .value("Serial", wavefront::ExecutionBackend::Serial)
      .value("ThreadedCPU", wavefront::ExecutionBackend::ThreadedCPU)
      .value("GPUAccelerated", wavefront::ExecutionBackend::GPUAccelerated)
      .value("Distributed", wavefront::ExecutionBackend::Distributed);

  py::enum_<wavefront::FieldRepresentation>(m, "FieldRepresentation")
      .value("RealScalar", wavefront::FieldRepresentation::RealScalar)
      .value("ComplexPhasor", wavefront::FieldRepresentation::ComplexPhasor);

  py::enum_<wavefront::GeometryShape>(m, "GeometryShape")
      .value("Box", wavefront::GeometryShape::Box)
      .value("Sphere", wavefront::GeometryShape::Sphere)
      .value("Layer", wavefront::GeometryShape::Layer);

  py::class_<wavefront::GridSpec>(m, "GridSpec")
      .def(py::init<>())
      .def_readwrite("dims", &wavefront::GridSpec::dims)
      .def_readwrite("shape", &wavefront::GridSpec::shape)
      .def_readwrite("spacing", &wavefront::GridSpec::spacing)
      .def_readwrite("origin", &wavefront::GridSpec::origin);

  py::class_<wavefront::SymbolicExpr>(m, "SymbolicExpr")
      .def(py::init<>())
      .def(py::init<std::string>(), py::arg("text"))
      .def_readwrite("text", &wavefront::SymbolicExpr::text);

  py::class_<wavefront::MediumLaw>(m, "MediumLaw")
      .def(py::init<>())
      .def_readwrite("density", &wavefront::MediumLaw::density)
      .def_readwrite("stiffness", &wavefront::MediumLaw::stiffness)
      .def_readwrite("damping", &wavefront::MediumLaw::damping)
      .def_readwrite("dispersion", &wavefront::MediumLaw::dispersion);

  py::class_<wavefront::GeometryRegion>(m, "GeometryRegion")
      .def(py::init<>())
      .def_readwrite("name", &wavefront::GeometryRegion::name)
      .def_readwrite("shape", &wavefront::GeometryRegion::shape)
      .def_readwrite("min_corner", &wavefront::GeometryRegion::min_corner)
      .def_readwrite("max_corner", &wavefront::GeometryRegion::max_corner)
      .def_readwrite("center", &wavefront::GeometryRegion::center)
      .def_readwrite("radius", &wavefront::GeometryRegion::radius)
      .def_readwrite("axis", &wavefront::GeometryRegion::axis)
      .def_readwrite("lower", &wavefront::GeometryRegion::lower)
      .def_readwrite("upper", &wavefront::GeometryRegion::upper)
      .def_readwrite("medium", &wavefront::GeometryRegion::medium);

  py::class_<wavefront::ProbeMonitorSpec>(m, "ProbeMonitorSpec")
      .def(py::init<>())
      .def_readwrite("name", &wavefront::ProbeMonitorSpec::name)
      .def_readwrite("index", &wavefront::ProbeMonitorSpec::index)
      .def_readwrite("component", &wavefront::ProbeMonitorSpec::component)
      .def_readwrite("capture_complex", &wavefront::ProbeMonitorSpec::capture_complex);

  py::class_<wavefront::SurfaceMonitorSpec>(m, "SurfaceMonitorSpec")
      .def(py::init<>())
      .def_readwrite("name", &wavefront::SurfaceMonitorSpec::name)
      .def_readwrite("axis", &wavefront::SurfaceMonitorSpec::axis)
      .def_readwrite("upper_face", &wavefront::SurfaceMonitorSpec::upper_face)
      .def_readwrite("component", &wavefront::SurfaceMonitorSpec::component);

  py::class_<wavefront::MonitorSpec>(m, "MonitorSpec")
      .def(py::init<>())
      .def_readwrite("probes", &wavefront::MonitorSpec::probes)
      .def_readwrite("surfaces", &wavefront::MonitorSpec::surfaces)
      .def_readwrite("snapshot_interval", &wavefront::MonitorSpec::snapshot_interval)
      .def_readwrite("spectrum_bins", &wavefront::MonitorSpec::spectrum_bins)
      .def_readwrite("enable_far_field", &wavefront::MonitorSpec::enable_far_field);

  py::class_<wavefront::BoundarySpec>(m, "BoundarySpec")
      .def(py::init<>())
      .def_readwrite("type", &wavefront::BoundarySpec::type)
      .def_readwrite("axis", &wavefront::BoundarySpec::axis)
      .def_readwrite("upper_face", &wavefront::BoundarySpec::upper_face)
      .def_readwrite("parameter", &wavefront::BoundarySpec::parameter);

  py::class_<wavefront::ProblemSpec>(m, "ProblemSpec")
      .def(py::init<>())
      .def_readwrite("grid", &wavefront::ProblemSpec::grid)
      .def_readwrite("medium", &wavefront::ProblemSpec::medium)
      .def_readwrite("geometry", &wavefront::ProblemSpec::geometry)
      .def_readwrite("boundaries", &wavefront::ProblemSpec::boundaries)
      .def_readwrite("monitors", &wavefront::ProblemSpec::monitors)
      .def_readwrite("source_term", &wavefront::ProblemSpec::source_term)
      .def_readwrite("field_components", &wavefront::ProblemSpec::field_components)
      .def_readwrite("wave_type", &wavefront::ProblemSpec::wave_type);

  py::class_<wavefront::SolverConfig>(m, "SolverConfig")
      .def(py::init<>())
      .def_readwrite("mode", &wavefront::SolverConfig::mode)
      .def_readwrite("family", &wavefront::SolverConfig::family)
      .def_readwrite("precision", &wavefront::SolverConfig::precision)
      .def_readwrite("backend", &wavefront::SolverConfig::backend)
      .def_readwrite("representation", &wavefront::SolverConfig::representation)
      .def_readwrite("cfl", &wavefront::SolverConfig::cfl)
      .def_readwrite("max_steps", &wavefront::SolverConfig::max_steps)
      .def_readwrite("threads", &wavefront::SolverConfig::threads)
      .def_readwrite("deterministic", &wavefront::SolverConfig::deterministic)
      .def_readwrite("spatial_order", &wavefront::SolverConfig::spatial_order)
      .def_readwrite("split_pml", &wavefront::SolverConfig::split_pml)
      .def_readwrite("reference_window", &wavefront::SolverConfig::reference_window)
      .def_readwrite("center_frequency", &wavefront::SolverConfig::center_frequency)
      .def_readwrite("allow_backend_fallback", &wavefront::SolverConfig::allow_backend_fallback)
      .def_readwrite("far_field_samples", &wavefront::SolverConfig::far_field_samples);

  py::class_<wavefront::ComplexValue>(m, "ComplexValue")
      .def(py::init<>())
      .def_readwrite("real", &wavefront::ComplexValue::real)
      .def_readwrite("imag", &wavefront::ComplexValue::imag)
      .def("magnitude", &wavefront::ComplexValue::magnitude)
      .def("phase", &wavefront::ComplexValue::phase);

  py::class_<wavefront::FieldSnapshot>(m, "FieldSnapshot")
      .def(py::init<>())
      .def_readwrite("step", &wavefront::FieldSnapshot::step)
      .def_readwrite("time", &wavefront::FieldSnapshot::time)
      .def_readwrite("shape", &wavefront::FieldSnapshot::shape)
      .def_readwrite("components", &wavefront::FieldSnapshot::components)
      .def_readwrite("representation", &wavefront::FieldSnapshot::representation)
      .def_readwrite("values", &wavefront::FieldSnapshot::values)
      .def_readwrite("complex_values", &wavefront::FieldSnapshot::complex_values);

  py::class_<wavefront::ProbeSample>(m, "ProbeSample")
      .def(py::init<>())
      .def_readwrite("name", &wavefront::ProbeSample::name)
      .def_readwrite("step", &wavefront::ProbeSample::step)
      .def_readwrite("time", &wavefront::ProbeSample::time)
      .def_readwrite("index", &wavefront::ProbeSample::index)
      .def_readwrite("component", &wavefront::ProbeSample::component)
      .def_readwrite("value", &wavefront::ProbeSample::value)
      .def_readwrite("complex_value", &wavefront::ProbeSample::complex_value);

  py::class_<wavefront::SpectrumSample>(m, "SpectrumSample")
      .def(py::init<>())
      .def_readwrite("frequency", &wavefront::SpectrumSample::frequency)
      .def_readwrite("magnitude", &wavefront::SpectrumSample::magnitude);

  py::class_<wavefront::SurfaceFluxResult>(m, "SurfaceFluxResult")
      .def(py::init<>())
      .def_readwrite("name", &wavefront::SurfaceFluxResult::name)
      .def_readwrite("samples", &wavefront::SurfaceFluxResult::samples)
      .def_readwrite("integrated_flux", &wavefront::SurfaceFluxResult::integrated_flux)
      .def_readwrite("reflected_proxy", &wavefront::SurfaceFluxResult::reflected_proxy)
      .def_readwrite("transmitted_proxy", &wavefront::SurfaceFluxResult::transmitted_proxy);

  py::class_<wavefront::FarFieldPattern>(m, "FarFieldPattern")
      .def(py::init<>())
      .def_readwrite("step", &wavefront::FarFieldPattern::step)
      .def_readwrite("time", &wavefront::FarFieldPattern::time)
      .def_readwrite("angles", &wavefront::FarFieldPattern::angles)
      .def_readwrite("amplitudes", &wavefront::FarFieldPattern::amplitudes);

  py::class_<PySolver>(m, "Solver")
      .def(py::init<const wavefront::ProblemSpec&, const wavefront::SolverConfig&>(), py::arg("problem"), py::arg("config"))
      .def("step", &PySolver::step)
      .def("run", &PySolver::run, py::arg("steps"))
      .def("sample", &PySolver::sample, py::arg("index"))
      .def("diagnostics_json", &PySolver::diagnostics_json)
      .def("field_snapshot", &PySolver::field_snapshot)
      .def("probe_history", &PySolver::probe_history, py::arg("name") = "")
      .def("probe_spectrum", &PySolver::probe_spectrum, py::arg("name"), py::arg("bins") = 0)
      .def("surface_flux", &PySolver::surface_flux, py::arg("name"))
      .def("far_field_pattern", &PySolver::far_field_pattern, py::arg("samples") = 0)
      .def("save_checkpoint", &PySolver::save_checkpoint, py::arg("path"))
      .def("load_checkpoint", &PySolver::load_checkpoint, py::arg("path"))
      .def("export_field_csv", &PySolver::export_field_csv, py::arg("path"));
}
