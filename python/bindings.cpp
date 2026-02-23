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
      .def_readwrite("boundaries", &wavefront::ProblemSpec::boundaries)
      .def_readwrite("source_term", &wavefront::ProblemSpec::source_term)
      .def_readwrite("field_components", &wavefront::ProblemSpec::field_components);

  py::class_<wavefront::SolverConfig>(m, "SolverConfig")
      .def(py::init<>())
      .def_readwrite("mode", &wavefront::SolverConfig::mode)
      .def_readwrite("precision", &wavefront::SolverConfig::precision)
      .def_readwrite("cfl", &wavefront::SolverConfig::cfl)
      .def_readwrite("max_steps", &wavefront::SolverConfig::max_steps)
      .def_readwrite("threads", &wavefront::SolverConfig::threads)
      .def_readwrite("deterministic", &wavefront::SolverConfig::deterministic)
      .def_readwrite("spatial_order", &wavefront::SolverConfig::spatial_order)
      .def_readwrite("split_pml", &wavefront::SolverConfig::split_pml)
      .def_readwrite("reference_window", &wavefront::SolverConfig::reference_window);

  py::class_<PySolver>(m, "Solver")
      .def(py::init<const wavefront::ProblemSpec&, const wavefront::SolverConfig&>(), py::arg("problem"), py::arg("config"))
      .def("step", &PySolver::step)
      .def("run", &PySolver::run, py::arg("steps"))
      .def("sample", &PySolver::sample, py::arg("index"))
      .def("diagnostics_json", &PySolver::diagnostics_json);
}
