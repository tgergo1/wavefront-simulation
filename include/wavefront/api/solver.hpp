#pragma once

#include <cstddef>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace wavefront {

enum class SolverMode {
  LinearApprox,
  NonlinearContinuum,
  MicroSurrogate,
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

struct BoundarySpec {
  BoundaryType type = BoundaryType::Dirichlet;
  std::size_t axis = 0;
  bool upper_face = false;
  SymbolicExpr parameter{"0.0"};
};

struct ProblemSpec {
  GridSpec grid;
  MediumLaw medium;
  std::vector<BoundarySpec> boundaries;
  SymbolicExpr source_term{"0.0"};
  std::size_t field_components = 1;
};

struct SolverConfig {
  SolverMode mode = SolverMode::LinearApprox;
  PrecisionMode precision = PrecisionMode::FastFloat64;
  double cfl = 0.4;
  std::size_t max_steps = 0;
  std::size_t threads = 1;
  bool deterministic = true;
  std::size_t spatial_order = 2;
  bool split_pml = true;
  std::size_t reference_window = 32;
};

class ISolver {
 public:
  virtual ~ISolver() = default;

  virtual void initialize(const ProblemSpec& problem, const SolverConfig& config) = 0;
  virtual void step() = 0;
  virtual void run(std::size_t steps) = 0;
  virtual std::vector<double> sample(std::span<const std::size_t> index) const = 0;
  virtual std::string diagnostics_json() const = 0;
};

std::unique_ptr<ISolver> make_solver(const ProblemSpec& problem, const SolverConfig& config);

}  // namespace wavefront
