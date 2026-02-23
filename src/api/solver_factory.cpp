#include "wavefront/api/solver.hpp"

namespace wavefront {

std::unique_ptr<ISolver> make_runtime_solver();

std::unique_ptr<ISolver> make_solver(const ProblemSpec& problem, const SolverConfig& config) {
  auto solver = make_runtime_solver();
  solver->initialize(problem, config);
  return solver;
}

}  // namespace wavefront
