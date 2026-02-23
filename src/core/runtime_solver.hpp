#pragma once

#include <memory>

#include "wavefront/api/solver.hpp"

namespace wavefront {

std::unique_ptr<ISolver> make_runtime_solver();

}  // namespace wavefront
