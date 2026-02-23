# Tutorial: N-D Setup

```cpp
#include "wavefront/api/solver.hpp"

wavefront::ProblemSpec p;
p.grid.dims = 3;
p.grid.shape = {96, 96, 96};
p.grid.spacing = {0.01, 0.01, 0.01};
p.grid.origin = {0.0, 0.0, 0.0};
p.field_components = 1;
p.medium.density.text = "1.0";
p.medium.stiffness.text = "2.0";
p.medium.damping.text = "0.005";
p.medium.dispersion.text = "0.02";
p.source_term.text = "sin(t) * exp(-x_0*x_0)";

p.boundaries = {
  {wavefront::BoundaryType::PML, 0, false, {"10.0"}},
  {wavefront::BoundaryType::PML, 0, true, {"10.0"}},
  {wavefront::BoundaryType::Periodic, 1, false, {"0"}},
  {wavefront::BoundaryType::Periodic, 1, true, {"0"}},
  {wavefront::BoundaryType::Neumann, 2, false, {"0"}},
  {wavefront::BoundaryType::Neumann, 2, true, {"0"}},
};

wavefront::SolverConfig c;
c.mode = wavefront::SolverMode::LinearApprox;
c.precision = wavefront::PrecisionMode::FastFloat64;
c.cfl = 0.35;
c.threads = 8;
c.spatial_order = 4;

auto solver = wavefront::make_solver(p, c);
solver->run(200);
auto center = solver->sample({48, 48, 48});
```
