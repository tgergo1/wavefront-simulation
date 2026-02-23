# API Reference

## C++ Runtime API

Public entrypoint: `/Users/gergely.toth/Work/wavefront-simulation/include/wavefront/api/solver.hpp`

### Core enums

- `wavefront::SolverMode`
- `wavefront::PrecisionMode`
- `wavefront::BoundaryType`

### Core structs

- `GridSpec`
- `SymbolicExpr`
- `MediumLaw`
- `BoundarySpec`
- `ProblemSpec`
- `SolverConfig`

### Solver lifecycle

1. Construct `ProblemSpec` and `SolverConfig`.
2. Build solver with `make_solver(problem, config)`.
3. Run with `step()` or `run(steps)`.
4. Observe state via `sample(index)` and `diagnostics_json()`.

## Compile-time API

Public entrypoint: `/Users/gergely.toth/Work/wavefront-simulation/include/wavefront/core/solver_nd.hpp`

Template:

`template<std::size_t N, typename Scalar, SolverMode Mode> class SolverND;`

## Python API

Package: `wavefront`

Exposed symbols:

- `SolverMode`, `PrecisionMode`, `BoundaryType`
- `GridSpec`, `SymbolicExpr`, `MediumLaw`, `BoundarySpec`, `ProblemSpec`, `SolverConfig`
- `Solver`

Constructor parity:

- `Solver(problem: ProblemSpec, config: SolverConfig)`

Methods parity:

- `step()`
- `run(steps: int)`
- `sample(index: list[int]) -> list[float]`
- `diagnostics_json() -> str`
