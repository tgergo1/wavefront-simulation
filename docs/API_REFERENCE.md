# API Reference

## C++ Runtime API

Public entrypoint: `/Users/gergely.toth/Work/wavefront-simulation/include/wavefront/api/solver.hpp`

### Core enums

- `wavefront::SolverMode`
- `wavefront::SolverFamily`
- `wavefront::WaveType`
- `wavefront::PrecisionMode`
- `wavefront::BoundaryType`
- `wavefront::ExecutionBackend`
- `wavefront::FieldRepresentation`
- `wavefront::GeometryShape`

### Core structs

- `GridSpec`
- `SymbolicExpr`
- `MediumLaw`
- `GeometryRegion`
- `BoundarySpec`
- `ProbeMonitorSpec`
- `SurfaceMonitorSpec`
- `MonitorSpec`
- `ProblemSpec`
- `SolverConfig`
- `ComplexValue`
- `FieldSnapshot`
- `ProbeSample`
- `SpectrumSample`
- `SurfaceFluxResult`
- `FarFieldPattern`

### Solver lifecycle

1. Construct `ProblemSpec` and `SolverConfig`.
   - Set `problem.wave_type = wavefront::WaveType::Longitudinal` for compressional / P-wave behaviour.
   - Use `field_components >= dims` when modelling vector longitudinal displacement fields.
2. Build solver with `make_solver(problem, config)`.
3. Run with `step()` or `run(steps)`.
4. Observe state via `sample(index)`, `field_snapshot()`, `probe_history(name)`, `probe_spectrum(name)`,
   `surface_flux(name)`, `far_field_pattern(samples)`, and `diagnostics_json()`.
5. Persist state via `save_checkpoint(path)`, `load_checkpoint(path)`, and `export_field_csv(path)`.

### Monitoring and results

- `FieldSnapshot field_snapshot()`
- `std::vector<ProbeSample> probe_history(std::string_view name = {})`
- `std::vector<SpectrumSample> probe_spectrum(std::string_view name, std::size_t bins = 0)`
- `SurfaceFluxResult surface_flux(std::string_view name)`
- `FarFieldPattern far_field_pattern(std::size_t samples = 0)`
- `void save_checkpoint(std::string_view path)`
- `void load_checkpoint(std::string_view path)`
- `void export_field_csv(std::string_view path)`

### Geometry and materials

- `ProblemSpec::geometry` accepts `GeometryRegion` entries with `Box`, `Sphere`, and `Layer` shapes.
- `ProblemSpec::monitors` accepts probe/surface monitor suites.
- Built-in material presets: `include/wavefront/material/library.hpp`

### Optimization and helper utilities

- Parameter sweeps and finite-difference gradients: `include/wavefront/optimization/sweep.hpp`
- Snapshot plotting helpers: `include/wavefront/utils/plotting.hpp`

## Compile-time API

Public entrypoint: `/Users/gergely.toth/Work/wavefront-simulation/include/wavefront/core/solver_nd.hpp`

Template:

`template<std::size_t N, typename Scalar, SolverMode Mode> class SolverND;`

## Python API

Package: `wavefront`

Exposed symbols:

- `SolverMode`, `SolverFamily`, `WaveType`, `PrecisionMode`, `BoundaryType`
- `ExecutionBackend`, `FieldRepresentation`, `GeometryShape`
- `GridSpec`, `SymbolicExpr`, `MediumLaw`, `GeometryRegion`, `BoundarySpec`
- `ProbeMonitorSpec`, `SurfaceMonitorSpec`, `MonitorSpec`, `ProblemSpec`, `SolverConfig`
- `ComplexValue`, `FieldSnapshot`, `ProbeSample`, `SpectrumSample`, `SurfaceFluxResult`, `FarFieldPattern`
- `Solver`

Constructor parity:

- `Solver(problem: ProblemSpec, config: SolverConfig)`

Methods parity:

- `step()`
- `run(steps: int)`
- `sample(index: list[int]) -> list[float]`
- `field_snapshot() -> FieldSnapshot`
- `probe_history(name: str = "") -> list[ProbeSample]`
- `probe_spectrum(name: str, bins: int = 0) -> list[SpectrumSample]`
- `surface_flux(name: str) -> SurfaceFluxResult`
- `far_field_pattern(samples: int = 0) -> FarFieldPattern`
- `save_checkpoint(path: str)`
- `load_checkpoint(path: str)`
- `export_field_csv(path: str)`
- `diagnostics_json() -> str`

Python helper modules:

- `wavefront.helpers`
- `wavefront.plotting`
