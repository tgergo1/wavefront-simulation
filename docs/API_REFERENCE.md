# API Reference

## C++ Runtime API

Public entrypoint: `include/wavefront/api/solver.hpp`

### Core enums

- `wavefront::SolverMode`
- `wavefront::SolverFamily`
- `wavefront::WaveType`
- `wavefront::PrecisionMode`
- `wavefront::BoundaryType`
- `wavefront::ExecutionBackend`
- `wavefront::FieldRepresentation`
- `wavefront::GeometryShape`

Key meanings:

- `SolverMode`: constitutive/runtime model (`LinearApprox`, `NonlinearContinuum`, `MicroSurrogate`)
- `SolverFamily`: propagation strategy (`TimeDomain`, `FrequencyDomain`, `AngularSpectrum`)
- `ExecutionBackend`: requested execution target (`Serial`, `ThreadedCPU`, `GPUAccelerated`, `Distributed`)
- `FieldRepresentation`: real scalar vs complex phasor result views
- `GeometryShape`: region primitive (`Box`, `Sphere`, `Layer`, `Polygon`, `SignedDistanceField`, `Fractal`)

### Core structs

- `GridSpec`
- `SymbolicExpr`
- `MediumLaw`
- `GeometryRegion`
- `WaveSourceSpec`
- `BoundarySpec`
- `ProbeMonitorSpec`
- `SurfaceMonitorSpec`
- `CollisionMonitorSpec`
- `MonitorSpec`
- `ProblemSpec`
- `SolverConfig`
- `ComplexValue`
- `FieldSnapshot`
- `ProbeSample`
- `SpectrumSample`
- `SurfaceFluxResult`
- `CollisionPairResult`
- `CollisionSurfaceResult`
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

### ProblemSpec fields

- `grid`: dimensions, shape, spacing, and origin
- `medium`: background symbolic density/stiffness/damping/dispersion laws
- `geometry`: region overrides applied on top of the background medium, including polygon, signed-distance, and finite fractal surfaces
- `boundaries`: per-axis face boundary conditions
- `monitors`: probe, surface, snapshot, spectrum, and far-field configuration
- `source_term`: symbolic source expression
- `sources`: explicit multi-source wave definitions with per-source `wave_id` and mergeable `wave_class`
- `field_components`: scalar field count or vector components for longitudinal runs
- `wave_type`: `Transverse` or `Longitudinal`

### SolverConfig fields

- `mode`: runtime constitutive model
- `family`: time-domain / frequency-domain / angular-spectrum behavior
- `precision`: `FastFloat64` or `ExactReference`
- `backend`: requested backend; unavailable backends can fall back when `allow_backend_fallback` is true
- `representation`: how snapshots/results should expose the field
- `cfl`: time-step scaling for explicit families
- `max_steps`: optional hard cap for `run()`
- `threads`: requested CPU worker count
- `deterministic`: use deterministic work partitioning
- `spatial_order`: finite-difference order (`2` or `4`)
- `split_pml`: enable split-field PML handling
- `reference_window`: number of points checked in exact-reference mode
- `center_frequency`: frequency used by phasor/frequency-family utilities
- `allow_backend_fallback`: allow unavailable GPU/distributed requests to fall back to CPU/serial
- `far_field_samples`: default sample count for `far_field_pattern()`

### Monitoring and results

- `FieldSnapshot field_snapshot()`
  - returns full-array values, shape, component count, step/time, and complex phasor data when enabled
- `std::vector<ProbeSample> probe_history(std::string_view name = {})`
  - returns all probes or only the named probe
- `std::vector<SpectrumSample> probe_spectrum(std::string_view name, std::size_t bins = 0)`
  - returns a DFT-style magnitude spectrum over the stored probe history
- `SurfaceFluxResult surface_flux(std::string_view name)`
  - returns integrated surface flux plus reflected/transmitted proxies
- `CollisionSurfaceResult collision_surface(std::string_view name)`
  - returns cumulative cross-wave collision activity, self activity, and pairwise wave/class summaries for a named collision monitor
- `FarFieldPattern far_field_pattern(std::size_t samples = 0)`
  - returns angular samples and amplitudes for a near-to-far diagnostic
- `std::string diagnostics_json()`
  - reports family, requested/active backend, representation, energy, reflection/absorption totals, and monitor counts
- `void save_checkpoint(std::string_view path)`
  - writes a restartable checkpoint of the live state
- `void load_checkpoint(std::string_view path)`
  - restores a checkpoint previously created by `save_checkpoint`
- `void export_field_csv(std::string_view path)`
  - writes field values and complex projections to CSV

### Geometry and materials

- `ProblemSpec::geometry` accepts `GeometryRegion` entries with `Box`, `Sphere`, `Layer`, `Polygon`, `SignedDistanceField`, and `Fractal` shapes.
- `ProblemSpec::monitors` accepts probe/surface monitor suites.
- Built-in material presets: `include/wavefront/material/library.hpp` (`air`, `water`, `glass`, `steel`, `honey`, `hyperhoney`, `oobleck`, `aerogel`, `ferrofluid`, `plasma`, `metamaterial`, `neutron_star_crust`, `strange_matter`)

GeometryRegion highlights:

- `name`: human-readable label
- `shape`: `Box`, `Sphere`, or `Layer`
- `min_corner` / `max_corner`: box extents
- `center` / `radius`: sphere definition
- `axis`, `lower`, `upper`: layer definition
- `medium`: symbolic material override used inside the region
- `vertices`: flattened 2-D polygon vertex list for `Polygon`
- `signed_distance`: symbolic signed-distance expression for `SignedDistanceField`
- `fractal_generator`, `fractal_iterations`, `fractal_scale`: finite fractal controls (`koch_snowflake` currently supported)

Surface monitor highlights:

- Standard boundary-face monitors still use `axis` and `upper_face`
- Arbitrary-surface monitors set `geometry_region` to a named region and optionally `shell_thickness`
- `SurfaceFluxResult` now also reports `peak_flux` and `phase_proxy`

Wave source highlights:

- `WaveSourceSpec` lets a problem define multiple symbolic sources instead of only `source_term`
- `wave_id` identifies a tracked wavefront; `wave_class` merges multiple source entries into one logical collision class
- when `sources` is empty, `source_term` remains the backward-compatible single-wave shorthand

Collision monitor highlights:

- `CollisionMonitorSpec` mirrors boundary-face and named-geometry surface selection from `SurfaceMonitorSpec`
- `CollisionSurfaceResult` separates `self_activity` from cross-wave `integrated_collision`
- pairwise results are exposed in both `wave_pairs` and merged `class_pairs`

### Optimization and helper utilities

- Parameter sweeps and finite-difference gradients: `include/wavefront/optimization/sweep.hpp`
- Snapshot plotting helpers: `include/wavefront/utils/plotting.hpp`

## Compile-time API

Public entrypoint: `include/wavefront/core/solver_nd.hpp`

Template:

`template<std::size_t N, typename Scalar, SolverMode Mode> class SolverND;`

## Python API

Package: `wavefront`

Exposed symbols:

- `SolverMode`, `SolverFamily`, `WaveType`, `PrecisionMode`, `BoundaryType`
- `ExecutionBackend`, `FieldRepresentation`, `GeometryShape`
- `GridSpec`, `SymbolicExpr`, `MediumLaw`, `GeometryRegion`, `BoundarySpec`
- `ProbeMonitorSpec`, `SurfaceMonitorSpec`, `MonitorSpec`, `ProblemSpec`, `SolverConfig`
- `CollisionMonitorSpec`, `WaveSourceSpec`
- `ComplexValue`, `FieldSnapshot`, `ProbeSample`, `SpectrumSample`, `SurfaceFluxResult`, `CollisionPairResult`, `CollisionSurfaceResult`, `FarFieldPattern`
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
- `collision_surface(name: str) -> CollisionSurfaceResult`
- `far_field_pattern(samples: int = 0) -> FarFieldPattern`
- `save_checkpoint(path: str)`
- `load_checkpoint(path: str)`
- `export_field_csv(path: str)`
- `diagnostics_json() -> str`

Python helper modules:

- `wavefront.helpers`
- `wavefront.plotting`

Common Python helper functions:

- `builtin_material(name)`
- `make_layer_region(name, axis, lower, upper, material)`
- `make_probe_monitor(name, index, component=0, capture_complex=False)`
- `make_surface_monitor(name, axis, upper_face, component=0)`
- `make_collision_monitor(name, axis, upper_face, component=0, threshold=0.0)`
- `make_geometry_collision_monitor(name, geometry_region, component=0, shell_thickness=0.0, threshold=0.0)`
- `make_wave_source(name, expression, wave_id=None, wave_class=None)`
- `snapshot_component_series(snapshot, component=0)`
- `snapshot_minmax(snapshot)`
- `normalize_series(values)`
