# Tutorial: Results, Monitors, Geometry, and Far-Field Outputs

This tutorial shows how to configure the higher-level runtime APIs added on top of the original
`step/run/sample` surface: geometry regions, probe/surface monitors, full-field snapshots,
spectra, far-field diagnostics, and restart/export helpers.

## C++ example

```cpp
#include "wavefront/api/solver.hpp"
#include "wavefront/material/library.hpp"

wavefront::ProblemSpec problem;
problem.grid.dims = 2;
problem.grid.shape = {72, 48};
problem.grid.spacing = {1.0 / 71.0, 1.0 / 47.0};
problem.grid.origin = {0.0, 0.0};
problem.field_components = 1;
problem.medium = wavefront::builtin_material("air").medium;
problem.source_term.text = "24.0*sin(24*t)*exp(-((x_0-0.18)*(x_0-0.18)+(x_1-0.50)*(x_1-0.50))/0.01)";
problem.boundaries = {
    {wavefront::BoundaryType::PML, 0, false, {"10.0"}},
    {wavefront::BoundaryType::PML, 0, true, {"10.0"}},
    {wavefront::BoundaryType::PML, 1, false, {"10.0"}},
    {wavefront::BoundaryType::PML, 1, true, {"10.0"}},
};

wavefront::GeometryRegion slab;
slab.name = "glass-slab";
slab.shape = wavefront::GeometryShape::Layer;
slab.axis = 0;
slab.lower = 0.52;
slab.upper = 0.76;
slab.medium = wavefront::builtin_material("glass").medium;
problem.geometry.push_back(slab);

problem.monitors.probes.push_back({"center", {36, 24}, 0, true});
problem.monitors.surfaces.push_back({"output", 0, true, 0});
problem.sources = {
    {"left-source", "left", "incident",
     {"20.0*sin(24*t)*exp(-((x_0-0.20)*(x_0-0.20)+(x_1-0.50)*(x_1-0.50))/0.01)"}},
    {"right-source", "right", "incident",
     {"20.0*sin(24*t)*exp(-((x_0-0.80)*(x_0-0.80)+(x_1-0.50)*(x_1-0.50))/0.01)"}},
};
problem.monitors.collisions.push_back({"slab-collision", 0, false, 0, "glass-slab", 0.03, 0.0});
problem.monitors.snapshot_interval = 1;
problem.monitors.spectrum_bins = 32;
problem.monitors.enable_far_field = true;

wavefront::SolverConfig config;
config.mode = wavefront::SolverMode::LinearApprox;
config.family = wavefront::SolverFamily::FrequencyDomain;
config.backend = wavefront::ExecutionBackend::ThreadedCPU;
config.representation = wavefront::FieldRepresentation::ComplexPhasor;
config.center_frequency = 4.0;
config.threads = 8;

auto solver = wavefront::make_solver(problem, config);
solver->run(120);

auto snapshot = solver->field_snapshot();
auto probe = solver->probe_history("center");
auto spectrum = solver->probe_spectrum("center", 32);
auto flux = solver->surface_flux("output");
auto collision = solver->collision_surface("slab-collision");
auto far_field = solver->far_field_pattern(64);

solver->save_checkpoint("state.chk");
solver->export_field_csv("field.csv");
```

## Python example

```python
import wavefront as wf

problem = wf.ProblemSpec()
problem.grid.dims = 2
problem.grid.shape = [72, 48]
problem.grid.spacing = [1.0 / 71.0, 1.0 / 47.0]
problem.grid.origin = [0.0, 0.0]
problem.field_components = 1
problem.medium = wf.builtin_material("air")
problem.source_term = wf.SymbolicExpr(
    "24.0*sin(24*t)*exp(-((x_0-0.18)*(x_0-0.18)+(x_1-0.50)*(x_1-0.50))/0.01)"
)
problem.geometry = [
    wf.make_layer_region("glass-slab", axis=0, lower=0.52, upper=0.76, material=wf.builtin_material("glass"))
]
problem.sources = [
    wf.make_wave_source(
        "left-source",
        "20.0*sin(24*t)*exp(-((x_0-0.20)*(x_0-0.20)+(x_1-0.50)*(x_1-0.50))/0.01)",
        wave_id="left",
        wave_class="incident",
    ),
    wf.make_wave_source(
        "right-source",
        "20.0*sin(24*t)*exp(-((x_0-0.80)*(x_0-0.80)+(x_1-0.50)*(x_1-0.50))/0.01)",
        wave_id="right",
        wave_class="incident",
    ),
]
problem.monitors.probes = [wf.make_probe_monitor("center", [36, 24], capture_complex=True)]
problem.monitors.surfaces = [wf.make_surface_monitor("output", axis=0, upper_face=True)]
problem.monitors.collisions = [
    wf.make_geometry_collision_monitor("slab-collision", "glass-slab", shell_thickness=0.03, threshold=0.0)
]
problem.monitors.snapshot_interval = 1
problem.monitors.spectrum_bins = 32
problem.monitors.enable_far_field = True

config = wf.SolverConfig()
config.mode = wf.SolverMode.LinearApprox
config.family = wf.SolverFamily.FrequencyDomain
config.backend = wf.ExecutionBackend.ThreadedCPU
config.representation = wf.FieldRepresentation.ComplexPhasor
config.center_frequency = 4.0

solver = wf.Solver(problem, config)
solver.run(120)

snapshot = solver.field_snapshot()
series = wf.snapshot_component_series(snapshot, 0)
probe = solver.probe_history("center")
spectrum = solver.probe_spectrum("center", 32)
flux = solver.surface_flux("output")
collision = solver.collision_surface("slab-collision")
far_field = solver.far_field_pattern(64)

solver.save_checkpoint("state.chk")
solver.export_field_csv("field.csv")
```

## Notes

- `field_snapshot()` is the bulk-array entrypoint for full-field post-processing.
- `probe_history()` and `probe_spectrum()` are intended for monitor-style workflows.
- `surface_flux()` provides transmission/reflection-style summaries without requiring a custom reducer.
- `collision_surface()` reports cross-wave collision activity separately from same-wave self activity and includes pairwise wave/class summaries.
- `far_field_pattern()` gives a lightweight angular output suitable for optics-style workflows.
- The end-to-end example pipeline in `examples/generate_readme_gifs.py` exercises these APIs and
  writes validation metrics to `docs/assets/_checks/validation-metrics.json`.
