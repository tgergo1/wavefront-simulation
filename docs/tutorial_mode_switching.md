# Tutorial: Mode Switching

The runtime API is mode-interchangeable. Keep the same `ProblemSpec` and switch only `SolverConfig::mode`.

```cpp
wavefront::SolverConfig c;
c.mode = wavefront::SolverMode::LinearApprox;
auto linear = wavefront::make_solver(problem, c);

c.mode = wavefront::SolverMode::NonlinearContinuum;
auto nonlinear = wavefront::make_solver(problem, c);

c.mode = wavefront::SolverMode::MicroSurrogate;
auto micro = wavefront::make_solver(problem, c);
```

All three support identical methods: `step`, `run`, `sample`, `field_snapshot`, `probe_history`,
`probe_spectrum`, `surface_flux`, `far_field_pattern`, `save_checkpoint`, `load_checkpoint`,
`export_field_csv`, and `diagnostics_json`.

`SolverConfig::family`, `SolverConfig::backend`, and `SolverConfig::representation` can also be
swapped independently without changing the problem definition, which makes it easy to compare
time-domain vs frequency-style behavior or real vs phasor result views over the same geometry and
monitor layout.
