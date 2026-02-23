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

All three support identical methods: `step`, `run`, `sample`, `diagnostics_json`.
