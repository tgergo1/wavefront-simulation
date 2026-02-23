# Tutorial: Precision and Reference Workflow

Use `FastFloat64` for production throughput and `ExactReference` for certified cross-checking.

```cpp
wavefront::SolverConfig c;
c.mode = wavefront::SolverMode::LinearApprox;
c.precision = wavefront::PrecisionMode::ExactReference;
c.reference_window = 64;

auto solver = wavefront::make_solver(problem, c);
solver->run(32);
std::string diagnostics = solver->diagnostics_json();
```

`diagnostics_json` includes `max_reference_error`, `energy`, `reflected_energy`, and `absorbed_energy`.

Recommended workflow:

1. Tune and scale in `FastFloat64`.
2. Replay representative windows in `ExactReference`.
3. Gate acceptance on bounded reference error and invariant checks.
