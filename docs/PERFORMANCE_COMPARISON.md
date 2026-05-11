# Performance regression suite and SOTA comparison

## Detailed performance regression coverage

The repository now includes dedicated performance regression tests in
`tests/benchmarks/test_performance_regression.cpp`.

Those tests measure representative workloads with:

- UTC wall-clock timestamps for benchmark start and finish
- monotonic wall-time durations using `std::chrono::steady_clock`
- millisecond summaries
- steps/second throughput
- cell-updates/second throughput
- nanoseconds per cell update
- solver diagnostics such as total energy and absorbed energy

The helper implementation lives in
`tests/benchmarks/performance_test_utils.hpp`.

### Covered scenarios

1. **Telemetry baseline**
   - 1D linear runtime solver workload
   - validates ISO-8601 UTC timestamps and nonzero duration/throughput metrics
2. **Mode comparison**
   - compares `LinearApprox`, `NonlinearContinuum`, and `MicroSurrogate`
   - uses the same heterogeneous 2D setup so regressions stay comparable
3. **Monitor-heavy longitudinal workload**
   - exercises longitudinal propagation together with probes, snapshots, spectra, and far-field sampling
   - ensures instrumentation-heavy runs still emit stable performance telemetry

### How to run

```bash
cmake -S . -B build -DWAVEFRONT_BUILD_TESTS=ON -DWAVEFRONT_BUILD_PYTHON=OFF
cmake --build build -j
./build/tests/wavefront_tests --test-case="*performance*" --success
```

`std::chrono::system_clock` is used for human-readable UTC timestamps, while
`std::chrono::steady_clock` is used for duration measurement so benchmark timing
is not distorted by wall-clock adjustments.

## Comparison with state-of-the-art similar libraries

This comparison is intentionally scope-aware. The regression tests in this
repository are **internal guardrails**, not cross-project published benchmarks.
They are meant to catch local slowdowns with repeatable workloads before making
direct claims against other projects.

| Project | Main numerical focus | Typical strength | How wavefront-simulation compares |
| --- | --- | --- | --- |
| wavefront-simulation | N-D wave simulation with interchangeable runtime modes and mixed validation/performance tests | Lightweight C++20 library with deterministic CPU execution and broad API coverage in one test binary | Best suited for fast regression checks across multiple wave abstractions in one repository |
| Devito | Symbolic finite-difference code generation for seismic and PDE workloads | Auto-generated optimized CPU/GPU kernels and strong geophysical workflows | Devito is stronger for large generated stencil kernels; wavefront-simulation is stronger for compact end-to-end API regression coverage |
| k-Wave | Acoustic and ultrasound time-domain simulation | Mature acoustics/photoacoustics workflows and biomedical focus | k-Wave is more specialized for acoustics; wavefront-simulation spans broader solver-mode experiments in a smaller C++ core |
| Meep | Electromagnetic FDTD on Yee grids | State-of-the-art open-source EM/photonics simulation | Meep is the stronger choice for Maxwell/photonics workloads; wavefront-simulation targets a more general wave-library validation workflow |
| SPECFEM | Spectral-element seismic simulation | High-fidelity large-scale seismology on complex domains | SPECFEM is stronger for high-end seismic production runs; wavefront-simulation is optimized for compact regression scenarios and API experimentation |

## Practical benchmarking guidance

- Use this repository's performance tests to detect regressions inside the
  current implementation.
- Use problem-matched external benchmarks before making any claim against
  Devito, k-Wave, Meep, or SPECFEM.
- Compare like-for-like on physics, discretization, grid size, dimensionality,
  boundary treatment, hardware, compiler flags, and output requirements.
- Treat cross-library comparisons as methodology documents unless all projects
  are benchmarked under the same controlled setup.

## Public reference points

- Devito: https://www.devitoproject.org/
- k-Wave: https://www.k-wave.org/
- Meep: https://meep.readthedocs.io/
- SPECFEM: https://geodynamics.org/resources/specfem3d/
