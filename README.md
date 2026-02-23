# wavefront-simulation

Scientific N-dimensional wave simulation library in C++20 with interchangeable solver modes, symbolic constitutive laws, deterministic CPU execution, exact-reference hooks, and Python bindings.

## Features

- Runtime interchangeable modes: `LinearApprox`, `NonlinearContinuum`, `MicroSurrogate`
- Compile-time solver API: `SolverND<N, Scalar, Mode>`
- N-D Cartesian domains and configurable spatial order (2nd / 4th)
- Boundary operators: Dirichlet, Neumann, Robin, Periodic, Impedance, PML
- Interface physics helpers for reflection/refraction/mode-conversion calculations
- Symbolic expression parser/compiler for medium and source terms
- Exact-reference infrastructure with `limitless` in `ExactReference` mode
- CMake package export and pybind11 Python module
- Unit/integration/verification/benchmark test matrix

## Build

```bash
cmake -S . -B build -DWAVEFRONT_BUILD_TESTS=ON -DWAVEFRONT_BUILD_PYTHON=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```

## Python wheel build

```bash
python -m pip install -U build
python -m build
```

## License

GPL-3.0-only (aligned with `limitless`). See `/Users/gergely.toth/Work/wavefront-simulation/LICENSE`.

## Documentation

- `/Users/gergely.toth/Work/wavefront-simulation/docs/API_REFERENCE.md`
- `/Users/gergely.toth/Work/wavefront-simulation/docs/MATHEMATICAL_DERIVATIONS.md`
- `/Users/gergely.toth/Work/wavefront-simulation/docs/VALIDATION_REPORT.md`
- `/Users/gergely.toth/Work/wavefront-simulation/docs/tutorial_nd_setup.md`
- `/Users/gergely.toth/Work/wavefront-simulation/docs/tutorial_mode_switching.md`
- `/Users/gergely.toth/Work/wavefront-simulation/docs/tutorial_precision_reference.md`
