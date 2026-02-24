# Validation Report

This project uses two validation layers:

1. C++ test suite (`ctest`) for solver correctness, invariants, benchmarks, and API parity.
2. Visualization-scenario gates in `/Users/gergely.toth/Work/wavefront-simulation/examples/generate_readme_gifs.py`, which fail GIF generation if physical checks are not met.

## CI/CTest Gates

- Parser determinism, symbolic binding, and factory interchangeability.
- Boundary/interface canonical checks.
- Deterministic reproducibility and conservative drift checks.
- Convergence trend checks.
- Published-case benchmark scaffolds (Yee/Berenger/Virieux) and performance guardrail.
- Exact-reference diagnostics finiteness.

## Visualization Physics Gates

The README GIF pipeline validates:

- Mode separation: nonlinear/micro outputs must diverge from linear under identical setup.
- Interface case: both reflected and transmitted waves must be nontrivial.
- Boundary case: periodic right-strip energy must exceed PML right-strip energy at early and late checkpoints; PML absorbed energy must be nonzero.
- Double-slit case: early far-right phantom energy must stay below threshold; near-wall slit transmission must dominate blocked transmission; far screen must show multiple fringes.
- 3D volume case: axis isotropy spread in homogeneous medium must remain bounded; mid-frame active-voxel support must exceed minimum.

## Latest Metrics Artifact

Latest generated metrics:

- `/Users/gergely.toth/Work/wavefront-simulation/docs/assets/_checks/validation-metrics.json`

Regenerate and revalidate with:

```bash
PYTHONPATH=build/python python3 examples/generate_readme_gifs.py
ctest --test-dir build --output-on-failure
```
