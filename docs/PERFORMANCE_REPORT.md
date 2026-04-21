# Performance report

## Investigation methods

The runtime hot paths were isolated with multiple complementary techniques:

1. **End-to-end timing**
   - Full `ctest` run before changes: **41.91s**
   - Full `ctest` run after changes: **3.75s**
2. **Targeted benchmark timing**
   - `Marmousi-2 inspired 1D two-layer simulation`: **0.34s → 0.13s**
   - `performance gate for float-mode throughput`: **0.17s → 0.02s**
3. **Function profiling (`gprof`)**
   - `CompiledExpression::evaluate_long_double()` dominated runtime
   - `resolve_variable()` and indexed-variable parsing were secondary hotspots
   - `GridLayout::unravel_index()` and stencil/context construction also showed up prominently
4. **System-call profiling (`strace -c`)**
   - Before: **2002 `clone3` calls** during a single representative benchmark
   - After: **0 `clone3` calls**
   - This identified thread creation overhead inside `deterministic_parallel_for()`
5. **Static code inspection**
   - Medium coefficients were being re-evaluated per cell per step even when they depended only on space or constants
   - Evaluation contexts were fully populated with field values, derivatives, and extra maps even for expressions that only needed `x` and/or `t`

## Confirmed bottlenecks

### 1. Repeated coefficient expression evaluation

Density, stiffness, damping, and dispersion expressions were re-evaluated on every update, even for static expressions such as:

- constants like `1.0`
- spatial-only laws like `2.25*(1.0 - 0.20*exp(-12.0*(x_0-1.92)*(x_0-1.92)))`

That forced repeated bytecode interpretation, variable lookup, and geometry-region selection.

### 2. Oversized expression contexts

The solver built a full `EvaluationContext` for every cell update:

- coordinates
- time
- all field components
- all directional derivatives
- `"component"` extra value

Most benchmark expressions only needed a small subset of that data.

### 3. Thread creation overhead on small workloads

`deterministic_parallel_for()` spawned threads even when each worker only received a tiny amount of work. On the 1-D benchmark cases this overhead was large enough to dominate syscall activity.

### 4. Non-optimized default single-config builds

On generators such as Unix Makefiles, the project previously configured with no explicit build type unless the caller set one manually.

## Implemented fixes

### Static coefficient caching

The runtime solver now:

- analyzes coefficient bytecode usage
- detects coefficients that are constant or spatial-only
- precomputes those values once during initialization
- reuses the cached per-cell coefficients during time stepping

### Requirement-driven context construction

The runtime solver now builds only the parts of `EvaluationContext` that an expression actually uses:

- position only when `x_*` appears
- time only when `t` appears
- field values only when `u_*` appears
- derivatives only when `du*_dx*` appears
- `"component"` only when requested

This removes a large amount of unnecessary map population and stencil work.

### Spatial metadata caching

Per-cell coordinates and geometry-region membership are cached during initialization so the solver does not need to rediscover them during every update.

### Parallelism heuristic

`deterministic_parallel_for()` now falls back to serial execution for very small partitions. This keeps large workloads parallel while avoiding repeated thread startup costs on small grids.

### Better default build mode

For single-config generators, CMake now defaults to `RelWithDebInfo`, which gives optimized builds without losing profiler-friendly symbols.

## Practical guidance

- For best performance, keep using optimized builds (`RelWithDebInfo` or `Release`)
- Spatially varying but time-independent media benefit the most from the new coefficient cache
- Small 1-D workloads are now much cheaper because they avoid gratuitous thread creation
- If you profile locally with `perf`, use the new default build mode or `-DCMAKE_BUILD_TYPE=RelWithDebInfo`
