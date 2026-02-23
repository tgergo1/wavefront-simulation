# Validation Report Template

This document defines reproducible validation gates used by CI.

## Unit gates

- Parser determinism and bytecode stability
- Symbol binding correctness (`x_i`, `t`, `u_j`, `duj_dxi`)
- Boundary operator correctness on canonical fixtures
- Solver factory interchangeability

## Verification gates

- Conservative linear energy drift bound (`< 5e-2` relative in default fixture)
- Deterministic reproducibility for fixed configuration
- Grid-refinement discrepancy reduction trend

## Benchmark gates

- Yee-style CFL stability sanity
- Berenger-style PML absorption sanity (`absorbed_energy > 0`)
- Virieux-style bounded interface coefficient sanity
- Performance guardrail threshold on baseline fixture

## Precision cross-validation gate

In `ExactReference`, `max_reference_error` must remain finite and non-negative.

## Notes

Current benchmark set is an automated replication scaffold with quantitative CI checks and should be extended with domain-specific datasets when available.
