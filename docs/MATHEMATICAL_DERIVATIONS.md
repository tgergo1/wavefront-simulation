# Mathematical Derivations

## Governing form

The solver family evolves fields `u_j(x, t)` by a generalized hyperbolic operator:

`d2u/dt2 = div(C(x, t, u, du) grad(u)) + S(x, t, u, du)`

where constitutive coefficients are supplied symbolically.

## Discretization

- Grid: Cartesian N-D
- Time: leapfrog/symplectic explicit update
- Space: central finite differences (2nd or 4th order)

Update skeleton at point `i`:

`u^{n+1}_i = 2u^n_i - u^{n-1}_i + dt^2 * RHS(u^n, x_i, t_n)`

## Mode definitions

- `LinearApprox`: linear constitutive response
- `NonlinearContinuum`: adds cubic response terms weighted by symbolic dispersion
- `MicroSurrogate`: adds gradient correction, memory-kernel attenuation, and anisotropy surrogate terms

## Interface relations

For impedance `Z = rho * c`, reflection and transmission amplitudes are:

- `R = (Z2 - Z1) / (Z2 + Z1)`
- `T = 2 Z2 / (Z2 + Z1)`

Snell relation for refraction angle:

`sin(theta2) = (c2/c1) * sin(theta1)`

## Exact-reference policy

`ExactReference` computes certified envelopes around float updates using exact-core arithmetic wrappers and tolerance-based intervals for non-rational operations.
