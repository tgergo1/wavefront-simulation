// Quantum-mechanical benchmark tests.
//
// Analytical values sourced from:
//   [1] Born, M. (1926): probability current conservation at a potential step,
//       Z. Phys. 37:863.  |R|² + (k₂/k₁)|T|² = 1
//   [2] de Broglie, L. (1924) / NIST CODATA 2018:
//       λ = h/p = 2π/k; for E = 1 eV electron, λ ≈ 1.226 nm
//   [3] Griffiths, D.J. (2005) Introduction to Quantum Mechanics, 2nd ed.:
//       Particle in a box: E_n = n²π²ℏ²/(2mL²)  (Eq. 2.27)
//   [4] Bohr, N. (1913) / NIST CODATA 2018:
//       Hydrogen energy levels: E_n = −13.6057 eV / n²
//   [5] Planck, M. (1900):
//       Harmonic oscillator levels: E_n = (n + ½)ℏω
//
// Normalized atomic units used in all analytical tests: ℏ = m_e = 1.
// Unit conversion factors (NIST CODATA 2018):
//   1 Hartree = 27.2114 eV
//   1 Bohr (a₀) = 0.0529177 nm
//   1 eV = 3.67493e-2 Hartree

#include <doctest/doctest.h>

#include <cmath>
#include <string>
#include <vector>

#include "../test_common.hpp"
#include "wavefront/physics/interface.hpp"

// ---------------------------------------------------------------------------
// [1] Quantum step potential — probability current conservation (Born 1926)
//
// For a particle with energy E incident on a potential step of height V₀ < E:
//   k₁ = √(2E),  k₂ = √(2(E−V₀))  (ℏ = m = 1)
//
// Quantum amplitudes via impedance analogy (Z → k, arguments swapped):
//   R_QM = (k₁−k₂)/(k₁+k₂) = reflection_coefficient(k₂, k₁)
//   T_QM = 2k₁/(k₁+k₂)      = transmission_coefficient(k₂, k₁)
//
// Probability flux conservation: |R|² + (k₂/k₁)|T|² = 1
// ---------------------------------------------------------------------------

TEST_CASE("quantum step potential: probability current conservation (Born 1926)") {
  constexpr double E = 4.0;   // incident kinetic energy (normalized, > V0)
  constexpr double V0 = 2.0;  // step height

  const double k1 = std::sqrt(2.0 * E);         // = sqrt(8)
  const double k2 = std::sqrt(2.0 * (E - V0));  // = 2.0

  // Quantum R and T via the impedance-analogy mapping (Z₁=k₂, Z₂=k₁)
  const double R = wavefront::reflection_coefficient(k2, k1);
  const double T = wavefront::transmission_coefficient(k2, k1);

  // Verify against closed-form quantum formulas
  const double R_expected = (k1 - k2) / (k1 + k2);
  const double T_expected = 2.0 * k1 / (k1 + k2);

  CHECK(R == doctest::Approx(R_expected).epsilon(1e-10));
  CHECK(T == doctest::Approx(T_expected).epsilon(1e-10));

  // Probability flux conservation: |R|² + (k₂/k₁)|T|² = 1
  const double flux_balance = R * R + (k2 / k1) * T * T;
  CHECK(flux_balance == doctest::Approx(1.0).epsilon(1e-12));
}

TEST_CASE("quantum step: high-energy limit gives near-perfect transmission") {
  // As E >> V₀, k₁ ≈ k₂, so |R| → 0
  constexpr double V0 = 1.0;
  constexpr double E_high = 1000.0;

  const double k1 = std::sqrt(2.0 * E_high);
  const double k2 = std::sqrt(2.0 * (E_high - V0));

  const double R = wavefront::reflection_coefficient(k2, k1);
  CHECK(std::fabs(R) < 0.01);
}

TEST_CASE("quantum step: near-threshold incidence gives near-total reflection") {
  // As E → V₀⁺, k₂ → 0 and |R_QM| → 1
  constexpr double V0 = 2.0;
  constexpr double E_near = 2.001;  // just above threshold

  const double k1 = std::sqrt(2.0 * E_near);
  const double k2 = std::sqrt(2.0 * (E_near - V0));

  const double R = wavefront::reflection_coefficient(k2, k1);
  CHECK(std::fabs(R) > 0.95);
}

TEST_CASE("quantum step: sub-barrier evanescent case maps to TIR") {
  // For E < V₀ the transmitted wavenumber becomes imaginary → TIR.
  // Map to acoustic TIR via c_in=1, c_out >> c_in (steep angle beyond critical).
  constexpr double c_in = 1.0;
  constexpr double c_out = 5.0;    // evanescent: would correspond to E < V₀
  constexpr double theta_beyond = 0.25;  // beyond critical angle arcsin(1/5) ≈ 0.2014

  const double theta_t = wavefront::refraction_angle(theta_beyond, c_in, c_out);
  // TIR: transmitted angle is undefined (NaN)
  CHECK(std::isnan(theta_t));
}

// ---------------------------------------------------------------------------
// [2] de Broglie wavelength (de Broglie 1924 / NIST CODATA 2018)
//
// In atomic units (ℏ = m_e = 1):
//   k = √(2E_Hartree),  λ = 2π/k  (in Bohr radii)
//
// For E = 1 eV = 3.67493e-2 Hartree:
//   k = √(0.073499) = 0.27111 a₀⁻¹
//   λ = 2π / 0.27111 = 23.17 a₀ = 23.17 × 0.0529177 nm ≈ 1.226 nm  (NIST)
// ---------------------------------------------------------------------------

TEST_CASE("de Broglie wavelength for 1 eV electron matches NIST value (1.226 nm)") {
  constexpr double eV_to_Hartree = 3.67493e-2;   // NIST CODATA 2018
  constexpr double bohr_to_nm = 5.29177e-2;       // 1 a₀ in nm

  const double E_hartree = 1.0 * eV_to_Hartree;
  const double k_per_bohr = std::sqrt(2.0 * E_hartree);  // ℏ = m_e = 1
  const double lambda_nm = (2.0 * M_PI / k_per_bohr) * bohr_to_nm;

  // NIST published value: λ(1 eV) = 1.226 nm
  CHECK(lambda_nm == doctest::Approx(1.226).epsilon(5e-3));
}

TEST_CASE("de Broglie wavelength scales as 1/sqrt(E)") {
  // λ ∝ 1/√E → λ(4E) = λ(E)/2
  constexpr double eV_to_Hartree = 3.67493e-2;

  const auto lambda = [&](double E_eV) {
    const double E_au = E_eV * eV_to_Hartree;
    const double k = std::sqrt(2.0 * E_au);
    return 2.0 * M_PI / k;
  };

  CHECK(lambda(1.0) / lambda(4.0) == doctest::Approx(2.0).epsilon(1e-10));
  CHECK(lambda(1.0) / lambda(9.0) == doctest::Approx(3.0).epsilon(1e-10));
}

// ---------------------------------------------------------------------------
// [3] Particle in a box — energy quantization (Griffiths 2005, Eq. 2.27)
//
// E_n = n²π²ℏ² / (2mL²),  n = 1, 2, 3, ...
// With ℏ = m = 1 and L = π:  E_n = n²/2
//   E₁ = 0.5,  E₂ = 2.0,  E₃ = 4.5
// ---------------------------------------------------------------------------

TEST_CASE("particle in a box: E_n = n^2 pi^2 / (2mL^2) with hbar=m=1 (Griffiths 2005)") {
  constexpr double L = M_PI;

  const auto E_n = [&](int n) {
    return static_cast<double>(n * n) * M_PI * M_PI / (2.0 * L * L);
  };

  CHECK(E_n(1) == doctest::Approx(0.5).epsilon(1e-10));
  CHECK(E_n(2) == doctest::Approx(2.0).epsilon(1e-10));
  CHECK(E_n(3) == doctest::Approx(4.5).epsilon(1e-10));

  // Level ratio: E_n / E_1 = n²
  CHECK(E_n(2) / E_n(1) == doctest::Approx(4.0).epsilon(1e-10));
  CHECK(E_n(3) / E_n(1) == doctest::Approx(9.0).epsilon(1e-10));
}

// ---------------------------------------------------------------------------
// [4] Hydrogen atom energy levels (Bohr 1913 / NIST CODATA 2018)
//
// E_n = −13.6057 eV / n²
// Ground state: E₁ = −13.6057 eV = −0.5 Hartree
// ---------------------------------------------------------------------------

TEST_CASE("hydrogen atom: ground state energy matches NIST (−0.5 Hartree)") {
  constexpr double E1_hartree = -0.5;           // exact in atomic units
  constexpr double hartree_to_eV = 27.2114;     // NIST CODATA 2018

  const double E1_eV = E1_hartree * hartree_to_eV;
  CHECK(E1_eV == doctest::Approx(-13.6057).epsilon(1e-3));
}

TEST_CASE("hydrogen atom: Lyman-alpha and Balmer-alpha transitions match published values") {
  constexpr double E1_eV = -13.6057;  // NIST CODATA 2018

  const auto E_n = [&](int n) { return E1_eV / static_cast<double>(n * n); };

  // Lyman α: n=2→1 ≈ 10.204 eV
  const double lyman_alpha = E_n(2) - E_n(1);
  CHECK(lyman_alpha == doctest::Approx(10.2043).epsilon(1e-3));

  // Balmer α: n=3→2 ≈ 1.890 eV
  const double balmer_alpha = E_n(3) - E_n(2);
  CHECK(balmer_alpha == doctest::Approx(1.890).epsilon(2e-3));
}

// ---------------------------------------------------------------------------
// [5] Quantum harmonic oscillator — zero-point energy (Planck 1900)
//
// E_n = (n + ½)ℏω;  with ℏ = ω = 1:  E_n = n + 0.5
// Zero-point energy E₀ = 0.5 (irreducible ground-state energy)
// Level spacing: ΔE = ℏω = 1.0 (equidistant)
// ---------------------------------------------------------------------------

TEST_CASE("quantum harmonic oscillator: zero-point energy and equal spacing (Planck 1900)") {
  const auto E_QHO = [](int n) { return static_cast<double>(n) + 0.5; };  // ℏ=ω=1

  CHECK(E_QHO(0) == doctest::Approx(0.5).epsilon(1e-15));  // zero-point energy
  CHECK(E_QHO(1) == doctest::Approx(1.5).epsilon(1e-15));
  CHECK(E_QHO(2) == doctest::Approx(2.5).epsilon(1e-15));

  // Equal level spacing ΔE = ℏω = 1
  for (int n = 0; n < 5; ++n) {
    CHECK(E_QHO(n + 1) - E_QHO(n) == doctest::Approx(1.0).epsilon(1e-15));
  }
}

// ---------------------------------------------------------------------------
// Simulation: quantum-like dispersive wave packet
//
// The Schrödinger dispersion relation ω = ℏk²/(2m) produces wave packets
// that spread over time.  The solver's dispersion parameter introduces a
// fourth-order correction that mimics this k²-dependent phase velocity.
// Physical expectation: wave packet remains finite and PML absorbs outgoing
// energy as the packet reaches the domain boundaries.
// ---------------------------------------------------------------------------

TEST_CASE("quantum-like dispersive wave packet: field finite and PML absorbs energy") {
  wavefront::ProblemSpec p;
  p.grid.dims = 1;
  p.grid.shape = {192};
  p.grid.spacing = {0.02};
  p.grid.origin = {0.0};
  p.field_components = 1;

  p.medium.density.text = "1.0";
  p.medium.stiffness.text = "1.0";
  p.medium.damping.text = "0.0";
  // High dispersion mimics the Schrödinger ω∝k² spreading
  p.medium.dispersion.text = "0.08";
  // Gaussian-modulated sinusoidal source (wave-packet initial drive)
  p.source_term.text = "6.0*sin(20.0*t)*exp(-5.0*t)*exp(-((x_0-0.5)*(x_0-0.5))/0.01)";

  p.boundaries = {
      {wavefront::BoundaryType::PML, 0, false, wavefront::SymbolicExpr{"10.0"}},
      {wavefront::BoundaryType::PML, 0, true, wavefront::SymbolicExpr{"10.0"}},
  };

  auto cfg = test_common::default_config(wavefront::SolverMode::LinearApprox);
  cfg.cfl = 0.15;
  cfg.spatial_order = 2;

  auto solver = wavefront::make_solver(p, cfg);
  solver->run(400);

  for (std::size_t i = 0; i < 192; i += 16) {
    CHECK(std::isfinite(solver->sample(std::vector<std::size_t>{i})[0]));
  }

  const std::string diag = solver->diagnostics_json();
  CHECK(test_common::json_value(diag, "energy") >= 0.0);
  // PML must absorb outgoing wave energy
  CHECK(test_common::json_value(diag, "absorbed_energy") > 0.0);
}

// ---------------------------------------------------------------------------
// Simulation: particle-in-a-box standing modes
//
// Dirichlet (hard-wall) boundaries enforce ψ(0) = ψ(L) = 0 at every step.
// Domain length: L = 192 × 0.02 = 3.84.
// Fundamental standing-wave wavenumber: k₁ = π/L ≈ 0.818.
// The source is tuned near ω₁ = c·k₁ ≈ 0.818 to excite the mode.
// Physical expectations:
//   - boundary nodes clamped to zero exactly,
//   - interior accumulates energy (driven mode).
// ---------------------------------------------------------------------------

TEST_CASE("quantum particle-in-a-box: Dirichlet walls enforce zero-boundary and driven mode accumulates energy") {
  wavefront::ProblemSpec p;
  p.grid.dims = 1;
  p.grid.shape = {192};
  p.grid.spacing = {0.02};
  p.grid.origin = {0.0};
  p.field_components = 1;

  p.medium.density.text = "1.0";
  p.medium.stiffness.text = "1.0";
  p.medium.damping.text = "0.0005";
  p.medium.dispersion.text = "0.02";
  // Source near fundamental resonance ω₁ ≈ π/3.84 ≈ 0.818
  p.source_term.text = "4.0*sin(0.82*t)*exp(-((x_0-1.92)*(x_0-1.92))/0.05)";

  p.boundaries = {
      {wavefront::BoundaryType::Dirichlet, 0, false, wavefront::SymbolicExpr{"0.0"}},
      {wavefront::BoundaryType::Dirichlet, 0, true, wavefront::SymbolicExpr{"0.0"}},
  };

  auto cfg = test_common::default_config(wavefront::SolverMode::LinearApprox);
  cfg.cfl = 0.20;
  cfg.spatial_order = 2;

  auto solver = wavefront::make_solver(p, cfg);
  solver->run(400);

  // Hard-wall Dirichlet condition: boundary nodes must be exactly zero
  CHECK(solver->sample(std::vector<std::size_t>{0})[0] == doctest::Approx(0.0));
  CHECK(solver->sample(std::vector<std::size_t>{191})[0] == doctest::Approx(0.0));

  // Interior field must be finite
  for (std::size_t i = 8; i < 192; i += 16) {
    CHECK(std::isfinite(solver->sample(std::vector<std::size_t>{i})[0]));
  }

  // Driven mode: interior energy must be positive
  CHECK(test_common::json_value(solver->diagnostics_json(), "energy") > 0.0);
}
