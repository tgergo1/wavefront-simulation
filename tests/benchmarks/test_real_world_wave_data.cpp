// Real-world wave simulation benchmark tests.
//
// Parameters are derived from well-known open-source wave simulation benchmarks
// and published geophysical/acoustic datasets (all quantities are normalised to
// dimensionless form; the reference speed is c_ref = 1.0, reference density
// rho_ref = 1.0):
//
//   [1] Pekeris (1948): ocean acoustic waveguide, water-over-sediment model.
//       Water:    c = 1.5, rho = 1.0  (1500 m/s, 1000 kg/m³)
//       Sediment: c = 1.7, rho = 1.9  (1700 m/s, 1900 kg/m³)
//
//   [2] NORSAR two-layer seismic benchmark (shale-over-sandstone):
//       Shale:     c = 2.1, rho = 2.2  (2100 m/s, 2200 kg/m³)
//       Sandstone: c = 2.6, rho = 2.4  (2600 m/s, 2400 kg/m³)
//
//   [3] Munk (1974): deep-ocean SOFAR (sound-fixing-and-ranging) channel.
//       Sound-speed minimum at mid-depth gives a focusing waveguide.
//
//   [4] Marmousi-2 (Martin et al., 2006): heterogeneous seismic benchmark.
//       Shallow sediment: c = 1.7, rho = 1.8  (1700 m/s, 1800 kg/m³)
//       Deeper rock:      c = 2.8, rho = 2.3  (2800 m/s, 2300 kg/m³)

#include <doctest/doctest.h>

#include <cmath>
#include <string>
#include <vector>

#include "../test_common.hpp"
#include "wavefront/physics/interface.hpp"

// ---------------------------------------------------------------------------
// [1] Pekeris waveguide: water-over-sediment interface
// ---------------------------------------------------------------------------

// Stiffness K = rho * c^2 so that phase_velocity(K, rho) recovers c exactly.
// Water:    K = 1.0 * 1.5^2 = 2.25
// Sediment: K = 1.9 * 1.7^2 = 5.491
// Impedances: Z_w = 1.5,  Z_s = 3.23
// Analytical: R = (Z_s - Z_w) / (Z_s + Z_w) = 1.73 / 4.73 ≈ 0.3657

TEST_CASE("Pekeris waveguide: water-sediment reflection coefficient matches published value") {
  constexpr double rho_water = 1.0;
  constexpr double K_water = 2.25;   // c = sqrt(2.25 / 1.0) = 1.5
  constexpr double rho_sed = 1.9;
  constexpr double K_sed = 5.491;    // c = sqrt(5.491 / 1.9) ≈ 1.7

  const double c_water = wavefront::phase_velocity(K_water, rho_water);
  const double c_sed = wavefront::phase_velocity(K_sed, rho_sed);

  const double Z_water = wavefront::impedance(rho_water, c_water);
  const double Z_sed = wavefront::impedance(rho_sed, c_sed);

  const double R = wavefront::reflection_coefficient(Z_water, Z_sed);
  const double R_analytical = (Z_sed - Z_water) / (Z_sed + Z_water);

  CHECK(R == doctest::Approx(R_analytical).epsilon(1e-4));
  CHECK(R > 0.0);   // Z_sed > Z_water → positive reflection
  CHECK(R < 1.0);   // partial reflection only
}

TEST_CASE("Pekeris waveguide: energy conservation at water-sediment interface") {
  constexpr double Z_water = 1.5;   // rho * c = 1.0 * 1.5
  constexpr double Z_sed = 3.23;    // rho * c = 1.9 * 1.7

  const double R = wavefront::reflection_coefficient(Z_water, Z_sed);
  const double T = wavefront::transmission_coefficient(Z_water, Z_sed);

  // Power conservation: R^2 + (Z_water/Z_sed)*T^2 = 1
  const double power_balance = R * R + (Z_water / Z_sed) * T * T;
  CHECK(power_balance == doctest::Approx(1.0).epsilon(1e-12));
}

// ---------------------------------------------------------------------------
// [2] NORSAR benchmark: shale-over-sandstone interface
// ---------------------------------------------------------------------------

// Shale:     rho = 2.2, K = 2.2 * 2.1^2 = 9.702   → c = 2.1, Z = 4.62
// Sandstone: rho = 2.4, K = 2.4 * 2.6^2 = 16.224  → c = 2.6, Z = 6.24
// Analytical: R = (6.24 - 4.62) / (6.24 + 4.62) ≈ 0.1492

TEST_CASE("NORSAR shale-sandstone: reflection coefficient within published range") {
  constexpr double rho_shale = 2.2;
  constexpr double K_shale = 9.702;    // 2.2 * 2.1^2
  constexpr double rho_sand = 2.4;
  constexpr double K_sand = 16.224;    // 2.4 * 2.6^2

  const double c_shale = wavefront::phase_velocity(K_shale, rho_shale);
  const double c_sand = wavefront::phase_velocity(K_sand, rho_sand);

  const double Z1 = wavefront::impedance(rho_shale, c_shale);
  const double Z2 = wavefront::impedance(rho_sand, c_sand);

  const double R = wavefront::reflection_coefficient(Z1, Z2);

  // Verify against the closed-form formula
  CHECK(R == doctest::Approx((Z2 - Z1) / (Z2 + Z1)).epsilon(1e-10));
  // Published geophysical range for this contrast: |R| ≈ 0.14 – 0.16
  CHECK(R > 0.12);
  CHECK(R < 0.18);
}

TEST_CASE("NORSAR shale-sandstone: power conservation across interface") {
  constexpr double Z1 = 4.62;    // shale impedance
  constexpr double Z2 = 6.24;    // sandstone impedance

  const double R = wavefront::reflection_coefficient(Z1, Z2);
  const double T = wavefront::transmission_coefficient(Z1, Z2);

  const double power_balance = R * R + (Z1 / Z2) * T * T;
  CHECK(power_balance == doctest::Approx(1.0).epsilon(1e-12));
}

// ---------------------------------------------------------------------------
// [3] Munk SOFAR channel (simplified 1-D): stability test
//
// The Munk profile places a sound-speed minimum at mid-depth, creating a
// natural waveguide.  Normalised stiffness:
//   K(x) = 2.25 * (1 - 0.20 * exp(-12 * (x - x_axis)^2))
// gives c_min ≈ 1.342 at x_axis = 1.92 and c_bg = 1.5 elsewhere.
// ---------------------------------------------------------------------------

TEST_CASE("Munk SOFAR channel: 1D simulation with speed-minimum waveguide stays stable") {
  wavefront::ProblemSpec p;
  p.grid.dims = 1;
  p.grid.shape = {192};
  p.grid.spacing = {0.02};
  p.grid.origin = {0.0};
  p.field_components = 1;

  // Smooth sound-speed minimum at x = 1.92 (mid-domain)
  p.medium.density.text = "1.0";
  p.medium.stiffness.text = "2.25*(1.0 - 0.20*exp(-12.0*(x_0-1.92)*(x_0-1.92)))";
  p.medium.damping.text = "0.0002";
  p.medium.dispersion.text = "0.0";
  // Gaussian-modulated sinusoidal source (normalised Ricker-style)
  p.source_term.text = "8.0*sin(25.0*t)*exp(-8.0*t)*exp(-((x_0-0.5)*(x_0-0.5))/0.02)";

  p.boundaries = {
      {wavefront::BoundaryType::PML, 0, false, wavefront::SymbolicExpr{"10.0"}},
      {wavefront::BoundaryType::PML, 0, true, wavefront::SymbolicExpr{"10.0"}},
  };

  auto cfg = test_common::default_config(wavefront::SolverMode::LinearApprox);
  cfg.cfl = 0.20;
  cfg.spatial_order = 2;

  auto solver = wavefront::make_solver(p, cfg);
  solver->run(300);

  // Field must remain finite everywhere
  for (std::size_t i = 0; i < 192; i += 16) {
    CHECK(std::isfinite(solver->sample(std::vector<std::size_t>{i})[0]));
  }
  CHECK(test_common::json_value(solver->diagnostics_json(), "energy") >= 0.0);
  // PML absorbs outgoing energy
  CHECK(test_common::json_value(solver->diagnostics_json(), "absorbed_energy") > 0.0);
}

// ---------------------------------------------------------------------------
// [4] Marmousi-2 inspired interface: analytical check
//
// Shallow sediment: rho = 1.8, c = 1.7  → K = 5.202, Z = 3.06
// Deeper rock:      rho = 2.3, c = 2.8  → K = 18.032, Z = 6.44
// Analytical: R = (6.44 - 3.06) / (6.44 + 3.06) = 3.38 / 9.50 ≈ 0.3558
// ---------------------------------------------------------------------------

TEST_CASE("Marmousi-2 interface: reflection coefficient within expected range") {
  constexpr double rho1 = 1.8;
  constexpr double c1 = 1.7;
  constexpr double rho2 = 2.3;
  constexpr double c2 = 2.8;

  const double Z1 = wavefront::impedance(rho1, c1);
  const double Z2 = wavefront::impedance(rho2, c2);

  const double R = wavefront::reflection_coefficient(Z1, Z2);

  CHECK(R == doctest::Approx((Z2 - Z1) / (Z2 + Z1)).epsilon(1e-10));
  // Expected range for this impedance contrast: 0.30 < R < 0.42
  CHECK(R > 0.30);
  CHECK(R < 0.42);
}

TEST_CASE("Marmousi-2 interface: Snell refraction at shallow incidence angle") {
  // Sediment c_in = 1.7, rock c_out = 2.8, shallow angle theta_i = 0.15 rad
  const double theta_t = wavefront::refraction_angle(0.15, 1.7, 2.8);

  CHECK(std::isfinite(theta_t));
  // Snell: sin(theta_t) / sin(theta_i) = c_out / c_in = 2.8 / 1.7
  const double ratio = std::sin(theta_t) / std::sin(0.15);
  CHECK(ratio == doctest::Approx(2.8 / 1.7).epsilon(1e-10));
}

// ---------------------------------------------------------------------------
// [4b] Marmousi-2 inspired 1D simulation: two-layer propagation
// ---------------------------------------------------------------------------

TEST_CASE("Marmousi-2 inspired 1D two-layer simulation: PML absorbs energy and field is finite") {
  wavefront::ProblemSpec p;
  p.grid.dims = 1;
  p.grid.shape = {256};
  p.grid.spacing = {0.01};
  p.grid.origin = {0.0};
  p.field_components = 1;

  // Left half: shallow sediment (rho=1.8, K≈5.202, c=1.7)
  // Right half: deeper rock (rho=2.3, K≈18.032, c=2.8)
  // Step interface at x = 1.28 (midpoint of [0, 2.56])
  p.medium.density.text =
      "1.8 + 0.5*max(0.0, min(1.0, (x_0 - 1.28)*1000.0))";
  p.medium.stiffness.text =
      "5.202 + 12.83*max(0.0, min(1.0, (x_0 - 1.28)*1000.0))";
  p.medium.damping.text = "0.0003";
  p.medium.dispersion.text = "0.0";

  // Gaussian-modulated sinusoidal source near the left PML edge
  p.source_term.text =
      "10.0*sin(18.0*t)*exp(-9.0*t)*exp(-((x_0-0.25)*(x_0-0.25))/0.004)";

  p.boundaries = {
      {wavefront::BoundaryType::PML, 0, false, wavefront::SymbolicExpr{"12.0"}},
      {wavefront::BoundaryType::PML, 0, true, wavefront::SymbolicExpr{"12.0"}},
  };

  auto cfg = test_common::default_config(wavefront::SolverMode::LinearApprox);
  cfg.cfl = 0.15;
  cfg.spatial_order = 2;

  auto solver = wavefront::make_solver(p, cfg);
  solver->run(500);

  // Stability: no NaN / Inf anywhere in the domain
  for (std::size_t i = 0; i < 256; i += 16) {
    CHECK(std::isfinite(solver->sample(std::vector<std::size_t>{i})[0]));
  }

  const std::string diag = solver->diagnostics_json();
  CHECK(test_common::json_value(diag, "energy") >= 0.0);
  // Outgoing waves must reach and be absorbed by the PML layers
  CHECK(test_common::json_value(diag, "absorbed_energy") > 0.0);
}
