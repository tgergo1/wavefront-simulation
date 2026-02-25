#include <doctest/doctest.h>

#include <cmath>
#include <limits>
#include <vector>

#include "../test_common.hpp"
#include "wavefront/physics/interface.hpp"

// ---------------------------------------------------------------------------
//  Snell's law
// ---------------------------------------------------------------------------

TEST_CASE("refraction angle obeys Snell's law for normal incidence") {
  // At θ_i = 0 the transmitted ray stays normal regardless of speed ratio
  const double theta = wavefront::refraction_angle(0.0, 1.0, 2.0);
  CHECK(theta == doctest::Approx(0.0).epsilon(1.0e-14));
}

TEST_CASE("refraction angle obeys Snell's law: sin(θ_t)/sin(θ_i) = c_out/c_in") {
  const double c_in = 1.0;
  const double c_out = 1.5;
  const double theta_i = 0.3;  // ~17 degrees

  const double theta_t = wavefront::refraction_angle(theta_i, c_in, c_out);
  CHECK(std::isfinite(theta_t));

  // Snell: sin(θ_t) / sin(θ_i) = c_out / c_in
  const double ratio = std::sin(theta_t) / std::sin(theta_i);
  CHECK(ratio == doctest::Approx(c_out / c_in).epsilon(1.0e-12));
}

TEST_CASE("refraction angle symmetry: swapping c_in/c_out inverts the relationship") {
  const double theta_i = 0.25;
  const double c1 = 1.0;
  const double c2 = 2.0;

  const double theta_12 = wavefront::refraction_angle(theta_i, c1, c2);
  const double theta_21 = wavefront::refraction_angle(theta_12, c2, c1);

  CHECK(theta_21 == doctest::Approx(theta_i).epsilon(1.0e-10));
}

// ---------------------------------------------------------------------------
//  Total internal reflection
// ---------------------------------------------------------------------------

TEST_CASE("total internal reflection returns NaN when critical angle exceeded") {
  // TIR occurs when c_out > c_in and sin(θ_i) > c_in/c_out
  // c_in = 1, c_out = 3 → critical angle = arcsin(1/3) ≈ 0.3398
  const double c_in = 1.0;
  const double c_out = 3.0;
  const double theta_beyond_critical = 0.4;  // > arcsin(1/3)

  const double theta_t = wavefront::refraction_angle(theta_beyond_critical, c_in, c_out);
  CHECK(std::isnan(theta_t));
}

TEST_CASE("no total internal reflection when c_out <= c_in for any angle") {
  // When c_out <= c_in, (c_out/c_in)*sin(θ_i) <= sin(θ_i) <= 1
  const double c_in = 2.0;
  const double c_out = 1.0;

  for (double theta = 0.0; theta < 1.5; theta += 0.1) {
    const double theta_t = wavefront::refraction_angle(theta, c_in, c_out);
    CHECK(std::isfinite(theta_t));
  }
}

// ---------------------------------------------------------------------------
//  Impedance and Fresnel coefficients
// ---------------------------------------------------------------------------

TEST_CASE("impedance Z = ρ·c") {
  const double rho = 2.5;
  const double c = 3.0;
  CHECK(wavefront::impedance(rho, c) == doctest::Approx(rho * c));
}

TEST_CASE("reflection coefficient R = (Z2 - Z1)/(Z2 + Z1)") {
  const double z1 = 2.0;
  const double z2 = 5.0;
  const double expected = (z2 - z1) / (z2 + z1);
  CHECK(wavefront::reflection_coefficient(z1, z2) == doctest::Approx(expected));
}

TEST_CASE("transmission coefficient T = 2·Z2/(Z1 + Z2)") {
  const double z1 = 2.0;
  const double z2 = 5.0;
  const double expected = 2.0 * z2 / (z1 + z2);
  CHECK(wavefront::transmission_coefficient(z1, z2) == doctest::Approx(expected));
}

TEST_CASE("reflection + transmission energy conservation: R² + (Z1/Z2)·T² = 1") {
  // For plane waves the energy conservation relation is |R|^2 + (z1/z2)|T|^2 = 1
  const double z1 = 3.0;
  const double z2 = 7.0;
  const double r = wavefront::reflection_coefficient(z1, z2);
  const double t = wavefront::transmission_coefficient(z1, z2);

  const double energy = r * r + (z1 / z2) * t * t;
  CHECK(energy == doctest::Approx(1.0).epsilon(1.0e-12));
}

TEST_CASE("impedance-matched media give zero reflection") {
  const double z = 4.0;
  CHECK(wavefront::reflection_coefficient(z, z) == doctest::Approx(0.0));
  CHECK(wavefront::transmission_coefficient(z, z) == doctest::Approx(1.0));
}

TEST_CASE("reflection coefficient is antisymmetric: R(z1,z2) = -R(z2,z1)") {
  const double z1 = 2.0;
  const double z2 = 8.0;
  CHECK(wavefront::reflection_coefficient(z1, z2) ==
        doctest::Approx(-wavefront::reflection_coefficient(z2, z1)));
}

TEST_CASE("reflection coefficient bounded: |R| <= 1") {
  for (double z1 = 0.5; z1 <= 10.0; z1 += 0.5) {
    for (double z2 = 0.5; z2 <= 10.0; z2 += 0.5) {
      const double r = wavefront::reflection_coefficient(z1, z2);
      CHECK(std::fabs(r) <= 1.0 + 1.0e-14);
    }
  }
}

// ---------------------------------------------------------------------------
//  Phase velocity
// ---------------------------------------------------------------------------

TEST_CASE("phase velocity c = sqrt(K/ρ)") {
  CHECK(wavefront::phase_velocity(4.0, 1.0) == doctest::Approx(2.0));
  CHECK(wavefront::phase_velocity(9.0, 1.0) == doctest::Approx(3.0));
  CHECK(wavefront::phase_velocity(1.0, 4.0) == doctest::Approx(0.5));
}

TEST_CASE("phase velocity rejects non-positive inputs") {
  CHECK_THROWS_AS(wavefront::phase_velocity(0.0, 1.0), std::invalid_argument);
  CHECK_THROWS_AS(wavefront::phase_velocity(-1.0, 1.0), std::invalid_argument);
  CHECK_THROWS_AS(wavefront::phase_velocity(1.0, 0.0), std::invalid_argument);
  CHECK_THROWS_AS(wavefront::phase_velocity(1.0, -1.0), std::invalid_argument);
}

// ---------------------------------------------------------------------------
//  compute_interface_flux: comprehensive checks
// ---------------------------------------------------------------------------

TEST_CASE("interface flux at normal incidence: no mode conversion") {
  const auto flux = wavefront::compute_interface_flux(1.0, 0.0, 1.0, 1.0, 4.0, 4.0);
  CHECK(std::isfinite(flux.reflected));
  CHECK(std::isfinite(flux.transmitted));
  // At θ=0 the sin(θ_out - θ_in) term vanishes → mode_conversion ≈ 0
  CHECK(flux.mode_conversion == doctest::Approx(0.0).epsilon(1.0e-10));
}

TEST_CASE("interface flux transmitted vanishes under total internal reflection") {
  // Need c_out > c_in for TIR: c_in = sqrt(1/1)=1, c_out = sqrt(100/1)=10
  // s = (10/1)*sin(0.8) ≈ 7.17 → TIR
  const double steep = 0.8;  // well beyond critical angle
  const auto flux = wavefront::compute_interface_flux(1.0, steep, 1.0, 1.0, 1.0, 100.0);
  CHECK(flux.transmitted == doctest::Approx(0.0));
  CHECK(flux.mode_conversion > 0.0);
}

TEST_CASE("interface flux energy partition: |reflected|² + transmitted fraction bounded") {
  const auto flux = wavefront::compute_interface_flux(1.0, 0.2, 2.0, 4.0, 3.0, 9.0);
  CHECK(std::fabs(flux.reflected) <= 1.5);
  CHECK(std::fabs(flux.transmitted) <= 3.0);
  CHECK(flux.mode_conversion >= 0.0);
}

TEST_CASE("interface flux identical media gives no reflection") {
  const auto flux = wavefront::compute_interface_flux(1.0, 0.1, 2.0, 4.0, 2.0, 4.0);
  CHECK(flux.reflected == doctest::Approx(0.0).epsilon(1.0e-12));
  CHECK(flux.transmitted == doctest::Approx(1.0).epsilon(1.0e-6));
}

TEST_CASE("interface flux scales linearly with incident amplitude") {
  const auto flux_a = wavefront::compute_interface_flux(2.0, 0.3, 1.0, 1.0, 2.0, 4.0);
  const auto flux_b = wavefront::compute_interface_flux(4.0, 0.3, 1.0, 1.0, 2.0, 4.0);

  CHECK(flux_b.reflected == doctest::Approx(2.0 * flux_a.reflected).epsilon(1.0e-12));
  CHECK(flux_b.transmitted == doctest::Approx(2.0 * flux_a.transmitted).epsilon(1.0e-12));
}

// ---------------------------------------------------------------------------
//  Parameter validation
// ---------------------------------------------------------------------------

TEST_CASE("impedance rejects non-positive arguments") {
  CHECK_THROWS_AS(wavefront::impedance(0.0, 1.0), std::invalid_argument);
  CHECK_THROWS_AS(wavefront::impedance(1.0, 0.0), std::invalid_argument);
  CHECK_THROWS_AS(wavefront::impedance(-1.0, 1.0), std::invalid_argument);
}

TEST_CASE("reflection_coefficient rejects non-positive impedances") {
  CHECK_THROWS_AS(wavefront::reflection_coefficient(0.0, 1.0), std::invalid_argument);
  CHECK_THROWS_AS(wavefront::reflection_coefficient(1.0, -1.0), std::invalid_argument);
}

TEST_CASE("transmission_coefficient rejects non-positive impedances") {
  CHECK_THROWS_AS(wavefront::transmission_coefficient(0.0, 1.0), std::invalid_argument);
  CHECK_THROWS_AS(wavefront::transmission_coefficient(1.0, -1.0), std::invalid_argument);
}

TEST_CASE("refraction_angle rejects non-positive velocities") {
  CHECK_THROWS_AS(wavefront::refraction_angle(0.3, 0.0, 1.0), std::invalid_argument);
  CHECK_THROWS_AS(wavefront::refraction_angle(0.3, 1.0, -1.0), std::invalid_argument);
}
