#include "wavefront/physics/interface.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace wavefront {
namespace {

constexpr double kPi = 3.14159265358979323846;

}

double phase_velocity(double stiffness, double density) {
  if (stiffness <= 0.0 || density <= 0.0) {
    throw std::invalid_argument("stiffness and density must be positive");
  }
  return std::sqrt(stiffness / density);
}

double impedance(double density, double phase_velocity_value) {
  if (density <= 0.0 || phase_velocity_value <= 0.0) {
    throw std::invalid_argument("density and phase velocity must be positive");
  }
  return density * phase_velocity_value;
}

double reflection_coefficient(double z_in, double z_out) {
  if (z_in <= 0.0 || z_out <= 0.0) {
    throw std::invalid_argument("impedances must be positive");
  }
  return (z_out - z_in) / (z_out + z_in);
}

double transmission_coefficient(double z_in, double z_out) {
  if (z_in <= 0.0 || z_out <= 0.0) {
    throw std::invalid_argument("impedances must be positive");
  }
  return (2.0 * z_out) / (z_in + z_out);
}

double refraction_angle(double theta_incident_radians, double c_in, double c_out) {
  if (c_in <= 0.0 || c_out <= 0.0) {
    throw std::invalid_argument("phase velocities must be positive");
  }

  const double s = (c_out / c_in) * std::sin(theta_incident_radians);
  if (std::fabs(s) > 1.0) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  return std::asin(std::clamp(s, -1.0, 1.0));
}

InterfaceScatteringResult compute_planar_interface_scattering(
    double incident_amplitude,
    double theta_incident_radians,
    double density_in,
    double stiffness_in,
    double density_out,
    double stiffness_out) {
  const double c_in = phase_velocity(stiffness_in, density_in);
  const double c_out = phase_velocity(stiffness_out, density_out);

  const double z_in = impedance(density_in, c_in);
  const double z_out = impedance(density_out, c_out);

  const double theta_out = refraction_angle(theta_incident_radians, c_in, c_out);

  InterfaceScatteringResult result;
  result.reflected_angle_radians = theta_incident_radians;
  result.transmitted_angle_radians = theta_out;
  result.total_internal_reflection = std::isnan(theta_out);

  if (result.total_internal_reflection) {
    const double r_sign = reflection_coefficient(z_in, z_out) < 0.0 ? -1.0 : 1.0;
    result.reflected_amplitude = incident_amplitude * r_sign;
    result.transmitted_amplitude = 0.0;
    result.reflected_power = incident_amplitude * incident_amplitude;
    result.transmitted_power = 0.0;
    result.phase_shift_radians = 0.5 * kPi;
    return result;
  }

  const double cos_in = std::fabs(std::cos(theta_incident_radians));
  const double cos_out = std::fabs(std::cos(theta_out));
  const double denominator = z_out * cos_in + z_in * cos_out;
  const double r = (z_out * cos_in - z_in * cos_out) / denominator;
  const double t = (2.0 * z_out * cos_in) / denominator;

  result.reflected_amplitude = incident_amplitude * r;
  result.transmitted_amplitude = incident_amplitude * t;
  result.reflected_power = incident_amplitude * incident_amplitude * r * r;
  result.transmitted_power =
      incident_amplitude * incident_amplitude * (z_in * cos_out) / (z_out * std::max(cos_in, 1.0e-12)) *
      t * t;
  result.phase_shift_radians = r < 0.0 ? kPi : 0.0;
  return result;
}

InterfaceFluxResult compute_interface_flux(
    double incident_amplitude,
    double theta_incident_radians,
    double density_in,
    double stiffness_in,
    double density_out,
    double stiffness_out) {
  const double c_in = phase_velocity(stiffness_in, density_in);
  const double c_out = phase_velocity(stiffness_out, density_out);
  const auto scattering = compute_planar_interface_scattering(
      incident_amplitude,
      theta_incident_radians,
      density_in,
      stiffness_in,
      density_out,
      stiffness_out);

  InterfaceFluxResult result;
  result.reflected = scattering.reflected_amplitude;
  result.transmitted = scattering.transmitted_amplitude;
  result.mode_conversion = scattering.total_internal_reflection
                               ? std::fabs(incident_amplitude)
                               : std::fabs(incident_amplitude) *
                                     std::fabs(std::sin(refraction_angle(theta_incident_radians, c_in, c_out) -
                                                        theta_incident_radians));
  return result;
}

}  // namespace wavefront
