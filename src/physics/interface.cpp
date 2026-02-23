#include "wavefront/physics/interface.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace wavefront {

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

InterfaceFluxResult compute_interface_flux(
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

  const double r = reflection_coefficient(z_in, z_out);
  const double t = transmission_coefficient(z_in, z_out);

  const double theta_out = refraction_angle(theta_incident_radians, c_in, c_out);
  const bool total_internal_reflection = std::isnan(theta_out);

  InterfaceFluxResult result;
  result.reflected = incident_amplitude * r;
  result.transmitted = total_internal_reflection ? 0.0 : incident_amplitude * t;
  result.mode_conversion = total_internal_reflection ? std::fabs(incident_amplitude)
                                                     : std::fabs(incident_amplitude) *
                                                           std::fabs(std::sin(theta_out - theta_incident_radians));
  return result;
}

}  // namespace wavefront
