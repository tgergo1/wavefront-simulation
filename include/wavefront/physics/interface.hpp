#pragma once

#include <cstddef>

namespace wavefront {

struct InterfaceFluxResult {
  double reflected = 0.0;
  double transmitted = 0.0;
  double mode_conversion = 0.0;
};

double phase_velocity(double stiffness, double density);
double impedance(double density, double phase_velocity);
double reflection_coefficient(double z_in, double z_out);
double transmission_coefficient(double z_in, double z_out);
double refraction_angle(double theta_incident_radians, double c_in, double c_out);

InterfaceFluxResult compute_interface_flux(
    double incident_amplitude,
    double theta_incident_radians,
    double density_in,
    double stiffness_in,
    double density_out,
    double stiffness_out);

}  // namespace wavefront
