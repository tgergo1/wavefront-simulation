#include <doctest/doctest.h>

#include <filesystem>
#include <fstream>
#include <string>

#include "../test_common.hpp"
#include "wavefront/api/problem_validation.hpp"
#include "wavefront/material/library.hpp"

TEST_CASE("runtime solver exposes snapshots monitors far-field and IO") {
  auto problem = test_common::default_problem_1d(48);
  problem.source_term.text = "sin(t)";
  problem.monitors.probes.push_back({"centre", {24}, 0, true});
  problem.monitors.surfaces.push_back({"left", 0, false, 0});
  problem.monitors.surfaces.push_back({"right", 0, true, 0});
  problem.monitors.spectrum_bins = 4;
  problem.monitors.enable_far_field = true;

  auto config = test_common::default_config(wavefront::SolverMode::LinearApprox);
  config.family = wavefront::SolverFamily::FrequencyDomain;
  config.backend = wavefront::ExecutionBackend::GPUAccelerated;
  config.representation = wavefront::FieldRepresentation::ComplexPhasor;
  config.center_frequency = 2.0;
  config.far_field_samples = 7;

  auto solver = wavefront::make_solver(problem, config);
  solver->run(8);

  const auto snapshot = solver->field_snapshot();
  CHECK(snapshot.step == 8);
  CHECK(snapshot.shape == std::vector<std::size_t>{48});
  CHECK(snapshot.values.size() == 48);
  CHECK(snapshot.complex_values.size() == 48);
  CHECK(snapshot.complex_values[24].magnitude() >= 0.0);

  const auto history = solver->probe_history("centre");
  CHECK(history.size() == 9);
  CHECK(history.back().complex_value.magnitude() >= 0.0);

  const auto spectrum = solver->probe_spectrum("centre", 4);
  CHECK(spectrum.size() == 4);
  CHECK(std::isfinite(spectrum[0].magnitude));

  const auto left_flux = solver->surface_flux("left");
  const auto right_flux = solver->surface_flux("right");
  CHECK(left_flux.samples == 9);
  CHECK(right_flux.samples == 9);
  CHECK(left_flux.reflected_proxy >= 0.0);
  CHECK(right_flux.transmitted_proxy >= 0.0);

  const auto far_field = solver->far_field_pattern(7);
  CHECK(far_field.angles.size() == 7);
  CHECK(far_field.amplitudes.size() == 7);

  const std::filesystem::path tmp_dir = std::filesystem::temp_directory_path() / "wavefront-extended-api";
  std::filesystem::create_directories(tmp_dir);
  const std::filesystem::path checkpoint_path = tmp_dir / "solver.chk";
  const std::filesystem::path csv_path = tmp_dir / "field.csv";

  solver->save_checkpoint(checkpoint_path.string());
  solver->export_field_csv(csv_path.string());

  auto restored = wavefront::make_solver(problem, config);
  restored->load_checkpoint(checkpoint_path.string());
  CHECK(restored->sample(std::vector<std::size_t>{24}).at(0) == doctest::Approx(solver->sample(std::vector<std::size_t>{24}).at(0)));

  std::ifstream csv(csv_path);
  REQUIRE(csv.good());
  std::string header;
  std::getline(csv, header);
  CHECK(header == "flat,component,value,real,imaginary");

  const std::string diagnostics = solver->diagnostics_json();
  CHECK(diagnostics.find("\"family\":\"FrequencyDomain\"") != std::string::npos);
  CHECK(diagnostics.find("\"requested_backend\":\"GPUAccelerated\"") != std::string::npos);
  CHECK(diagnostics.find("\"active_backend\":\"ThreadedCPU\"") != std::string::npos);
}

TEST_CASE("geometry regions and extended validation are supported") {
  auto problem = test_common::default_problem_1d(64);
  problem.source_term.text = "sin(t) * exp(-pow(x_0-0.64, 2))";
  problem.geometry.push_back({"layer", wavefront::GeometryShape::Layer, {}, {}, {}, 0.0, 0, 0.4, 0.8, wavefront::builtin_material("water").medium});
  problem.monitors.probes.push_back({"probe", {32}, 0, false});

  auto config = test_common::default_config(wavefront::SolverMode::LinearApprox);
  config.family = wavefront::SolverFamily::AngularSpectrum;
  config.backend = wavefront::ExecutionBackend::Serial;

  auto baseline = test_common::default_problem_1d(64);
  baseline.source_term.text = problem.source_term.text;

  auto solver_with_geometry = wavefront::make_solver(problem, config);
  auto solver_without_geometry = wavefront::make_solver(baseline, config);
  solver_with_geometry->run(10);
  solver_without_geometry->run(10);

  const double with_geometry = solver_with_geometry->sample(std::vector<std::size_t>{32}).at(0);
  const double without_geometry = solver_without_geometry->sample(std::vector<std::size_t>{32}).at(0);
  CHECK(with_geometry != doctest::Approx(without_geometry));
  CHECK(solver_with_geometry->diagnostics_json().find("\"geometry_regions\":1") != std::string::npos);
  CHECK(solver_with_geometry->diagnostics_json().find("\"active_backend\":\"Serial\"") != std::string::npos);

  auto invalid = problem;
  invalid.monitors.probes.front().index.clear();
  CHECK_FALSE(wavefront::validate_problem(invalid, config).empty());

  invalid = problem;
  invalid.geometry.front().upper = invalid.geometry.front().lower;
  CHECK_FALSE(wavefront::validate_problem(invalid, config).empty());
}

TEST_CASE("polygon and signed-distance regions participate in runtime geometry and validation") {
  wavefront::ProblemSpec problem;
  problem.grid.dims = 2;
  problem.grid.shape = {40, 40};
  problem.grid.spacing = {0.03, 0.03};
  problem.grid.origin = {0.0, 0.0};
  problem.field_components = 1;
  problem.medium.density.text = "1.0";
  problem.medium.stiffness.text = "1.0";
  problem.medium.damping.text = "0.0005";
  problem.medium.dispersion.text = "0.0";
  problem.source_term.text =
      "10*sin(18*t)*exp(-6*t)*exp(-((x_0-0.45)*(x_0-0.45)+(x_1-0.5)*(x_1-0.5))/0.01)";
  problem.boundaries = {
      {wavefront::BoundaryType::Periodic, 0, false, wavefront::SymbolicExpr{"0.0"}},
      {wavefront::BoundaryType::Periodic, 0, true, wavefront::SymbolicExpr{"0.0"}},
      {wavefront::BoundaryType::Periodic, 1, false, wavefront::SymbolicExpr{"0.0"}},
      {wavefront::BoundaryType::Periodic, 1, true, wavefront::SymbolicExpr{"0.0"}},
  };

  wavefront::GeometryRegion polygon;
  polygon.name = "poly";
  polygon.shape = wavefront::GeometryShape::Polygon;
  polygon.vertices = {0.35, 0.35, 0.75, 0.45, 0.60, 0.78, 0.30, 0.70};
  polygon.medium = wavefront::builtin_material("water").medium;
  problem.geometry.push_back(polygon);

  wavefront::GeometryRegion sdf;
  sdf.name = "bubble";
  sdf.shape = wavefront::GeometryShape::SignedDistanceField;
  sdf.signed_distance = wavefront::SymbolicExpr{"sqrt((x_0-0.7)*(x_0-0.7)+(x_1-0.3)*(x_1-0.3)) - 0.12"};
  sdf.medium = wavefront::builtin_material("glass").medium;
  problem.geometry.push_back(sdf);

  problem.monitors.surfaces.push_back({"poly-shell", 0, false, 0, "poly", 0.03});

  auto baseline = problem;
  baseline.geometry.clear();
  baseline.monitors.surfaces.clear();

  auto config = test_common::default_config(wavefront::SolverMode::LinearApprox);
  config.spatial_order = 2;

  auto solver_with_geometry = wavefront::make_solver(problem, config);
  auto solver_without_geometry = wavefront::make_solver(baseline, config);
  solver_with_geometry->run(14);
  solver_without_geometry->run(14);

  const double with_geometry = solver_with_geometry->sample(std::vector<std::size_t>{20, 20}).at(0);
  const double without_geometry = solver_without_geometry->sample(std::vector<std::size_t>{20, 20}).at(0);
  CHECK(with_geometry != doctest::Approx(without_geometry));

  const auto polygon_flux = solver_with_geometry->surface_flux("poly-shell");
  CHECK(polygon_flux.samples == 15);
  CHECK(polygon_flux.integrated_flux >= 0.0);
  CHECK(polygon_flux.peak_flux >= 0.0);

  auto invalid = problem;
  invalid.monitors.surfaces.front().geometry_region = "missing";
  CHECK_FALSE(wavefront::validate_problem(invalid, config).empty());
}

TEST_CASE("fractal geometry regions can drive arbitrary-surface monitors") {
  wavefront::ProblemSpec problem;
  problem.grid.dims = 2;
  problem.grid.shape = {48, 48};
  problem.grid.spacing = {0.025, 0.025};
  problem.grid.origin = {0.0, 0.0};
  problem.field_components = 1;
  problem.medium.density.text = "1.0";
  problem.medium.stiffness.text = "1.0";
  problem.medium.damping.text = "0.0005";
  problem.medium.dispersion.text = "0.0";
  problem.source_term.text =
      "12*sin(20*t)*exp(-8*t)*exp(-((x_0-0.5)*(x_0-0.5)+(x_1-0.5)*(x_1-0.5))/0.008)";
  problem.boundaries = {
      {wavefront::BoundaryType::Periodic, 0, false, wavefront::SymbolicExpr{"0.0"}},
      {wavefront::BoundaryType::Periodic, 0, true, wavefront::SymbolicExpr{"0.0"}},
      {wavefront::BoundaryType::Periodic, 1, false, wavefront::SymbolicExpr{"0.0"}},
      {wavefront::BoundaryType::Periodic, 1, true, wavefront::SymbolicExpr{"0.0"}},
  };

  wavefront::GeometryRegion fractal;
  fractal.name = "snowflake";
  fractal.shape = wavefront::GeometryShape::Fractal;
  fractal.center = {0.6, 0.55};
  fractal.radius = 0.14;
  fractal.fractal_generator = "koch_snowflake";
  fractal.fractal_iterations = 2;
  fractal.medium = wavefront::builtin_material("steel").medium;
  problem.geometry.push_back(fractal);
  problem.monitors.surfaces.push_back({"snow-shell", 0, false, 0, "snowflake", 0.025});

  auto config = test_common::default_config(wavefront::SolverMode::LinearApprox);
  config.spatial_order = 2;
  auto solver = wavefront::make_solver(problem, config);
  solver->run(12);

  const auto flux = solver->surface_flux("snow-shell");
  CHECK(flux.samples == 13);
  CHECK(flux.integrated_flux >= 0.0);
  CHECK(std::isfinite(flux.phase_proxy));
  CHECK(flux.peak_flux >= 0.0);
}
