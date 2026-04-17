#include <doctest/doctest.h>

#include <string>

#include "wavefront/api/problem_validation.hpp"
#include "wavefront/api/solver.hpp"

namespace {

wavefront::ProblemSpec valid_problem() {
  wavefront::ProblemSpec p;
  p.grid.dims = 1;
  p.grid.shape = {8};
  p.grid.spacing = {0.1};
  p.grid.origin = {0.0};
  p.field_components = 1;
  return p;
}

wavefront::SolverConfig valid_config() {
  wavefront::SolverConfig c;
  c.mode = wavefront::SolverMode::LinearApprox;
  c.precision = wavefront::PrecisionMode::FastFloat64;
  c.cfl = 0.4;
  c.threads = 1;
  c.spatial_order = 2;
  return c;
}

}  // namespace

TEST_CASE("validation: zero dims is fatal") {
  auto p = valid_problem();
  p.grid.dims = 0;
  auto issues = wavefront::validate_problem(p, valid_config());
  CHECK(!issues.empty());
  CHECK(issues[0].fatal);
}

TEST_CASE("validation: shape rank mismatch") {
  auto p = valid_problem();
  p.grid.shape = {8, 8};  // dims=1 but shape has 2 entries
  auto issues = wavefront::validate_problem(p, valid_config());
  CHECK(!issues.empty());
}

TEST_CASE("validation: spacing rank mismatch") {
  auto p = valid_problem();
  p.grid.spacing = {0.1, 0.1};  // dims=1 but spacing has 2 entries
  auto issues = wavefront::validate_problem(p, valid_config());
  CHECK(!issues.empty());
}

TEST_CASE("validation: origin wrong size") {
  auto p = valid_problem();
  p.grid.origin = {0.0, 0.0};  // dims=1 but origin has 2 entries
  auto issues = wavefront::validate_problem(p, valid_config());
  CHECK(!issues.empty());
}

TEST_CASE("validation: shape entries too small") {
  auto p = valid_problem();
  p.grid.shape = {2};  // less than 3
  auto issues = wavefront::validate_problem(p, valid_config());
  CHECK(!issues.empty());
}

TEST_CASE("validation: spacing non-positive") {
  auto p = valid_problem();
  p.grid.spacing = {0.0};
  auto issues = wavefront::validate_problem(p, valid_config());
  CHECK(!issues.empty());
}

TEST_CASE("validation: field_components is zero") {
  auto p = valid_problem();
  p.field_components = 0;
  auto issues = wavefront::validate_problem(p, valid_config());
  CHECK(!issues.empty());
}

TEST_CASE("validation: boundary axis out of range") {
  auto p = valid_problem();
  p.boundaries.push_back({wavefront::BoundaryType::Dirichlet, 5, false, wavefront::SymbolicExpr{"0.0"}});
  auto issues = wavefront::validate_problem(p, valid_config());
  CHECK(!issues.empty());
}

TEST_CASE("validation: cfl out of range") {
  auto p = valid_problem();
  auto c = valid_config();
  c.cfl = 0.0;
  auto issues = wavefront::validate_problem(p, c);
  CHECK(!issues.empty());

  c.cfl = 1.5;
  issues = wavefront::validate_problem(p, c);
  CHECK(!issues.empty());
}

TEST_CASE("validation: threads zero") {
  auto p = valid_problem();
  auto c = valid_config();
  c.threads = 0;
  auto issues = wavefront::validate_problem(p, c);
  CHECK(!issues.empty());
}

TEST_CASE("validation: spatial_order invalid") {
  auto p = valid_problem();
  auto c = valid_config();
  c.spatial_order = 3;
  auto issues = wavefront::validate_problem(p, c);
  CHECK(!issues.empty());
}

TEST_CASE("validation: ExactReference with reference_window zero") {
  auto p = valid_problem();
  auto c = valid_config();
  c.precision = wavefront::PrecisionMode::ExactReference;
  c.reference_window = 0;
  auto issues = wavefront::validate_problem(p, c);
  CHECK(!issues.empty());
}

TEST_CASE("validation: throw_if_invalid throws on bad config") {
  auto p = valid_problem();
  auto c = valid_config();
  c.threads = 0;
  CHECK_THROWS_AS(wavefront::throw_if_invalid(p, c), std::invalid_argument);
}

TEST_CASE("validation: throw_if_invalid does not throw on valid config") {
  CHECK_NOTHROW(wavefront::throw_if_invalid(valid_problem(), valid_config()));
}

TEST_CASE("validation: valid problem produces no issues") {
  auto issues = wavefront::validate_problem(valid_problem(), valid_config());
  CHECK(issues.empty());
}
