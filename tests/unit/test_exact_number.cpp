#include <doctest/doctest.h>

#include <cmath>
#include <stdexcept>
#include <string>

#include "wavefront/exact/exact_number.hpp"

using namespace wavefront::exact;

// ---------------------------------------------------------------------------
//  ExactNumber constructors
// ---------------------------------------------------------------------------

TEST_CASE("exact: default constructor is zero") {
  ExactNumber n;
  CHECK(n.to_long_double() == 0.0L);
  CHECK(n.str() == "0");
}

TEST_CASE("exact: int64 constructor") {
  ExactNumber n(static_cast<std::int64_t>(42));
  CHECK(n.to_long_double() == 42.0L);
}

TEST_CASE("exact: long double constructor") {
  ExactNumber n(3.14L);
  CHECK(n.to_long_double() == doctest::Approx(3.14));
}

TEST_CASE("exact: string constructor") {
  ExactNumber n(std::string("2.718"));
  CHECK(n.to_long_double() == doctest::Approx(2.718));
  CHECK(!n.str().empty());
}

// ---------------------------------------------------------------------------
//  ExactNumber arithmetic
// ---------------------------------------------------------------------------

TEST_CASE("exact: addition") {
  ExactNumber a(2.0L);
  ExactNumber b(3.0L);
  ExactNumber c = a + b;
  CHECK(c.to_long_double() == doctest::Approx(5.0));
}

TEST_CASE("exact: subtraction") {
  ExactNumber a(5.0L);
  ExactNumber b(3.0L);
  ExactNumber c = a - b;
  CHECK(c.to_long_double() == doctest::Approx(2.0));
}

TEST_CASE("exact: multiplication") {
  ExactNumber a(3.0L);
  ExactNumber b(4.0L);
  ExactNumber c = a * b;
  CHECK(c.to_long_double() == doctest::Approx(12.0));
}

TEST_CASE("exact: division") {
  ExactNumber a(10.0L);
  ExactNumber b(4.0L);
  ExactNumber c = a / b;
  CHECK(c.to_long_double() == doctest::Approx(2.5));
}

TEST_CASE("exact: division by zero throws") {
  ExactNumber a(1.0L);
  ExactNumber b(0.0L);
  CHECK_THROWS_AS(a / b, std::domain_error);
}

// ---------------------------------------------------------------------------
//  CertifiedInterval
// ---------------------------------------------------------------------------

TEST_CASE("exact: CertifiedInterval::contains") {
  CertifiedInterval iv{1.0L, 3.0L};
  CHECK(iv.contains(2.0L));
  CHECK(iv.contains(1.0L));
  CHECK(iv.contains(3.0L));
  CHECK_FALSE(iv.contains(0.5L));
  CHECK_FALSE(iv.contains(3.5L));
}

TEST_CASE("exact: CertifiedInterval::width") {
  CertifiedInterval iv{1.0L, 4.0L};
  CHECK(iv.width() == doctest::Approx(3.0));
}

TEST_CASE("exact: CertifiedInterval::around") {
  auto iv = CertifiedInterval::around(5.0L, 2.0L);
  CHECK(iv.lower == doctest::Approx(3.0));
  CHECK(iv.upper == doctest::Approx(7.0));
  CHECK(iv.contains(5.0L));
}

TEST_CASE("exact: CertifiedInterval::around with negative radius uses zero") {
  auto iv = CertifiedInterval::around(5.0L, -1.0L);
  CHECK(iv.lower == doctest::Approx(5.0));
  CHECK(iv.upper == doctest::Approx(5.0));
}

// ---------------------------------------------------------------------------
//  certify_nonlinear_operation
// ---------------------------------------------------------------------------

TEST_CASE("exact: certify_nonlinear_operation produces valid interval") {
  ExactNumber exact_val(3.0L);
  auto iv = certify_nonlinear_operation(exact_val, 3.0, 1.0e-10L, 1.0e-12L);
  CHECK(iv.contains(3.0L));
  CHECK(iv.width() > 0.0L);
}

TEST_CASE("exact: certify_nonlinear_operation with offset approximation") {
  ExactNumber exact_val(3.0L);
  auto iv = certify_nonlinear_operation(exact_val, 3.1, 1.0e-10L, 1.0e-12L);
  CHECK(iv.contains(3.0L));
}
