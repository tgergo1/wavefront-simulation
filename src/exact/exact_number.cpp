#include "wavefront/exact/exact_number.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>

#if WAVEFRONT_HAS_LIMITLESS
#if __has_include(<limitless.hpp>)
#ifndef LIMITLESS_IMPLEMENTATION
#define LIMITLESS_IMPLEMENTATION
#endif
#include <limitless.hpp>
#define WAVEFRONT_LIMITLESS_READY 1
#else
#define WAVEFRONT_LIMITLESS_READY 0
#endif
#else
#define WAVEFRONT_LIMITLESS_READY 0
#endif

namespace wavefront::exact {
namespace {

std::string to_string_precise(long double value) {
  std::ostringstream out;
  out.precision(std::numeric_limits<long double>::max_digits10);
  out << value;
  return out.str();
}

}  // namespace

ExactNumber::ExactNumber() : value_(0.0L), exact_repr_("0") {}

ExactNumber::ExactNumber(std::int64_t value)
    : value_(static_cast<long double>(value)), exact_repr_(to_string_precise(static_cast<long double>(value))) {}

ExactNumber::ExactNumber(long double value) : value_(value), exact_repr_(to_string_precise(value)) {}

ExactNumber::ExactNumber(const std::string& value) : value_(std::stold(value)), exact_repr_(value) {}

std::string ExactNumber::str() const {
  return exact_repr_;
}

long double ExactNumber::to_long_double() const {
  return value_;
}

ExactNumber operator+(const ExactNumber& lhs, const ExactNumber& rhs) {
  ExactNumber out(lhs.value_ + rhs.value_);
#if WAVEFRONT_LIMITLESS_READY
  limitless::number a = static_cast<double>(lhs.value_);
  limitless::number b = static_cast<double>(rhs.value_);
  out.exact_repr_ = (a + b).str();
#endif
  return out;
}

ExactNumber operator-(const ExactNumber& lhs, const ExactNumber& rhs) {
  ExactNumber out(lhs.value_ - rhs.value_);
#if WAVEFRONT_LIMITLESS_READY
  limitless::number a = static_cast<double>(lhs.value_);
  limitless::number b = static_cast<double>(rhs.value_);
  out.exact_repr_ = (a - b).str();
#endif
  return out;
}

ExactNumber operator*(const ExactNumber& lhs, const ExactNumber& rhs) {
  ExactNumber out(lhs.value_ * rhs.value_);
#if WAVEFRONT_LIMITLESS_READY
  limitless::number a = static_cast<double>(lhs.value_);
  limitless::number b = static_cast<double>(rhs.value_);
  out.exact_repr_ = (a * b).str();
#endif
  return out;
}

ExactNumber operator/(const ExactNumber& lhs, const ExactNumber& rhs) {
  if (rhs.value_ == 0.0L) {
    throw std::domain_error("division by zero in ExactNumber");
  }

  ExactNumber out(lhs.value_ / rhs.value_);
#if WAVEFRONT_LIMITLESS_READY
  limitless::number a = static_cast<double>(lhs.value_);
  limitless::number b = static_cast<double>(rhs.value_);
  out.exact_repr_ = (a / b).str();
#endif
  return out;
}

bool CertifiedInterval::contains(long double value) const {
  return value >= lower && value <= upper;
}

long double CertifiedInterval::width() const {
  return upper - lower;
}

CertifiedInterval CertifiedInterval::around(long double center, long double radius) {
  const long double safe_radius = std::max<long double>(0.0L, radius);
  return {center - safe_radius, center + safe_radius};
}

CertifiedInterval certify_nonlinear_operation(
    const ExactNumber& exact_value,
    long double approximation,
    long double relative_tolerance,
    long double absolute_floor) {
  const long double exact = exact_value.to_long_double();
  const long double scale = std::max(std::fabs(exact), absolute_floor);
  const long double radius = std::max(std::fabs(exact - approximation), relative_tolerance * scale);
  return CertifiedInterval::around(exact, radius);
}

}  // namespace wavefront::exact
