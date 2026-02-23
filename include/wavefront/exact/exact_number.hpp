#pragma once

#include <cstdint>
#include <functional>
#include <string>

namespace wavefront::exact {

class ExactNumber {
 public:
  ExactNumber();
  explicit ExactNumber(std::int64_t value);
  explicit ExactNumber(long double value);
  explicit ExactNumber(const std::string& value);

  std::string str() const;
  long double to_long_double() const;

  friend ExactNumber operator+(const ExactNumber& lhs, const ExactNumber& rhs);
  friend ExactNumber operator-(const ExactNumber& lhs, const ExactNumber& rhs);
  friend ExactNumber operator*(const ExactNumber& lhs, const ExactNumber& rhs);
  friend ExactNumber operator/(const ExactNumber& lhs, const ExactNumber& rhs);

 private:
  long double value_ = 0.0L;
  std::string exact_repr_ = "0";
};

struct CertifiedInterval {
  long double lower = 0.0L;
  long double upper = 0.0L;

  bool contains(long double value) const;
  long double width() const;

  static CertifiedInterval around(long double center, long double radius);
};

CertifiedInterval certify_nonlinear_operation(
    const ExactNumber& exact_value,
    long double approximation,
    long double relative_tolerance,
    long double absolute_floor);

}  // namespace wavefront::exact
