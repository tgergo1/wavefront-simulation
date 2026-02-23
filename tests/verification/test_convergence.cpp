#include <doctest/doctest.h>

#include <vector>

#include "../test_common.hpp"

namespace {

std::vector<double> downsample_by_2(const std::vector<double>& fine) {
  std::vector<double> coarse;
  coarse.reserve(fine.size() / 2);
  for (std::size_t i = 0; i + 1 < fine.size(); i += 2) {
    coarse.push_back(fine[i]);
  }
  return coarse;
}

}  // namespace

TEST_CASE("manufactured-style grid refinement reduces discrepancy") {
  auto p32 = test_common::default_problem_1d(32);
  auto p64 = test_common::default_problem_1d(64);
  auto p128 = test_common::default_problem_1d(128);

  auto c = test_common::default_config(wavefront::SolverMode::LinearApprox);

  auto s32 = wavefront::make_solver(p32, c);
  auto s64 = wavefront::make_solver(p64, c);
  auto s128 = wavefront::make_solver(p128, c);

  s32->run(18);
  s64->run(36);
  s128->run(72);

  const auto y32 = test_common::sample_line(*s32, 32);
  const auto y64 = test_common::sample_line(*s64, 64);
  const auto y128 = test_common::sample_line(*s128, 128);

  const auto y128_to_64 = downsample_by_2(y128);
  const auto y64_to_32 = downsample_by_2(y64);

  const double e32 = test_common::l2_error(y32, y64_to_32);
  const double e64 = test_common::l2_error(y64, y128_to_64);

  CHECK(e64 <= e32);
}
