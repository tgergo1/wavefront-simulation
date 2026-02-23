#include <doctest/doctest.h>

#include <cmath>

#include "wavefront/symbolic/expression.hpp"

TEST_CASE("symbolic parser is deterministic") {
  const auto expr_a = wavefront::CompiledExpression::compile("sin(x_0) + 2*u_0 - 3");
  const auto expr_b = wavefront::CompiledExpression::compile(" sin ( x_0 )+2 * u_0-3 ");

  CHECK(expr_a.canonical_form() == expr_b.canonical_form());
  CHECK(expr_a.bytecode().size() == expr_b.bytecode().size());
}

TEST_CASE("symbolic evaluator binds x,t,u and derivative symbols") {
  const auto expr = wavefront::CompiledExpression::compile("u_0 + 0.5*x_0 + t + du0_dx0");

  wavefront::EvaluationContext context;
  context.x = {4.0L};
  context.t = 2.0L;
  context.u = {3.0L};
  context.derivatives["du0_dx0"] = -1.0L;

  const long double value = expr.evaluate_long_double(context);
  CHECK(value == doctest::Approx(3.0 + 0.5 * 4.0 + 2.0 - 1.0));
}

TEST_CASE("symbolic evaluator supports binary functions") {
  const auto expr = wavefront::CompiledExpression::compile("max(pow(x_0,2), min(5, u_0))");

  wavefront::EvaluationContext context;
  context.x = {3.0L};
  context.u = {4.0L};

  CHECK(expr.evaluate_double(context) == doctest::Approx(9.0));
}
