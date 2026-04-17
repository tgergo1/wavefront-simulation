#include <doctest/doctest.h>

#include <cmath>
#include <stdexcept>
#include <string>

#include "wavefront/symbolic/expression.hpp"

// ---------------------------------------------------------------------------
//  Scientific notation parsing
// ---------------------------------------------------------------------------

TEST_CASE("expression: parse scientific notation") {
  const auto expr = wavefront::CompiledExpression::compile("1.5e2");
  wavefront::EvaluationContext ctx;
  CHECK(expr.evaluate_double(ctx) == doctest::Approx(150.0));
}

TEST_CASE("expression: parse scientific notation with explicit plus sign") {
  const auto expr = wavefront::CompiledExpression::compile("3.0E+1");
  wavefront::EvaluationContext ctx;
  CHECK(expr.evaluate_double(ctx) == doctest::Approx(30.0));
}

TEST_CASE("expression: parse scientific notation with negative exponent") {
  const auto expr = wavefront::CompiledExpression::compile("5.0e-2");
  wavefront::EvaluationContext ctx;
  CHECK(expr.evaluate_double(ctx) == doctest::Approx(0.05));
}

// ---------------------------------------------------------------------------
//  Error handling: unsupported token
// ---------------------------------------------------------------------------

TEST_CASE("expression: unsupported token throws") {
  CHECK_THROWS_AS(wavefront::CompiledExpression::compile("2 @ 3"), std::invalid_argument);
}

// ---------------------------------------------------------------------------
//  Error handling: mismatched parentheses
// ---------------------------------------------------------------------------

TEST_CASE("expression: mismatched right paren throws") {
  CHECK_THROWS_AS(wavefront::CompiledExpression::compile("3 + 2)"), std::invalid_argument);
}

TEST_CASE("expression: mismatched left paren throws") {
  CHECK_THROWS_AS(wavefront::CompiledExpression::compile("(3 + 2"), std::invalid_argument);
}

// ---------------------------------------------------------------------------
//  Operators and precedence
// ---------------------------------------------------------------------------

TEST_CASE("expression: power operator right-associative") {
  const auto expr = wavefront::CompiledExpression::compile("2^3^2");
  wavefront::EvaluationContext ctx;
  // Right-associative: 2^(3^2) = 2^9 = 512
  CHECK(expr.evaluate_double(ctx) == doctest::Approx(512.0));
}

TEST_CASE("expression: division operator") {
  const auto expr = wavefront::CompiledExpression::compile("10/4");
  wavefront::EvaluationContext ctx;
  CHECK(expr.evaluate_double(ctx) == doctest::Approx(2.5));
}

TEST_CASE("expression: power via caret") {
  const auto expr = wavefront::CompiledExpression::compile("3^2");
  wavefront::EvaluationContext ctx;
  CHECK(expr.evaluate_double(ctx) == doctest::Approx(9.0));
}

// ---------------------------------------------------------------------------
//  Unary negation
// ---------------------------------------------------------------------------

TEST_CASE("expression: unary negation at start of expression") {
  const auto expr = wavefront::CompiledExpression::compile("-5");
  wavefront::EvaluationContext ctx;
  CHECK(expr.evaluate_double(ctx) == doctest::Approx(-5.0));
}

TEST_CASE("expression: unary negation after operator") {
  const auto expr = wavefront::CompiledExpression::compile("3 * -2");
  wavefront::EvaluationContext ctx;
  CHECK(expr.evaluate_double(ctx) == doctest::Approx(-6.0));
}

TEST_CASE("expression: unary negation after left paren") {
  const auto expr = wavefront::CompiledExpression::compile("(-5) + 3");
  wavefront::EvaluationContext ctx;
  CHECK(expr.evaluate_double(ctx) == doctest::Approx(-2.0));
}

// ---------------------------------------------------------------------------
//  All single-argument functions
// ---------------------------------------------------------------------------

TEST_CASE("expression: all unary functions produce correct results") {
  wavefront::EvaluationContext ctx;
  ctx.x = {0.5L};

  {
    const auto expr = wavefront::CompiledExpression::compile("tan(x_0)");
    CHECK(expr.evaluate_double(ctx) == doctest::Approx(std::tan(0.5)));
  }
  {
    const auto expr = wavefront::CompiledExpression::compile("exp(x_0)");
    CHECK(expr.evaluate_double(ctx) == doctest::Approx(std::exp(0.5)));
  }
  {
    const auto expr = wavefront::CompiledExpression::compile("log(x_0)");
    CHECK(expr.evaluate_double(ctx) == doctest::Approx(std::log(0.5)));
  }
  {
    const auto expr = wavefront::CompiledExpression::compile("sqrt(x_0)");
    CHECK(expr.evaluate_double(ctx) == doctest::Approx(std::sqrt(0.5)));
  }
  {
    const auto expr = wavefront::CompiledExpression::compile("abs(x_0)");
    CHECK(expr.evaluate_double(ctx) == doctest::Approx(std::fabs(0.5)));
  }
  {
    const auto expr = wavefront::CompiledExpression::compile("tanh(x_0)");
    CHECK(expr.evaluate_double(ctx) == doctest::Approx(std::tanh(0.5)));
  }
  {
    const auto expr = wavefront::CompiledExpression::compile("cos(x_0)");
    CHECK(expr.evaluate_double(ctx) == doctest::Approx(std::cos(0.5)));
  }
}

// ---------------------------------------------------------------------------
//  Binary functions
// ---------------------------------------------------------------------------

TEST_CASE("expression: min function") {
  wavefront::EvaluationContext ctx;
  const auto expr = wavefront::CompiledExpression::compile("min(3, 7)");
  CHECK(expr.evaluate_double(ctx) == doctest::Approx(3.0));
}

// ---------------------------------------------------------------------------
//  Variable resolution: all prefixes
// ---------------------------------------------------------------------------

TEST_CASE("expression: resolves plain x prefix (without underscore)") {
  // x0 without underscore should also be parsed
  const auto expr = wavefront::CompiledExpression::compile("x0 + 1");
  wavefront::EvaluationContext ctx;
  ctx.x = {5.0L};
  CHECK(expr.evaluate_double(ctx) == doctest::Approx(6.0));
}

TEST_CASE("expression: resolves plain u prefix (without underscore)") {
  const auto expr = wavefront::CompiledExpression::compile("u0 + 1");
  wavefront::EvaluationContext ctx;
  ctx.u = {3.0L};
  CHECK(expr.evaluate_double(ctx) == doctest::Approx(4.0));
}

TEST_CASE("expression: resolves extra variables") {
  const auto expr = wavefront::CompiledExpression::compile("component + 1");
  wavefront::EvaluationContext ctx;
  ctx.extra["component"] = 2.0L;
  CHECK(expr.evaluate_double(ctx) == doctest::Approx(3.0));
}

TEST_CASE("expression: unbound variable throws") {
  const auto expr = wavefront::CompiledExpression::compile("unknown_var");
  wavefront::EvaluationContext ctx;
  CHECK_THROWS_AS(expr.evaluate_double(ctx), std::invalid_argument);
}

TEST_CASE("expression: variable index out of range throws") {
  const auto expr = wavefront::CompiledExpression::compile("x_5");
  wavefront::EvaluationContext ctx;
  ctx.x = {1.0L};  // only x_0 exists
  CHECK_THROWS_AS(expr.evaluate_double(ctx), std::out_of_range);
}

// ---------------------------------------------------------------------------
//  Empty expression
// ---------------------------------------------------------------------------

TEST_CASE("expression: default-constructed expression evaluates to zero") {
  wavefront::CompiledExpression expr;
  wavefront::EvaluationContext ctx;
  CHECK(expr.evaluate_long_double(ctx) == 0.0L);
}

// ---------------------------------------------------------------------------
//  Comma in function call
// ---------------------------------------------------------------------------

TEST_CASE("expression: comma outside function call throws") {
  CHECK_THROWS_AS(wavefront::CompiledExpression::compile("1, 2"), std::invalid_argument);
}

// ---------------------------------------------------------------------------
//  Expression that does not resolve to a single result
// ---------------------------------------------------------------------------

TEST_CASE("expression: unbalanced expression throws") {
  CHECK_THROWS_AS(wavefront::CompiledExpression::compile("1 2"), std::invalid_argument);
}

TEST_CASE("expression: invalid number literal with trailing 'e' throws") {
  // A bare "." is tokenized as a number but from_chars cannot parse it
  CHECK_THROWS_AS(wavefront::CompiledExpression::compile("."), std::invalid_argument);
}

TEST_CASE("expression: pow binary function") {
  wavefront::EvaluationContext ctx;
  const auto expr = wavefront::CompiledExpression::compile("pow(2, 3)");
  CHECK(expr.evaluate_double(ctx) == doctest::Approx(8.0));
}

TEST_CASE("expression: max binary function") {
  wavefront::EvaluationContext ctx;
  const auto expr = wavefront::CompiledExpression::compile("max(3, 7)");
  CHECK(expr.evaluate_double(ctx) == doctest::Approx(7.0));
}

TEST_CASE("expression: derivatives context variable resolution") {
  const auto expr = wavefront::CompiledExpression::compile("du0_dx0 + 1");
  wavefront::EvaluationContext ctx;
  ctx.derivatives["du0_dx0"] = 5.0L;
  CHECK(expr.evaluate_double(ctx) == doctest::Approx(6.0));
}

TEST_CASE("expression: u_ prefix variable resolution") {
  const auto expr = wavefront::CompiledExpression::compile("u_0 + 1");
  wavefront::EvaluationContext ctx;
  ctx.u = {7.0L};
  CHECK(expr.evaluate_double(ctx) == doctest::Approx(8.0));
}

TEST_CASE("expression: time variable t") {
  const auto expr = wavefront::CompiledExpression::compile("t + 1");
  wavefront::EvaluationContext ctx;
  ctx.t = 2.0L;
  CHECK(expr.evaluate_double(ctx) == doctest::Approx(3.0));
}

TEST_CASE("expression: canonical_form returns non-empty string") {
  const auto expr = wavefront::CompiledExpression::compile("sin(x_0) + 1");
  CHECK(!expr.canonical_form().empty());
}
