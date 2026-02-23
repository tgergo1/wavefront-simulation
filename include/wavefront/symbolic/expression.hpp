#pragma once

#include <cstddef>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace wavefront {

enum class OpCode {
  PushConstant,
  PushVariable,
  Add,
  Sub,
  Mul,
  Div,
  Pow,
  Neg,
  Func1,
  Func2,
};

struct Instruction {
  OpCode op = OpCode::PushConstant;
  long double constant_value = 0.0L;
  std::size_t operand_index = 0;
  std::string symbol;
};

struct EvaluationContext {
  std::vector<long double> x;
  long double t = 0.0L;
  std::vector<long double> u;
  std::unordered_map<std::string, long double> derivatives;
  std::unordered_map<std::string, long double> extra;
};

class CompiledExpression {
 public:
  CompiledExpression() = default;

  static CompiledExpression compile(std::string_view expression);

  long double evaluate_long_double(const EvaluationContext& context) const;
  double evaluate_double(const EvaluationContext& context) const;

  const std::string& canonical_form() const;
  const std::vector<Instruction>& bytecode() const;

 private:
  static long double resolve_variable(std::string_view name, const EvaluationContext& context);

  std::string canonical_;
  std::vector<Instruction> bytecode_;
};

}  // namespace wavefront
