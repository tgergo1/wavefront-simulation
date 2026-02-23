#include "wavefront/symbolic/expression.hpp"

#include <cmath>
#include <cctype>
#include <charconv>
#include <limits>
#include <sstream>
#include <stack>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>

namespace wavefront {
namespace {

enum class TokenType {
  Number,
  Identifier,
  Operator,
  LeftParen,
  RightParen,
  Comma,
  Function,
};

struct Token {
  TokenType type = TokenType::Number;
  std::string text;
  long double number = 0.0L;
};

bool is_identifier_start(char c) {
  return std::isalpha(static_cast<unsigned char>(c)) != 0 || c == '_';
}

bool is_identifier_char(char c) {
  return std::isalnum(static_cast<unsigned char>(c)) != 0 || c == '_';
}

std::vector<Token> tokenize(std::string_view expr) {
  std::vector<Token> tokens;
  std::size_t i = 0;

  while (i < expr.size()) {
    const char c = expr[i];
    if (std::isspace(static_cast<unsigned char>(c)) != 0) {
      ++i;
      continue;
    }

    if (std::isdigit(static_cast<unsigned char>(c)) != 0 || c == '.') {
      std::size_t begin = i;
      bool saw_exp = false;
      ++i;
      while (i < expr.size()) {
        const char d = expr[i];
        if (std::isdigit(static_cast<unsigned char>(d)) != 0 || d == '.') {
          ++i;
          continue;
        }
        if ((d == 'e' || d == 'E') && !saw_exp) {
          saw_exp = true;
          ++i;
          if (i < expr.size() && (expr[i] == '+' || expr[i] == '-')) {
            ++i;
          }
          continue;
        }
        break;
      }

      const std::string number_text(expr.substr(begin, i - begin));
      long double value = 0.0L;
#if defined(__cpp_lib_to_chars) && __cpp_lib_to_chars >= 201611L
      const auto result = std::from_chars(number_text.data(), number_text.data() + number_text.size(), value);
      if (result.ec != std::errc{}) {
        throw std::invalid_argument("Invalid number literal: " + number_text);
      }
#else
      try {
        value = std::stold(number_text);
      } catch (...) {
        throw std::invalid_argument("Invalid number literal: " + number_text);
      }
#endif
      tokens.push_back(Token{TokenType::Number, number_text, value});
      continue;
    }

    if (is_identifier_start(c)) {
      const std::size_t begin = i;
      ++i;
      while (i < expr.size() && is_identifier_char(expr[i])) {
        ++i;
      }
      tokens.push_back(Token{TokenType::Identifier, std::string(expr.substr(begin, i - begin)), 0.0L});
      continue;
    }

    if (c == '(') {
      tokens.push_back(Token{TokenType::LeftParen, "(", 0.0L});
      ++i;
      continue;
    }
    if (c == ')') {
      tokens.push_back(Token{TokenType::RightParen, ")", 0.0L});
      ++i;
      continue;
    }
    if (c == ',') {
      tokens.push_back(Token{TokenType::Comma, ",", 0.0L});
      ++i;
      continue;
    }

    if (c == '+' || c == '-' || c == '*' || c == '/' || c == '^') {
      tokens.push_back(Token{TokenType::Operator, std::string(1, c), 0.0L});
      ++i;
      continue;
    }

    throw std::invalid_argument(std::string("Unsupported token: ") + c);
  }

  return tokens;
}

int precedence(std::string_view op) {
  if (op == "neg") {
    return 5;
  }
  if (op == "^") {
    return 4;
  }
  if (op == "*" || op == "/") {
    return 3;
  }
  if (op == "+" || op == "-") {
    return 2;
  }
  return 0;
}

bool right_associative(std::string_view op) {
  return op == "^" || op == "neg";
}

bool is_function_name(std::string_view name) {
  static const std::unordered_map<std::string, int> funcs = {
      {"sin", 1}, {"cos", 1}, {"tan", 1}, {"exp", 1}, {"log", 1}, {"sqrt", 1}, {"abs", 1}, {"tanh", 1},
      {"min", 2}, {"max", 2}, {"pow", 2},
  };
  return funcs.find(std::string(name)) != funcs.end();
}

int function_arity(std::string_view name) {
  static const std::unordered_map<std::string, int> funcs = {
      {"sin", 1}, {"cos", 1}, {"tan", 1}, {"exp", 1}, {"log", 1}, {"sqrt", 1}, {"abs", 1}, {"tanh", 1},
      {"min", 2}, {"max", 2}, {"pow", 2},
  };
  const auto it = funcs.find(std::string(name));
  if (it == funcs.end()) {
    throw std::invalid_argument("Unknown function: " + std::string(name));
  }
  return it->second;
}

std::vector<Token> to_rpn(const std::vector<Token>& input) {
  std::vector<Token> output;
  std::vector<Token> stack;
  stack.reserve(input.size());

  TokenType prev = TokenType::Comma;

  for (std::size_t i = 0; i < input.size(); ++i) {
    Token token = input[i];

    if (token.type == TokenType::Identifier && i + 1 < input.size() && input[i + 1].type == TokenType::LeftParen &&
        is_function_name(token.text)) {
      token.type = TokenType::Function;
    }

    switch (token.type) {
      case TokenType::Number:
      case TokenType::Identifier:
        output.push_back(token);
        prev = token.type;
        break;
      case TokenType::Function:
        stack.push_back(token);
        prev = token.type;
        break;
      case TokenType::Comma:
        while (!stack.empty() && stack.back().type != TokenType::LeftParen) {
          output.push_back(stack.back());
          stack.pop_back();
        }
        if (stack.empty()) {
          throw std::invalid_argument("Comma outside function argument list");
        }
        prev = token.type;
        break;
      case TokenType::Operator: {
        if (token.text == "-" &&
            (prev == TokenType::Operator || prev == TokenType::LeftParen || prev == TokenType::Comma || i == 0 ||
             prev == TokenType::Function)) {
          token.text = "neg";
        }

        while (!stack.empty() && stack.back().type == TokenType::Operator) {
          const Token& top = stack.back();
          const bool should_pop = right_associative(token.text)
                                      ? (precedence(token.text) < precedence(top.text))
                                      : (precedence(token.text) <= precedence(top.text));
          if (!should_pop) {
            break;
          }
          output.push_back(top);
          stack.pop_back();
        }

        stack.push_back(token);
        prev = TokenType::Operator;
        break;
      }
      case TokenType::LeftParen:
        stack.push_back(token);
        prev = token.type;
        break;
      case TokenType::RightParen:
        while (!stack.empty() && stack.back().type != TokenType::LeftParen) {
          output.push_back(stack.back());
          stack.pop_back();
        }
        if (stack.empty()) {
          throw std::invalid_argument("Mismatched parenthesis");
        }
        stack.pop_back();
        if (!stack.empty() && stack.back().type == TokenType::Function) {
          output.push_back(stack.back());
          stack.pop_back();
        }
        prev = token.type;
        break;
    }
  }

  while (!stack.empty()) {
    if (stack.back().type == TokenType::LeftParen || stack.back().type == TokenType::RightParen) {
      throw std::invalid_argument("Mismatched parenthesis");
    }
    output.push_back(stack.back());
    stack.pop_back();
  }

  return output;
}

long double parse_indexed_variable(std::string_view name, std::string_view prefix, const std::vector<long double>& values) {
  if (!name.starts_with(prefix)) {
    return std::numeric_limits<long double>::quiet_NaN();
  }

  std::string index_text(name.substr(prefix.size()));
  if (index_text.empty()) {
    return std::numeric_limits<long double>::quiet_NaN();
  }

  std::size_t index = 0;
  try {
    index = static_cast<std::size_t>(std::stoul(index_text));
  } catch (...) {
    return std::numeric_limits<long double>::quiet_NaN();
  }

  if (index >= values.size()) {
    throw std::out_of_range("Variable index out of range: " + std::string(name));
  }
  return values[index];
}

long double eval_func1(std::string_view name, long double value) {
  if (name == "sin") {
    return std::sin(value);
  }
  if (name == "cos") {
    return std::cos(value);
  }
  if (name == "tan") {
    return std::tan(value);
  }
  if (name == "exp") {
    return std::exp(value);
  }
  if (name == "log") {
    return std::log(value);
  }
  if (name == "sqrt") {
    return std::sqrt(value);
  }
  if (name == "abs") {
    return std::fabs(value);
  }
  if (name == "tanh") {
    return std::tanh(value);
  }
  throw std::invalid_argument("Unsupported unary function: " + std::string(name));
}

long double eval_func2(std::string_view name, long double lhs, long double rhs) {
  if (name == "min") {
    return std::min(lhs, rhs);
  }
  if (name == "max") {
    return std::max(lhs, rhs);
  }
  if (name == "pow") {
    return std::pow(lhs, rhs);
  }
  throw std::invalid_argument("Unsupported binary function: " + std::string(name));
}

}  // namespace

CompiledExpression CompiledExpression::compile(std::string_view expression) {
  CompiledExpression compiled;
  const auto tokens = tokenize(expression);
  const auto rpn = to_rpn(tokens);

  std::ostringstream canonical;
  bool first = true;
  std::vector<Instruction> program;
  program.reserve(rpn.size());

  std::size_t stack_depth = 0;
  for (const auto& token : rpn) {
    if (!first) {
      canonical << ' ';
    }
    canonical << token.text;
    first = false;

    Instruction instruction;
    if (token.type == TokenType::Number) {
      instruction.op = OpCode::PushConstant;
      instruction.constant_value = token.number;
      ++stack_depth;
    } else if (token.type == TokenType::Identifier) {
      instruction.op = OpCode::PushVariable;
      instruction.symbol = token.text;
      ++stack_depth;
    } else if (token.type == TokenType::Operator) {
      if (token.text == "neg") {
        if (stack_depth < 1) {
          throw std::invalid_argument("Invalid unary expression stack state");
        }
        instruction.op = OpCode::Neg;
      } else {
        if (stack_depth < 2) {
          throw std::invalid_argument("Invalid binary expression stack state");
        }
        if (token.text == "+") {
          instruction.op = OpCode::Add;
        } else if (token.text == "-") {
          instruction.op = OpCode::Sub;
        } else if (token.text == "*") {
          instruction.op = OpCode::Mul;
        } else if (token.text == "/") {
          instruction.op = OpCode::Div;
        } else if (token.text == "^") {
          instruction.op = OpCode::Pow;
        } else {
          throw std::invalid_argument("Unsupported operator: " + token.text);
        }
        --stack_depth;
      }
    } else if (token.type == TokenType::Function) {
      instruction.symbol = token.text;
      const int arity = function_arity(token.text);
      instruction.operand_index = static_cast<std::size_t>(arity);
      instruction.op = (arity == 1) ? OpCode::Func1 : OpCode::Func2;
      if (stack_depth < static_cast<std::size_t>(arity)) {
        throw std::invalid_argument("Function arity stack underflow: " + token.text);
      }
      stack_depth -= static_cast<std::size_t>(arity - 1);
    } else {
      throw std::invalid_argument("Unexpected token in RPN program");
    }

    program.push_back(std::move(instruction));
  }

  if (stack_depth != 1) {
    throw std::invalid_argument("Expression compilation did not resolve to a single result");
  }

  compiled.canonical_ = canonical.str();
  compiled.bytecode_ = std::move(program);
  return compiled;
}

long double CompiledExpression::resolve_variable(std::string_view name, const EvaluationContext& context) {
  if (name == "t") {
    return context.t;
  }

  if (const long double x_u = parse_indexed_variable(name, "x_", context.x); !std::isnan(x_u)) {
    return x_u;
  }
  if (const long double x_plain = parse_indexed_variable(name, "x", context.x); !std::isnan(x_plain)) {
    return x_plain;
  }
  if (const long double u_u = parse_indexed_variable(name, "u_", context.u); !std::isnan(u_u)) {
    return u_u;
  }
  if (const long double u_plain = parse_indexed_variable(name, "u", context.u); !std::isnan(u_plain)) {
    return u_plain;
  }

  if (const auto it = context.derivatives.find(std::string(name)); it != context.derivatives.end()) {
    return it->second;
  }
  if (const auto it = context.extra.find(std::string(name)); it != context.extra.end()) {
    return it->second;
  }

  throw std::invalid_argument("Unbound variable in expression: " + std::string(name));
}

long double CompiledExpression::evaluate_long_double(const EvaluationContext& context) const {
  if (bytecode_.empty()) {
    return 0.0L;
  }

  std::vector<long double> stack;
  stack.reserve(bytecode_.size());

  for (const auto& instruction : bytecode_) {
    switch (instruction.op) {
      case OpCode::PushConstant:
        stack.push_back(instruction.constant_value);
        break;
      case OpCode::PushVariable:
        stack.push_back(resolve_variable(instruction.symbol, context));
        break;
      case OpCode::Add: {
        const long double rhs = stack.back();
        stack.pop_back();
        const long double lhs = stack.back();
        stack.pop_back();
        stack.push_back(lhs + rhs);
        break;
      }
      case OpCode::Sub: {
        const long double rhs = stack.back();
        stack.pop_back();
        const long double lhs = stack.back();
        stack.pop_back();
        stack.push_back(lhs - rhs);
        break;
      }
      case OpCode::Mul: {
        const long double rhs = stack.back();
        stack.pop_back();
        const long double lhs = stack.back();
        stack.pop_back();
        stack.push_back(lhs * rhs);
        break;
      }
      case OpCode::Div: {
        const long double rhs = stack.back();
        stack.pop_back();
        const long double lhs = stack.back();
        stack.pop_back();
        stack.push_back(lhs / rhs);
        break;
      }
      case OpCode::Pow: {
        const long double rhs = stack.back();
        stack.pop_back();
        const long double lhs = stack.back();
        stack.pop_back();
        stack.push_back(std::pow(lhs, rhs));
        break;
      }
      case OpCode::Neg: {
        const long double rhs = stack.back();
        stack.pop_back();
        stack.push_back(-rhs);
        break;
      }
      case OpCode::Func1: {
        const long double rhs = stack.back();
        stack.pop_back();
        stack.push_back(eval_func1(instruction.symbol, rhs));
        break;
      }
      case OpCode::Func2: {
        const long double rhs = stack.back();
        stack.pop_back();
        const long double lhs = stack.back();
        stack.pop_back();
        stack.push_back(eval_func2(instruction.symbol, lhs, rhs));
        break;
      }
    }
  }

  if (stack.size() != 1) {
    throw std::runtime_error("Expression evaluation produced invalid stack state");
  }
  return stack.back();
}

double CompiledExpression::evaluate_double(const EvaluationContext& context) const {
  return static_cast<double>(evaluate_long_double(context));
}

const std::string& CompiledExpression::canonical_form() const {
  return canonical_;
}

const std::vector<Instruction>& CompiledExpression::bytecode() const {
  return bytecode_;
}

}  // namespace wavefront
