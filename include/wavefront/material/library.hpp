#pragma once

#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "wavefront/api/solver.hpp"

namespace wavefront {

struct MaterialPreset {
  std::string name;
  MediumLaw medium;
};

inline std::vector<MaterialPreset> builtin_materials() {
  return {
      {"air", {{"1.225"}, {"0.014"}, {"0.0001"}, {"0.0"}}},
      {"water", {{"1000.0"}, {"2.25e9"}, {"0.001"}, {"0.0"}}},
      {"glass", {{"2500.0"}, {"7.0e10"}, {"0.0005"}, {"0.0"}}},
      {"steel", {{"7850.0"}, {"2.0e11"}, {"0.0002"}, {"0.0"}}},
  };
}

inline MaterialPreset builtin_material(std::string_view name) {
  for (const auto& preset : builtin_materials()) {
    if (preset.name == name) {
      return preset;
    }
  }
  throw std::out_of_range("unknown builtin material");
}

}  // namespace wavefront
