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
      {"honey", {{"1420.0"}, {"3.2e9"}, {"0.15"}, {"0.02"}}},
      {"hyperhoney", {{"1550.0"}, {"3.8e9"}, {"0.22"}, {"0.08"}}},
      {"oobleck", {{"1650.0"}, {"4.6e9"}, {"0.35"}, {"0.05"}}},
      {"aerogel", {{"120.0"}, {"1.5e7"}, {"0.03"}, {"0.01"}}},
      {"ferrofluid", {{"1210.0"}, {"1.8e9"}, {"0.09"}, {"0.12"}}},
      {"plasma", {{"0.18"}, {"8.0e4"}, {"0.005"}, {"0.25"}}},
      {"metamaterial", {{"950.0"}, {"7.5e8"}, {"0.12"}, {"0.4"}}},
      {"neutron_star_crust", {{"4.0e17"}, {"1.0e29"}, {"1.0e-6"}, {"0.001"}}},
      {"strange_matter", {{"8.0e17"}, {"3.2e29"}, {"5.0e-7"}, {"0.02"}}},
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
