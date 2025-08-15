/* SPDX-FileCopyrightText: Copyright (c) 2025, the adamantine authors.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include <Boundary.hh>
#include <utils.hh>

#include <algorithm>

namespace adamantine
{
Boundary::Boundary(boost::property_tree::ptree const &database,
                   std::vector<dealii::types::boundary_id> const &boundary_ids)
{
  // Check that the boundary ids provided by the user match the ones from the
  // domain.
  for (auto const &child_pair : database)
  {
    std::string main_string = child_pair.first;
    std::string substring = "boundary_";
    size_t pos = main_string.find(substring);
    // Check if the substring was found
    if (pos != std::string::npos)
    {
      main_string.erase(pos, substring.length());
      auto it = std::find(boundary_ids.begin(), boundary_ids.end(),
                          std::stoi(main_string));
      ASSERT_THROW(it != boundary_ids.end(),
                   "Provided boundary id " + main_string + " is not valid.");
    }
  }

  _types.resize(*std::max_element(boundary_ids.begin(), boundary_ids.end()) + 1,
                BoundaryType::invalid);
  // Set the default boundary condition. It can be overwritten for given
  // boundary ids.
  // PropertyTreeInput boundary.type
  auto default_boundary_type_str = database.get_optional<std::string>("type");
  if (default_boundary_type_str)
  {
    BoundaryType default_boundary =
        parse_boundary_line(*default_boundary_type_str);
    for (auto &boundary : _types)
    {
      boundary = default_boundary;
    }
  }

  // Check if the user wants to use different boundary conditions for different
  // part of the domain's boundary.
  for (auto const id : boundary_ids)
  {
    // PropertyTreeInput boundary.boundary_id.type
    auto boundary_type_str = database.get_optional<std::string>(
        "boundary_" + std::to_string(id) + ".type");
    if (boundary_type_str)
    {
      _types[id] = parse_boundary_line(*boundary_type_str);
    }
  }

  // Check if the user wants to use different for the printed surface
  // PropertyTreeInput boundary.printed_surface.type
  auto surface_boundary_type_str =
      database.get_optional<std::string>("printed_surface.type");
  if (surface_boundary_type_str)
  {
    _types.back() = parse_boundary_line(*surface_boundary_type_str);
  }
}

std::vector<dealii::types::boundary_id>
Boundary::get_boundary_ids(BoundaryType type) const
{
  std::vector<dealii::types::boundary_id> ids;
  for (unsigned int i = 0; i < _types.size(); ++i)
  {
    if (_types[i] & type)
    {
      ids.push_back(i);
    }
  }

  return ids;
}

BoundaryType Boundary::parse_boundary_line(std::string boundary_type_str)
{
  BoundaryType boundary_type = BoundaryType::invalid;
  std::string delimiter = ",";

  std::unordered_map<std::string, BoundaryType> const bc_map{
      {"adiabatic", BoundaryType::adiabatic},
      {"radiative", BoundaryType::radiative},
      {"convective", BoundaryType::convective},
      {"clamped", BoundaryType::clamped},
      {"traction_free", BoundaryType::traction_free}};

  size_t pos_str = 0;
  while ((pos_str = boundary_type_str.find(delimiter)) != std::string::npos)
  {
    std::string boundary = boundary_type_str.substr(0, pos_str);
    boundary_type |= bc_map.find(boundary)->second;
    boundary_type_str.erase(0, pos_str + delimiter.length());
  }
  boundary_type |= bc_map.find(boundary_type_str)->second;

  return boundary_type;
}
} // namespace adamantine
