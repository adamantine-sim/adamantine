/* SPDX-FileCopyrightText: Copyright (c) 2025, the adamantine authors.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include <Boundary.hh>
#include <utils.hh>

#include <algorithm>

namespace adamantine
{
Boundary::Boundary(boost::property_tree::ptree const &database,
                   std::vector<dealii::types::boundary_id> const &boundary_ids,
                   bool const mechanical_only)
{
  _types.resize(*std::max_element(boundary_ids.begin(), boundary_ids.end()) + 1,
                BoundaryType::invalid);
  // Set the default boundary condition. It can be overwritten for given
  // boundary ids. It is the boundary used for the current layer. If we solve a
  // mechanical problem only, it is not necessary to set a global boundary type.
  // It is traction-free by default.
  if (!mechanical_only)
  {
    // PropertyTreeInput boundary.type
    std::string boundary_type_str = database.get<std::string>("type");
    BoundaryType default_boundary = parse_boundary_line(boundary_type_str);
    for (auto &boundary : _types)
    {
      boundary = default_boundary;
    }
  }
  // Check if the user wants to use different boundaries for different part
  // of the domain's boundary.
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

  // Lambda function to parse the string
  auto parse_boundary_type =
      [](std::string const &boundary, BoundaryType &boundary_type)
  {
    // Parse thermal bc
    if (boundary == "adiabatic")
    {
      boundary_type = BoundaryType::adiabatic;
    }
    else
    {
      if (boundary == "radiative")
      {
        boundary_type |= BoundaryType::radiative;
      }
      else if (boundary == "convective")
      {
        boundary_type |= BoundaryType::convective;
      }
      else
      {
        ASSERT_THROW(boundary == "clamped", "Unknown boundary type.");
      }
    }

    // Parse mechanical bc
    if (boundary == "clamped")
    {
      boundary_type |= BoundaryType::clamped;
    }
  };

  size_t pos_str = 0;
  while ((pos_str = boundary_type_str.find(delimiter)) != std::string::npos)
  {
    std::string boundary = boundary_type_str.substr(0, pos_str);
    parse_boundary_type(boundary, boundary_type);
    boundary_type_str.erase(0, pos_str + delimiter.length());
  }
  parse_boundary_type(boundary_type_str, boundary_type);

  return boundary_type;
}
} // namespace adamantine
