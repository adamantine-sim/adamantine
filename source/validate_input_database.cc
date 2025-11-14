/* SPDX-FileCopyrightText: Copyright (c) 2021 - 2025, the adamantine authors.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include <Boundary.hh>
#include <types.hh>
#include <utils.hh>
#include <validate_input_database.hh>

#include <boost/algorithm/string/predicate.hpp>

#include <algorithm>

namespace adamantine
{
void validate_input_database(boost::property_tree::ptree &database)
{
  // Tree: physics
  // PropertyTreeInput physics.thermal
  bool const use_thermal_physics = database.get<bool>("physics.thermal");
  // PropertyTreeInput physics.mechanical
  bool const use_mechanical_physics = database.get<bool>("physics.mechanical");
  ASSERT_THROW(use_thermal_physics || use_mechanical_physics,
               "Both thermal and mechanical physics are disabled");

  // Tree: boundary
  // Read the boundary condition type and check for disallowed combinations.
  // Store if we use radiative or convective boundary so we can check later that
  // the appropriate material properties are saved.
  BoundaryType boundary_type = BoundaryType::invalid;
  // Parse the string
  std::string delimiter = ",";
  auto parse_boundary_type = [&](std::string const &boundary,
                                 BoundaryType &line_boundary_type,
                                 bool &thermal_bc, bool &mechanical_bc)
  {
    if (boundary == "adiabatic")
    {
      ASSERT_THROW(!(line_boundary_type & BoundaryType::radiative) ||
                       !(line_boundary_type & BoundaryType::convective),
                   "adiabatic condition cannot be combined with another type "
                   "of thermal boundary condition.");
      line_boundary_type |= BoundaryType::adiabatic;
      thermal_bc = true;
    }
    else
    {
      if (boundary == "radiative")
      {
        ASSERT_THROW(!(line_boundary_type & BoundaryType::adiabatic),
                     "adiabatic condition cannot be combined with another type "
                     "of thermal boundary condition.");
        line_boundary_type |= BoundaryType::radiative;
        boundary_type |= BoundaryType::radiative;
        thermal_bc = true;
      }
      else if (boundary == "convective")
      {
        ASSERT_THROW(!(line_boundary_type & BoundaryType::adiabatic),
                     "adiabatic condition cannot be combined with another type "
                     "of thermal boundary condition.");
        line_boundary_type |= BoundaryType::convective;
        boundary_type |= BoundaryType::convective;
        thermal_bc = true;
      }
      else if (boundary == "clamped")
      {
        ASSERT_THROW(
            !(line_boundary_type & BoundaryType::traction_free),
            "Mechanical boundary conditions cannot be combined together.");
        line_boundary_type |= BoundaryType::clamped;
        mechanical_bc = true;
      }
      else if (boundary == "traction_free")
      {
        ASSERT_THROW(
            !(line_boundary_type & BoundaryType::clamped),
            "Mechanical boundary conditions cannot be combined together.");
        line_boundary_type |= BoundaryType::traction_free;
        mechanical_bc = true;
      }
      else
      {
        ASSERT_THROW(false, "Unknown boundary type.");
      }
    }
  };
  auto const boundary_database = database.get_child("boundary");
  for (auto const &child_pair : boundary_database)
  {
    size_t pos_str = 0;
    bool thermal_bc = false;
    bool mechanical_bc = false;
    BoundaryType line_boundary_type = BoundaryType::invalid;
    std::string boundary_type_str = "invalid";
    if (child_pair.first == "type")
    {
      boundary_type_str = child_pair.second.data();
    }
    else
    {
      boundary_type_str =
          boundary_database.get<std::string>(child_pair.first + ".type");
    }

    while ((pos_str = boundary_type_str.find(delimiter)) != std::string::npos)
    {
      std::string boundary = boundary_type_str.substr(0, pos_str);
      parse_boundary_type(boundary, line_boundary_type, thermal_bc,
                          mechanical_bc);
      boundary_type_str.erase(0, pos_str + delimiter.length());
    }
    parse_boundary_type(boundary_type_str, line_boundary_type, thermal_bc,
                        mechanical_bc);

    if (use_thermal_physics)
    {
      ASSERT_THROW(thermal_bc, "Missing thermal boundary condition.");
    }
    if (use_mechanical_physics)
    {
      ASSERT_THROW(mechanical_bc, "Missing mechanical boundary condition.");
    }
  }

  // Tree: discretization.thermal
  if (use_thermal_physics)
  {
    // PropertyTreeInput discretization.thermal.fe_degree
    unsigned int const fe_degree =
        database.get<unsigned int>("discretization.thermal.fe_degree");
    ASSERT_THROW(fe_degree > 0 && fe_degree < 6,
                 "fe_degree should be between 1 and 5.");

    // PropertyTreeInput discretization.thermal.quadrature
    boost::optional<std::string> quadrature_type_optional =
        database.get_optional<std::string>("discretization.thermal.quadrature");

    if (quadrature_type_optional)
    {
      std::string quadrature_type = quadrature_type_optional.get();
      if (!((boost::iequals(quadrature_type, "gauss") ||
             (boost::iequals(quadrature_type, "lobatto")))))
      {
        ASSERT_THROW(false, "Unknown quadrature type.");
      }
    }
  }

  // Tree: geometry
  unsigned int dim = database.get<unsigned int>("geometry.dim");
  ASSERT_THROW((dim == 2) || (dim == 3), "dim should be 2 or 3");

  bool use_powder = database.get("geometry.use_powder", false);

  if (use_powder)
  {
    double powder_layer = database.get<double>("geometry.powder_layer");
    ASSERT_THROW(powder_layer >= 0.0, "powder_layer must be non-negative.");
  }

  bool material_deposition =
      database.get("geometry.material_deposition", false);

  if (material_deposition)
  {
    std::string method =
        database.get<std::string>("geometry.material_deposition_method");

    ASSERT_THROW(
        (method == "file" || method == "scan_paths"),
        "Method type for material deposition, '" + method +
            "', is not recognized. Valid options are: 'file' and 'scan_paths'");

    if (method == "file")
    {
      ASSERT_THROW(database.count("geometry.material_deposition_file") != 0,
                   "If the material deposition method is 'file', "
                   "'material_deposition_file' must be given.");
    }
    else
    {
      ASSERT_THROW(database.get_child("geometry").count("deposition_length") !=
                       0,
                   "If the material deposition method is 'scan_path', "
                   "'deposition_length' must be given.");
      ASSERT_THROW(database.get_child("geometry").count("deposition_height") !=
                       0,
                   "If the material deposition method is 'scan_path', "
                   "'deposition_height' must be given.");
      if (dim == 3)
      {
        ASSERT_THROW(database.get_child("geometry").count("deposition_width") !=
                         0,
                     "If the material deposition method is 'scan_path', "
                     "'deposition_width' must be given.");
      }
      ASSERT_THROW(
          database.get_child("geometry").count("deposition_lead_time") != 0,
          "If the material deposition method is 'scan_path', "
          "'deposition_lead_time' must be given.");
    }
  }

  bool import_mesh = database.get<bool>("geometry.import_mesh");
  if (import_mesh)
  {
    ASSERT_THROW(database.get_child("geometry").count("mesh_file") != 0,
                 "If the the mesh is imported, 'mesh_file' must be given.");
    ASSERT_THROW(database.get_child("geometry").count("mesh_format") != 0,
                 "If the the mesh is imported, 'mesh_format' must be given.");
  }
  else
  {
    ASSERT_THROW(database.get_child("geometry").count("length") != 0,
                 "If the the mesh is not imported, 'length' must be given.");
    ASSERT_THROW(database.get_child("geometry").count("height") != 0,
                 "If the the mesh is not imported, 'height' must be given.");
    if (dim == 3)
    {
      ASSERT_THROW(database.get_child("geometry").count("width") != 0,
                   "If the the mesh is not imported, 'width' must be given.");
    }
  }

  // Tree: materials
  unsigned int n_materials =
      database.get<unsigned int>("materials.n_materials");

  std::string property_format =
      database.get<std::string>("materials.property_format");
  ASSERT_THROW((property_format == "table") ||
                   (property_format == "polynomial"),
               "property_format should be table or polynomial.");

  for (dealii::types::material_id id = 0; id < n_materials; ++id)
  {
    ASSERT_THROW(database.get_child("materials")
                         .count("material_" + std::to_string(id)) != 0,
                 "Number of material subtrees does not match the set number of "
                 "materials.");

    bool has_a_valid_state = false;
    for (unsigned int state_index = 0;
         state_index < material_state_names.size(); ++state_index)
    {
      if (database.get_child("materials")
              .get_child("material_" + std::to_string(id))
              .count(material_state_names.at(state_index)) != 0)
      {
        has_a_valid_state = true;

        ASSERT_THROW(database.get_child("materials")
                             .get_child("material_" + std::to_string(id))
                             .get_child(material_state_names.at(state_index))
                             .count("density") != 0,
                     "Each state needs a user-specified density.");
        ASSERT_THROW(database.get_child("materials")
                             .get_child("material_" + std::to_string(id))
                             .get_child(material_state_names.at(state_index))
                             .count("specific_heat") != 0,
                     "Each state needs a user-specified specific heat.");
        ASSERT_THROW(database.get_child("materials")
                             .get_child("material_" + std::to_string(id))
                             .get_child(material_state_names.at(state_index))
                             .count("thermal_conductivity_x") != 0,
                     "Each state needs a user-specified specific thermal "
                     "conductivity x.");
        ASSERT_THROW(database.get_child("materials")
                             .get_child("material_" + std::to_string(id))
                             .get_child(material_state_names.at(state_index))
                             .count("thermal_conductivity_z") != 0,
                     "Each state needs a user-specified specific thermal "
                     "conductivity z.");

        if (dim == 3)
        {
          ASSERT_THROW(database.get_child("materials")
                               .get_child("material_" + std::to_string(id))
                               .get_child(material_state_names.at(state_index))
                               .count("thermal_conductivity_y") != 0,
                       "Each state needs a user-specified specific thermal "
                       "conductivity y.");
        }

        if (boundary_type & BoundaryType::convective)
        {
          ASSERT_THROW(database.get_child("materials")
                               .get_child("material_" + std::to_string(id))
                               .get_child(material_state_names.at(state_index))
                               .count("convection_heat_transfer_coef") != 0,
                       "Convective BCs require a user-specified convection "
                       "heat transfer coefficient.");
        }

        if (boundary_type & BoundaryType::radiative)
        {
          ASSERT_THROW(database.get_child("materials")
                               .get_child("material_" + std::to_string(id))
                               .get_child(material_state_names.at(state_index))
                               .count("emissivity") != 0,
                       "Radiative BCs require a user-specified emissivity.");
        }

        // For now I'm leaving the error checking for the polynomials and tables
        // for individual properties to the MaterialProperty class. I don't
        // think it makes sense to duplicate that logic here.
      }
    }

    ASSERT_THROW(
        has_a_valid_state == true,
        "Material without any valid state (solid, powder, or liquid).");

    if (boundary_type & BoundaryType::convective)
    {
      ASSERT_THROW(
          database.get_child("materials")
                  .get_child("material_" + std::to_string(id))
                  .count("convection_temperature_infty") != 0,
          "Convective BCs require setting 'convection_temperature_infty'.");
    }

    if (boundary_type & BoundaryType::radiative)
    {
      ASSERT_THROW(
          database.get_child("materials")
                  .get_child("material_" + std::to_string(id))
                  .count("radiation_temperature_infty") != 0,
          "Radiative BCs require setting 'radiation_temperature_infty'.");
    }
  }

  // Tree: memory_space
  boost::optional<std::string> memory_space_optional =
      database.get_optional<std::string>("memory_space");
  if (memory_space_optional)
  {
    std::string memory_space = memory_space_optional.get();
    ASSERT_THROW(
        (memory_space == "device" || memory_space == "host"),
        "Method type for memory space, '" + memory_space +
            "', is not recognized. Valid options are: 'host' and 'device'");
  }

  // Tree: post_processor
  ASSERT_THROW(database.get_child("post_processor").count("filename_prefix") !=
                   0,
               "The filename prefix for the postprocessor must be specified.");

  // Tree: refinement
  ASSERT_THROW(database.count("refinement") != 0,
               "A refinement section of the input file must exist.");

  // Tree: sources
  unsigned int n_beams = database.get<unsigned int>("sources.n_beams");
  for (unsigned int beam_index = 0; beam_index < n_beams; ++beam_index)
  {
    std::string beam_type = database.get<std::string>(
        "sources.beam_" + std::to_string(beam_index) + ".type");
    ASSERT_THROW(boost::iequals(beam_type, "goldak") ||
                     boost::iequals(beam_type, "electron_beam") ||
                     boost::iequals(beam_type, "cube") ||
                     boost::iequals(beam_type, "gaussian"),
                 "Beam type, '" + beam_type +
                     "', is not recognized. Valid options are: 'goldak', "
                     "'electron_beam', 'cube', and gaussian.");
    ASSERT_THROW(database.get_child("sources")
                         .get_child("beam_" + std::to_string(beam_index))
                         .count("scan_path_file") != 0,
                 "A scan path file for each beam must be given.");

    std::string file_format =
        database.get<std::string>("sources.beam_" + std::to_string(beam_index) +
                                  ".scan_path_file_format");
    ASSERT_THROW(boost::iequals(file_format, "segment") ||
                     boost::iequals(file_format, "event_series"),
                 "Scan path file format, '" + file_format +
                     "', is not recognized. Valid options are: 'segment' and "
                     "'event_series'.");
    ASSERT_THROW(database.get<double>("sources.beam_" +
                                      std::to_string(beam_index) + ".depth") >=
                     0.0,
                 "Heat source depth must be non-negative.");

    double absorption_efficiency =
        database.get<double>("sources.beam_" + std::to_string(beam_index) +
                             ".absorption_efficiency");
    ASSERT_THROW(absorption_efficiency >= 0.0 && absorption_efficiency <= 1.0,
                 "Heat source absorption efficiency must be between 0 and 1.");
  }

  // Tree: time_stepping
  std::string time_stepping_method =
      database.get<std::string>("time_stepping.method");
  ASSERT_THROW(boost::iequals(time_stepping_method, "forward_euler") ||
                   boost::iequals(time_stepping_method, "rk_third_order") ||
                   boost::iequals(time_stepping_method, "rk_fourth_order"),
               "Time stepping method, '" + time_stepping_method +
                   "', is not recognized. Valid options are: 'forward_euler', "
                   "'rk_third_order', and 'rk_fourth_order'.");

  if (database.get("time.scan_path_for_duration", false))
  {
    ASSERT_THROW(database.get<double>("time_stepping.duration") >= 0.0,
                 "Time stepping duration must be non-negative.");
  }

  ASSERT_THROW(database.get<double>("time_stepping.time_step") >= 0.0,
               "Time step must be non-negative.");

  // Tree: experiment
  // I'm not checking for the existence of the experimental files here, that's
  // still done in `adamantine::read_experimental_data_point_cloud` and
  // `adamantine::RayTracing`.
  boost::optional<boost::property_tree::ptree &> experiment_optional_database =
      database.get_child_optional("experiment");
  if (experiment_optional_database)
  {
    bool experiment_active =
        database.get("experiment.read_in_experimental_data", false);
    if (experiment_active)
    {
      ASSERT_THROW(database.get_child("experiment").count("file") != 0,
                   "If reading experimental data, a file must be given.");

      ASSERT_THROW(database.get_child("experiment").count("last_frame") != 0,
                   "If reading experimental data, a last frame index "
                   "must be given.");

      std::string experiment_format =
          database.get<std::string>("experiment.format");
      ASSERT_THROW(boost::iequals(experiment_format, "point_cloud") ||
                       boost::iequals(experiment_format, "ray"),
                   "Experiment format must be 'point_cloud' or 'ray'.");

      unsigned int first_frame_index =
          database.get<unsigned int>("experiment.first_frame", 0);
      unsigned int last_frame_index =
          database.get<unsigned int>("experiment.last_frame");
      ASSERT_THROW(last_frame_index >= first_frame_index,
                   "When reading experimental data, the last frame index "
                   "cannot be lower than the first frame index.");

      unsigned int first_camera_id =
          database.get<unsigned int>("experiment.first_camera_id");
      unsigned int last_camera_id =
          database.get<unsigned int>("experiment.last_camera_id");
      ASSERT_THROW(last_camera_id >= first_camera_id,
                   "When reading experimental data, the last camera id cannot "
                   "be lower than the first camera id.");

      ASSERT_THROW(
          database.get_child("experiment").count("log_filename") != 0,
          "If reading experimental data, a log filename must be given.");
    }
  }

  // Tree: ensemble
  // We check the input in ensemble_management.cc

  // Tree: data_assimilation
  boost::optional<double> convergence_tolerance =
      database.get_optional<double>("data_assimilation.convergence_tolerance");
  if (convergence_tolerance)
  {
    ASSERT_THROW(
        convergence_tolerance.get() >= 0.0,
        "The data assimilation convergene tolerance must be non-negative.");
  }

  std::string localization_cutoff_function_str =
      database.get("data_assimilation.localization_cutoff_function", "none");

  if (!(boost::iequals(localization_cutoff_function_str, "gaspari_cohn") ||
        boost::iequals(localization_cutoff_function_str, "step_function") ||
        boost::iequals(localization_cutoff_function_str, "none")))
  {
    ASSERT_THROW(false, "Unknown localization cutoff function. Valid options "
                        "are 'gaspari_cohn', 'step_function', and 'none'.");
  }

  // Tree: units
  boost::optional<std::string> mesh_unit =
      database.get_optional<std::string>("units.mesh");
  if (mesh_unit && (!(boost::iequals(mesh_unit.get(), "millimeter") ||
                      boost::iequals(mesh_unit.get(), "centimeter") ||
                      boost::iequals(mesh_unit.get(), "inch") ||
                      boost::iequals(mesh_unit.get(), "meter"))))
  {
    ASSERT_THROW(false, "Unknown unit associated with the mesh. Valid options "
                        "are `millimeter`, `centimeter`, `inch`, and `meter`");
  }

  boost::optional<std::string> heat_source_power_unit =
      database.get_optional<std::string>("units.heat_source.power");
  if (heat_source_power_unit &&
      (!(boost::iequals(heat_source_power_unit.get(), "milliwatt") ||
         boost::iequals(heat_source_power_unit.get(), "watt"))))
  {
    ASSERT_THROW(false, "Unknown unit associated with the power of the heat "
                        "source. Valid options are `milliwatt`, and `watt`");
  }

  boost::optional<std::string> heat_source_velocity_unit =
      database.get_optional<std::string>("units.heat_source.velocity");
  if (heat_source_velocity_unit &&
      (!(boost::iequals(heat_source_velocity_unit.get(), "millimeter/second") ||
         boost::iequals(heat_source_velocity_unit.get(), "centimeter/second") ||
         boost::iequals(heat_source_velocity_unit.get(), "meter/second"))))
  {
    ASSERT_THROW(false, "Unknown unit associated with the velocity of the heat "
                        "source. Valid options are `millimeter/second`, "
                        "`centimeter/second`, and `meter/second`");
  }

  boost::optional<std::string> heat_source_dimension_unit =
      database.get_optional<std::string>("units.heat_source.dimension");
  if (heat_source_dimension_unit &&
      (!(boost::iequals(heat_source_dimension_unit.get(), "millimeter") ||
         boost::iequals(heat_source_dimension_unit.get(), "centimeter") ||
         boost::iequals(heat_source_dimension_unit.get(), "inch") ||
         boost::iequals(heat_source_dimension_unit.get(), "meter"))))
  {
    ASSERT_THROW(
        false,
        "Unknown unit associated with the dimension of the heat source. Valid "
        "options are `millimeter`, `centimeter`, `inch`, and `meter`");
  }

  boost::optional<std::string> heat_source_scan_path_unit =
      database.get_optional<std::string>("units.heat_source.scan_path");
  if (heat_source_scan_path_unit &&
      (!(boost::iequals(heat_source_scan_path_unit.get(), "millimeter") ||
         boost::iequals(heat_source_scan_path_unit.get(), "centimeter") ||
         boost::iequals(heat_source_scan_path_unit.get(), "inch") ||
         boost::iequals(heat_source_scan_path_unit.get(), "meter"))))
  {
    ASSERT_THROW(false,
                 "Unknown unit associated with the scan path. Valid options "
                 "are `millimeter`, `centimeter`, `inch`, and `meter`");
  }
}
} // namespace adamantine
