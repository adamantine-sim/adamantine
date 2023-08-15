/* Copyright (c) 2021, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

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
               "Error: both thermal and mechanical physics are disabled");

  // Tree: boundary
  // Read the boundary condition type and check for disallowed combinations
  // PropertyTreeInput boundary.type
  BoundaryType boundary_type = BoundaryType::invalid;
  std::string boundary_type_str = database.get<std::string>("boundary.type");
  // Parse the string
  size_t pos_str = 0;
  std::string boundary;
  std::string delimiter = ",";
  auto parse_boundary_type = [&](std::string const &boundary)
  {
    if (boundary == "adiabatic")
    {
      ASSERT_THROW(
          boundary_type == BoundaryType::invalid,
          "Error: Adiabatic condition cannot be combined with another type.");
      boundary_type = BoundaryType::adiabatic;
    }
    else
    {
      ASSERT_THROW(
          boundary_type != BoundaryType::adiabatic,
          "Error: Adiabatic condition cannot be combined with another type.");

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
        ASSERT_THROW(false, "Error: Unknown boundary type.");
      }
    }
  };
  while ((pos_str = boundary_type_str.find(delimiter)) != std::string::npos)
  {
    boundary = boundary_type_str.substr(0, pos_str);
    parse_boundary_type(boundary);
    boundary_type_str.erase(0, pos_str + delimiter.length());
  }
  parse_boundary_type(boundary_type_str);

  // Tree: discretization.thermal
  if (use_thermal_physics)
  {
    // PropertyTreeInput discretization.thermal.fe_degree
    unsigned int const fe_degree =
        database.get<unsigned int>("discretization.thermal.fe_degree");
    ASSERT_THROW(fe_degree > 0 && fe_degree < 11,
                 "Error: fe_degree should be between 1 and 10.");

    // PropertyTreeInput discretization.thermal.quadrature
    boost::optional<std::string> quadrature_type_optional =
        database.get_optional<std::string>("discretization.thermal.quadrature");

    if (quadrature_type_optional)
    {
      std::string quadrature_type = quadrature_type_optional.get();
      if (!((boost::iequals(quadrature_type, "gauss") ||
             (boost::iequals(quadrature_type, "lobatto")))))
      {
        ASSERT_THROW(false, "Error: Unknown quadrature type.");
      }
    }
  }

  // Tree: geometry
  unsigned int dim = database.get<unsigned int>("geometry.dim");
  ASSERT_THROW((dim == 2) || (dim == 3), "Error: dim should be 2 or 3");

  boost::optional<double> material_height_optional =
      database.get_optional<double>("geometry.material_height");

  if (material_height_optional)
  {
    double material_height = material_height_optional.get();
    ASSERT_THROW(material_height >= 0.0,
                 "Error: Material height must be non-negative.");
  }

  bool use_powder = database.get("geometry.use_powder", false);

  if (use_powder)
  {
    double powder_layer = database.get<double>("geometry.powder_layer");
    ASSERT_THROW(powder_layer >= 0.0,
                 "Error: Powder layer must be non-negative.");
  }

  bool material_deposition =
      database.get("geometry.material_deposition", false);

  if (material_deposition)
  {
    std::string method =
        database.get<std::string>("geometry.material_deposition_method");

    ASSERT_THROW((method == "file" || method == "scan_paths"),
                 "Error: Method type for material deposition, '" + method +
                     "', is not recognized. Valid options are: 'file' "
                     "and 'scan_paths'");

    if (method == "file")
    {
      ASSERT_THROW(database.count("geometry.material_deposition_file") != 0,
                   "Error: If the material deposition method is 'file', "
                   "'material_deposition_file' must be given.");
    }
    else
    {
      ASSERT_THROW(database.get_child("geometry").count("deposition_length") !=
                       0,
                   "Error: If the material deposition method is 'scan_path', "
                   "'deposition_length' must be given.");
      ASSERT_THROW(database.get_child("geometry").count("deposition_height") !=
                       0,
                   "Error: If the material deposition method is 'scan_path', "
                   "'deposition_height' must be given.");
      if (dim == 3)
      {
        ASSERT_THROW(database.get_child("geometry").count("deposition_width") !=
                         0,
                     "Error: If the material deposition method is 'scan_path', "
                     "'deposition_width' must be given.");
      }
      ASSERT_THROW(
          database.get_child("geometry").count("deposition_lead_time") != 0,
          "Error: If the material deposition method is 'scan_path', "
          "'deposition_lead_time' must be given.");
    }
  }

  bool import_mesh = database.get<bool>("geometry.import_mesh");
  if (import_mesh)
  {
    ASSERT_THROW(database.get_child("geometry").count("mesh_file") != 0,
                 "Error: If the the mesh is imported, "
                 "'mesh_file' must be given.");
    ASSERT_THROW(database.get_child("geometry").count("mesh_format") != 0,
                 "Error: If the the mesh is imported, "
                 "'mesh_format' must be given.");
  }
  else
  {
    ASSERT_THROW(database.get_child("geometry").count("length") != 0,
                 "Error: If the the mesh is not imported, "
                 "'length' must be given.");
    ASSERT_THROW(database.get_child("geometry").count("height") != 0,
                 "Error: If the the mesh is not imported, "
                 "'height' must be given.");
    if (dim == 3)
    {
      ASSERT_THROW(database.get_child("geometry").count("width") != 0,
                   "Error: If the the mesh is not imported, "
                   "'width' must be given.");
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
    ASSERT_THROW(
        database.get_child("materials")
                .count("material_" + std::to_string(id)) != 0,
        "Error: Number of material subtrees does not match the set number of "
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
                     "Error: Each state needs a user-specified density.");
        ASSERT_THROW(database.get_child("materials")
                             .get_child("material_" + std::to_string(id))
                             .get_child(material_state_names.at(state_index))
                             .count("specific_heat") != 0,
                     "Error: Each state needs a user-specified specific heat.");
        ASSERT_THROW(
            database.get_child("materials")
                    .get_child("material_" + std::to_string(id))
                    .get_child(material_state_names.at(state_index))
                    .count("thermal_conductivity_x") != 0,
            "Error: Each state needs a user-specified specific thermal "
            "conductivity x.");
        ASSERT_THROW(
            database.get_child("materials")
                    .get_child("material_" + std::to_string(id))
                    .get_child(material_state_names.at(state_index))
                    .count("thermal_conductivity_z") != 0,
            "Error: Each state needs a user-specified specific thermal "
            "conductivity z.");

        if (dim == 3)
        {
          ASSERT_THROW(
              database.get_child("materials")
                      .get_child("material_" + std::to_string(id))
                      .get_child(material_state_names.at(state_index))
                      .count("thermal_conductivity_y") != 0,
              "Error: Each state needs a user-specified specific thermal "
              "conductivity y.");
        }

        if (boundary_type & BoundaryType::convective)
        {
          ASSERT_THROW(
              database.get_child("materials")
                      .get_child("material_" + std::to_string(id))
                      .get_child(material_state_names.at(state_index))
                      .count("convection_heat_transfer_coef") != 0,
              "Error: Convective BCs require a user-specified convection "
              "heat transfer coefficient.");
        }

        if (boundary_type & BoundaryType::radiative)
        {
          ASSERT_THROW(
              database.get_child("materials")
                      .get_child("material_" + std::to_string(id))
                      .get_child(material_state_names.at(state_index))
                      .count("emissivity") != 0,
              "Error: Radiative BCs require a user-specified emissivity.");
        }

        // For now I'm leaving the error checking for the polynomials and tables
        // for individual properties to the MaterialProperty class. I don't
        // think it makes sense to duplicate that logic here.
      }
    }

    ASSERT_THROW(
        has_a_valid_state == true,
        "Error: Material without any valid state (solid, powder, or liquid).");

    if (boundary_type & BoundaryType::convective)
    {
      ASSERT_THROW(database.get_child("materials")
                           .get_child("material_" + std::to_string(id))
                           .count("convection_temperature_infty") != 0,
                   "Error: Convective BCs require setting "
                   "'convection_temperature_infty'.");
    }

    if (boundary_type & BoundaryType::radiative)
    {
      ASSERT_THROW(database.get_child("materials")
                           .get_child("material_" + std::to_string(id))
                           .count("radiation_temperature_infty") != 0,
                   "Error: Convective BCs require setting "
                   "'radiation_temperature_infty'.");
    }
  }

  // Tree: memory_space
  boost::optional<std::string> memory_space_optional =
      database.get_optional<std::string>("memory_space");
  if (memory_space_optional)
  {
    std::string memory_space = memory_space_optional.get();
    ASSERT_THROW((memory_space == "device" || memory_space == "host"),
                 "Error: Method type for memory space, '" + memory_space +
                     "', is not recognized. Valid options are: 'host' "
                     "and 'device'");
  }

  // Tree: post_processor
  ASSERT_THROW(
      database.get_child("post_processor").count("filename_prefix") != 0,
      "Error: The filename prefix for the postprocessor must be specified.");

  // Tree: refinement
  ASSERT_THROW(database.count("refinement") != 0,
               "Error: A refinement section of the input file must exist.");

  boost::optional<double> beam_cutoff_optional =
      database.get_optional<double>("beam_cutoff");
  if (beam_cutoff_optional)
  {
    ASSERT_THROW(beam_cutoff_optional.get() >= 0.0,
                 "Error: The refinement beam cutoff must be non-negative.");
  }

  // Tree: sources
  unsigned int n_beams = database.get<unsigned int>("sources.n_beams");
  for (unsigned int beam_index = 0; beam_index < n_beams; ++beam_index)
  {
    std::string beam_type = database.get<std::string>(
        "sources.beam_" + std::to_string(beam_index) + ".type");
    ASSERT_THROW(boost::iequals(beam_type, "goldak") ||
                     boost::iequals(beam_type, "electron_beam") ||
                     boost::iequals(beam_type, "cube"),
                 "Error: Beam type, '" + beam_type +
                     "', is not recognized. Valid options are: 'goldak', "
                     "'electron_beam', and 'cube'.");
    ASSERT_THROW(database.get_child("sources")
                         .get_child("beam_" + std::to_string(beam_index))
                         .count("scan_path_file") != 0,
                 "Error: A scan path file for each beam must be given.");

    std::string file_format =
        database.get<std::string>("sources.beam_" + std::to_string(beam_index) +
                                  ".scan_path_file_format");
    ASSERT_THROW(boost::iequals(file_format, "segment") ||
                     boost::iequals(file_format, "event_series"),
                 "Error: Scan path file format, '" + file_format +
                     "', is not recognized. Valid options are: 'segment' and "
                     "'event_series'.");
    ASSERT_THROW(database.get<double>("sources.beam_" +
                                      std::to_string(beam_index) + ".depth") >=
                     0.0,
                 "Error: Heat source depth must be non-negative.");

    double absorption_efficiency =
        database.get<double>("sources.beam_" + std::to_string(beam_index) +
                             ".absorption_efficiency");
    ASSERT_THROW(
        absorption_efficiency >= 0.0 && absorption_efficiency <= 1.0,
        "Error: Heat source absorption efficiency must be between 0 and 1.");
  }

  // Tree: time_stepping
  std::string time_stepping_method =
      database.get<std::string>("time_stepping.method");
  ASSERT_THROW(
      boost::iequals(time_stepping_method, "forward_euler") ||
          boost::iequals(time_stepping_method, "rk_third_order") ||
          boost::iequals(time_stepping_method, "rk_fourth_order") ||
          boost::iequals(time_stepping_method, "heun_euler") ||
          boost::iequals(time_stepping_method, "bogacki_shampine") ||
          boost::iequals(time_stepping_method, "dopri") ||
          boost::iequals(time_stepping_method, "fehlberg") ||
          boost::iequals(time_stepping_method, "cash_karp") ||
          boost::iequals(time_stepping_method, "backward_euler") ||
          boost::iequals(time_stepping_method, "implicit_midpoint") ||
          boost::iequals(time_stepping_method, "crank_nicolson") ||
          boost::iequals(time_stepping_method, "sdirk2"),
      "Error: Time stepping method, '" + time_stepping_method +
          "', is not recognized. Valid options are: 'forward_euler', "
          "'rk_third_order', 'rk_fourth_order', 'heun_euler', "
          "'bogacki_shampine', 'dopri', 'fehlberg', 'cash_karp', "
          "'backward_euler', 'implicit_midpoint', 'crank_nicolson', and "
          "'sdirk2'.");

  ASSERT_THROW(database.get<double>("time_stepping.duration") >= 0.0,
               "Error: Time stepping duration must be non-negative.");

  ASSERT_THROW(database.get<double>("time_stepping.time_step") >= 0.0,
               "Error: Time step must be non-negative.");

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
      ASSERT_THROW(
          database.get_child("experiment").count("file") != 0,
          "Error: If reading experimental data, a file must be given.");

      ASSERT_THROW(database.get_child("experiment").count("last_frame") != 0,
                   "Error: If reading experimental data, a last frame index "
                   "must be given.");

      std::string experiment_format =
          database.get<std::string>("experiment.format");
      ASSERT_THROW(boost::iequals(experiment_format, "point_cloud") ||
                       boost::iequals(experiment_format, "ray"),
                   "Error: Experiment format must be 'point_cloud' or 'ray'.");

      unsigned int first_frame_index =
          database.get<unsigned int>("experiment.first_frame", 0);
      unsigned int last_frame_index =
          database.get<unsigned int>("experiment.last_frame");
      ASSERT_THROW(
          last_frame_index >= first_frame_index,
          "Error: When reading experimental data, the last frame index "
          "cannot be lower than the first frame index.");

      unsigned int first_camera_id =
          database.get<unsigned int>("experiment.first_camera_id");
      unsigned int last_camera_id =
          database.get<unsigned int>("experiment.last_camera_id");
      ASSERT_THROW(last_camera_id >= first_camera_id,
                   "Error: When reading experimental data, the last camera id "
                   "cannot be lower than the first camera id.");

      std::string data_columns =
          database.get<std::string>("experiment.data_columns");
      ASSERT_THROW(std::count(data_columns.begin(), data_columns.end(), ',') ==
                       dim,
                   "Error: The experimental data column indices in the input "
                   "file do not have the correct number of entries.");

      ASSERT_THROW(database.get_child("experiment").count("log_filename") != 0,
                   "Error: If reading experimental data, a log filename must "
                   "be given.");
    }
  }

  // Tree: ensemble
  boost::optional<double> initial_temperature_stddev =
      database.get_optional<double>("ensemble.initial_temperature_stddev");
  if (initial_temperature_stddev)
  {
    ASSERT_THROW(initial_temperature_stddev.get() >= 0.0,
                 "Error: The standard deviation for the initial temperature "
                 "must be non-negative.");
  }
  boost::optional<double> new_material_temperature_stddev =
      database.get_optional<double>("ensemble.new_material_temperature_stddev");
  if (new_material_temperature_stddev)
  {
    ASSERT_THROW(
        new_material_temperature_stddev.get() >= 0.0,
        "Error: The standard deviation for the new material temperature "
        "must be non-negative.");
  }
  boost::optional<double> beam_0_max_power_stddev =
      database.get_optional<double>("ensembe.beam_0_max_power_stddev");
  if (beam_0_max_power_stddev)
  {
    ASSERT_THROW(beam_0_max_power_stddev.get() >= 0.0,
                 "Error: The standard deviation for the beam 0 max power "
                 "must be non-negative.");
  }

  // Tree: data_assimilation
  boost::optional<double> convergence_tolerance =
      database.get_optional<double>("data_assimilation.convergence_tolerance");
  if (convergence_tolerance)
  {
    ASSERT_THROW(convergence_tolerance.get() >= 0.0,
                 "Error: The data assimilation convergene tolerance must be "
                 "non-negative.");
  }

  std::string localization_cutoff_function_str =
      database.get("data_assimilation.localization_cutoff_function", "none");

  if (!(boost::iequals(localization_cutoff_function_str, "gaspari_cohn") ||
        boost::iequals(localization_cutoff_function_str, "step_function") ||
        boost::iequals(localization_cutoff_function_str, "none")))
  {
    ASSERT_THROW(false,
                 "Error: Unknown localization cutoff function. Valid options "
                 "are 'gaspari_cohn', 'step_function', and 'none'.");
  }
}
} // namespace adamantine
