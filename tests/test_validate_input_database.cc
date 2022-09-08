/* Copyright (c) 2021, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#define BOOST_TEST_MODULE validate_input_database

#include <validate_input_database.hh>

#include "main.cc"

namespace adamantine
{

BOOST_AUTO_TEST_CASE(expected_passes)
{
  boost::property_tree::ptree database;

  // Bare-bones required inputs
  database.put("boundary.type", "adiabatic");
  database.put("discretization.thermal.fe_degree", "1");
  database.put("geometry.dim", "3");
  database.put("discretization.thermal.quadrature", "gauss");
  database.put("geometry.import_mesh", "false");
  database.put("geometry.length", 1.0);
  database.put("geometry.height", 1.0);
  database.put("geometry.width", 1.0);
  database.put("materials.n_materials", 1);
  database.put("materials.property_format", "polynomial");
  database.put("materials.material_0.solid.thermal_conductivity_x", 10.);
  database.put("materials.material_0.solid.thermal_conductivity_y", 10.);
  database.put("materials.material_0.solid.thermal_conductivity_z", 10.);
  database.put("materials.material_0.solid.density", 10.);
  database.put("materials.material_0.solid.specific_heat", 10.);
  database.put("physics.thermal", true);
  database.put("physics.mechanical", true);
  database.put("post_processor.filename_prefix", "output");
  database.put("refinement.n_heat_refinements", 0);
  database.put("sources.n_beams", 1);
  database.put("sources.beam_0.type", "goldak");
  database.put("sources.beam_0.scan_path_file", "sp.txt");
  database.put("sources.beam_0.scan_path_file_format", "segment");
  database.put("sources.beam_0.depth", 0.1);
  database.put("sources.beam_0.absorption_efficiency", 0.1);
  database.put("time_stepping.method", "forward_euler");
  database.put("time_stepping.duration", 1.0);
  database.put("time_stepping.time_step", 0.1);
  validate_input_database(database);
}

BOOST_AUTO_TEST_CASE(parse_test_input_files)
{
  boost::property_tree::ptree database;

  std::string filename = "bare_plate_L_da.info";
  boost::property_tree::info_parser::read_info(filename, database);
  validate_input_database(database);

  filename = "bare_plate_L_ensemble.info";
  boost::property_tree::info_parser::read_info(filename, database);
  validate_input_database(database);

  filename = "demo_316_short.info";
  boost::property_tree::info_parser::read_info(filename, database);
  validate_input_database(database);

  filename = "integration_2d.info";
  boost::property_tree::info_parser::read_info(filename, database);
  validate_input_database(database);
}

BOOST_AUTO_TEST_CASE(expected_failures)
{
  boost::property_tree::ptree database;

  // Start with a valid set of bare-bones inputs
  database.put("boundary.type", "adiabatic");
  database.put("discretization.thermal.fe_degree", "1");
  database.put("geometry.dim", "3");
  database.put("discretization.thermal.quadrature", "gauss");
  database.put("geometry.import_mesh", "false");
  database.put("geometry.length", 1.0);
  database.put("geometry.height", 1.0);
  database.put("geometry.width", 1.0);
  database.put("materials.n_materials", 1);
  database.put("materials.property_format", "polynomial");
  database.put("materials.material_0.solid.thermal_conductivity_x", 10.);
  database.put("materials.material_0.solid.thermal_conductivity_y", 10.);
  database.put("materials.material_0.solid.thermal_conductivity_z", 10.);
  database.put("materials.material_0.solid.density", 10.);
  database.put("materials.material_0.solid.specific_heat", 10.);
  database.put("physics.thermal", true);
  database.put("physics.mechanical", true);
  database.put("post_processor.filename_prefix", "output");
  database.put("refinement.n_heat_refinements", 0);
  database.put("sources.n_beams", 1);
  database.put("sources.beam_0.type", "goldak");
  database.put("sources.beam_0.scan_path_file", "sp.txt");
  database.put("sources.beam_0.scan_path_file_format", "segment");
  database.put("sources.beam_0.depth", 0.1);
  database.put("sources.beam_0.absorption_efficiency", 0.1);
  database.put("time_stepping.method", "forward_euler");
  database.put("time_stepping.duration", 1.0);
  database.put("time_stepping.time_step", 0.1);

  // Check 0: The base database (this one should be valid)
  validate_input_database(database);

  // We purposefully reset the database fully after each check to make the
  // checks independent. Even though carrying some state between checks might
  // reduce the number of lines, the increase in complexity isn't worth it.

  // Check 1: Invalid BC combination
  database.get_child("boundary").erase("type");
  database.put("boundary.type", "adiabatic,convective");
  BOOST_CHECK_THROW(validate_input_database(database), std::runtime_error);
  database.get_child("boundary").erase("type");
  database.put("boundary.type", "adiabatic");

  // Check 2: Invalid fe degree
  database.get_child("discretization").erase("thermal.fe_degree");
  database.put("discretization.thermal.fe_degree", 11);
  BOOST_CHECK_THROW(validate_input_database(database), std::runtime_error);
  database.get_child("discretization").erase("thermal.fe_degree");
  database.put("discretization.thermal.fe_degree", 1);

  // Check 3: Invalid quadrature type
  database.get_child("discretization").erase("thermal.quadrature");
  database.put("discretization.thermal.quadrature", "gass");
  BOOST_CHECK_THROW(validate_input_database(database), std::runtime_error);
  database.get_child("discretization").erase("thermal.quadrature");
  database.put("discretization.thermal.quadrature", "gauss");

  // Check 4: Invalid number of dimensions
  database.get_child("geometry").erase("dim");
  database.put("geometry.dim", 1);
  BOOST_CHECK_THROW(validate_input_database(database), std::runtime_error);
  database.get_child("geometry").erase("dim");
  database.put("geometry.dim", 11);
  BOOST_CHECK_THROW(validate_input_database(database), std::runtime_error);
  database.get_child("geometry").erase("dim");
  database.put("geometry.dim", 3);

  // Check 5: 'use_powder' true, but no 'powder_layer'
  database.get_child("geometry").erase("use_powder");
  database.put("geometry.use_powder", true);
  BOOST_CHECK_THROW(validate_input_database(database), std::runtime_error);
  database.get_child("geometry").erase("use_powder");
  database.put("geometry.use_powder", false);

  // Check 6: 'material_deposition' true, but invalid
  // 'material_deposition_method'
  database.get_child("geometry").erase("material_deposition");
  database.put("geometry.material_deposition", true);
  database.put("geometry.material_deposition_method", "ffile");
  BOOST_CHECK_THROW(validate_input_database(database), std::runtime_error);
  database.get_child("geometry").erase("material_deposition");
  database.get_child("geometry").erase("material_deposition_method");

  // Check 7: 'material_deposition' true, but missing deposition inputs
  database.get_child("geometry").erase("material_deposition_method");
  database.put("geometry.material_deposition", true);
  database.put("geometry.material_deposition_method", "scan_paths");
  BOOST_CHECK_THROW(validate_input_database(database), std::runtime_error);
  database.put("geometry.deposition_length", 0.1);
  database.put("geometry.deposition_height", 0.1);
  database.put("geometry.deposition_width", 0.1);
  BOOST_CHECK_THROW(validate_input_database(database), std::runtime_error);
  database.get_child("geometry").erase("deposition_length");
  database.put("geometry.lead_time", 0.1);
  BOOST_CHECK_THROW(validate_input_database(database), std::runtime_error);
  database.get_child("geometry").erase("deposition_height");
  BOOST_CHECK_THROW(validate_input_database(database), std::runtime_error);
  database.put("geometry.height", 0.1);
  database.get_child("geometry").erase("deposition_width");
  BOOST_CHECK_THROW(validate_input_database(database), std::runtime_error);
  database.get_child("geometry").erase("material_deposition");
  database.get_child("geometry").erase("material_deposition_method");
  database.get_child("geometry").erase("deposition_length");
  database.get_child("geometry").erase("deposition_height");
  database.get_child("geometry").erase("deposition_width");
  database.get_child("geometry").erase("lead_time");

  // Check 8: 'import_mesh' true, but missing mesh reading inputs
  database.get_child("geometry").erase("import_mesh");
  database.put("geometry.import_mesh", true);
  database.put("geometry.mesh_file", "mesh.vtk");
  BOOST_CHECK_THROW(validate_input_database(database), std::runtime_error);
  database.get_child("geometry").erase("mesh_file");
  database.put("geometry.mesh_format", "vtk");
  BOOST_CHECK_THROW(validate_input_database(database), std::runtime_error);
  database.get_child("geometry").erase("import_mesh");
  database.put("geometry.import_mesh", false);
  database.get_child("geometry").erase("mesh_file");
  database.get_child("geometry").erase("mesh_format");

  // Check 9: 'import_mesh' true, incorrect mesh format
  database.get_child("geometry").erase("import_mesh");
  database.put("geometry.import_mesh", true);
  database.put("geometry.mesh_file", "mesh.vtk");
  database.put("geometry.mesh_format", "vvtk");
  BOOST_CHECK_THROW(validate_input_database(database), std::runtime_error);
  database.get_child("geometry").erase("import_mesh");
  database.put("geometry.import_mesh", false);
  database.get_child("geometry").erase("mesh_file");
  database.get_child("geometry").erase("mesh_format");

  // Check 10: 'import_mesh' false, missing inputs
  database.get_child("geometry").erase("length");
  BOOST_CHECK_THROW(validate_input_database(database), std::runtime_error);
  database.put("geometry.length", 10.0);
  database.get_child("geometry").erase("height");
  BOOST_CHECK_THROW(validate_input_database(database), std::runtime_error);
  database.put("geometry.height", 10.0);
  database.get_child("geometry").erase("width");
  BOOST_CHECK_THROW(validate_input_database(database), std::runtime_error);
  database.put("geometry.width", 10.0);

  // Check 11: 'n_materials' missing
  database.get_child("materials").erase("n_materials");
  BOOST_CHECK_THROW(validate_input_database(database), std::runtime_error);
  database.put("materials.n_materials", 1);

  // Check 12: Invalid material properties format
  database.get_child("materials").erase("property_format");
  BOOST_CHECK_THROW(validate_input_database(database), std::runtime_error);
  database.put("materials.property_format", "ppoly");
  BOOST_CHECK_THROW(validate_input_database(database), std::runtime_error);
  database.put("materials.property_format", "polynomial");

  // Check 13: Non-consecutive materials
  database.get_child("materials").erase("n_materials");
  database.put("materials.n_materials", 2);
  database.put("materials.material_2.solid.thermal_conductivity_x", 10.);
  database.put("materials.material_2.solid.thermal_conductivity_y", 10.);
  database.put("materials.material_2.solid.thermal_conductivity_z", 10.);
  database.put("materials.material_2.solid.density", 10.);
  database.put("materials.material_2.solid.specific_heat", 10.);
  BOOST_CHECK_THROW(validate_input_database(database), std::runtime_error);
  database.get_child("materials").erase("n_materials");
  database.put("materials.n_materials", 1);
  database.get_child("materials")
      .get_child("material_2")
      .get_child("solid")
      .erase("thermal_conductivity_x");
  database.get_child("materials")
      .get_child("material_2")
      .get_child("solid")
      .erase("thermal_conductivity_y");
  database.get_child("materials")
      .get_child("material_2")
      .get_child("solid")
      .erase("thermal_conductivity_z");
  database.get_child("materials")
      .get_child("material_2")
      .get_child("solid")
      .erase("density");
  database.get_child("materials")
      .get_child("material_2")
      .get_child("solid")
      .erase("specific_heat");

  // Check 14: Material without any phases
  database.get_child("materials")
      .get_child("material_0")
      .get_child("solid")
      .erase("thermal_conductivity_x");
  database.get_child("materials")
      .get_child("material_0")
      .get_child("solid")
      .erase("thermal_conductivity_y");
  database.get_child("materials")
      .get_child("material_0")
      .get_child("solid")
      .erase("thermal_conductivity_z");
  database.get_child("materials")
      .get_child("material_0")
      .get_child("solid")
      .erase("density");
  database.get_child("materials")
      .get_child("material_0")
      .get_child("solid")
      .erase("specific_heat");
  BOOST_CHECK_THROW(validate_input_database(database), std::runtime_error);
  database.put("materials.material_0.solid.thermal_conductivity_x", 10.);
  database.put("materials.material_0.solid.thermal_conductivity_y", 10.);
  database.put("materials.material_0.solid.thermal_conductivity_z", 10.);
  database.put("materials.material_0.solid.density", 10.);
  database.put("materials.material_0.solid.specific_heat", 10.);

  // Check 15: Missing always required material properties
  database.get_child("materials")
      .get_child("material_0")
      .get_child("solid")
      .erase("thermal_conductivity_x");
  BOOST_CHECK_THROW(validate_input_database(database), std::runtime_error);
  database.put("materials.material_0.solid.thermal_conductivity_x", 10.);
  database.get_child("materials")
      .get_child("material_0")
      .get_child("solid")
      .erase("thermal_conductivity_y");
  BOOST_CHECK_THROW(validate_input_database(database), std::runtime_error);
  database.put("materials.material_0.solid.thermal_conductivity_y", 10.);
  database.get_child("materials")
      .get_child("material_0")
      .get_child("solid")
      .erase("thermal_conductivity_z");
  BOOST_CHECK_THROW(validate_input_database(database), std::runtime_error);
  database.put("materials.material_0.solid.thermal_conductivity_z", 10.);
  database.get_child("materials")
      .get_child("material_0")
      .get_child("solid")
      .erase("density");
  BOOST_CHECK_THROW(validate_input_database(database), std::runtime_error);
  database.put("materials.material_0.solid.density", 10.);
  database.get_child("materials")
      .get_child("material_0")
      .get_child("solid")
      .erase("specific_heat");
  BOOST_CHECK_THROW(validate_input_database(database), std::runtime_error);
  database.put("materials.material_0.solid.specific_heat", 10.);

  // Check 16: Missing required material properties with convective BCs
  database.get_child("boundary").erase("type");
  database.put("boundary.type", "convective");
  database.put("materials.material_0.solid.convection_heat_transfer_coef", 10.);
  BOOST_CHECK_THROW(validate_input_database(database), std::runtime_error);
  database.get_child("materials")
      .get_child("material_0")
      .get_child("solid")
      .erase("convection_heat_transfer_coef");
  database.put("materials.material_0.convection_temperature_infty", 10.);
  BOOST_CHECK_THROW(validate_input_database(database), std::runtime_error);
  database.get_child("materials")
      .get_child("material_0")
      .get_child("solid")
      .erase("convection_temperature_infty");
  database.get_child("boundary").erase("type");
  database.put("boundary.type", "adiabatic");

  // Check 17: Missing required material properties with radiative BCs
  database.get_child("boundary").erase("type");
  database.put("boundary.type", "radiative");
  database.put("materials.material_0.solid.emissivity", 10.);
  BOOST_CHECK_THROW(validate_input_database(database), std::runtime_error);
  database.get_child("materials")
      .get_child("material_0")
      .get_child("solid")
      .erase("emissivity");
  database.put("materials.material_0.radiation_temperature_infty", 10.);
  BOOST_CHECK_THROW(validate_input_database(database), std::runtime_error);
  database.get_child("materials")
      .get_child("material_0")
      .get_child("solid")
      .erase("radiation_temperature_infty");
  database.get_child("boundary").erase("type");
  database.put("boundary.type", "adiabatic");

  // Check 18: Invalid memory space
  database.erase("memory_space");
  database.put("memory_space", "hhost");
  BOOST_CHECK_THROW(validate_input_database(database), std::runtime_error);
  database.erase("memory_space");

  // TODO check physics

  // Check 19: Missing postprocessor filename
  database.get_child("post_processor").erase("filename_prefix");
  BOOST_CHECK_THROW(validate_input_database(database), std::runtime_error);
  database.put("post_processor.filename_prefix", "output");

  // Check 20: Missing refinement block
  database.get_child("refinement").erase("n_heat_refinements");
  database.erase("refinement");
  BOOST_CHECK_THROW(validate_input_database(database), std::runtime_error);
  database.put("refinement.n_heat_refinements", 0);

  // Check 21: Missing 'n_beams'
  database.get_child("sources").erase("n_beams");
  BOOST_CHECK_THROW(validate_input_database(database), std::runtime_error);
  database.put("sources.n_beams", 1);

  // Check 22: Missing heat source type
  database.get_child("sources").get_child("beam_0").erase("type");
  BOOST_CHECK_THROW(validate_input_database(database), std::runtime_error);
  database.put("sources.beam_0.type", "goldak");

  // Check 22: Invalid heat source type
  database.get_child("sources").get_child("beam_0").erase("type");
  database.put("sources.beam_0.type", "gold");
  BOOST_CHECK_THROW(validate_input_database(database), std::runtime_error);
  database.put("sources.beam_0.type", "goldak");

  // Check 23: Missing scan path file
  database.get_child("sources").get_child("beam_0").erase("scan_path_file");
  BOOST_CHECK_THROW(validate_input_database(database), std::runtime_error);
  database.put("sources.beam_0.scan_path_file", "sp.txt");

  // Check 24: Missing scan path file format
  database.get_child("sources").get_child("beam_0").erase(
      "scan_path_file_format");
  BOOST_CHECK_THROW(validate_input_database(database), std::runtime_error);
  database.put("sources.beam_0.scan_path_file_format", "segment");

  // Check 25: Invalid scan path file format
  database.get_child("sources").get_child("beam_0").erase(
      "scan_path_file_format");
  database.put("sources.beam_0.scan_path_file_format", "seg");
  BOOST_CHECK_THROW(validate_input_database(database), std::runtime_error);
  database.put("sources.beam_0.scan_path_file_format", "segment");

  // Check 24: Missing beam depth
  database.get_child("sources").get_child("beam_0").erase("depth");
  BOOST_CHECK_THROW(validate_input_database(database), std::runtime_error);
  database.put("sources.beam_0.depth", 0.1);

  // Check 25: Missing beam absorption efficiency
  database.get_child("sources").get_child("beam_0").erase(
      "absorption_efficiency");
  BOOST_CHECK_THROW(validate_input_database(database), std::runtime_error);
  database.put("sources.beam_0.absorption_efficiency", 0.1);

  // Check 26: Missing time stepping method
  database.get_child("time_stepping").erase("method");
  BOOST_CHECK_THROW(validate_input_database(database), std::runtime_error);
  database.put("time_stepping.method", "forward_euler");

  // Check 27: Invalid time stepping method
  database.get_child("time_stepping").erase("method");
  database.put("time_stepping.method", "4_step_jump");
  BOOST_CHECK_THROW(validate_input_database(database), std::runtime_error);
  database.put("time_stepping.method", "forward_euler");

  // Check 28: Missing experimental inputs
  database.put("experiment.read_in_experimental_data", true);
  database.put("experiment.file", "file.csv");
  database.put("experiment.last_frame", 1);
  database.put("experiment.first_camera_id", 0);
  database.put("experiment.last_camera_id", 1);
  database.put("experiment.data_columns", "1,2,3,4");
  BOOST_CHECK_THROW(validate_input_database(database), std::runtime_error);
  database.get_child("experiment").erase("file");
  database.put("experiment.log_filename", "log.txt");
  BOOST_CHECK_THROW(validate_input_database(database), std::runtime_error);
  database.get_child("experiment").erase("last_frame");
  BOOST_CHECK_THROW(validate_input_database(database), std::runtime_error);
  database.put("experiment.last_frame", 1);
  database.get_child("experiment").erase("first_camera_id");
  BOOST_CHECK_THROW(validate_input_database(database), std::runtime_error);
  database.put("experiment.first_camera_id", 0);
  database.get_child("experiment").erase("last_camera_id");
  BOOST_CHECK_THROW(validate_input_database(database), std::runtime_error);
  database.put("experiment.first_camera_id", 0);
  database.get_child("experiment").erase("data_columns");
  BOOST_CHECK_THROW(validate_input_database(database), std::runtime_error);
  database.get_child("experiment").erase("file");
  database.get_child("experiment").erase("last_frame");
  database.get_child("experiment").erase("first_camera_id");
  database.get_child("experiment").erase("last_camera_id");
  database.get_child("experiment").erase("read_in_experimental_data");

  // Check 29: Last experimental frame index smaller than first frame
  database.put("experiment.read_in_experimental_data", true);
  database.put("experiment.first_frame", 4);
  database.put("experiment.last_frame", 3);
  BOOST_CHECK_THROW(validate_input_database(database), std::runtime_error);
  database.get_child("experiment").erase("last_frame");
  database.get_child("experiment").erase("first_frame");
  database.get_child("experiment").erase("read_in_experimental_data");

  // Check 30: Last experimental camera id smaller than first camera id
  database.put("experiment.read_in_experimental_data", true);
  database.put("experiment.first_camera_id", 4);
  database.put("experiment.last_camera_id", 3);
  BOOST_CHECK_THROW(validate_input_database(database), std::runtime_error);
  database.get_child("experiment").erase("first_camera_id");
  database.get_child("experiment").erase("last_camera_id");
  database.get_child("experiment").erase("read_in_experimental_data");

  // Check 31: Incorrect number of entries for the experimental data column
  // indices
  database.put("experiment.read_in_experimental_data", true);
  database.put("experiment.data_columns", "1,2,3");
  BOOST_CHECK_THROW(validate_input_database(database), std::runtime_error);
  database.put("experiment.data_columns", "1,2,3,4");
  database.put("geometry.dim", 2);
  BOOST_CHECK_THROW(validate_input_database(database), std::runtime_error);
  database.get_child("experiment").erase("data_columns");
  database.get_child("geometry").erase("dim");
  database.put("geometry.dim", 3);
  database.get_child("experiment").erase("read_in_experimental_data");

  // Final Check: This should be back to the base database (this should be
  // valid)
  validate_input_database(database);
}

} // namespace adamantine
