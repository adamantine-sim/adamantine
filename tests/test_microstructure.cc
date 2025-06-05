/* SPDX-FileCopyrightText: Copyright (c)  2025, the adamantine authors.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include <Geometry.hh>
#include <MaterialProperty.hh>
#include <Microstructure.hh>
#include <ThermalPhysics.hh>

#include <deal.II/base/mpi.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/mapping_q1.h>

#include <filesystem>
#include <fstream>

#define BOOST_TEST_MODULE MicroStructure

#include "main.cc"

namespace utf = boost::unit_test;

void check_results(std::string const &filename_prefix,
                   std::string const &gold_filename)
{
  std::string filename(filename_prefix + ".txt");
  std::ifstream file(filename);
  std::ifstream gold_file(gold_filename);

  std::vector<std::string> lines;
  std::vector<std::string> gold_lines;
  std::string line;

  while (std::getline(file, line))
  {
    lines.push_back(line);
  }
  while (std::getline(gold_file, line))
  {
    gold_lines.push_back(line);
  }

  BOOST_TEST_REQUIRE(lines.size() == gold_lines.size());
  for (unsigned int i = 0; i < lines.size(); ++i)
  {
    BOOST_TEST(lines[i] == gold_lines[i]);
  }

  std::filesystem::remove(filename);
}

BOOST_AUTO_TEST_CASE(G_and_R)
{
  MPI_Comm communicator = MPI_COMM_WORLD;
  std::string filename_prefix("G_R");

  // Geometry database
  boost::property_tree::ptree geometry_database;
  geometry_database.put("import_mesh", false);
  geometry_database.put("length", 10e-3);
  geometry_database.put("length_divisions", 1);
  geometry_database.put("height", 10e-3);
  geometry_database.put("height_divisions", 1);

  boost::optional<boost::property_tree::ptree const &> units_optional_database;
  adamantine::Geometry<2> geometry(communicator, geometry_database,
                                   units_optional_database);

  // Build MaterialProperty
  boost::property_tree::ptree material_property_database;
  material_property_database.put("property_format", "polynomial");
  material_property_database.put("n_materials", 1);
  material_property_database.put("material_0.solid.density", 1.);
  material_property_database.put("material_0.powder.density", 10.);
  material_property_database.put("material_0.liquid.density", 1.);
  material_property_database.put("material_0.solid.specific_heat", 1.);
  material_property_database.put("material_0.powder.specific_heat", 2.);
  material_property_database.put("material_0.liquid.specific_heat", 1.);
  material_property_database.put("material_0.solid.thermal_conductivity_x", 1.);
  material_property_database.put("material_0.solid.thermal_conductivity_z", 1.);
  material_property_database.put("material_0.powder.thermal_conductivity_x",
                                 1.);
  material_property_database.put("material_0.powder.thermal_conductivity_z",
                                 1.);
  material_property_database.put("material_0.liquid.thermal_conductivity_x",
                                 1.);
  material_property_database.put("material_0.liquid.thermal_conductivity_z",
                                 1.);
  material_property_database.put("material_0.liquidus", 100.);
  adamantine::MaterialProperty<2, 4, adamantine::SolidLiquidPowder,
                               dealii::MemorySpace::Host>
      material_properties(communicator, geometry.get_triangulation(),
                          material_property_database);

  // Other generic input parameters
  boost::property_tree::ptree database;
  // Source database
  database.put("sources.n_beams", 1);
  database.put("sources.beam_0.depth", 1e100);
  database.put("sources.beam_0.diameter", 1e100);
  database.put("sources.beam_0.max_power", 1e300);
  database.put("sources.beam_0.scan_path_file", "scan_path.txt");
  database.put("sources.beam_0.absorption_efficiency", 0.3);
  database.put("sources.beam_0.type", "electron_beam");
  database.put("sources.beam_0.scan_path_file",
               "scan_path_test_thermal_physics.txt");
  database.put("sources.beam_0.scan_path_file_format", "segment");
  // Boundary database
  database.put("boundary.type", "adiabatic");

  // Time-stepping database
  database.put("time_stepping.method", "rk_fourth_order");

  // Build ThermalPhysics
  adamantine::ThermalPhysics<2, 4, 2, adamantine::SolidLiquidPowder,
                             dealii::MemorySpace::Host, dealii::QGauss<1>>
      physics(communicator, database, geometry, material_properties);
  physics.setup();
  dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host> solution;

  // Now that the setup is done we can compute G, R, and the cooling rate
  {
    adamantine::Microstructure<2> microstructure(communicator, filename_prefix);
    physics.initialize_dof_vector(1000., solution);
    microstructure.set_old_temperature(solution);
    physics.initialize_dof_vector(50., solution);
    microstructure.compute_G_and_R(material_properties,
                                   physics.get_dof_handler(), solution, 1.1);
  }

  // Compare the new file with the gold file
  unsigned int const rank =
      dealii::Utilities::MPI::this_mpi_process(communicator);
  if (rank == 0)
  {
    check_results(filename_prefix, "microstructure_G_R_gold_1.txt");
  }
  MPI_Barrier(communicator);

  // Now that the setup is done we can compute G, R, and the cooling rate
  {
    adamantine::Microstructure<2> microstructure(communicator, filename_prefix);
    physics.initialize_dof_vector(1000., solution);
    // Get the support points
    auto dofs_points_map = dealii::DoFTools::map_dofs_to_support_points(
        dealii::MappingQ1<2, 2>(), physics.get_dof_handler());
    auto locally_owned_elements = solution.locally_owned_elements();
    for (auto index : locally_owned_elements)
    {
      auto const &point = dofs_points_map[index];
      solution[index] += std::pow(point[0] - 1., 2) + std::pow(point[1], 2);
    }
    microstructure.set_old_temperature(solution);

    physics.initialize_dof_vector(50., solution);
    for (auto index : locally_owned_elements)
    {
      auto const &point = dofs_points_map[index];
      solution[index] += std::pow(point[0] - 1., 2) + std::pow(point[1], 2);
    }
    microstructure.compute_G_and_R(material_properties,
                                   physics.get_dof_handler(), solution, 1.1);
  }

  if (rank == 0)
  {
    check_results(filename_prefix, "microstructure_G_R_gold_2.txt");
  }
}
