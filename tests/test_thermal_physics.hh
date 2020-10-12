/* Copyright (c) 2016 - 2020, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#include <Geometry.hh>
#include <ThermalPhysics.hh>

#include <deal.II/base/quadrature_lib.h>

template <typename MemorySpaceType>
void thermal_2d(boost::property_tree::ptree &database, double time_step)
{
  MPI_Comm communicator = MPI_COMM_WORLD;

  // Geometry database
  boost::property_tree::ptree geometry_database;
  geometry_database.put("import_mesh", false);
  geometry_database.put("length", 12e-3);
  geometry_database.put("length_divisions", 4);
  geometry_database.put("height", 6e-3);
  geometry_database.put("height_divisions", 5);
  // Build Geometry
  adamantine::Geometry<2> geometry(communicator, geometry_database);
  // Material property
  database.put("materials.property_format", "polynomial");
  database.put("materials.n_materials", 1);
  database.put("materials.material_0.solid.density", 1.);
  database.put("materials.material_0.powder.density", 1.);
  database.put("materials.material_0.liquid.density", 1.);
  database.put("materials.material_0.solid.specific_heat", 1.);
  database.put("materials.material_0.powder.specific_heat", 1.);
  database.put("materials.material_0.liquid.specific_heat", 1.);
  database.put("materials.material_0.solid.thermal_conductivity", 1.);
  database.put("materials.material_0.powder.thermal_conductivity", 1.);
  database.put("materials.material_0.liquid.thermal_conductivity", 1.);
  // Source database
  database.put("sources.n_beams", 1);
  database.put("sources.beam_0.depth", 1e100);
  database.put("sources.beam_0.diameter", 1e100);
  database.put("sources.beam_0.max_power", 1e300);
  database.put("sources.beam_0.absorption_efficiency", 0.1);
  database.put("sources.beam_0.type", "electron_beam");

  // Build ThermalPhysics
  adamantine::ThermalPhysics<2, 2, MemorySpaceType, dealii::QGauss<1>> physics(
      communicator, database, geometry);
  physics.setup_dofs();
  physics.compute_inverse_mass_matrix();

  dealii::LA::distributed::Vector<double, MemorySpaceType> solution;
  physics.initialize_dof_vector(solution);
  std::vector<adamantine::Timer> timers(6);
  double time = 0;
  while (time < 0.1)
    time = physics.evolve_one_time_step(time, time_step, solution, timers);

  double const tolerance = 1e-3;
  BOOST_CHECK(time == 0.1);
  BOOST_CHECK_CLOSE(solution.l2_norm(), 0.291705, tolerance);

  physics.initialize_dof_vector(1000., solution);
  BOOST_CHECK(solution.l1_norm() == 1000. * solution.size());
}

template <typename MemorySpaceType>
void thermal_2d_manufactured_solution()
{
  MPI_Comm communicator = MPI_COMM_WORLD;

  // Geometry database
  boost::property_tree::ptree geometry_database;
  geometry_database.put("import_mesh", false);
  geometry_database.put("length", 1e3);
  geometry_database.put("length_divisions", 4);
  geometry_database.put("height", 6e3);
  geometry_database.put("height_divisions", 5);
  // Build Geometry
  adamantine::Geometry<2> geometry(communicator, geometry_database);

  boost::property_tree::ptree database;
  // Material property
  database.put("materials.property_format", "polynomial");
  database.put("materials.n_materials", 1);
  database.put("materials.material_0.solid.density", 1.);
  database.put("materials.material_0.powder.density", 1.);
  database.put("materials.material_0.liquid.density", 1.);
  database.put("materials.material_0.solid.specific_heat", 1.);
  database.put("materials.material_0.powder.specific_heat", 1.);
  database.put("materials.material_0.liquid.specific_heat", 1.);
  database.put("materials.material_0.solid.thermal_conductivity", 1.);
  database.put("materials.material_0.powder.thermal_conductivity", 1.);
  database.put("materials.material_0.liquid.thermal_conductivity", 1.);
  // Source database
  database.put("sources.n_beams", 1);
  database.put("sources.beam_0.depth", 1e100);
  database.put("sources.beam_0.diameter", 1e100);
  database.put("sources.beam_0.max_power", 1e300);
  database.put("sources.beam_0.absorption_efficiency",
               0.1 / 0.29317423955177113);
  database.put("sources.beam_0.scan_path_file",
               "scan_path_test_thermal_physics.txt");
  database.put("sources.beam_0.type", "electron_beam");

  // Time-stepping database
  database.put("time_stepping.method", "rk_fourth_order");
  // Build ThermalPhysics
  adamantine::ThermalPhysics<2, 2, MemorySpaceType, dealii::QGauss<1>> physics(
      communicator, database, geometry);
  physics.setup_dofs();
  physics.compute_inverse_mass_matrix();

  dealii::LA::distributed::Vector<double, MemorySpaceType> solution;
  std::vector<adamantine::Timer> timers(6);
  physics.initialize_dof_vector(solution);
  double time = physics.evolve_one_time_step(0., 0.1, solution, timers);

  double const tolerance = 1e-5;
  BOOST_CHECK(time == 0.1);
  if (std::is_same<MemorySpaceType, dealii::MemorySpace::Host>::value)
  {
    for (unsigned int i = 0; i < solution.size(); ++i)
      BOOST_CHECK_CLOSE(solution[i], 0.1, tolerance);
  }
  else
  {
    dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host>
        solution_host(solution.get_partitioner());
    solution_host.import(solution, dealii::VectorOperation::insert);
    for (unsigned int i = 0; i < solution.size(); ++i)
      BOOST_CHECK_CLOSE(solution_host[i], 0.1, tolerance);
  }
}

template <typename MemorySpaceType>
void initial_temperature()
{
  MPI_Comm communicator = MPI_COMM_WORLD;

  // Geometry database
  boost::property_tree::ptree geometry_database;
  geometry_database.put("import_mesh", false);
  geometry_database.put("length", 12e-3);
  geometry_database.put("length_divisions", 1);
  geometry_database.put("height", 6e-3);
  geometry_database.put("height_divisions", 1);
  // Build Geometry
  adamantine::Geometry<2> geometry(communicator, geometry_database);
  boost::property_tree::ptree database;
  // Material property
  database.put("materials.property_format", "polynomial");
  database.put("materials.n_materials", 1);
  database.put("materials.material_0.solid.density", 1.);
  database.put("materials.material_0.powder.density", 10.);
  database.put("materials.material_0.liquid.density", 1.);
  database.put("materials.material_0.solid.specific_heat", 1.);
  database.put("materials.material_0.powder.specific_heat", 2.);
  database.put("materials.material_0.liquid.specific_heat", 1.);
  database.put("materials.material_0.solid.thermal_conductivity", 1.);
  database.put("materials.material_0.powder.thermal_conductivity", 1.);
  database.put("materials.material_0.liquid.thermal_conductivity", 1.);
  // Source database
  database.put("sources.n_beams", 1);
  database.put("sources.beam_0.depth", 1e100);
  database.put("sources.beam_0.diameter", 1e100);
  database.put("sources.beam_0.max_power", 1e300);
  database.put("sources.beam_0.scan_path_file", "scan_path.txt");
  database.put("sources.beam_0.absorption_efficiency", 0.3);
  database.put("sources.beam_0.type", "electron_beam");

  // Time-stepping database
  database.put("time_stepping.method", "rk_fourth_order");
  // Build ThermalPhysics
  adamantine::ThermalPhysics<2, 2, MemorySpaceType, dealii::QGauss<1>> physics(
      communicator, database, geometry);
  physics.setup_dofs();
  physics.compute_inverse_mass_matrix();

  dealii::LA::distributed::Vector<double, MemorySpaceType> solution;
  physics.initialize_dof_vector(1000., solution);
  BOOST_CHECK(solution.l1_norm() == 1000. * solution.size());
}
