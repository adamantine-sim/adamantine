/* Copyright (c) 2016 - 2021, the adamantine authors.
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
  database.put("materials.material_0.solid.thermal_conductivity_x", 1.);
  database.put("materials.material_0.solid.thermal_conductivity_z", 1.);
  database.put("materials.material_0.powder.thermal_conductivity_x", 1.);
  database.put("materials.material_0.powder.thermal_conductivity_z", 1.);
  database.put("materials.material_0.liquid.thermal_conductivity_x", 1.);
  database.put("materials.material_0.liquid.thermal_conductivity_z", 1.);
  // Source database
  database.put("sources.n_beams", 1);
  database.put("sources.beam_0.depth", 1e100);
  database.put("sources.beam_0.diameter", 1e100);
  database.put("sources.beam_0.max_power", 1e300);
  database.put("sources.beam_0.absorption_efficiency", 0.1);
  database.put("sources.beam_0.type", "electron_beam");
  database.put("sources.beam_0.scan_path_file",
               "scan_path_test_thermal_physics.txt");
  database.put("sources.beam_0.scan_path_file_format", "segment");
  // Boundary database
  database.put("boundary.type", "adiabatic");

  // Build ThermalPhysics
  adamantine::ThermalPhysics<2, 2, MemorySpaceType, dealii::QGauss<1>> physics(
      communicator, database, geometry);
  physics.setup_dofs();
  physics.update_material_deposition_orientation();
  physics.compute_inverse_mass_matrix();

  dealii::LA::distributed::Vector<double, MemorySpaceType> solution;
  physics.initialize_dof_vector(solution);
  physics.get_state_from_material_properties();
  std::vector<adamantine::Timer> timers(adamantine::Timing::n_timers);
  double time = 0;
  while (time < 0.1)
  {
    time = physics.evolve_one_time_step(time, time_step, solution, timers);
  }

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
  database.put("materials.material_0.solid.thermal_conductivity_x", 1.);
  database.put("materials.material_0.solid.thermal_conductivity_z", 1.);
  database.put("materials.material_0.powder.thermal_conductivity_x", 1.);
  database.put("materials.material_0.powder.thermal_conductivity_z", 1.);
  database.put("materials.material_0.liquid.thermal_conductivity_x", 1.);
  database.put("materials.material_0.liquid.thermal_conductivity_z", 1.);
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
  database.put("sources.beam_0.scan_path_file_format", "segment");
  // Boundary database
  database.put("boundary.type", "adiabatic");

  // Time-stepping database
  database.put("time_stepping.method", "rk_fourth_order");
  // Build ThermalPhysics
  adamantine::ThermalPhysics<2, 2, MemorySpaceType, dealii::QGauss<1>> physics(
      communicator, database, geometry);
  physics.setup_dofs();
  physics.update_material_deposition_orientation();
  physics.compute_inverse_mass_matrix();

  dealii::LA::distributed::Vector<double, MemorySpaceType> solution;
  std::vector<adamantine::Timer> timers(adamantine::Timing::n_timers);
  physics.initialize_dof_vector(solution);
  physics.get_state_from_material_properties();
  double time = physics.evolve_one_time_step(0., 0.1, solution, timers);

  double const tolerance = 1e-5;

  BOOST_CHECK(time == 0.1);

  BOOST_CHECK_SMALL(std::abs(physics.get_current_source_height() - 0.0),
                    tolerance);

  if (std::is_same<MemorySpaceType, dealii::MemorySpace::Host>::value)
  {
    for (unsigned int i = 0; i < solution.locally_owned_size(); ++i)
      BOOST_CHECK_CLOSE(solution.local_element(i), 0.1, tolerance);
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
  database.put("materials.material_0.solid.thermal_conductivity_x", 1.);
  database.put("materials.material_0.solid.thermal_conductivity_z", 1.);
  database.put("materials.material_0.powder.thermal_conductivity_x", 1.);
  database.put("materials.material_0.powder.thermal_conductivity_z", 1.);
  database.put("materials.material_0.liquid.thermal_conductivity_x", 1.);
  database.put("materials.material_0.liquid.thermal_conductivity_z", 1.);
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
  adamantine::ThermalPhysics<2, 2, MemorySpaceType, dealii::QGauss<1>> physics(
      communicator, database, geometry);
  physics.setup_dofs();
  physics.update_material_deposition_orientation();
  physics.compute_inverse_mass_matrix();

  dealii::LA::distributed::Vector<double, MemorySpaceType> solution;
  physics.initialize_dof_vector(1000., solution);
  physics.get_state_from_material_properties();
  BOOST_CHECK(solution.l1_norm() == 1000. * solution.size());
}

template <typename MemorySpaceType>
void energy_conservation()
{
  MPI_Comm communicator = MPI_COMM_WORLD;

  // Geometry database
  boost::property_tree::ptree geometry_database;
  geometry_database.put("import_mesh", false);
  geometry_database.put("length", 10);
  geometry_database.put("length_divisions", 10);
  geometry_database.put("height", 10);
  geometry_database.put("height_divisions", 10);
  // Build Geometry
  adamantine::Geometry<2> geometry(communicator, geometry_database);
  boost::property_tree::ptree database;
  // Material property
  database.put("materials.property_format", "polynomial");
  database.put("materials.n_materials", 1);
  database.put("materials.material_0.solid.density", 0.5);
  database.put("materials.material_0.powder.density", 0.5);
  database.put("materials.material_0.liquid.density", 0.5);
  database.put("materials.material_0.solid.specific_heat", 4.);
  database.put("materials.material_0.powder.specific_heat", 4.);
  database.put("materials.material_0.liquid.specific_heat", 4.);
  database.put("materials.material_0.solid.thermal_conductivity_x", 2.);
  database.put("materials.material_0.solid.thermal_conductivity_z", 2.);
  database.put("materials.material_0.powder.thermal_conductivity_x", 2.);
  database.put("materials.material_0.powder.thermal_conductivity_z", 2.);
  database.put("materials.material_0.liquid.thermal_conductivity_x", 2.);
  database.put("materials.material_0.liquid.thermal_conductivity_z", 2.);
  // Source database
  database.put("sources.n_beams", 1);
  database.put("sources.beam_0.type", "cube");
  database.put("sources.beam_0.start_time", 0);
  database.put("sources.beam_0.end_time", 5);
  database.put("sources.beam_0.value", 5);
  database.put("sources.beam_0.min_x", 4);
  database.put("sources.beam_0.min_y", 4);
  database.put("sources.beam_0.max_x", 6);
  database.put("sources.beam_0.max_y", 6);
  // Time-stepping database
  database.put("time_stepping.method", "forward_euler");
  // Boundary database
  database.put("boundary.type", "adiabatic");
  // Build ThermalPhysics
  adamantine::ThermalPhysics<2, 2, dealii::MemorySpace::Host, dealii::QGauss<1>>
      physics(communicator, database, geometry);
  physics.setup_dofs();
  physics.update_material_deposition_orientation();
  physics.compute_inverse_mass_matrix();

  dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host> solution;
  double constexpr initial_temperature = 10;
  double constexpr final_temperature = 10.5;
  physics.initialize_dof_vector(initial_temperature, solution);
  physics.get_state_from_material_properties();
  std::vector<adamantine::Timer> timers(adamantine::Timing::n_timers);
  double time = 0;
  while (time < 100)
  {
    time = physics.evolve_one_time_step(time, 0.05, solution, timers);
  }

  double max = -1;
  double min = 1e4;
  if (std::is_same<MemorySpaceType, dealii::MemorySpace::Host>::value)
  {
    for (auto v : solution)
    {
      if (max < v)
        max = v;
      if (min > v)
        min = v;
    }
  }
  else
  {
    dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host>
        solution_host(solution.get_partitioner());
    solution_host.import(solution, dealii::VectorOperation::insert);
    for (auto v : solution_host)
    {
      if (max < v)
        max = v;
      if (min > v)
        min = v;
    }
  }

  double constexpr tolerance = 1e-9;
  BOOST_CHECK_CLOSE(solution.mean_value(), final_temperature, tolerance);
  BOOST_CHECK_CLOSE(min, max, tolerance);
}

template <typename MemorySpaceType>
void radiation_bcs()
{
  MPI_Comm communicator = MPI_COMM_WORLD;

  // Geometry database
  boost::property_tree::ptree geometry_database;
  geometry_database.put("import_mesh", false);
  geometry_database.put("length", 5);
  geometry_database.put("length_divisions", 5);
  geometry_database.put("height", 5);
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
  database.put("materials.material_0.solid.thermal_conductivity_x", 1.);
  database.put("materials.material_0.solid.thermal_conductivity_z", 1.);
  database.put("materials.material_0.powder.thermal_conductivity_x", 1.);
  database.put("materials.material_0.powder.thermal_conductivity_z", 1.);
  database.put("materials.material_0.liquid.thermal_conductivity_x", 1.);
  database.put("materials.material_0.liquid.thermal_conductivity_z", 1.);
  database.put("materials.material_0.solid.emissivity", 1.);
  database.put("materials.material_0.powder.emissivity", 1.);
  database.put("materials.material_0.liquid.emissivity", 1.);
  database.put("materials.material_0.solid.radiation_heat_transfer_coef", 1.);
  database.put("materials.material_0.powder.radiation_heat_transfer_coef", 1.);
  database.put("materials.material_0.liquid.radiation_heat_transfer_coef", 1.);
  database.put("materials.material_0.solid.convection_heat_transfer_coef", 1.);
  database.put("materials.material_0.powder.convection_heat_transfer_coef", 1.);
  database.put("materials.material_0.liquid.convection_heat_transfer_coef", 1.);
  database.put("materials.material_0.radiation_temperature_infty", 20.0);
  database.put("materials.material_0.convection_temperature_infty", 0.0);
  // Source database
  database.put("sources.n_beams", 0);
  // Time-stepping database
  database.put("time_stepping.method", "forward_euler");
  // Boundary database
  database.put("boundary.type", "radiative");
  // Build ThermalPhysics
  adamantine::ThermalPhysics<2, 2, dealii::MemorySpace::Host, dealii::QGauss<1>>
      physics(communicator, database, geometry);
  physics.setup_dofs();
  physics.update_material_deposition_orientation();
  physics.compute_inverse_mass_matrix();

  dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host> solution;
  double constexpr initial_temperature = 10;
  physics.initialize_dof_vector(initial_temperature, solution);
  physics.get_state_from_material_properties();
  std::vector<adamantine::Timer> timers(adamantine::Timing::n_timers);
  double time = 0;
  while (time < 100)
  {
    time = physics.evolve_one_time_step(time, 0.05, solution, timers);
  }

  double max = -1;
  double min = 1e4;
  if (std::is_same<MemorySpaceType, dealii::MemorySpace::Host>::value)
  {
    for (auto v : solution)
    {
      if (max < v)
        max = v;
      if (min > v)
        min = v;
    }
  }
  else
  {
    dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host>
        solution_host(solution.get_partitioner());
    solution_host.import(solution, dealii::VectorOperation::insert);
    for (auto v : solution_host)
    {
      if (max < v)
        max = v;
      if (min > v)
        min = v;
    }
  }

  BOOST_CHECK(min >= 10. && min <= 20.);
  BOOST_CHECK(max > 10. && max <= 20.);
}

template <typename MemorySpaceType>
void convection_bcs()
{
  MPI_Comm communicator = MPI_COMM_WORLD;

  // Geometry database
  boost::property_tree::ptree geometry_database;
  geometry_database.put("import_mesh", false);
  geometry_database.put("length", 5);
  geometry_database.put("length_divisions", 5);
  geometry_database.put("height", 5);
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
  database.put("materials.material_0.solid.thermal_conductivity_x", 1.);
  database.put("materials.material_0.solid.thermal_conductivity_z", 1.);
  database.put("materials.material_0.powder.thermal_conductivity_x", 1.);
  database.put("materials.material_0.powder.thermal_conductivity_z", 1.);
  database.put("materials.material_0.liquid.thermal_conductivity_x", 1.);
  database.put("materials.material_0.liquid.thermal_conductivity_z", 1.);
  database.put("materials.material_0.solid.emissivity", 1.);
  database.put("materials.material_0.powder.emissivity", 1.);
  database.put("materials.material_0.liquid.emissivity", 1.);
  database.put("materials.material_0.solid.radiation_heat_transfer_coef", 1.);
  database.put("materials.material_0.powder.radiation_heat_transfer_coef", 1.);
  database.put("materials.material_0.liquid.radiation_heat_transfer_coef", 1.);
  database.put("materials.material_0.solid.convection_heat_transfer_coef", 1.);
  database.put("materials.material_0.powder.convection_heat_transfer_coef", 1.);
  database.put("materials.material_0.liquid.convection_heat_transfer_coef", 1.);
  database.put("materials.material_0.radiation_temperature_infty", 0.0);
  database.put("materials.material_0.convection_temperature_infty", 20.0);
  // Source database
  database.put("sources.n_beams", 0);
  // Time-stepping database
  database.put("time_stepping.method", "forward_euler");
  // Boundary database
  database.put("boundary.type", "convective");
  // Build ThermalPhysics
  adamantine::ThermalPhysics<2, 2, dealii::MemorySpace::Host, dealii::QGauss<1>>
      physics(communicator, database, geometry);
  physics.setup_dofs();
  physics.update_material_deposition_orientation();
  physics.compute_inverse_mass_matrix();

  dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host> solution;
  double constexpr initial_temperature = 10;
  physics.initialize_dof_vector(initial_temperature, solution);
  physics.get_state_from_material_properties();
  std::vector<adamantine::Timer> timers(adamantine::Timing::n_timers);
  double time = 0;
  while (time < 10)
  {
    time = physics.evolve_one_time_step(time, 0.05, solution, timers);
  }

  double max = -1;
  double min = 1e4;
  if (std::is_same<MemorySpaceType, dealii::MemorySpace::Host>::value)
  {
    for (auto v : solution)
    {
      if (max < v)
        max = v;
      if (min > v)
        min = v;
    }
  }
  else
  {
    dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host>
        solution_host(solution.get_partitioner());
    solution_host.import(solution, dealii::VectorOperation::insert);
    for (auto v : solution_host)
    {
      if (max < v)
        max = v;
      if (min > v)
        min = v;
    }
  }

  BOOST_CHECK(min >= 10. && min <= 20.);
  BOOST_CHECK(max > 10. && max <= 20.);
}
