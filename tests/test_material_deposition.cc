/* Copyright (c) 2021 - 2024, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#include "MaterialStates.hh"
#define BOOST_TEST_MODULE MaterialDeposition

#include <Geometry.hh>
#include <MaterialProperty.hh>
#include <ThermalPhysics.hh>
#include <Timer.hh>
#include <material_deposition.hh>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_q.h>

#include "main.cc"

namespace utf = boost::unit_test;

BOOST_AUTO_TEST_CASE(read_material_deposition_file_2d, *utf::tolerance(1e-13))
{
  // Geometry database
  boost::property_tree::ptree geometry_database;
  geometry_database.put("material_deposition", true);
  geometry_database.put("material_deposition_file",
                        "material_deposition_2d.txt");

  std::vector<dealii::BoundingBox<2>> bounding_boxes_ref;
  bounding_boxes_ref.emplace_back(
      std::make_pair(dealii::Point<2>(0., 0.), dealii::Point<2>(1., 1.)));
  bounding_boxes_ref.emplace_back(
      std::make_pair(dealii::Point<2>(3., 2.4), dealii::Point<2>(7., 10.)));
  bounding_boxes_ref.emplace_back(
      std::make_pair(dealii::Point<2>(4., 2.4), dealii::Point<2>(7., 9.)));
  bounding_boxes_ref.emplace_back(
      std::make_pair(dealii::Point<2>(4., 3.4), dealii::Point<2>(6., 9.)));
  std::vector<double> time_ref = {1., 2., 3.4, 4.98};
  std::vector<double> cos_ref = {std::cos(0.), std::cos(3.14), std::cos(1.57),
                                 std::cos(-1.57)};
  std::vector<double> sin_ref = {std::sin(0.), std::sin(3.14), std::sin(1.57),
                                 std::sin(-1.57)};

  auto [bounding_boxes, time, deposition_cos, deposition_sin] =
      adamantine::read_material_deposition<2>(geometry_database);

  BOOST_TEST(time == time_ref);
  BOOST_TEST(deposition_cos == cos_ref);
  BOOST_TEST(deposition_sin == sin_ref);
  BOOST_TEST(bounding_boxes.size() == bounding_boxes_ref.size());
  for (unsigned int i = 0; i < bounding_boxes_ref.size(); ++i)
  {
    auto points = bounding_boxes[i].get_boundary_points();
    auto points_ref = bounding_boxes_ref[i].get_boundary_points();
    for (int d = 0; d < 2; ++d)
    {
      BOOST_TEST(points.first[d] == points_ref.first[d]);
      BOOST_TEST(points.second[d] == points_ref.second[d]);
    }
  }
}

BOOST_AUTO_TEST_CASE(read_material_deposition_file_3d, *utf::tolerance(1e-13))
{
  // Geometry database
  boost::property_tree::ptree geometry_database;
  geometry_database.put("material_deposition", true);
  geometry_database.put("material_deposition_file",
                        "material_deposition_3d.txt");

  std::vector<dealii::BoundingBox<3>> bounding_boxes_ref;
  bounding_boxes_ref.emplace_back(std::make_pair(dealii::Point<3>(0., 0., 0.),
                                                 dealii::Point<3>(1., 1., 1.)));
  bounding_boxes_ref.emplace_back(std::make_pair(
      dealii::Point<3>(3., 2.4, 1.), dealii::Point<3>(7., 10., 2.)));
  bounding_boxes_ref.emplace_back(std::make_pair(dealii::Point<3>(4., 2.4, 2.),
                                                 dealii::Point<3>(7., 9., 4.)));
  bounding_boxes_ref.emplace_back(std::make_pair(dealii::Point<3>(4., 3.4, 3.),
                                                 dealii::Point<3>(6., 9., 8.)));
  std::vector<double> time_ref = {1., 2., 3.4, 4.98};
  std::vector<double> cos_ref = {std::cos(0.), std::cos(-3.14), std::cos(1.57),
                                 std::cos(-1.57)};
  std::vector<double> sin_ref = {std::sin(0.), std::sin(-3.14), std::sin(1.57),
                                 std::sin(-1.57)};

  auto [bounding_boxes, time, deposition_cos, deposition_sin] =
      adamantine::read_material_deposition<3>(geometry_database);

  BOOST_TEST(time == time_ref);
  BOOST_TEST(deposition_cos == cos_ref);
  BOOST_TEST(deposition_sin == sin_ref);
  BOOST_TEST(bounding_boxes.size() == bounding_boxes_ref.size());
  for (unsigned int i = 0; i < bounding_boxes_ref.size(); ++i)
  {
    auto points = bounding_boxes[i].get_boundary_points();
    auto points_ref = bounding_boxes_ref[i].get_boundary_points();
    for (int d = 0; d < 2; ++d)
    {
      BOOST_TEST(points.first[d] == points_ref.first[d]);
      BOOST_TEST(points.second[d] == points_ref.second[d]);
    }
  }
}

BOOST_AUTO_TEST_CASE(get_elements_to_activate_2d)
{
  MPI_Comm communicator = MPI_COMM_WORLD;
  if (dealii::Utilities::MPI::n_mpi_processes(communicator) == 1)
  {
    // Geometry database
    boost::property_tree::ptree geometry_database;
    geometry_database.put("import_mesh", false);
    geometry_database.put("length", 12);
    geometry_database.put("length_divisions", 12);
    geometry_database.put("height", 6);
    geometry_database.put("height_divisions", 6);

    adamantine::Geometry<2> geometry(communicator, geometry_database);
    dealii::parallel::distributed::Triangulation<2> const &tria =
        geometry.get_triangulation();

    dealii::hp::FECollection<2> fe_collection;
    fe_collection.push_back(dealii::FE_Q<2>(1));
    fe_collection.push_back(dealii::FE_Nothing<2>());
    dealii::DoFHandler<2> dof_handler(tria);
    dof_handler.distribute_dofs(fe_collection);
    for (auto const &cell :
         dealii::filter_iterators(dof_handler.active_cell_iterators(),
                                  dealii::IteratorFilters::LocallyOwnedCell()))
      cell->set_active_fe_index(1);

    std::vector<dealii::BoundingBox<2>> bounding_boxes;
    bounding_boxes.emplace_back(
        std::make_pair(dealii::Point<2>(0.1, 0.1), dealii::Point<2>(0.2, 0.2)));
    bounding_boxes.emplace_back(
        std::make_pair(dealii::Point<2>(1.1, 0.1), dealii::Point<2>(2.2, 0.2)));
    bounding_boxes.emplace_back(
        std::make_pair(dealii::Point<2>(4.5, 4.5), dealii::Point<2>(5.5, 5.5)));

    auto elements_to_activate =
        adamantine::get_elements_to_activate(dof_handler, bounding_boxes);

    BOOST_TEST(elements_to_activate.size() == bounding_boxes.size());

    std::vector<std::vector<dealii::CellId>> cell_id_ref(3);
    cell_id_ref[0].push_back(dealii::CellId("0_0:"));
    cell_id_ref[1].push_back(dealii::CellId("1_0:"));
    cell_id_ref[1].push_back(dealii::CellId("4_0:"));
    cell_id_ref[2].push_back(dealii::CellId("40_0:"));
    cell_id_ref[2].push_back(dealii::CellId("41_0:"));
    cell_id_ref[2].push_back(dealii::CellId("42_0:"));
    cell_id_ref[2].push_back(dealii::CellId("43_0:"));

    for (unsigned int i = 0; i < cell_id_ref.size(); ++i)
    {
      BOOST_TEST(elements_to_activate[i].size() == cell_id_ref[i].size());
      for (unsigned int j = 0; j < cell_id_ref[i].size(); ++j)
      {
        BOOST_TEST(elements_to_activate[i][j]->id() == cell_id_ref[i][j]);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE(get_elements_to_activate_3d)
{
  MPI_Comm communicator = MPI_COMM_WORLD;
  if (dealii::Utilities::MPI::n_mpi_processes(communicator) == 1)
  {
    // Geometry database
    boost::property_tree::ptree geometry_database;
    geometry_database.put("import_mesh", false);
    geometry_database.put("length", 12);
    geometry_database.put("length_divisions", 12);
    geometry_database.put("width", 6);
    geometry_database.put("width_divisions", 6);
    geometry_database.put("height", 6);
    geometry_database.put("height_divisions", 6);

    adamantine::Geometry<3> geometry(communicator, geometry_database);
    dealii::parallel::distributed::Triangulation<3> const &tria =
        geometry.get_triangulation();

    dealii::hp::FECollection<3> fe_collection;
    fe_collection.push_back(dealii::FE_Q<3>(1));
    fe_collection.push_back(dealii::FE_Nothing<3>());
    dealii::DoFHandler<3> dof_handler(tria);
    dof_handler.distribute_dofs(fe_collection);
    for (auto const &cell :
         dealii::filter_iterators(dof_handler.active_cell_iterators(),
                                  dealii::IteratorFilters::LocallyOwnedCell()))
      cell->set_active_fe_index(1);

    std::vector<dealii::BoundingBox<3>> bounding_boxes;
    bounding_boxes.emplace_back(std::make_pair(
        dealii::Point<3>(0.1, 0.1, 0.1), dealii::Point<3>(0.2, 0.2, 0.2)));
    bounding_boxes.emplace_back(std::make_pair(
        dealii::Point<3>(1.1, 0.1, 0.1), dealii::Point<3>(2.2, 0.2, 0.2)));
    bounding_boxes.emplace_back(std::make_pair(
        dealii::Point<3>(4.5, 4.5, 4.5), dealii::Point<3>(5.5, 5.5, 5.5)));

    auto elements_to_activate =
        adamantine::get_elements_to_activate(dof_handler, bounding_boxes);

    BOOST_TEST(elements_to_activate.size() == bounding_boxes.size());

    std::vector<std::vector<dealii::CellId>> cell_id_ref(3);
    cell_id_ref[0].push_back(dealii::CellId("0_0:"));
    cell_id_ref[1].push_back(dealii::CellId("1_0:"));
    cell_id_ref[1].push_back(dealii::CellId("8_0:"));
    cell_id_ref[2].push_back(dealii::CellId("272_0:"));
    cell_id_ref[2].push_back(dealii::CellId("273_0:"));
    cell_id_ref[2].push_back(dealii::CellId("276_0:"));
    cell_id_ref[2].push_back(dealii::CellId("277_0:"));
    cell_id_ref[2].push_back(dealii::CellId("274_0:"));
    cell_id_ref[2].push_back(dealii::CellId("275_0:"));
    cell_id_ref[2].push_back(dealii::CellId("278_0:"));
    cell_id_ref[2].push_back(dealii::CellId("279_0:"));

    for (unsigned int i = 0; i < cell_id_ref.size(); ++i)
    {
      BOOST_TEST(elements_to_activate[i].size() == cell_id_ref[i].size());
      for (unsigned int j = 0; j < cell_id_ref[i].size(); ++j)
      {
        BOOST_TEST(elements_to_activate[i][j]->id() == cell_id_ref[i][j]);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE(material_deposition)
{
  int constexpr dim = 3;
  MPI_Comm communicator = MPI_COMM_WORLD;

  double const initial_temperature = 300.;

  boost::property_tree::ptree database;
  // Geometry database
  database.put("geometry.import_mesh", false);
  database.put("geometry.length", 10);
  database.put("geometry.length_divisions", 10);
  database.put("geometry.width", 10);
  database.put("geometry.width_divisions", 10);
  database.put("geometry.height", 10);
  database.put("geometry.height_divisions", 10);
  database.put("geometry.material_height", 6.);
  database.put("geometry.material_deposition", true);
  database.put("geometry.material_deposition_file",
               "material_path_test_material_deposition.txt");
  // Build Geometry
  boost::property_tree::ptree geometry_database =
      database.get_child("geometry");
  adamantine::Geometry<dim> geometry(communicator, geometry_database);

  // MaterialProperty database
  database.put("materials.property_format", "polynomial");
  database.put("materials.initial_temperature", initial_temperature);
  database.put("materials.new_material_temperature", initial_temperature);
  database.put("materials.n_materials", 1);
  database.put("materials.material_0.solid.density", 1.);
  database.put("materials.material_0.liquid.density", 1.);
  database.put("materials.material_0.solid.specific_heat", 1.);
  database.put("materials.material_0.liquid.specific_heat", 1.);
  database.put("materials.material_0.solid.thermal_conductivity_x", 1.);
  database.put("materials.material_0.solid.thermal_conductivity_z", 1.);
  database.put("materials.material_0.liquid.thermal_conductivity_x", 1.);
  database.put("materials.material_0.liquid.thermal_conductivity_z", 1.);
  // Build MaterialProperty
  boost::property_tree::ptree material_property_database =
      database.get_child("materials");
  adamantine::MaterialProperty<dim, 1, adamantine::SolidLiquidPowder,
                               dealii::MemorySpace::Host>
      material_properties(communicator, geometry.get_triangulation(),
                          material_property_database);

  // Source database
  database.put("sources.n_beams", 0);
  // Time-stepping database
  database.put("time_stepping.method", "forward_euler");
  // Boundary database
  database.put("boundary.type", "adiabatic");

  // Build ThermalPhysics
  adamantine::ThermalPhysics<dim, 1, dim, adamantine::SolidLiquidPowder,
                             dealii::MemorySpace::Host, dealii::QGauss<1>>
      thermal_physics(communicator, database, geometry, material_properties);
  thermal_physics.setup();
  auto &dof_handler = thermal_physics.get_dof_handler();

  auto [material_deposition_boxes, deposition_times, deposition_cos,
        deposition_sin] =
      adamantine::read_material_deposition<dim>(geometry_database);
  dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host> solution;
  thermal_physics.initialize_dof_vector(0., solution);
  std::vector<adamantine::Timer> timers(adamantine::Timing::n_timers);
  std::vector<unsigned int> n_cells_ref = {610, 620, 630, 650, 650,
                                           660, 670, 680, 720, 720};
  double const time_step = 0.1;
  double time =
      thermal_physics.evolve_one_time_step(0., time_step, solution, timers);
  double const eps = time_step / 1e12;
  // The build is too slow in debug mode when using sanitizer. In that case
  // reduce the size of the loop
#ifdef ADAMANTINE_DEBUG
  unsigned int const i_max = 2;
#else
  unsigned int const i_max = 10;
#endif
  for (unsigned int i = 0; i < i_max; ++i)
  {
    auto activation_start =
        std::lower_bound(deposition_times.begin(), deposition_times.end(),
                         time - eps) -
        deposition_times.begin();
    auto activation_end =
        std::lower_bound(deposition_times.begin(), deposition_times.end(),
                         time + time_step - eps) -
        deposition_times.begin();
    if (activation_start < activation_end)
    {
      auto elements_to_activate = adamantine::get_elements_to_activate(
          dof_handler, material_deposition_boxes);

      std::vector<bool> has_melted(deposition_cos.size(), false);

      thermal_physics.add_material(
          elements_to_activate, deposition_cos, deposition_sin, has_melted,
          activation_start, activation_end, initial_temperature, solution);
    }

    time =
        thermal_physics.evolve_one_time_step(time, time_step, solution, timers);

    unsigned int n_cells = 0;
    for (auto const &cell : dealii::filter_iterators(
             dof_handler.active_cell_iterators(),
             dealii::IteratorFilters::LocallyOwnedCell(),
             dealii::IteratorFilters::ActiveFEIndexEqualTo(0)))
    {
      (void)cell;
      ++n_cells;
    }
    BOOST_TEST(dealii::Utilities::MPI::sum(n_cells, communicator) ==
               n_cells_ref[i]);
  }
}

std::pair<Kokkos::View<adamantine::ScanPathSegment *, Kokkos::HostSpace>,
          adamantine::ScanPath>
create_scan_path(std::string file_name)
{
  std::vector<adamantine::ScanPathSegment> scan_path_segments =
      adamantine::ScanPath::read_file(file_name, "segment");
  Kokkos::View<adamantine::ScanPathSegment *, Kokkos::HostSpace>
      scan_path_segments_view("scan_path_segments", scan_path_segments.size());
  Kokkos::deep_copy(
      scan_path_segments_view,
      Kokkos::View<adamantine::ScanPathSegment *, Kokkos::HostSpace>{
          scan_path_segments.data(), scan_path_segments.size()});
  return {scan_path_segments_view, {scan_path_segments_view}};
}

BOOST_AUTO_TEST_CASE(deposition_from_scan_path_2d, *utf::tolerance(1e-13))
{
  auto [scan_path_segments, scan_path] = create_scan_path("scan_path.txt");

  boost::property_tree::ptree database;
  database.put("deposition_length", 0.0005);
  database.put("deposition_height", 0.1);
  database.put("deposition_width", 0.1);
  database.put("deposition_lead_time", 0.0);

  auto [bounding_boxes, deposition_times, deposition_cos, deposition_sin] =
      adamantine::deposition_along_scan_path<2>(database, scan_path);

  // Check the first and last boxes
  BOOST_TEST(bounding_boxes.at(0).get_boundary_points().first(0) == -0.00025);
  BOOST_TEST(bounding_boxes.at(0).get_boundary_points().first(1) == 0.0);
  BOOST_TEST(bounding_boxes.at(0).get_boundary_points().second(0) == 0.00025);
  BOOST_TEST(bounding_boxes.at(0).get_boundary_points().second(1) == 0.1);

  BOOST_TEST(bounding_boxes.at(4).get_boundary_points().first(0) ==
             (0.002 - 0.00025));
  BOOST_TEST(bounding_boxes.at(4).get_boundary_points().first(1) == 0.0);
  BOOST_TEST(bounding_boxes.at(4).get_boundary_points().second(0) ==
             (0.002 + 0.00025));
  BOOST_TEST(bounding_boxes.at(4).get_boundary_points().second(1) == 0.1);

  // Check the times
  BOOST_TEST(deposition_times.at(0) == 1.0e-6);
  BOOST_TEST(deposition_times.at(1) == 6.26e-4);
  BOOST_TEST(deposition_times.at(2) == 1.251e-3);
  BOOST_TEST(deposition_times.at(3) == 1.876e-3);
  BOOST_TEST(deposition_times.at(4) == 2.501e-3);

  // Check the deposition cosines
  BOOST_TEST(deposition_cos.at(0) == 1.);
  BOOST_TEST(deposition_cos.at(1) == 1.);
  BOOST_TEST(deposition_cos.at(2) == 1.);
  BOOST_TEST(deposition_cos.at(3) == 1.);
  BOOST_TEST(deposition_cos.at(4) == 1.);

  // Check the deposition sines
  BOOST_TEST(deposition_sin.at(0) == 0.);
  BOOST_TEST(deposition_sin.at(1) == 0.);
  BOOST_TEST(deposition_sin.at(2) == 0.);
  BOOST_TEST(deposition_sin.at(3) == 0.);
  BOOST_TEST(deposition_sin.at(4) == 0.);
}

BOOST_AUTO_TEST_CASE(deposition_from_scan_path_3d, *utf::tolerance(1e-13))
{
  auto [scan_path_segments, scan_path] = create_scan_path("scan_path.txt");

  boost::property_tree::ptree database;
  database.put("deposition_length", 0.0005);
  database.put("deposition_height", 0.1);
  database.put("deposition_width", 0.1);
  database.put("deposition_lead_time", 0.0);

  auto [bounding_boxes, deposition_times, deposition_cos, deposition_sin] =
      adamantine::deposition_along_scan_path<3>(database, scan_path);

  // Check the first and last boxes
  BOOST_TEST(bounding_boxes.at(0).get_boundary_points().first(0) == -0.00025);
  BOOST_TEST(bounding_boxes.at(0).get_boundary_points().first(1) == 0.05);
  BOOST_TEST(bounding_boxes.at(0).get_boundary_points().first(2) == 0.0);
  BOOST_TEST(bounding_boxes.at(0).get_boundary_points().second(0) == 0.00025);
  BOOST_TEST(bounding_boxes.at(0).get_boundary_points().second(1) == 0.15);
  BOOST_TEST(bounding_boxes.at(0).get_boundary_points().second(2) == 0.1);

  BOOST_TEST(bounding_boxes.at(4).get_boundary_points().first(0) ==
             (0.002 - 0.00025));
  BOOST_TEST(bounding_boxes.at(4).get_boundary_points().first(1) == 0.05);
  BOOST_TEST(bounding_boxes.at(4).get_boundary_points().first(2) == 0.0);
  BOOST_TEST(bounding_boxes.at(4).get_boundary_points().second(0) ==
             (0.002 + 0.00025));
  BOOST_TEST(bounding_boxes.at(4).get_boundary_points().second(1) == 0.15);
  BOOST_TEST(bounding_boxes.at(4).get_boundary_points().second(2) == 0.1);

  // Check the times
  BOOST_TEST(deposition_times.at(0) == 1.0e-6);
  BOOST_TEST(deposition_times.at(1) == 6.26e-4);
  BOOST_TEST(deposition_times.at(2) == 1.251e-3);
  BOOST_TEST(deposition_times.at(3) == 1.876e-3);
  BOOST_TEST(deposition_times.at(4) == 2.501e-3);

  // Check the deposition cosines
  BOOST_TEST(deposition_cos.at(0) == 1.);
  BOOST_TEST(deposition_cos.at(1) == 1.);
  BOOST_TEST(deposition_cos.at(2) == 1.);
  BOOST_TEST(deposition_cos.at(3) == 1.);
  BOOST_TEST(deposition_cos.at(4) == 1.);

  // Check the deposition sines
  BOOST_TEST(deposition_sin.at(0) == 0.);
  BOOST_TEST(deposition_sin.at(1) == 0.);
  BOOST_TEST(deposition_sin.at(2) == 0.);
  BOOST_TEST(deposition_sin.at(3) == 0.);
  BOOST_TEST(deposition_sin.at(4) == 0.);
}

BOOST_AUTO_TEST_CASE(deposition_from_L_scan_path_3d, *utf::tolerance(1e-13))
{
  auto [scan_path_segments, scan_path] = create_scan_path("scan_path_L.txt");

  boost::property_tree::ptree database;
  database.put("deposition_length", 0.0005);
  database.put("deposition_height", 0.1);
  database.put("deposition_width", 0.1);
  database.put("deposition_lead_time", 0.0);

  auto [bounding_boxes, deposition_times, deposition_cos, deposition_sin] =
      adamantine::deposition_along_scan_path<3>(database, scan_path);

  // Check the first and last boxes for each segment
  BOOST_TEST(bounding_boxes.at(0).get_boundary_points().first(0) == -0.00025);
  BOOST_TEST(bounding_boxes.at(0).get_boundary_points().first(1) == -0.05);
  BOOST_TEST(bounding_boxes.at(0).get_boundary_points().first(2) == 0.0);
  BOOST_TEST(bounding_boxes.at(0).get_boundary_points().second(0) == 0.00025);
  BOOST_TEST(bounding_boxes.at(0).get_boundary_points().second(1) == 0.05);
  BOOST_TEST(bounding_boxes.at(0).get_boundary_points().second(2) == 0.1);

  BOOST_TEST(bounding_boxes.at(4).get_boundary_points().first(0) ==
             (0.002 - 0.00025));
  BOOST_TEST(bounding_boxes.at(4).get_boundary_points().first(1) == -0.05);
  BOOST_TEST(bounding_boxes.at(4).get_boundary_points().first(2) == 0.0);
  BOOST_TEST(bounding_boxes.at(4).get_boundary_points().second(0) ==
             (0.002 + 0.00025));
  BOOST_TEST(bounding_boxes.at(4).get_boundary_points().second(1) == 0.05);
  BOOST_TEST(bounding_boxes.at(4).get_boundary_points().second(2) == 0.1);

  BOOST_TEST(bounding_boxes.at(5).get_boundary_points().first(0) ==
             (0.002 - 0.05));
  BOOST_TEST(bounding_boxes.at(5).get_boundary_points().first(1) == -0.00025);
  BOOST_TEST(bounding_boxes.at(5).get_boundary_points().first(2) == 0.0);
  BOOST_TEST(bounding_boxes.at(5).get_boundary_points().second(0) ==
             (0.002 + 0.05));
  BOOST_TEST(bounding_boxes.at(5).get_boundary_points().second(1) == 0.00025);
  BOOST_TEST(bounding_boxes.at(5).get_boundary_points().second(2) == 0.1);

  BOOST_TEST(bounding_boxes.at(9).get_boundary_points().first(0) ==
             (0.002 - 0.05));
  BOOST_TEST(bounding_boxes.at(9).get_boundary_points().first(1) == 0.00175);
  BOOST_TEST(bounding_boxes.at(9).get_boundary_points().first(2) == 0.0);
  BOOST_TEST(bounding_boxes.at(9).get_boundary_points().second(0) ==
             0.002 + 0.05);
  BOOST_TEST(bounding_boxes.at(9).get_boundary_points().second(1) == 0.00205);
  BOOST_TEST(bounding_boxes.at(9).get_boundary_points().second(2) == 0.1);

  // Check the times
  BOOST_TEST(deposition_times.at(0) == 1.0e-6);
  BOOST_TEST(deposition_times.at(1) == 6.26e-4);
  BOOST_TEST(deposition_times.at(2) == 1.251e-3);
  BOOST_TEST(deposition_times.at(3) == 1.876e-3);
  BOOST_TEST(deposition_times.at(4) == 2.501e-3);
  BOOST_TEST(deposition_times.at(5) == 2.501e-3);
  BOOST_TEST(deposition_times.at(6) == 3.126e-3);
  BOOST_TEST(deposition_times.at(7) == 3.751e-3);
  BOOST_TEST(deposition_times.at(8) == 4.376e-3);
  BOOST_TEST(deposition_times.at(9) == 4.876e-3);

  // Check the deposition cosines
  BOOST_TEST(deposition_cos.at(0) == 1.);
  BOOST_TEST(deposition_cos.at(1) == 1.);
  BOOST_TEST(deposition_cos.at(2) == 1.);
  BOOST_TEST(deposition_cos.at(3) == 1.);
  BOOST_TEST(deposition_cos.at(4) == 1.);
  BOOST_TEST(deposition_cos.at(5) == 0.);
  BOOST_TEST(deposition_cos.at(6) == 0.);
  BOOST_TEST(deposition_cos.at(7) == 0.);
  BOOST_TEST(deposition_cos.at(8) == 0.);
  BOOST_TEST(deposition_cos.at(9) == 0.);

  // Check the deposition sines
  BOOST_TEST(deposition_sin.at(0) == 0.);
  BOOST_TEST(deposition_sin.at(1) == 0.);
  BOOST_TEST(deposition_sin.at(2) == 0.);
  BOOST_TEST(deposition_sin.at(3) == 0.);
  BOOST_TEST(deposition_sin.at(4) == 0.);
  BOOST_TEST(deposition_sin.at(5) == 1.);
  BOOST_TEST(deposition_sin.at(6) == 1.);
  BOOST_TEST(deposition_sin.at(7) == 1.);
  BOOST_TEST(deposition_sin.at(8) == 1.);
  BOOST_TEST(deposition_sin.at(8) == 1.);
}

BOOST_AUTO_TEST_CASE(deposition_from_diagonal_scan_path_3d,
                     *utf::tolerance(1e-10))
{
  auto [scan_path_segments, scan_path] =
      create_scan_path("scan_path_diagonal.txt");

  boost::property_tree::ptree database;
  database.put("deposition_length", 0.0005);
  database.put("deposition_height", 0.1);
  database.put("deposition_width", 0.1);
  database.put("deposition_lead_time", 0.0);

  auto [bounding_boxes, deposition_times, deposition_cos, deposition_sin] =
      adamantine::deposition_along_scan_path<3>(database, scan_path);

  BOOST_TEST(bounding_boxes.at(0).get_boundary_points().first(0) ==
             -0.00022360679775);
  BOOST_TEST(bounding_boxes.at(0).get_boundary_points().first(1) ==
             -0.050111803398874992);
  BOOST_TEST(bounding_boxes.at(0).get_boundary_points().first(2) == 0);
  BOOST_TEST(bounding_boxes.at(0).get_boundary_points().second(0) ==
             0.00022360679775);
  BOOST_TEST(bounding_boxes.at(0).get_boundary_points().second(1) ==
             0.050111803398874992);
  BOOST_TEST(bounding_boxes.at(0).get_boundary_points().second(2) == 0.1);

  BOOST_TEST(bounding_boxes.at(5).get_boundary_points().first(0) ==
             0.00201246117975);
  BOOST_TEST(bounding_boxes.at(5).get_boundary_points().first(1) ==
             -0.0489937694101251);
  BOOST_TEST(bounding_boxes.at(5).get_boundary_points().first(2) == 0.);
  BOOST_TEST(bounding_boxes.at(5).get_boundary_points().second(0) ==
             0.00222360679775);
  BOOST_TEST(bounding_boxes.at(5).get_boundary_points().second(1) ==
             0.051111803398874993);
  BOOST_TEST(bounding_boxes.at(5).get_boundary_points().second(2) == 0.1);

  // Check the times
  BOOST_TEST(deposition_times.at(0) == 1.0e-6);
  BOOST_TEST(deposition_times.at(1) == 6.26e-4);
  BOOST_TEST(deposition_times.at(2) == 1.251e-3);
  BOOST_TEST(deposition_times.at(3) == 1.876e-3);
  BOOST_TEST(deposition_times.at(4) == 2.501e-3);
  BOOST_TEST(deposition_times.at(5) == 0.002961042485937369);

  // Check the deposition cosines and sines
  for (unsigned int i = 0; i < 6; ++i)
  {
    BOOST_TEST(deposition_cos.at(0) == 2. / std::sqrt(5.));
    BOOST_TEST(deposition_sin.at(0) == 1. / std::sqrt(5.));
  }
}

BOOST_AUTO_TEST_CASE(merge_multiple_deposition_paths, *utf::tolerance(1e-10))
{
  dealii::BoundingBox<2> box_0(
      std::make_pair(dealii::Point<2>(0., 0.), dealii::Point<2>(1., 1.)));
  dealii::BoundingBox<2> box_1(
      std::make_pair(dealii::Point<2>(1., 1.), dealii::Point<2>(2., 2.)));
  dealii::BoundingBox<2> box_2(
      std::make_pair(dealii::Point<2>(2., 2.), dealii::Point<2>(3., 3.)));
  dealii::BoundingBox<2> box_3(
      std::make_pair(dealii::Point<2>(3., 3.), dealii::Point<2>(4., 4.)));

  std::vector<dealii::BoundingBox<2>> bb_p1 = {box_0, box_2};
  std::vector<dealii::BoundingBox<2>> bb_p2 = {box_1, box_3};
  std::vector<dealii::BoundingBox<2>> bb_ref = {box_0, box_1, box_2, box_3};

  std::vector<double> time_p1 = {0., 2.};
  std::vector<double> time_p2 = {1., 3.};
  std::vector<double> time_ref = {0., 1., 2., 3.};

  std::vector<double> cos_p1 = {0., 0.2};
  std::vector<double> cos_p2 = {0.1, 0.3};
  std::vector<double> cos_ref = {0., 0.1, 0.2, 0.3};

  std::vector<double> sin_p1 = {-0., -0.2};
  std::vector<double> sin_p2 = {-0.1, -0.3};
  std::vector<double> sin_ref = {-0., -0.1, -0.2, -0.3};

  std::vector<
      std::tuple<std::vector<dealii::BoundingBox<2>>, std::vector<double>,
                 std::vector<double>, std::vector<double>>>
      multiple_paths;

  multiple_paths.push_back({bb_p1, time_p1, cos_p1, sin_p1});
  multiple_paths.push_back({bb_p2, time_p2, cos_p2, sin_p2});

  auto [bounding_boxes, time, cos, sin] =
      adamantine::merge_deposition_paths(multiple_paths);

  for (unsigned int i = 0; i < 4; ++i)
  {
    BOOST_TEST(bounding_boxes[i].get_boundary_points().first(0) ==
               bb_ref[i].get_boundary_points().first(0));
    BOOST_TEST(bounding_boxes[i].get_boundary_points().first(1) ==
               bb_ref[i].get_boundary_points().first(1));
    BOOST_TEST(bounding_boxes[i].get_boundary_points().second(0) ==
               bb_ref[i].get_boundary_points().second(0));
    BOOST_TEST(bounding_boxes[i].get_boundary_points().second(1) ==
               bb_ref[i].get_boundary_points().second(1));

    BOOST_TEST(time[i] == time_ref[i]);

    BOOST_TEST(cos[i] == cos_ref[i]);

    BOOST_TEST(sin[i] == sin_ref[i]);
  }
}
