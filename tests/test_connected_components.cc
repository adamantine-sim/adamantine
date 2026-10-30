/* SPDX-FileCopyrightText: Copyright (c) 2025, the adamantine authors.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include "connected_components.hh"

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/geometry_info.h>
#include <deal.II/base/types.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/grid/grid_generator.h>

#include <set>

#include "main.cc"

unsigned int z_layer_from_point(const dealii::Point<3> &p,
                                const unsigned int n_layers)
{
  const double z = p[2];
  return std::min(n_layers - 1U,
                  static_cast<unsigned int>(std::floor(z * n_layers)));
}

dealii::types::boundary_id
find_top_boundary_id(const dealii::Triangulation<3> &triangulation,
                     const MPI_Comm comm)
{
  dealii::types::boundary_id local_id = dealii::numbers::invalid_boundary_id;
  for (const auto &cell : triangulation.active_cell_iterators())
    if (!cell->is_artificial())
      for (unsigned int f = 0; f < dealii::GeometryInfo<3>::faces_per_cell; ++f)
        if (cell->at_boundary(f) &&
            std::abs(cell->face(f)->center()[2] - 1.0) < 1e-12)
          local_id = cell->face(f)->boundary_id();

  const auto id = dealii::Utilities::MPI::min(local_id, comm);
  BOOST_REQUIRE(id != dealii::numbers::invalid_boundary_id);
  return id;
}

BOOST_AUTO_TEST_CASE(connected_components)
{
  MPI_Comm comm = MPI_COMM_WORLD;

  // --------------------------------------------------------------------------
  // Build a nx x nx x 2*nx cube on [0,1]^3.
  // With colorize=true, deal.II assigns distinct boundary ids to the 6 faces.
  // --------------------------------------------------------------------------
  dealii::parallel::distributed::Triangulation<3> triangulation(comm);

  constexpr unsigned int nx = 64;
  constexpr unsigned int ny = nx;
  constexpr unsigned int nz = 2 * nx;
  dealii::GridGenerator::subdivided_hyper_rectangle(
      triangulation, {nx, ny, nz}, dealii::Point<3>(0.0, 0.0, 0.0),
      dealii::Point<3>(1.0, 1.0, 1.0),
      /*colorize=*/true);

  dealii::DoFHandler<3> dof_handler(triangulation);

  // One FE per z-layer.
  dealii::hp::FECollection<3> fe_collection;
  dealii::FE_DGQ<3> fe(0);
  fe_collection.push_back(fe); // fe_index 0
  fe_collection.push_back(fe); // fe_index 1
  fe_collection.push_back(fe); // fe_index 2
  fe_collection.push_back(fe); // fe_index 3

  // --------------------------------------------------------------------------
  // Assign active FE indices by z-layer, on locally owned cells.
  // Then distribute DoFs so ghost cells get consistent active_fe_index() info.
  // --------------------------------------------------------------------------
  for (const auto &cell : dof_handler.active_cell_iterators())
    if (cell->is_locally_owned())
      cell->set_active_fe_index(
          z_layer_from_point(cell->center(), fe_collection.size()));

  dof_handler.distribute_dofs(fe_collection);

  // --------------------------------------------------------------------------
  // Detect the boundary id of the top face z=1 from the triangulation itself.
  // This avoids hard-coding the numeric boundary id.
  // --------------------------------------------------------------------------
  dealii::types::boundary_id local_top_boundary_id =
      dealii::numbers::invalid_boundary_id;

  for (const auto &cell : triangulation.active_cell_iterators())
    if (!cell->is_artificial())
      for (unsigned int f = 0; f < dealii::GeometryInfo<3>::faces_per_cell; ++f)
        if (cell->at_boundary(f) &&
            std::abs(cell->face(f)->center()[2] - 1.0) < 1e-12)
          local_top_boundary_id = cell->face(f)->boundary_id();

  const unsigned int top_boundary_id_int = dealii::Utilities::MPI::min(
      static_cast<unsigned int>(local_top_boundary_id), comm);

  if (top_boundary_id_int ==
      static_cast<unsigned int>(dealii::numbers::invalid_boundary_id))
    throw std::runtime_error("Could not determine top boundary id.");

  const dealii::types::boundary_id top_boundary_id =
      static_cast<dealii::types::boundary_id>(top_boundary_id_int);

  // --------------------------------------------------------------------------
  // Run the connected-components routine.
  // --------------------------------------------------------------------------

  for (unsigned int target_fe_index = 0; target_fe_index < fe_collection.size();
       ++target_fe_index)
  {
    auto result = adamantine::ConnectedComponents::find_components(
        dof_handler, {top_boundary_id}, target_fe_index, comm);

    // --------------------------------------------------------------------------
    // Consistency checks.
    // --------------------------------------------------------------------------
    BOOST_TEST(result.size() == 1);
    const unsigned int n_layers =
        static_cast<unsigned int>(fe_collection.size());
    for (const auto &cell : result[0]._locally_owned_cells)
    {
      BOOST_TEST(cell->active_fe_index() == target_fe_index);
      BOOST_TEST(z_layer_from_point(cell->center(), n_layers) ==
                 target_fe_index);
    }

    unsigned int n_expected_local_cells = 0;
    for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (cell->is_locally_owned() &&
          cell->active_fe_index() == target_fe_index)
        ++n_expected_local_cells;
    }
    BOOST_TEST(result[0]._locally_owned_cells.size() == n_expected_local_cells);

    const bool expected_touches_top = (target_fe_index == n_layers - 1);
    BOOST_TEST(result[0]._touches_target_boundary == expected_touches_top);
  }
}

BOOST_AUTO_TEST_CASE(disconnected_components)
{
  MPI_Comm comm = MPI_COMM_WORLD;

  // Two disconnected regions with FE index 0:
  //
  //   x = 0              x = 1
  //   +-------+---+-------+
  //   |   0   | 1 |   1   |  z > 0.5
  //   |   0   | 1 |   1   |
  //   |   0   | 1 |   0   |  z < 0.5
  //   +-------+---+-------+
  //
  // The left component touches the top boundary; the right one does not.
  // FE index 1 separates the two components.

  dealii::parallel::distributed::Triangulation<3> triangulation(comm);

  constexpr unsigned int nx = 6;
  constexpr unsigned int ny = 2;
  constexpr unsigned int nz = 4;

  dealii::GridGenerator::subdivided_hyper_rectangle(
      triangulation, {nx, ny, nz}, dealii::Point<3>(0., 0., 0.),
      dealii::Point<3>(1., 1., 1.), /*colorize=*/true);

  dealii::DoFHandler<3> dof_handler(triangulation);

  dealii::hp::FECollection<3> fe_collection;
  fe_collection.push_back(dealii::FE_DGQ<3>(0)); // FE index 0
  fe_collection.push_back(dealii::FE_DGQ<3>(0)); // FE index 1

  for (const auto &cell : dof_handler.active_cell_iterators())
    if (cell->is_locally_owned())
    {
      const auto center = cell->center();

      const bool left_region = center[0] < 1.0 / 3.0;
      const bool right_region = center[0] > 2.0 / 3.0 && center[2] < 0.5;

      cell->set_active_fe_index((left_region || right_region) ? 0U : 1U);
    }

  dof_handler.distribute_dofs(fe_collection);

  // Find the top boundary ID without assuming a particular numeric ID.
  dealii::types::boundary_id top_boundary_id =
      find_top_boundary_id(triangulation, comm);

  const auto result = adamantine::ConnectedComponents::find_components(
      dof_handler, {top_boundary_id}, 0, comm);

  // The two FE-0 regions must remain separate components.
  BOOST_REQUIRE_EQUAL(result.size(), 2);

  unsigned int n_touching_top = 0;
  unsigned int n_result_local_cells = 0;

  for (const auto &component : result)
  {
    if (component._touches_target_boundary)
      ++n_touching_top;

    for (const auto &cell : component._locally_owned_cells)
    {
      ++n_result_local_cells;

      BOOST_TEST(cell->active_fe_index() == 0U);

      // A component must not contain cells from both regions.
      const bool is_left_region = cell->center()[0] < 1.0 / 3.0;
      BOOST_TEST(is_left_region == component._touches_target_boundary);
    }
  }

  // Exactly one component touches the top boundary.
  BOOST_TEST(n_touching_top == 1U);

  // Every locally owned FE-0 cell must occur in the result.
  unsigned int n_expected_local_cells = 0;
  for (const auto &cell : dof_handler.active_cell_iterators())
    if (cell->is_locally_owned() && cell->active_fe_index() == 0)
      ++n_expected_local_cells;

  BOOST_TEST(n_result_local_cells == n_expected_local_cells);
}

// Components must be connected across hanging-node (coarse/fine) faces.
//
// The left half (x < 0.5) is refined (twice near x = 0, once elsewhere), the
// right half stays coarse. Case 1: all cells have FE index 0, so the single
// component spans coarse/fine interfaces. Case 2: a FE index 1 slab separates
// a refined left region from a coarse right region, which only touches the top
// boundary for the left one.
BOOST_AUTO_TEST_CASE(adaptive_mesh_refinement)
{
  MPI_Comm comm = MPI_COMM_WORLD;

  dealii::parallel::distributed::Triangulation<3> triangulation(comm);
  dealii::GridGenerator::subdivided_hyper_rectangle(
      triangulation, {4, 4, 4}, dealii::Point<3>(0., 0., 0.),
      dealii::Point<3>(1., 1., 1.), /*colorize=*/true);

  for (unsigned int level = 0; level < 2; ++level)
  {
    for (const auto &cell : triangulation.active_cell_iterators())
      if (cell->is_locally_owned() &&
          cell->center()[0] < (level == 0 ? 0.5 : 0.25))
        cell->set_refine_flag();
    triangulation.execute_coarsening_and_refinement();
  }

  const auto top_boundary_id = find_top_boundary_id(triangulation, comm);

  dealii::hp::FECollection<3> fe_collection;
  fe_collection.push_back(dealii::FE_DGQ<3>(0));
  fe_collection.push_back(dealii::FE_DGQ<3>(0));

  // The mesh must actually contain hanging nodes for this test to be useful.
  unsigned int local_n_coarser_faces = 0;
  for (const auto &cell : triangulation.active_cell_iterators())
    if (cell->is_locally_owned())
      for (unsigned int f = 0; f < dealii::GeometryInfo<3>::faces_per_cell; ++f)
        if (!cell->at_boundary(f) && cell->neighbor_is_coarser(f))
          ++local_n_coarser_faces;
  BOOST_TEST(dealii::Utilities::MPI::sum(local_n_coarser_faces, comm) > 0);

  // Case 1: a single FE index.
  {
    dealii::DoFHandler<3> dof_handler(triangulation);
    for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
        cell->set_active_fe_index(0);
    dof_handler.distribute_dofs(fe_collection);

    const auto result = adamantine::ConnectedComponents::find_components(
        dof_handler, {top_boundary_id}, 0, comm);

    BOOST_TEST(result.size() == 1);
    BOOST_TEST(result[0]._touches_target_boundary);

    unsigned int n_result_cells = 0;
    for (const auto &c : result)
      n_result_cells += c._locally_owned_cells.size();
    unsigned int n_local_cells = 0;
    for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
        ++n_local_cells;
    BOOST_TEST(n_result_cells == n_local_cells);
  }

  // Case 2: FE index 1 slab for 0.5 <= x < 0.75; FE 0 on both sides. The right
  // FE 0 region (x >= 0.75) only extends up to z < 0.5.
  {
    dealii::DoFHandler<3> dof_handler(triangulation);
    for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
      {
        const auto center = cell->center();
        const bool left = center[0] < 0.5;
        const bool right = center[0] > 0.75 && center[2] < 0.5;
        cell->set_active_fe_index((left || right) ? 0U : 1U);
      }
    dof_handler.distribute_dofs(fe_collection);

    const auto result = adamantine::ConnectedComponents::find_components(
        dof_handler, {top_boundary_id}, 0, comm);

    BOOST_TEST(result.size() == 2);

    unsigned int n_touching_top = 0;
    unsigned int n_result_cells = 0;
    for (const auto &component : result)
    {
      if (component._touches_target_boundary)
        ++n_touching_top;
      for (const auto &cell : component._locally_owned_cells)
      {
        ++n_result_cells;
        BOOST_TEST(cell->active_fe_index() == 0);
        const bool is_left = cell->center()[0] < 0.5;
        BOOST_TEST(is_left == component._touches_target_boundary);
      }
    }
    BOOST_TEST(n_touching_top == 1);

    unsigned int n_expected = 0;
    for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_locally_owned() && cell->active_fe_index() == 0)
        ++n_expected;
    BOOST_TEST(n_result_cells == n_expected);
  }
}

// Components that share no face but touch at >= 3 non-collinear vertices must
// be merged. The mesh is 3x3x1 cells; (i,j) are the cell indices in x and y.
// Diagonal neighbors share an edge, i.e. two collinear vertices.
BOOST_AUTO_TEST_CASE(vertex_contacts)
{
  MPI_Comm comm = MPI_COMM_WORLD;

  dealii::parallel::distributed::Triangulation<3> triangulation(comm);
  dealii::GridGenerator::subdivided_hyper_rectangle(
      triangulation, {3, 3, 1}, dealii::Point<3>(0., 0., 0.),
      dealii::Point<3>(1., 1., 1.), /*colorize=*/true);
  const auto top_boundary_id = find_top_boundary_id(triangulation, comm);

  dealii::hp::FECollection<3> fe_collection;
  fe_collection.push_back(dealii::FE_DGQ<3>(0));
  fe_collection.push_back(dealii::FE_DGQ<3>(0));

  auto run = [&](const std::set<std::pair<int, int>> &fe0_cells,
                 const unsigned int expected_components)
  {
    dealii::DoFHandler<3> dof_handler(triangulation);
    for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
      {
        const std::pair<int, int> ij(
            static_cast<int>(std::floor(cell->center()[0] * 3.)),
            static_cast<int>(std::floor(cell->center()[1] * 3.)));
        cell->set_active_fe_index(fe0_cells.count(ij) ? 0U : 1U);
      }
    dof_handler.distribute_dofs(fe_collection);

    const auto result = adamantine::ConnectedComponents::find_components(
        dof_handler, {top_boundary_id}, 0, comm);

    BOOST_TEST(result.size() == expected_components);
    unsigned int n_result_cells = 0;
    for (const auto &component : result)
      n_result_cells += component._locally_owned_cells.size();
    unsigned int n_expected = 0;
    for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_locally_owned() && cell->active_fe_index() == 0)
        ++n_expected;
    BOOST_TEST(n_result_cells == n_expected);
  };

  // U-shaped region plus (1,2): (1,2) touches (0,1) and (2,1) diagonally at
  // four non-collinear vertices -> one component.
  run({{0, 1}, {0, 0}, {1, 0}, {2, 0}, {2, 1}, {1, 2}}, 1);

  // Only one diagonal contact (two collinear vertices) -> two components.
  run({{0, 1}, {0, 0}, {1, 2}}, 2);
}
