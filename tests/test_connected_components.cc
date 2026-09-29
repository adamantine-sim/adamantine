/* SPDX-FileCopyrightText: Copyright (c) 2025, the adamantine authors.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include <ConnectedComponents.hh>

#include <deal.II/distributed/tria.h>
#include <deal.II/base/geometry_info.h>
#include <deal.II/base/types.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/base/conditional_ostream.h>

#include "main.cc"

  unsigned int
  z_layer_from_point(const dealii::Point<3> &p, const unsigned int n_layers)
  {
    const double z = p[2];
    return std::min(n_layers - 1U,
                    static_cast<unsigned int>(std::floor(z * n_layers)));
  }

  BOOST_AUTO_TEST_CASE(connected_components) {
    MPI_Comm comm = MPI_COMM_WORLD;

    // --------------------------------------------------------------------------
    // Build a nx x nx x 2*nx cube on [0,1]^3.
    // With colorize=true, deal.II assigns distinct boundary ids to the 6 faces.
    // --------------------------------------------------------------------------
    dealii::parallel::distributed::Triangulation<3> triangulation(comm);

    constexpr unsigned int nx = 64;
    constexpr unsigned int ny = nx;
    constexpr unsigned int nz = 2*nx;
    dealii::GridGenerator::subdivided_hyper_rectangle(triangulation,
                                              {nx, ny, nz},
                                              dealii::Point<3>(0.0, 0.0, 0.0),
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
        cell->set_active_fe_index(z_layer_from_point(cell->center(), fe_collection.size()));

    dof_handler.distribute_dofs(fe_collection);

    // --------------------------------------------------------------------------
    // Detect the boundary id of the top face z=1 from the triangulation itself.
    // This avoids hard-coding the numeric boundary id.
    // --------------------------------------------------------------------------
    dealii::types::boundary_id local_top_boundary_id = dealii::numbers::invalid_boundary_id;

    for (const auto &cell : triangulation.active_cell_iterators())
      if (!cell->is_artificial())
        for (unsigned int f = 0; f < dealii::GeometryInfo<3>::faces_per_cell; ++f)
          if (cell->at_boundary(f) &&
              std::abs(cell->face(f)->center()[2] - 1.0) < 1e-12)
            local_top_boundary_id = cell->face(f)->boundary_id();

    const unsigned int top_boundary_id_int =
      dealii::Utilities::MPI::min(static_cast<unsigned int>(local_top_boundary_id), comm);

    if (top_boundary_id_int ==
        static_cast<unsigned int>(dealii::numbers::invalid_boundary_id))
      throw std::runtime_error("Could not determine top boundary id.");

    const dealii::types::boundary_id top_boundary_id =
      static_cast<dealii::types::boundary_id>(top_boundary_id_int);

    // --------------------------------------------------------------------------
    // Run the connected-components routine.
    // --------------------------------------------------------------------------

for (unsigned int target_fe_index = 0; target_fe_index<fe_collection.size(); ++ target_fe_index) {
auto result  = adamantine::ConnectedComponents::find_components(
        dof_handler, {top_boundary_id}, target_fe_index, comm);

    // --------------------------------------------------------------------------
    // Consistency checks.
    // --------------------------------------------------------------------------
    BOOST_TEST(result.size() == 1);
    const unsigned int n_layers = static_cast<unsigned int>(fe_collection.size());
       for (const auto &cell : result[0].locally_owned_cells)
          {
            BOOST_TEST(cell->active_fe_index() == target_fe_index);
            BOOST_TEST(z_layer_from_point(cell->center(), n_layers) == target_fe_index);
          }

    unsigned int n_expected_local_cells = 0;
    for (const auto& cell: dof_handler.active_cell_iterators()) {
      if(cell->is_locally_owned() && cell->active_fe_index() == target_fe_index)
       ++n_expected_local_cells;
    }
    BOOST_TEST(result[0].locally_owned_cells.size() == n_expected_local_cells);

    const bool expected_touches_top = (target_fe_index == n_layers - 1);
    BOOST_TEST(result[0].touches_target_boundary == expected_touches_top);
  }
}
