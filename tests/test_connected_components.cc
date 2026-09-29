/* SPDX-FileCopyrightText: Copyright (c) 2025, the adamantine authors.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

//#include <deal.II/base
#include <ConnectedComponents.hh>

#include <deal.II/distributed/tria.h>
#include <deal.II/base/geometry_info.h>
#include <deal.II/base/types.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/base/conditional_ostream.h>

namespace
{
  unsigned int
  z_layer_from_point(const dealii::Point<3> &p, const unsigned int n_layers)
  {
    const double z = p[2];
    return std::min(n_layers - 1U,
                    static_cast<unsigned int>(std::floor(z * n_layers)));
  }

  void
  run_test(const MPI_Comm comm)
  {
    constexpr unsigned int nx = 64;
    constexpr unsigned int ny = nx;
    constexpr unsigned int nz = 2*nx;

    const unsigned int my_rank = dealii::Utilities::MPI::this_mpi_process(comm);
    dealii::ConditionalOStream pcout(std::cout, my_rank == 0);

    // --------------------------------------------------------------------------
    // Build a 2 x 2 x 4 cube on [0,1]^3.
    // With colorize=true, deal.II assigns distinct boundary ids to the 6 faces.
    // --------------------------------------------------------------------------
    dealii::parallel::distributed::Triangulation<3> triangulation(comm);

    dealii::GridGenerator::subdivided_hyper_rectangle(triangulation,
                                              {nx, ny, nz},
                                              dealii::Point<3>(0.0, 0.0, 0.0),
                                              dealii::Point<3>(1.0, 1.0, 1.0),
                                              /*colorize=*/true);

    dealii::DoFHandler<3> dof_handler(triangulation);

    std::cout << "n_cells: " << triangulation.n_cells() << std::endl;

    // One FE per z-layer. Using different polynomial degrees is a simple way
    // to guarantee distinct FE indices.
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

    pcout << "Top boundary id = " << static_cast<unsigned int>(top_boundary_id)
          << '\n';

    // --------------------------------------------------------------------------
    // Run the connected-components routine.
    // --------------------------------------------------------------------------

for (unsigned int target_fe_index = 0; target_fe_index<4; ++ target_fe_index) {
auto result  = adamantine::ConnectedComponents::find_components(
        dof_handler, {top_boundary_id}, target_fe_index, comm);

    // --------------------------------------------------------------------------
    // Local consistency checks.
    // --------------------------------------------------------------------------
    unsigned int local_fail = 0;

    for (unsigned int c = 0; c < result.size(); ++c)
      {
        const auto &component = result[c];

        for (const auto &cell : component.locally_owned_cells)
          {
            if (cell->active_fe_index() != target_fe_index)
              local_fail = 1;

            if (z_layer_from_point(cell->center(), static_cast<unsigned int>(fe_collection.size())) != target_fe_index)
              local_fail = 1;
          }
      }

    // --------------------------------------------------------------------------
    // Global checks by FE index.
    // Since each z-layer has a unique FE index, we expect up to one component per layer
    // (zero components are acceptable for FEs that were intentionally filtered out).
    // --------------------------------------------------------------------------

    const unsigned int n_layers_check = static_cast<unsigned int>(fe_collection.size());
    std::vector<unsigned int> local_cells_per_fe(n_layers_check, 0U);
    std::vector<unsigned int> components_per_fe(n_layers_check, 0U);
    std::vector<unsigned int> touches_top_per_fe(n_layers_check, 0U);

    for (const auto &component : result)
      {
        if (target_fe_index >= n_layers_check)
          {
            local_fail = 1;
            continue;
          }

        ++components_per_fe[target_fe_index];
        local_cells_per_fe[target_fe_index] += component.locally_owned_cells.size();

        if (component.touches_target_boundary)
          touches_top_per_fe[target_fe_index] = 1U;
      }

    for (unsigned int fe_index = 0; fe_index < n_layers_check; ++fe_index)
      {
        // Zero components are acceptable if the FE was filtered out. More than one
        // component per FE is an error.
        if (components_per_fe[fe_index] > 1U)
          local_fail = 1;

        if (components_per_fe[fe_index] == 1U)
          {
            const unsigned int expected_touches_top = (fe_index == n_layers_check - 1U ? 1U : 0U);
            if (touches_top_per_fe[fe_index] != expected_touches_top)
              local_fail = 1;
          }
      }

    // Compute global fail for the test and print detailed diagnostics if failing.
    const unsigned int global_fail = dealii::Utilities::MPI::max(local_fail, comm);

    // --------------------------------------------------------------------------
    // Print a short summary.
    // --------------------------------------------------------------------------
    pcout << "Found " << result.size() << " components\n";
    for (unsigned int c = 0; c < result.size(); ++c)
      {
        const auto &component = result[c];
        const unsigned int global_n_cells =
          dealii::Utilities::MPI::sum(static_cast<unsigned int>(component.locally_owned_cells.size()),
                              comm);

        pcout << "  component " << c
              << ": fe_index=" << target_fe_index
              << ", touches_top=" << std::boolalpha
              << component.touches_target_boundary
              << ", global_cells=" << global_n_cells
              << ", local_cells=" << component.locally_owned_cells.size()
              << '\n';

        // Print a small sample of local cell ids and verify mapping correctness
        unsigned int sample = 0;
        for (const auto &cell : component.locally_owned_cells)
          {
            if (sample++ >= 5)
              break;
            pcout << "    sample cell id=" << cell->active_cell_index()
                  << ", active_fe_index=" << cell->active_fe_index()
                  << '\n';
          }
      }

    if (global_fail)
      {
        pcout << "TEST FAILURE: local_fail=" << local_fail << " result.size()=" << result.size() << "\n";

        // Print per-FE diagnostics
        for (unsigned int fe_index = 0; fe_index < n_layers_check; ++fe_index)
          {
            pcout << "FE " << fe_index << ": components_per_fe=" << components_per_fe[fe_index]
                  << ", local_cells_per_fe=" << local_cells_per_fe[fe_index]
                  << ", touches_top=" << touches_top_per_fe[fe_index] << '\n';
          }

        // Search for mismatched component mappings
        for (unsigned int c = 0; c < result.size(); ++c)
          {
            const auto &component = result[c];
            for (const auto &cell : component.locally_owned_cells)
              {
                const unsigned int id = cell->active_cell_index();
                if (cell->active_fe_index() != target_fe_index)
                  pcout << "  FE_MISMATCH: cell " << id << " fe_index=" << cell->active_fe_index()
                        << " target_fe_index=" << target_fe_index << "\n";

                if (z_layer_from_point(cell->center(), static_cast<unsigned int>(fe_collection.size())) != target_fe_index)
                  pcout << "  Z_LAYER_MISMATCH: cell " << id << " center_z=" << cell->center()[2]
                        << " expected_layer=" << z_layer_from_point(cell->center(), static_cast<unsigned int>(fe_collection.size()))
                        << " target_fe_index=" << target_fe_index << "\n";
              }
          }

        throw std::runtime_error("Connected-components test failed.");
      }

    pcout << "Connected-components test passed.\n";
  }
}
} // namespace

int
main(int argc, char **argv)
{
  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  run_test(MPI_COMM_WORLD);
}

