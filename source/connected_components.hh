/* SPDX-FileCopyrightText: Copyright (c) 2026, the adamantine authors.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#ifndef CONNECTED_COMPONENTS_HH
#define CONNECTED_COMPONENTS_HH

#include <deal.II/base/types.h>
#include <deal.II/dofs/dof_handler.h>

#include <vector>

namespace adamantine
{
namespace ConnectedComponents
{

/**
 * Data structure storing all cell indices for a component and whether it
 * touches a boundary we care about.
 */
template <int dim>
struct ComponentInfo
{
  bool _touches_target_boundary = false;

  // Only locally owned cells on this rank are stored.
  std::vector<typename dealii::DoFHandler<dim>::active_cell_iterator>
      _locally_owned_cells;
};

/**
 * Find connected components of locally owned active cells with the given FE
 * index. Connectivity is defined through face adjacency, including across MPI
 * ranks through ghost neighbors.
 *
 * The returned vector stores, on each rank, only the locally owned cells that
 * belong to each global component together with a flag indicating whether that
 * global component touches one of the target boundaries.
 */
template <int dim>
std::vector<ComponentInfo<dim>> find_components(
    const dealii::DoFHandler<dim> &dof_handler,
    const std::vector<dealii::types::boundary_id> &target_boundary_id,
    const unsigned int target_fe_index, const MPI_Comm mpi_communicator);

} // namespace ConnectedComponents
} // namespace adamantine

#endif
