/* SPDX-FileCopyrightText: Copyright (c) 2025, the adamantine authors.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#ifndef MICROSTRUCTURE_HH
#define MICROSTRUCTURE_HH

#include <MaterialProperty.hh>
#include <types.hh>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <fstream>
#include <string>

namespace adamantine
{
/**
 * This class computes data need for microstructure simulation.
 */
template <int dim>
class Microstructure
{
public:
  /**
   * Constructor.
   */
  Microstructure(MPI_Comm communicator, std::string const &filename_prefix);

  /**
   * On each rank, the destructor closes their own temporary file. On rank zero,
   * the destructor concatenates the temporary files into a single new file, and
   * it removes all the temporary files.
   */
  ~Microstructure();

  /**
   * Save the temperature before calling evolve_one_time_step().
   */
  void set_old_temperature(
      dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host>
          &old_temperature);

  /**
   * Compute the following data
   * - Temperature Gradient: G (K/m) = sqrt(G_x^2+G_y^2+G_z^2)
   * - Cooling Rate (K/s) = |dT/dt|
   * - Interface Velocity: R (m/s) = cooling rate / G
   */
  template <int p_order, typename MaterialStates, typename MemorySpaceType>
  void compute_G_and_R(
      MaterialProperty<dim, p_order, MaterialStates, MemorySpaceType> const
          &material_properties,
      dealii::DoFHandler<dim> const &dof_handler,
      dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host> const
          &temperature,
      double delta_t);

private:
  /**
   * Local MPI communicator.
   */
  MPI_Comm _communicator;
  /**
   * Prefix of the output filename.
   */
  std::string _filename_prefix;
  /**
   * Temporary output file.
   */
  std::ofstream _file;
  /**
   * Temperature at the previous time step.
   */
  dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host>
      _old_temperature;
};

template <int dim>
template <int p_order, typename MaterialStates, typename MemorySpaceType>
void Microstructure<dim>::compute_G_and_R(
    MaterialProperty<dim, p_order, MaterialStates, MemorySpaceType> const
        &material_properties,
    dealii::DoFHandler<dim> const &dof_handler,
    dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host> const
        &temperature,
    double delta_t)
{
  // In general, the quadrature points and the support points are different.
  // Here, we just care about G, the cooling rate, and R at some points. We
  // don't really care which points exactly. So we can use a different
  // quadrature that matches the support points. The big advantage is that we
  // can easily compute which dof crossed the liquidus and compute the
  // temperature gradient only if necessary.
  // We are interested in the following quantities:
  // - Temperature Gradient: G (K/m) = sqrt(G_x^2+G_y^2+G_z^2)
  // - Cooling Rate (K/s) = |dT/dt|
  // - Interface Velocity: R (m/s) = cooling rate / G

  // We are going to evaluate the temperature fields, so we need to update the
  // ghost values.
  _old_temperature.update_ghost_values();
  temperature.update_ghost_values();

  // Create a quadrature formula that matches the support points of the finite
  // element.
  auto const &fe = dof_handler.get_fe();
  dealii::QGaussLobatto<dim> quadrature_formula(fe.degree + 1);
  dealii::FEValues<dim> fe_values(fe, quadrature_formula,
                                  dealii::update_values |
                                      dealii::update_gradients |
                                      dealii::update_quadrature_points);
  unsigned int const n_dofs_per_cell = fe.n_dofs_per_cell();
  std::vector<dealii::types::global_dof_index> local_dof_indices(
      n_dofs_per_cell);
  std::vector<bool> relevant_indices(n_dofs_per_cell);
  std::vector<double> G(n_dofs_per_cell);
  std::vector<double> cooling_rate(n_dofs_per_cell);
  std::vector<unsigned int> quad_to_support_pts(n_dofs_per_cell);

  for (const auto &cell : dof_handler.active_cell_iterators() |
                              dealii::IteratorFilters::LocallyOwnedCell() |
                              dealii::IteratorFilters::ActiveFEIndexEqualTo(0))
  {
    cell->get_dof_indices(local_dof_indices);

    double const liquidus =
        material_properties.get(cell->material_id(), Property::liquidus);

    bool skip_cell = true;
    for (unsigned int const i : fe_values.dof_indices())
    {
      dealii::types::global_dof_index const dof = local_dof_indices[i];
      // Compute the cooling rate if the cell solidifies
      if ((_old_temperature[dof] > liquidus) && (temperature[dof] <= liquidus))
      {
        relevant_indices[i] = true;
        cooling_rate[i] = _old_temperature[dof] - temperature[dof] / delta_t;
        skip_cell = false;
      }
      else
      {
        relevant_indices[i] = false;
      }
    }

    // The cell did not solidified, there is nothing to do.
    if (skip_cell)
    {
      continue;
    }

    // Compute G
    fe_values.reinit(cell);
    for (unsigned int const q_point : fe_values.quadrature_point_indices())
    {
      G[q_point] = 0;

      // Compute the components of the gradient
      std::array<double, dim> grad = {0.};
      for (unsigned int const i : fe_values.dof_indices())
      {
        dealii::types::global_dof_index const dof = local_dof_indices[i];
        for (int d = 0; d < dim; ++d)
        {
          grad[d] += temperature[dof] * fe_values.shape_grad(i, q_point)[d];
        }
        // The order of the quadrature points and the support points is
        // different. This is problematic because the cooling rate is using a
        // different ordering than G. So we need to create a map. We use the
        // fact that at one support point, the value of the shape function is
        // one. At the others the value is zero.
        if (fe_values.shape_value(i, q_point) - 1.0 > -1e-12)
        {
          quad_to_support_pts[q_point] = i;
        }
      }

      // Compute G
      for (int d = 0; d < dim; ++d)
      {
        G[q_point] += std::pow(grad[d], 2);
      }
      G[q_point] = std::sqrt(G[q_point]);
    }

    // Write the data on file.
    for (unsigned int const q_point : fe_values.quadrature_point_indices())
    {
      unsigned int fe_index = quad_to_support_pts[q_point];
      if (relevant_indices[fe_index])
      {
        for (int d = 0; d < dim; ++d)
        {
          _file << fe_values.quadrature_point(q_point)[d] << " ";
        }
        _file << G[q_point] << " " << cooling_rate[fe_index] << " "
              << cooling_rate[fe_index] / G[q_point] << "\n";
      }
    }
  }
}
} // namespace adamantine

#endif
