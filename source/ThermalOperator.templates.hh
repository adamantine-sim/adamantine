/* Copyright (c) 2016, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#ifndef _THERMAL_OPERATOR_TEMPLATES_HH_
#define _THERMAL_OPERATOR_TEMPLATES_HH_

#include "ThermalOperator.hh"
#include <deal.II/base/index_set.h>
#include <deal.II/matrix_free/fe_evaluation.h>

namespace adamantine
{

template <int dim, int fe_degree, typename NumberType>
ThermalOperator<dim, fe_degree, NumberType>::ThermalOperator(
    boost::mpi::communicator &communicator,
    std::shared_ptr<MaterialProperty> material_properties)
    : _communicator(communicator), _material_properties(material_properties)
{
}

template <int dim, int fe_degree, typename NumberType>
template <typename QuadratureType>
void ThermalOperator<dim, fe_degree, NumberType>::reinit(
    dealii::DoFHandler<dim> const &dof_handler,
    dealii::ConstraintMatrix const &constraint_matrix,
    QuadratureType const &quad)
{
  _data.reinit(dof_handler, constraint_matrix, quad);
  _inverse_mass_matrix.reinit(_data.get_locally_owned_set(), _communicator);
  // TODO: for now we only solve linear problem so we can evaluate the thermal
  // conductivity once. Since the thermal conductivity is independent of the
  // current temperature, we use a dummy temperature vector.
  dealii::LA::distributed::Vector<NumberType> dummy;
  evaluate_thermal_conductivity(dummy);
}

template <int dim, int fe_degree, typename NumberType>
void ThermalOperator<dim, fe_degree, NumberType>::clear()
{
  _data.clear();
  _inverse_mass_matrix.reinit(0);
}

template <int dim, int fe_degree, typename NumberType>
void ThermalOperator<dim, fe_degree, NumberType>::vmult(
    dealii::LA::distributed::Vector<NumberType> &dst,
    dealii::LA::distributed::Vector<NumberType> const &src) const
{
  dst = 0.;
  vmult_add(dst, src);
}

template <int dim, int fe_degree, typename NumberType>
void ThermalOperator<dim, fe_degree, NumberType>::Tvmult(
    dealii::LA::distributed::Vector<NumberType> &dst,
    dealii::LA::distributed::Vector<NumberType> const &src) const
{
  dst = 0.;
  Tvmult_add(dst, src);
}

template <int dim, int fe_degree, typename NumberType>
void ThermalOperator<dim, fe_degree, NumberType>::vmult_add(
    dealii::LA::distributed::Vector<NumberType> &dst,
    dealii::LA::distributed::Vector<NumberType> const &src) const
{
  // Execute the matrix-free matrix-vector multiplication
  _data.cell_loop(&ThermalOperator::local_apply, this, dst, src);

  // Because cell_loop resolves the constraints, the constrained dofs are not
  // called they stay at zero. Thus, we need to force the value on the
  // constrained dofs by hand. The variable scaling is used so that we get the
  // right order of magnitude.
  // TODO: for now the value of scaling is set to 1
  NumberType const scaling = 1.;
  std::vector<dealii::types::global_dof_index> const &constrained_dofs =
      _data.get_constrained_dofs();
  for (auto &dof : constrained_dofs)
    dst[dof] += scaling * src[dof];
}

template <int dim, int fe_degree, typename NumberType>
void ThermalOperator<dim, fe_degree, NumberType>::Tvmult_add(
    dealii::LA::distributed::Vector<NumberType> &dst,
    dealii::LA::distributed::Vector<NumberType> const &src) const
{
  // The system of equation is symmetric so we can use vmult_add
  vmult_add(dst, src);
}

template <int dim, int fe_degree, typename NumberType>
void ThermalOperator<dim, fe_degree, NumberType>::local_apply(
    dealii::MatrixFree<dim, NumberType> const &data,
    dealii::LA::distributed::Vector<NumberType> &dst,
    dealii::LA::distributed::Vector<NumberType> const &src,
    std::pair<unsigned int, unsigned int> const &cell_range) const
{
  dealii::FEEvaluation<dim, fe_degree, fe_degree + 1, 1, NumberType> fe_eval(
      data);

  // Loop over the "cells". Note that we don't really work on a cell but on a
  // set of quadrature point.
  for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
  {
    // Reinit fe_eval on the current cell
    fe_eval.reinit(cell);
    // Store in a local vector the local values of src
    fe_eval.read_dof_values(src);
    // Evaluate the only the function gradients on the reference cell
    fe_eval.evaluate(false, true);
    // Apply the Jacobian of the transformation, multiply by the variable
    // coefficients and the quadrature points
    for (unsigned int q = 0; q < fe_eval.n_q_points; ++q)
      fe_eval.submit_gradient(
          _thermal_conductivity(cell, q) * fe_eval.get_gradient(q), q);
    // Sum over the quadrature points.
    fe_eval.integrate(false, true);
    fe_eval.distribute_local_to_global(dst);
  }
}

template <int dim, int fe_degree, typename NumberType>
void ThermalOperator<dim, fe_degree, NumberType>::evaluate_thermal_conductivity(
    dealii::LA::distributed::Vector<NumberType> const &state)
{
  unsigned int const n_cells = _data.n_macro_cells();
  dealii::FEEvaluation<dim, fe_degree, fe_degree + 1, 1, NumberType> fe_eval(
      _data);
  _thermal_conductivity.reinit(n_cells, fe_eval.n_q_points);
  for (unsigned int cell = 0; cell < n_cells; ++cell)
  {
    for (unsigned int q = 0; q < fe_eval.n_q_points; ++q)
      for (unsigned int i = 0; i < _data.n_components_filled(cell); ++i)
      {
        typename dealii::DoFHandler<dim>::cell_iterator cell_it =
            _data.get_cell_iterator(cell, i);
        // Cast to Triangulation<dim>::cell_iterator to access the material_id
        typename dealii::Triangulation<dim>::active_cell_iterator cell_tria(
            cell_it);
        _data.get_cell_iterator(cell, i);
        _thermal_conductivity(cell, q)[i] =
            _material_properties->get<dim, NumberType>(
                cell_tria, Property::thermal_conductivity, state);
      }
  }
}
}

#endif
