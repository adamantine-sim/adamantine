/* Copyright (c) 2016, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#include <ThermalOperator.hh>
#include <instantiation.hh>

#include <deal.II/base/index_set.h>
#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/matrix_free/fe_evaluation.h>

namespace adamantine
{

template <int dim, int fe_degree, typename NumberType>
ThermalOperator<dim, fe_degree, NumberType>::ThermalOperator(
    MPI_Comm const &communicator,
    std::shared_ptr<MaterialProperty<dim>> material_properties)
    : _communicator(communicator), _material_properties(material_properties),
      _inverse_mass_matrix(new dealii::LA::distributed::Vector<NumberType>())
{
  _matrix_free_data.tasks_parallel_scheme =
      dealii::MatrixFree<dim, NumberType>::AdditionalData::partition_color;
}

template <int dim, int fe_degree, typename NumberType>
template <typename QuadratureType>
void ThermalOperator<dim, fe_degree, NumberType>::setup_dofs(
    dealii::DoFHandler<dim> const &dof_handler,
    dealii::AffineConstraints<double> const &affine_constraints,
    QuadratureType const &quad)
{
  _matrix_free.reinit(dof_handler, affine_constraints, quad, _matrix_free_data);
}

template <int dim, int fe_degree, typename NumberType>
void ThermalOperator<dim, fe_degree, NumberType>::reinit(
    dealii::DoFHandler<dim> const &dof_handler,
    dealii::AffineConstraints<double> const &affine_constraints)
{
  // Compute the inverse of the mass matrix
  dealii::QGaussLobatto<1> mass_matrix_quad(fe_degree + 1);
  dealii::MatrixFree<dim, NumberType> mass_matrix_free;
  mass_matrix_free.reinit(dof_handler, affine_constraints, mass_matrix_quad,
                          _matrix_free_data);
  mass_matrix_free.initialize_dof_vector(*_inverse_mass_matrix);
  dealii::VectorizedArray<NumberType> one =
      dealii::make_vectorized_array(static_cast<NumberType>(1.));
  dealii::FEEvaluation<dim, fe_degree, fe_degree + 1, 1, NumberType> fe_eval(
      mass_matrix_free);
  unsigned int const n_q_points = fe_eval.n_q_points;
  for (unsigned int cell = 0; cell < mass_matrix_free.n_macro_cells(); ++cell)
  {
    fe_eval.reinit(cell);
    for (unsigned int q = 0; q < n_q_points; ++q)
      fe_eval.submit_value(one, q);
    fe_eval.integrate(true, false);
    fe_eval.distribute_local_to_global(*_inverse_mass_matrix);
  }
  _inverse_mass_matrix->compress(dealii::VectorOperation::add);
  unsigned int const local_size = _inverse_mass_matrix->local_size();
  for (unsigned int k = 0; k < local_size; ++k)
  {
    if (_inverse_mass_matrix->local_element(k) > 1e-15)
      _inverse_mass_matrix->local_element(k) =
          1. / _inverse_mass_matrix->local_element(k);
    else
      _inverse_mass_matrix->local_element(k) = 0.;
  }
}

template <int dim, int fe_degree, typename NumberType>
void ThermalOperator<dim, fe_degree, NumberType>::clear()
{
  _matrix_free.clear();
  _inverse_mass_matrix->reinit(0);
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
  _matrix_free.cell_loop(&ThermalOperator::local_apply, this, dst, src);

  // Because cell_loop resolves the constraints, the constrained dofs are not
  // called they stay at zero. Thus, we need to force the value on the
  // constrained dofs by hand. The variable scaling is used so that we get the
  // right order of magnitude.
  // TODO: for now the value of scaling is set to 1
  NumberType const scaling = 1.;
  std::vector<unsigned int> const &constrained_dofs =
      _matrix_free.get_constrained_dofs();
  for (auto &dof : constrained_dofs)
    dst.local_element(dof) += scaling * src.local_element(dof);
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
  dealii::Tensor<1, dim> unit_tensor;
  for (unsigned int i = 0; i < dim; ++i)
    unit_tensor[i] = 1.;

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
      fe_eval.submit_gradient(-_thermal_conductivity(cell, q) *
                                  (fe_eval.get_gradient(q) * _alpha(cell, q) +
                                   unit_tensor * _beta(cell, q)),
                              q);
    // Sum over the quadrature points.
    fe_eval.integrate(false, true);
    fe_eval.distribute_local_to_global(dst);
  }
}

template <int dim, int fe_degree, typename NumberType>
void ThermalOperator<dim, fe_degree, NumberType>::evaluate_material_properties(
    dealii::LA::distributed::Vector<NumberType> const &state)
{
  // Update the state of the materials
  _material_properties->update_state(_matrix_free.get_dof_handler(), state);

  unsigned int const n_cells = _matrix_free.n_macro_cells();
  dealii::FEEvaluation<dim, fe_degree, fe_degree + 1, 1, NumberType> fe_eval(
      _matrix_free);
  _alpha.reinit(n_cells, fe_eval.n_q_points);
  _beta.reinit(n_cells, fe_eval.n_q_points);
  _thermal_conductivity.reinit(n_cells, fe_eval.n_q_points);
  for (unsigned int cell = 0; cell < n_cells; ++cell)
    for (unsigned int q = 0; q < fe_eval.n_q_points; ++q)
      for (unsigned int i = 0; i < _matrix_free.n_components_filled(cell); ++i)
      {
        typename dealii::DoFHandler<dim>::cell_iterator cell_it =
            _matrix_free.get_cell_iterator(cell, i);
        // Cast to Triangulation<dim>::cell_iterator to access the material_id
        typename dealii::Triangulation<dim>::active_cell_iterator cell_tria(
            cell_it);

        _thermal_conductivity(cell, q)[i] = _material_properties->get(
            cell_tria, Property::thermal_conductivity, state);

        double liquid_ratio = _material_properties->get_state_ratio(
            cell_tria, MaterialState::liquid);

        // If there is less than 1e-13% of liquid, assume that there is no
        // liquid.
        if (liquid_ratio < 1e-15)
        {
          _alpha(cell, q)[i] =
              1. /
              (_material_properties->get(cell_tria, Property::density, state) *
               _material_properties->get(cell_tria, Property::specific_heat,
                                         state));
        }
        else
        {
          // If there is less than 1e-13% of solid, assume that there is no
          // solid. Otherwise, we have a mix of liquid and solid (mushy zone).
          if (liquid_ratio > (1 - 1e-15))
          {
            _alpha(cell, q)[i] =
                1. / (_material_properties->get(cell_tria, Property::density,
                                                state) *
                      _material_properties->get(
                          cell_tria, Property::specific_heat, state));
            _beta(cell, q)[i] =
                _material_properties->get_liquid_beta(cell_tria);
          }
          else
          {
            _alpha(cell, q)[i] =
                _material_properties->get_mushy_alpha(cell_tria);
            _beta(cell, q)[i] = _material_properties->get_mushy_beta(cell_tria);
          }
        }
      }
}
} // namespace adamantine

INSTANTIATE_DIM_FEDEGREE_NUM(TUPLE(ThermalOperator))

// Instantiate the function template.
namespace adamantine
{
template void ThermalOperator<2, 1, float>::setup_dofs<dealii::QGauss<1>>(
    dealii::DoFHandler<2> const &, dealii::AffineConstraints<double> const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<2, 2, float>::setup_dofs<dealii::QGauss<1>>(
    dealii::DoFHandler<2> const &, dealii::AffineConstraints<double> const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<2, 3, float>::setup_dofs<dealii::QGauss<1>>(
    dealii::DoFHandler<2> const &, dealii::AffineConstraints<double> const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<2, 4, float>::setup_dofs<dealii::QGauss<1>>(
    dealii::DoFHandler<2> const &, dealii::AffineConstraints<double> const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<2, 5, float>::setup_dofs<dealii::QGauss<1>>(
    dealii::DoFHandler<2> const &, dealii::AffineConstraints<double> const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<2, 6, float>::setup_dofs<dealii::QGauss<1>>(
    dealii::DoFHandler<2> const &, dealii::AffineConstraints<double> const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<2, 7, float>::setup_dofs<dealii::QGauss<1>>(
    dealii::DoFHandler<2> const &, dealii::AffineConstraints<double> const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<2, 8, float>::setup_dofs<dealii::QGauss<1>>(
    dealii::DoFHandler<2> const &, dealii::AffineConstraints<double> const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<2, 9, float>::setup_dofs<dealii::QGauss<1>>(
    dealii::DoFHandler<2> const &, dealii::AffineConstraints<double> const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<2, 10, float>::setup_dofs<dealii::QGauss<1>>(
    dealii::DoFHandler<2> const &, dealii::AffineConstraints<double> const &,
    dealii::QGauss<1> const &);

template void
    ThermalOperator<2, 1, float>::setup_dofs<dealii::QGaussLobatto<1>>(
        dealii::DoFHandler<2> const &,
        dealii::AffineConstraints<double> const &,
        dealii::QGaussLobatto<1> const &);
template void
    ThermalOperator<2, 2, float>::setup_dofs<dealii::QGaussLobatto<1>>(
        dealii::DoFHandler<2> const &,
        dealii::AffineConstraints<double> const &,
        dealii::QGaussLobatto<1> const &);
template void
    ThermalOperator<2, 3, float>::setup_dofs<dealii::QGaussLobatto<1>>(
        dealii::DoFHandler<2> const &,
        dealii::AffineConstraints<double> const &,
        dealii::QGaussLobatto<1> const &);
template void
    ThermalOperator<2, 4, float>::setup_dofs<dealii::QGaussLobatto<1>>(
        dealii::DoFHandler<2> const &,
        dealii::AffineConstraints<double> const &,
        dealii::QGaussLobatto<1> const &);
template void
    ThermalOperator<2, 5, float>::setup_dofs<dealii::QGaussLobatto<1>>(
        dealii::DoFHandler<2> const &,
        dealii::AffineConstraints<double> const &,
        dealii::QGaussLobatto<1> const &);
template void
    ThermalOperator<2, 6, float>::setup_dofs<dealii::QGaussLobatto<1>>(
        dealii::DoFHandler<2> const &,
        dealii::AffineConstraints<double> const &,
        dealii::QGaussLobatto<1> const &);
template void
    ThermalOperator<2, 7, float>::setup_dofs<dealii::QGaussLobatto<1>>(
        dealii::DoFHandler<2> const &,
        dealii::AffineConstraints<double> const &,
        dealii::QGaussLobatto<1> const &);
template void
    ThermalOperator<2, 8, float>::setup_dofs<dealii::QGaussLobatto<1>>(
        dealii::DoFHandler<2> const &,
        dealii::AffineConstraints<double> const &,
        dealii::QGaussLobatto<1> const &);
template void
    ThermalOperator<2, 9, float>::setup_dofs<dealii::QGaussLobatto<1>>(
        dealii::DoFHandler<2> const &,
        dealii::AffineConstraints<double> const &,
        dealii::QGaussLobatto<1> const &);
template void
    ThermalOperator<2, 10, float>::setup_dofs<dealii::QGaussLobatto<1>>(
        dealii::DoFHandler<2> const &,
        dealii::AffineConstraints<double> const &,
        dealii::QGaussLobatto<1> const &);

template void ThermalOperator<2, 1, double>::setup_dofs<dealii::QGauss<1>>(
    dealii::DoFHandler<2> const &, dealii::AffineConstraints<double> const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<2, 2, double>::setup_dofs<dealii::QGauss<1>>(
    dealii::DoFHandler<2> const &, dealii::AffineConstraints<double> const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<2, 3, double>::setup_dofs<dealii::QGauss<1>>(
    dealii::DoFHandler<2> const &, dealii::AffineConstraints<double> const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<2, 4, double>::setup_dofs<dealii::QGauss<1>>(
    dealii::DoFHandler<2> const &, dealii::AffineConstraints<double> const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<2, 5, double>::setup_dofs<dealii::QGauss<1>>(
    dealii::DoFHandler<2> const &, dealii::AffineConstraints<double> const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<2, 6, double>::setup_dofs<dealii::QGauss<1>>(
    dealii::DoFHandler<2> const &, dealii::AffineConstraints<double> const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<2, 7, double>::setup_dofs<dealii::QGauss<1>>(
    dealii::DoFHandler<2> const &, dealii::AffineConstraints<double> const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<2, 8, double>::setup_dofs<dealii::QGauss<1>>(
    dealii::DoFHandler<2> const &, dealii::AffineConstraints<double> const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<2, 9, double>::setup_dofs<dealii::QGauss<1>>(
    dealii::DoFHandler<2> const &, dealii::AffineConstraints<double> const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<2, 10, double>::setup_dofs<dealii::QGauss<1>>(
    dealii::DoFHandler<2> const &, dealii::AffineConstraints<double> const &,
    dealii::QGauss<1> const &);

template void
    ThermalOperator<2, 1, double>::setup_dofs<dealii::QGaussLobatto<1>>(
        dealii::DoFHandler<2> const &,
        dealii::AffineConstraints<double> const &,
        dealii::QGaussLobatto<1> const &);
template void
    ThermalOperator<2, 2, double>::setup_dofs<dealii::QGaussLobatto<1>>(
        dealii::DoFHandler<2> const &,
        dealii::AffineConstraints<double> const &,
        dealii::QGaussLobatto<1> const &);
template void
    ThermalOperator<2, 3, double>::setup_dofs<dealii::QGaussLobatto<1>>(
        dealii::DoFHandler<2> const &,
        dealii::AffineConstraints<double> const &,
        dealii::QGaussLobatto<1> const &);
template void
    ThermalOperator<2, 4, double>::setup_dofs<dealii::QGaussLobatto<1>>(
        dealii::DoFHandler<2> const &,
        dealii::AffineConstraints<double> const &,
        dealii::QGaussLobatto<1> const &);
template void
    ThermalOperator<2, 5, double>::setup_dofs<dealii::QGaussLobatto<1>>(
        dealii::DoFHandler<2> const &,
        dealii::AffineConstraints<double> const &,
        dealii::QGaussLobatto<1> const &);
template void
    ThermalOperator<2, 6, double>::setup_dofs<dealii::QGaussLobatto<1>>(
        dealii::DoFHandler<2> const &,
        dealii::AffineConstraints<double> const &,
        dealii::QGaussLobatto<1> const &);
template void
    ThermalOperator<2, 7, double>::setup_dofs<dealii::QGaussLobatto<1>>(
        dealii::DoFHandler<2> const &,
        dealii::AffineConstraints<double> const &,
        dealii::QGaussLobatto<1> const &);
template void
    ThermalOperator<2, 8, double>::setup_dofs<dealii::QGaussLobatto<1>>(
        dealii::DoFHandler<2> const &,
        dealii::AffineConstraints<double> const &,
        dealii::QGaussLobatto<1> const &);
template void
    ThermalOperator<2, 9, double>::setup_dofs<dealii::QGaussLobatto<1>>(
        dealii::DoFHandler<2> const &,
        dealii::AffineConstraints<double> const &,
        dealii::QGaussLobatto<1> const &);
template void
    ThermalOperator<2, 10, double>::setup_dofs<dealii::QGaussLobatto<1>>(
        dealii::DoFHandler<2> const &,
        dealii::AffineConstraints<double> const &,
        dealii::QGaussLobatto<1> const &);

template void ThermalOperator<3, 1, float>::setup_dofs<dealii::QGauss<1>>(
    dealii::DoFHandler<3> const &, dealii::AffineConstraints<double> const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<3, 2, float>::setup_dofs<dealii::QGauss<1>>(
    dealii::DoFHandler<3> const &, dealii::AffineConstraints<double> const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<3, 3, float>::setup_dofs<dealii::QGauss<1>>(
    dealii::DoFHandler<3> const &, dealii::AffineConstraints<double> const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<3, 4, float>::setup_dofs<dealii::QGauss<1>>(
    dealii::DoFHandler<3> const &, dealii::AffineConstraints<double> const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<3, 5, float>::setup_dofs<dealii::QGauss<1>>(
    dealii::DoFHandler<3> const &, dealii::AffineConstraints<double> const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<3, 6, float>::setup_dofs<dealii::QGauss<1>>(
    dealii::DoFHandler<3> const &, dealii::AffineConstraints<double> const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<3, 7, float>::setup_dofs<dealii::QGauss<1>>(
    dealii::DoFHandler<3> const &, dealii::AffineConstraints<double> const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<3, 8, float>::setup_dofs<dealii::QGauss<1>>(
    dealii::DoFHandler<3> const &, dealii::AffineConstraints<double> const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<3, 9, float>::setup_dofs<dealii::QGauss<1>>(
    dealii::DoFHandler<3> const &, dealii::AffineConstraints<double> const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<3, 10, float>::setup_dofs<dealii::QGauss<1>>(
    dealii::DoFHandler<3> const &, dealii::AffineConstraints<double> const &,
    dealii::QGauss<1> const &);

template void
    ThermalOperator<3, 1, float>::setup_dofs<dealii::QGaussLobatto<1>>(
        dealii::DoFHandler<3> const &,
        dealii::AffineConstraints<double> const &,
        dealii::QGaussLobatto<1> const &);
template void
    ThermalOperator<3, 2, float>::setup_dofs<dealii::QGaussLobatto<1>>(
        dealii::DoFHandler<3> const &,
        dealii::AffineConstraints<double> const &,
        dealii::QGaussLobatto<1> const &);
template void
    ThermalOperator<3, 3, float>::setup_dofs<dealii::QGaussLobatto<1>>(
        dealii::DoFHandler<3> const &,
        dealii::AffineConstraints<double> const &,
        dealii::QGaussLobatto<1> const &);
template void
    ThermalOperator<3, 4, float>::setup_dofs<dealii::QGaussLobatto<1>>(
        dealii::DoFHandler<3> const &,
        dealii::AffineConstraints<double> const &,
        dealii::QGaussLobatto<1> const &);
template void
    ThermalOperator<3, 5, float>::setup_dofs<dealii::QGaussLobatto<1>>(
        dealii::DoFHandler<3> const &,
        dealii::AffineConstraints<double> const &,
        dealii::QGaussLobatto<1> const &);
template void
    ThermalOperator<3, 6, float>::setup_dofs<dealii::QGaussLobatto<1>>(
        dealii::DoFHandler<3> const &,
        dealii::AffineConstraints<double> const &,
        dealii::QGaussLobatto<1> const &);
template void
    ThermalOperator<3, 7, float>::setup_dofs<dealii::QGaussLobatto<1>>(
        dealii::DoFHandler<3> const &,
        dealii::AffineConstraints<double> const &,
        dealii::QGaussLobatto<1> const &);
template void
    ThermalOperator<3, 8, float>::setup_dofs<dealii::QGaussLobatto<1>>(
        dealii::DoFHandler<3> const &,
        dealii::AffineConstraints<double> const &,
        dealii::QGaussLobatto<1> const &);
template void
    ThermalOperator<3, 9, float>::setup_dofs<dealii::QGaussLobatto<1>>(
        dealii::DoFHandler<3> const &,
        dealii::AffineConstraints<double> const &,
        dealii::QGaussLobatto<1> const &);
template void
    ThermalOperator<3, 10, float>::setup_dofs<dealii::QGaussLobatto<1>>(
        dealii::DoFHandler<3> const &,
        dealii::AffineConstraints<double> const &,
        dealii::QGaussLobatto<1> const &);

template void ThermalOperator<3, 1, double>::setup_dofs<dealii::QGauss<1>>(
    dealii::DoFHandler<3> const &, dealii::AffineConstraints<double> const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<3, 2, double>::setup_dofs<dealii::QGauss<1>>(
    dealii::DoFHandler<3> const &, dealii::AffineConstraints<double> const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<3, 3, double>::setup_dofs<dealii::QGauss<1>>(
    dealii::DoFHandler<3> const &, dealii::AffineConstraints<double> const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<3, 4, double>::setup_dofs<dealii::QGauss<1>>(
    dealii::DoFHandler<3> const &, dealii::AffineConstraints<double> const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<3, 5, double>::setup_dofs<dealii::QGauss<1>>(
    dealii::DoFHandler<3> const &, dealii::AffineConstraints<double> const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<3, 6, double>::setup_dofs<dealii::QGauss<1>>(
    dealii::DoFHandler<3> const &, dealii::AffineConstraints<double> const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<3, 7, double>::setup_dofs<dealii::QGauss<1>>(
    dealii::DoFHandler<3> const &, dealii::AffineConstraints<double> const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<3, 8, double>::setup_dofs<dealii::QGauss<1>>(
    dealii::DoFHandler<3> const &, dealii::AffineConstraints<double> const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<3, 9, double>::setup_dofs<dealii::QGauss<1>>(
    dealii::DoFHandler<3> const &, dealii::AffineConstraints<double> const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<3, 10, double>::setup_dofs<dealii::QGauss<1>>(
    dealii::DoFHandler<3> const &, dealii::AffineConstraints<double> const &,
    dealii::QGauss<1> const &);

template void
    ThermalOperator<3, 1, double>::setup_dofs<dealii::QGaussLobatto<1>>(
        dealii::DoFHandler<3> const &,
        dealii::AffineConstraints<double> const &,
        dealii::QGaussLobatto<1> const &);
template void
    ThermalOperator<3, 2, double>::setup_dofs<dealii::QGaussLobatto<1>>(
        dealii::DoFHandler<3> const &,
        dealii::AffineConstraints<double> const &,
        dealii::QGaussLobatto<1> const &);
template void
    ThermalOperator<3, 3, double>::setup_dofs<dealii::QGaussLobatto<1>>(
        dealii::DoFHandler<3> const &,
        dealii::AffineConstraints<double> const &,
        dealii::QGaussLobatto<1> const &);
template void
    ThermalOperator<3, 4, double>::setup_dofs<dealii::QGaussLobatto<1>>(
        dealii::DoFHandler<3> const &,
        dealii::AffineConstraints<double> const &,
        dealii::QGaussLobatto<1> const &);
template void
    ThermalOperator<3, 5, double>::setup_dofs<dealii::QGaussLobatto<1>>(
        dealii::DoFHandler<3> const &,
        dealii::AffineConstraints<double> const &,
        dealii::QGaussLobatto<1> const &);
template void
    ThermalOperator<3, 6, double>::setup_dofs<dealii::QGaussLobatto<1>>(
        dealii::DoFHandler<3> const &,
        dealii::AffineConstraints<double> const &,
        dealii::QGaussLobatto<1> const &);
template void
    ThermalOperator<3, 7, double>::setup_dofs<dealii::QGaussLobatto<1>>(
        dealii::DoFHandler<3> const &,
        dealii::AffineConstraints<double> const &,
        dealii::QGaussLobatto<1> const &);
template void
    ThermalOperator<3, 8, double>::setup_dofs<dealii::QGaussLobatto<1>>(
        dealii::DoFHandler<3> const &,
        dealii::AffineConstraints<double> const &,
        dealii::QGaussLobatto<1> const &);
template void
    ThermalOperator<3, 9, double>::setup_dofs<dealii::QGaussLobatto<1>>(
        dealii::DoFHandler<3> const &,
        dealii::AffineConstraints<double> const &,
        dealii::QGaussLobatto<1> const &);
template void
    ThermalOperator<3, 10, double>::setup_dofs<dealii::QGaussLobatto<1>>(
        dealii::DoFHandler<3> const &,
        dealii::AffineConstraints<double> const &,
        dealii::QGaussLobatto<1> const &);
} // namespace adamantine
