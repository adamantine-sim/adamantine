/* Copyright (c) 2016, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#ifndef _THERMAL_OPERATOR_HH_
#define _THERMAL_OPERATOR_HH_

#include "MaterialProperty.hh"
#include "Operator.hh"
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <boost/mpi.hpp>

namespace adamantine
{

template <int dim, int fe_degree, typename NumberType>
class ThermalOperator : public Operator<NumberType>
{
public:
  ThermalOperator(boost::mpi::communicator &communicator,
                  std::shared_ptr<MaterialProperty> material_properties);

  /**
   * Reinit must be called after the constructor. The reason is that we cannot
   * create a templated constructor.
   */
  template <typename QuadratureType>
  void reinit(dealii::DoFHandler<dim> const &dof_handler,
              dealii::ConstraintMatrix const &constraint_matrix,
              QuadratureType const &quad);

  void clear();

  dealii::types::global_dof_index m() const override;

  dealii::types::global_dof_index n() const override;

  /**
   * This performs \f$ dst = -\nabla k \nabla src \f$.
   */
  void
  vmult(dealii::LA::distributed::Vector<NumberType> &dst,
        dealii::LA::distributed::Vector<NumberType> const &src) const override;

  void
  Tvmult(dealii::LA::distributed::Vector<NumberType> &dst,
         dealii::LA::distributed::Vector<NumberType> const &src) const override;

  void vmult_add(
      dealii::LA::distributed::Vector<NumberType> &dst,
      dealii::LA::distributed::Vector<NumberType> const &src) const override;

  void Tvmult_add(
      dealii::LA::distributed::Vector<NumberType> &dst,
      dealii::LA::distributed::Vector<NumberType> const &src) const override;

private:
  void
  local_apply(dealii::MatrixFree<dim, NumberType> const &data,
              dealii::LA::distributed::Vector<NumberType> &dst,
              dealii::LA::distributed::Vector<NumberType> const &src,
              std::pair<unsigned int, unsigned int> const &cell_range) const;

  void evaluate_thermal_conductivity(
      dealii::LA::distributed::Vector<NumberType> const &state);

  boost::mpi::communicator _communicator;
  dealii::Table<2, dealii::VectorizedArray<NumberType>> _thermal_conductivity;
  std::shared_ptr<MaterialProperty> _material_properties;
  dealii::MatrixFree<dim, NumberType> _data;
  /**
   * Compute the inverse of the mass matrix using an inexact Gauss-Lobatto
   * quadrature. This inexact quadrature makes the mass matrix and therefore
   * also its inverse, a diagonal matrix.
   */
  dealii::LA::distributed::Vector<double> _inverse_mass_matrix;
};

template <int dim, int fe_degree, typename NumberType>
inline dealii::types::global_dof_index
ThermalOperator<dim, fe_degree, NumberType>::m() const
{
  _data.get_vector_partitioner()->size();
}

template <int dim, int fe_degree, typename NumberType>
inline dealii::types::global_dof_index
ThermalOperator<dim, fe_degree, NumberType>::n() const
{
  _data.get_vector_partitioner()->size();
}
}

#endif
