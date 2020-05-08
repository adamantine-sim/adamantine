/* Copyright (c) 2016 - 2019, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#ifndef THERMAL_OPERATOR_HH
#define THERMAL_OPERATOR_HH

#include <MaterialProperty.hh>
#include <Operator.hh>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/matrix_free/matrix_free.h>

namespace adamantine
{
/**
 * This class is the operator associated with the heat equation, i.e., vmult
 * performs \f$ dst = -\nabla k \nabla src \f$.
 */
template <int dim, int fe_degree>
class ThermalOperator : public Operator
{
public:
  ThermalOperator(MPI_Comm const &communicator,
                  std::shared_ptr<MaterialProperty<dim>> material_properties);

  /**
   * Associate the AffineConstraints<double> and the MatrixFree objects to the
   * underlying Triangulation.
   */
  template <typename QuadratureType>
  void setup_dofs(dealii::DoFHandler<dim> const &dof_handler,
                  dealii::AffineConstraints<double> const &affine_constraints,
                  QuadratureType const &quad);

  /**
   * Compute the inverse of the mass matrix and update the material properties.
   */
  void reinit(dealii::DoFHandler<dim> const &dof_handler,
              dealii::AffineConstraints<double> const &affine_constraints);

  /**
   * Clear the MatrixFree object and resize the inverse of the mass matrix to
   * zero.
   */
  void clear();

  dealii::types::global_dof_index m() const override;

  dealii::types::global_dof_index n() const override;

  /**
   * Return a shared pointer to the inverse of the mass matrix.
   */
  std::shared_ptr<dealii::LA::distributed::Vector<double>>
  get_inverse_mass_matrix() const;

  /**
   * Return a shared pointer to the underlying MatrixFree object.
   */
  dealii::MatrixFree<dim, double> const &get_matrix_free() const;

  void vmult(dealii::LA::distributed::Vector<double> &dst,
             dealii::LA::distributed::Vector<double> const &src) const override;

  void
  Tvmult(dealii::LA::distributed::Vector<double> &dst,
         dealii::LA::distributed::Vector<double> const &src) const override;

  void
  vmult_add(dealii::LA::distributed::Vector<double> &dst,
            dealii::LA::distributed::Vector<double> const &src) const override;

  void
  Tvmult_add(dealii::LA::distributed::Vector<double> &dst,
             dealii::LA::distributed::Vector<double> const &src) const override;

  void jacobian_vmult(
      dealii::LA::distributed::Vector<double> &dst,
      dealii::LA::distributed::Vector<double> const &src) const override;

  /**
   * Evaluate the material properties for a given state field.
   */
  void evaluate_material_properties(
      dealii::LA::distributed::Vector<double> const &state);

private:
  /**
   * Apply the operator on a given set of quadrature points.
   */
  void
  local_apply(dealii::MatrixFree<dim, double> const &data,
              dealii::LA::distributed::Vector<double> &dst,
              dealii::LA::distributed::Vector<double> const &src,
              std::pair<unsigned int, unsigned int> const &cell_range) const;

  /**
   * MPI communicator.
   */
  MPI_Comm const &_communicator;
  /**
   * Data to configure the MatrixFree object.
   */
  typename dealii::MatrixFree<dim, double>::AdditionalData _matrix_free_data;
  /**
   * Store the \f$ \alpha \f$ coefficient described in
   * MaterialProperty::compute_constants()
   */
  dealii::Table<2, dealii::VectorizedArray<double>> _alpha;
  /**
   * Store the \f$ \beta \f$ coefficient described in
   * MaterialProperty::compute_constants()
   */
  dealii::Table<2, dealii::VectorizedArray<double>> _beta;
  /**
   * Table of thermal conductivity coefficient.
   */
  dealii::Table<2, dealii::VectorizedArray<double>> _thermal_conductivity;
  /**
   * Material properties associated with the domain.
   */
  std::shared_ptr<MaterialProperty<dim>> _material_properties;
  /**
   * Underlying MatrixFree object.
   */
  dealii::MatrixFree<dim, double> _matrix_free;
  /**
   * The inverse of the mass matrix is computed using an inexact Gauss-Lobatto
   * quadrature. This inexact quadrature makes the mass matrix and therefore
   * also its inverse, a diagonal matrix.
   */
  std::shared_ptr<dealii::LA::distributed::Vector<double>> _inverse_mass_matrix;
};

template <int dim, int fe_degree>
inline dealii::types::global_dof_index
ThermalOperator<dim, fe_degree>::m() const
{
  return _matrix_free.get_vector_partitioner()->size();
}

template <int dim, int fe_degree>
inline dealii::types::global_dof_index
ThermalOperator<dim, fe_degree>::n() const
{
  return _matrix_free.get_vector_partitioner()->size();
}

template <int dim, int fe_degree>
inline std::shared_ptr<dealii::LA::distributed::Vector<double>>
ThermalOperator<dim, fe_degree>::get_inverse_mass_matrix() const
{
  return _inverse_mass_matrix;
}

template <int dim, int fe_degree>
inline dealii::MatrixFree<dim, double> const &
ThermalOperator<dim, fe_degree>::get_matrix_free() const
{
  return _matrix_free;
}

template <int dim, int fe_degree>
inline void ThermalOperator<dim, fe_degree>::jacobian_vmult(
    dealii::LA::distributed::Vector<double> &dst,
    dealii::LA::distributed::Vector<double> const &src) const
{
  vmult(dst, src);
}
} // namespace adamantine

#endif
