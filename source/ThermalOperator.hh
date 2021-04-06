/* Copyright (c) 2016 - 2021, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#ifndef THERMAL_OPERATOR_HH
#define THERMAL_OPERATOR_HH

#include <HeatSource.hh>
#include <MaterialProperty.hh>
#include <ThermalOperatorBase.hh>

#include <deal.II/matrix_free/matrix_free.h>

namespace adamantine
{
/**
 * This class is the operator associated with the heat equation, i.e., vmult
 * performs \f$ dst = -\nabla k \nabla src \f$.
 */
template <int dim, int fe_degree, typename MemorySpaceType>
class ThermalOperator final : public ThermalOperatorBase<dim, MemorySpaceType>
{
public:
  ThermalOperator(
      MPI_Comm const &communicator,
      std::shared_ptr<MaterialProperty<dim>> material_properties,
      std::vector<std::shared_ptr<HeatSource<dim>>> const &heat_sources);

  /**
   * Associate the AffineConstraints<double> and the MatrixFree objects to the
   * underlying Triangulation.
   */
  void reinit(dealii::DoFHandler<dim> const &dof_handler,
              dealii::AffineConstraints<double> const &affine_constraints,
              dealii::hp::QCollection<1> const &quad) override;

  /**
   * Compute the inverse of the mass matrix and update the material properties.
   */
  void compute_inverse_mass_matrix(
      dealii::DoFHandler<dim> const &dof_handler,
      dealii::AffineConstraints<double> const &affine_constraints,
      dealii::hp::FECollection<dim> const &fe_collection) override;

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
  std::shared_ptr<dealii::LA::distributed::Vector<double, MemorySpaceType>>
  get_inverse_mass_matrix() const override;

  /**
   * Return a shared pointer to the underlying MatrixFree object.
   */
  dealii::MatrixFree<dim, double> const &get_matrix_free() const;

  void vmult(dealii::LA::distributed::Vector<double, MemorySpaceType> &dst,
             dealii::LA::distributed::Vector<double, MemorySpaceType> const
                 &src) const override;

  void Tvmult(dealii::LA::distributed::Vector<double, MemorySpaceType> &dst,
              dealii::LA::distributed::Vector<double, MemorySpaceType> const
                  &src) const override;

  void vmult_add(dealii::LA::distributed::Vector<double, MemorySpaceType> &dst,
                 dealii::LA::distributed::Vector<double, MemorySpaceType> const
                     &src) const override;

  void Tvmult_add(dealii::LA::distributed::Vector<double, MemorySpaceType> &dst,
                  dealii::LA::distributed::Vector<double, MemorySpaceType> const
                      &src) const override;

  void
  jacobian_vmult(dealii::LA::distributed::Vector<double, MemorySpaceType> &dst,
                 dealii::LA::distributed::Vector<double, MemorySpaceType> const
                     &src) const override;

  void initialize_dof_vector(
      dealii::LA::distributed::Vector<double, MemorySpaceType> &vector)
      const override;
  /**
   * Extract the stateful properties from the _material_properties object and
   * populate new vectors with the correct order.
   */
  void extract_stateful_material_properties() override;
  /**
   * Modify the properties from the MaterialProperty object to
   * match the evolved values in ThermalOperator. The main purpose of this is to
   * update the material state variables in the MaterialProperty object.
   */
  void sync_material_properties(
      dealii::LA::distributed::Vector<double, MemorySpaceType> const
          &temperature) override;

  /**
   * Update the time and the current height, to be used to calculate the source
   * term.
   */
  void update_time_and_height(double time, double height) override;
  /**
   * Changes the stateful material properties to the designated values for the
   * entire domain. This is primarily for testing purposes.
   */
  void set_stateful_material_property(
      const double powder_ratio,
      const dealii::types::material_id material_id) const;

private:
  /**
   * Apply the operator on a given set of quadrature points.
   */
  void local_apply(
      dealii::MatrixFree<dim, double> const &data,
      dealii::LA::distributed::Vector<double, MemorySpaceType> &dst,
      dealii::LA::distributed::Vector<double, MemorySpaceType> const &src,
      std::pair<unsigned int, unsigned int> const &cell_range) const;

  /**
   * Calculates the inverse of the density times the heat capacity.
   */
  dealii::VectorizedArray<double>
  get_inv_rho_cp(unsigned int cell, unsigned int q,
                 std::array<dealii::VectorizedArray<double>,
                            static_cast<unsigned int>(MaterialState::SIZE)>
                     state_ratios,
                 dealii::VectorizedArray<double> temperature) const;

  /**
   * Calculates the thermal conductivity.
   */
  dealii::VectorizedArray<double> get_thermal_conductivity(
      unsigned int cell, unsigned int q,
      std::array<dealii::VectorizedArray<double>,
                 static_cast<unsigned int>(MaterialState::SIZE)>
          state_ratios,
      dealii::VectorizedArray<double> temperature) const;
  /**
   * Calculate the state ratios given the temperature and the current powder
   * ratio. Even though this is const, it modifies the mutable _powder_ratio
   * member (this method is const because it is called in local_apply which has
   * to be const).
   */
  void
  update_state_ratios(unsigned int cell, unsigned int q,
                      dealii::VectorizedArray<double> temperature,
                      std::array<dealii::VectorizedArray<double>,
                                 static_cast<unsigned int>(MaterialState::SIZE)>
                          &state_ratios) const;
  /**
   * MPI communicator.
   */
  MPI_Comm const &_communicator;
  /**
   * Data to configure the MatrixFree object.
   */
  typename dealii::MatrixFree<dim, double>::AdditionalData _matrix_free_data;

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
  std::shared_ptr<dealii::LA::distributed::Vector<double, MemorySpaceType>>
      _inverse_mass_matrix;
  /**
   * Map between the cell iterator and the position in _inv_rho_cp table.
   */
  std::map<typename dealii::DoFHandler<dim>::cell_iterator,
           std::pair<unsigned int, unsigned int>>
      _cell_it_to_mf_cell_map;

  double _time;
  double _current_height;
  std::vector<std::shared_ptr<HeatSource<dim>>> _heat_sources;

  /**
   * Table of the powder fraction, mutable so that it can be changed in
   * local_apply, which is const.
   */
  mutable dealii::Table<2, dealii::VectorizedArray<double>> _powder_ratio;
  /**
   * Table of the material index, mutable so that it can be changed in
   * local_apply, which is const.
   */
  mutable dealii::Table<2, std::array<dealii::types::material_id,
                                      dealii::VectorizedArray<double>::size()>>
      _material_id;
};

template <int dim, int fe_degree, typename MemorySpaceType>
inline dealii::types::global_dof_index
ThermalOperator<dim, fe_degree, MemorySpaceType>::m() const
{
  return _matrix_free.get_vector_partitioner()->size();
}

template <int dim, int fe_degree, typename MemorySpaceType>
inline dealii::types::global_dof_index
ThermalOperator<dim, fe_degree, MemorySpaceType>::n() const
{
  return _matrix_free.get_vector_partitioner()->size();
}

template <int dim, int fe_degree, typename MemorySpaceType>
inline std::shared_ptr<dealii::LA::distributed::Vector<double, MemorySpaceType>>
ThermalOperator<dim, fe_degree, MemorySpaceType>::get_inverse_mass_matrix()
    const
{
  return _inverse_mass_matrix;
}

template <int dim, int fe_degree, typename MemorySpaceType>
inline dealii::MatrixFree<dim, double> const &
ThermalOperator<dim, fe_degree, MemorySpaceType>::get_matrix_free() const
{
  return _matrix_free;
}

template <int dim, int fe_degree, typename MemorySpaceType>
inline void ThermalOperator<dim, fe_degree, MemorySpaceType>::jacobian_vmult(
    dealii::LA::distributed::Vector<double, MemorySpaceType> &dst,
    dealii::LA::distributed::Vector<double, MemorySpaceType> const &src) const
{
  vmult(dst, src);
}

template <int dim, int fe_degree, typename MemorySpaceType>
inline void
ThermalOperator<dim, fe_degree, MemorySpaceType>::initialize_dof_vector(
    dealii::LA::distributed::Vector<double, MemorySpaceType> &vector) const
{
  _matrix_free.initialize_dof_vector(vector);
}

template <int dim, int fe_degree, typename MemorySpaceType>
inline void
ThermalOperator<dim, fe_degree, MemorySpaceType>::update_time_and_height(
    double time, double height)
{
  _time = time;
  _current_height = height;
}

} // namespace adamantine

#endif
