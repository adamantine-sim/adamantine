/* Copyright (c) 2016 - 2023, the adamantine authors.
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

#include <deal.II/base/aligned_vector.h>
#include <deal.II/base/vectorization.h>
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
      MPI_Comm const &communicator, BoundaryType boundary_type,
      MaterialProperty<dim, MemorySpaceType> &material_properties,
      std::vector<std::shared_ptr<HeatSource<dim>>> const &heat_sources);

  /**
   * Associate the AffineConstraints<double> and the MatrixFree objects to the
   * underlying Triangulation.
   */
  void reinit(dealii::DoFHandler<dim> const &dof_handler,
              dealii::AffineConstraints<double> const &affine_constraints,
              dealii::hp::QCollection<1> const &quad) override;

  /**
   * Compute the inverse of the mass matrix.
   */
  void compute_inverse_mass_matrix(
      dealii::DoFHandler<dim> const &dof_handler,
      dealii::AffineConstraints<double> const &affine_constraints) override;

  /**
   * Resize the inverse of the mass matrix to zero.
   */
  void clear() override;

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

  void get_state_from_material_properties() override;

  void set_state_to_material_properties() override;

  /**
   * Set the deposition cosine and sine angles and convert the data from
   * std::vector to dealii::Table<2, dealii::VectorizedArray>
   */
  void set_material_deposition_orientation(
      std::vector<double> const &deposition_cos,
      std::vector<double> const &deposition_sin) override;

  void set_time_and_source_height(double t, double height) override;

private:
  /**
   * Update the ratios of the material state.
   */
  void
  update_state_ratios(unsigned int cell, unsigned int q,
                      dealii::VectorizedArray<double> temperature,
                      std::array<dealii::VectorizedArray<double>,
                                 static_cast<unsigned int>(MaterialState::SIZE)>
                          &state_ratios) const;

  /**
   * Update the ratios of the material state at the face quadrature points.
   */
  void update_face_state_ratios(
      unsigned int face, unsigned int q,
      dealii::VectorizedArray<double> temperature,
      std::array<dealii::VectorizedArray<double>,
                 static_cast<unsigned int>(MaterialState::SIZE)> &state_ratios)
      const;
  /**
   * Return the value of \f$ \frac{1}{\rho C_p} \f$ for a given matrix-free
   * cell/face and quadrature point.
   */
  dealii::VectorizedArray<double> get_inv_rho_cp(
      std::array<dealii::types::material_id,
                 dealii::VectorizedArray<double>::size()> const &material_id,
      std::array<dealii::VectorizedArray<double>,
                 static_cast<unsigned int>(MaterialState::SIZE)> const
          &state_ratios,
      dealii::VectorizedArray<double> const &temperature,
      dealii::AlignedVector<dealii::VectorizedArray<double>> const
          &temperature_powers) const;

  /**
   * Apply the operator on a given set of quadrature points inside each cell.
   */
  void cell_local_apply(
      dealii::MatrixFree<dim, double> const &data,
      dealii::LA::distributed::Vector<double, MemorySpaceType> &dst,
      dealii::LA::distributed::Vector<double, MemorySpaceType> const &src,
      std::pair<unsigned int, unsigned int> const &cell_range) const;

  /**
   * Apply the operator on a given set of quadrature points on each face.
   */
  void face_local_apply(
      dealii::MatrixFree<dim, double> const &data,
      dealii::LA::distributed::Vector<double, MemorySpaceType> &dst,
      dealii::LA::distributed::Vector<double, MemorySpaceType> const &src,
      std::pair<unsigned int, unsigned int> const &face_range) const;

  /**
   * Apply the mass operator on a given set of quadrature points.
   */
  void cell_local_mass(
      dealii::MatrixFree<dim, double> const &data,
      dealii::LA::distributed::Vector<double, MemorySpaceType> &dst,
      dealii::LA::distributed::Vector<double, MemorySpaceType> const &src,
      std::pair<unsigned int, unsigned int> const &cell_range) const;

  /**
   * MPI communicator.
   */
  MPI_Comm const &_communicator;
  /**
   * Type of boundary.
   */
  BoundaryType _boundary_type;
  /**
   * Current height of the heat sources.
   */
  double _current_source_height = 0.;
  /**
   * Data to configure the MatrixFree object.
   */
  typename dealii::MatrixFree<dim, double>::AdditionalData _matrix_free_data;
  /**
   * Table of thermal conductivity coefficient.
   */
  dealii::Table<2, dealii::VectorizedArray<double>> _thermal_conductivity;
  /**
   * Material properties associated with the domain.
   */
  MaterialProperty<dim, MemorySpaceType> &_material_properties;
  /**
   * Vector of heat sources.
   */
  std::vector<std::shared_ptr<HeatSource<dim>>> _heat_sources;
  /**
   * Underlying MatrixFree object.
   */
  dealii::MatrixFree<dim, double> _matrix_free;
  /**
   * Non-owning pointer to the AffineConstraints from ThermalPhysics.
   */
  dealii::AffineConstraints<double> const *_affine_constraints;
  /**
   * The inverse of the mass matrix is computed using an inexact
   * Gauss-Lobatto quadrature. This inexact quadrature makes the mass matrix
   * and therefore also its inverse, a diagonal matrix.
   */
  std::shared_ptr<dealii::LA::distributed::Vector<double, MemorySpaceType>>
      _inverse_mass_matrix;
  /**
   * Map between the cell iterator and the position in _inv_rho_cp table.
   */
  std::map<typename dealii::DoFHandler<dim>::cell_iterator,
           std::pair<unsigned int, unsigned int>>
      _cell_it_to_mf_cell_map;
  /**
   * Table of the powder fraction inside cells; mutable so that it can be
   * changed in cell_local_apply which is const.
   */
  mutable dealii::Table<2, dealii::VectorizedArray<double>> _liquid_ratio;
  /**
   * Table of the powder fraction inside cells; mutable so that it can be
   * changed in cell_local_apply which is const.
   */
  mutable dealii::Table<2, dealii::VectorizedArray<double>> _powder_ratio;
  /**
   * Table of the powder fraction on faces; mutable so that it can be changed in
   * face_local_apply which is const.
   */
  mutable dealii::Table<2, dealii::VectorizedArray<double>> _face_powder_ratio;
  /**
   * Table of the material index inside cells; mutable so that it can be changed
   * in cell_local_apply which is const.
   */
  mutable dealii::Table<2, std::array<dealii::types::material_id,
                                      dealii::VectorizedArray<double>::size()>>
      _material_id;
  /**
   * Table of the material index on faces; mutable so that it can be changed in
   * face_local_apply which is const.
   */
  mutable dealii::Table<2, std::array<dealii::types::material_id,
                                      dealii::VectorizedArray<double>::size()>>
      _face_material_id;
  /**
   * Table of the material deposition cosine angles.
   */
  dealii::Table<2, dealii::VectorizedArray<double>> _deposition_cos;
  /**
   * Table of the material deposition cosine angles.
   */
  dealii::Table<2, dealii::VectorizedArray<double>> _deposition_sin;
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
ThermalOperator<dim, fe_degree, MemorySpaceType>::set_time_and_source_height(
    double t, double height)
{
  _current_source_height = height;
  for (auto &beam : _heat_sources)
    beam->update_time(t);
}
} // namespace adamantine

#endif
