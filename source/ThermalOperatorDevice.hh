/* Copyright (c) 2016 - 2022, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#ifndef THERMAL_OPERATOR_DEVICE_HH
#define THERMAL_OPERATOR_DEVIcE_HH

#include <MaterialProperty.hh>
#include <ThermalOperatorBase.hh>

#include <deal.II/lac/cuda_vector.h>
#include <deal.II/matrix_free/cuda_matrix_free.h>

namespace adamantine
{
template <int dim, int fe_degree, typename MemorySpaceType>
class ThermalOperatorDevice final
    : public ThermalOperatorBase<dim, MemorySpaceType>
{
public:
  ThermalOperatorDevice(MPI_Comm const &communicator,
                        BoundaryType boundary_type,
                        std::shared_ptr<MaterialProperty<dim, MemorySpaceType>>
                            material_properties);

  void reinit(dealii::DoFHandler<dim> const &dof_handler,
              dealii::AffineConstraints<double> const &affine_constraints,
              dealii::hp::QCollection<1> const &q_collection) override;

  void compute_inverse_mass_matrix(
      dealii::DoFHandler<dim> const &dof_handler,
      dealii::AffineConstraints<double> const &affine_constraints,
      dealii::hp::FECollection<dim> const &fe_collection) override;

  void clear() override;

  dealii::types::global_dof_index m() const override;

  dealii::types::global_dof_index n() const override;

  dealii::CUDAWrappers::MatrixFree<dim, double> const &get_matrix_free() const;

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

  std::shared_ptr<dealii::LA::distributed::Vector<double, MemorySpaceType>>
  get_inverse_mass_matrix() const override;

  void initialize_dof_vector(
      dealii::LA::distributed::Vector<double, MemorySpaceType> &vector)
      const override;

  void evaluate_material_properties(
      dealii::LA::distributed::Vector<double, MemorySpaceType> const &state)
      override;

  void get_state_from_material_properties() override;

  void set_state_to_material_properties() override;

  /**
   * Set the deposition cosine and sine angles and convert the data from
   * std::vector to dealii::LinearAlgebra::CUDAWrappers::Vector<double>
   */
  void set_material_deposition_orientation(
      std::vector<double> const &deposition_cos,
      std::vector<double> const &deposition_sin);

  void set_time_and_source_height(double, double) override
  {
    // TODO
  }

  /**
   * Update \f$ \frac{1}{\rho C_p} \f$ on the cells using the values computed at
   * the quadrature points.
   */
  void update_inv_rho_cp_cell();

  /**
   * Return the value of \f$ \frac{1}{\rho C_p} \f$ for a given cell and
   * quadrature point.
   */
  double
  get_inv_rho_cp(typename dealii::DoFHandler<dim>::cell_iterator const &cell,
                 unsigned int q) const override;

private:
  MPI_Comm const &_communicator;
  BoundaryType _boundary_type;
  dealii::types::global_dof_index _m;
  unsigned int _n_owned_cells;
  typename dealii::CUDAWrappers::MatrixFree<dim, double>::AdditionalData
      _matrix_free_data;
  std::shared_ptr<MaterialProperty<dim, MemorySpaceType>> _material_properties;
  dealii::CUDAWrappers::MatrixFree<dim, double> _matrix_free;
  MemoryBlock<double, dealii::MemorySpace::CUDA> _liquid_ratio;
  MemoryBlock<double, dealii::MemorySpace::CUDA> _powder_ratio;
  MemoryBlock<double, dealii::MemorySpace::CUDA> _material_id;
  MemoryBlock<double, dealii::MemorySpace::CUDA> _inv_rho_cp;
  dealii::LinearAlgebra::CUDAWrappers::Vector<double> _deposition_cos;
  dealii::LinearAlgebra::CUDAWrappers::Vector<double> _deposition_sin;
  std::map<typename dealii::DoFHandler<dim>::cell_iterator,
           std::vector<unsigned int>>
      _cell_it_to_mf_pos;
  std::shared_ptr<dealii::LA::distributed::Vector<double, MemorySpaceType>>
      _inverse_mass_matrix;
  std::map<typename dealii::DoFHandler<dim>::cell_iterator, double>
      _inv_rho_cp_cells;
};

template <int dim, int fe_degree, typename MemorySpaceType>
inline dealii::types::global_dof_index
ThermalOperatorDevice<dim, fe_degree, MemorySpaceType>::m() const
{
  return _m;
}

template <int dim, int fe_degree, typename MemorySpaceType>
inline dealii::types::global_dof_index
ThermalOperatorDevice<dim, fe_degree, MemorySpaceType>::n() const
{
  // Operator must be square
  return _m;
}

template <int dim, int fe_degree, typename MemorySpaceType>
inline std::shared_ptr<dealii::LA::distributed::Vector<double, MemorySpaceType>>
ThermalOperatorDevice<dim, fe_degree,
                      MemorySpaceType>::get_inverse_mass_matrix() const
{
  return _inverse_mass_matrix;
}

template <int dim, int fe_degree, typename MemorySpaceType>
inline dealii::CUDAWrappers::MatrixFree<dim, double> const &
ThermalOperatorDevice<dim, fe_degree, MemorySpaceType>::get_matrix_free() const
{
  return _matrix_free;
}

template <int dim, int fe_degree, typename MemorySpaceType>
inline void
ThermalOperatorDevice<dim, fe_degree, MemorySpaceType>::jacobian_vmult(
    dealii::LA::distributed::Vector<double, MemorySpaceType> &dst,
    dealii::LA::distributed::Vector<double, MemorySpaceType> const &src) const
{
  vmult(dst, src);
}

template <int dim, int fe_degree, typename MemorySpaceType>
inline double
ThermalOperatorDevice<dim, fe_degree, MemorySpaceType>::get_inv_rho_cp(
    typename dealii::DoFHandler<dim>::cell_iterator const &cell,
    unsigned int) const
{
  auto inv_rho_cp = _inv_rho_cp_cells.find(cell);
  ASSERT(inv_rho_cp != _inv_rho_cp_cells.end(), "Internal error");

  return inv_rho_cp->second;
}

} // namespace adamantine

#endif
