/* SPDX-FileCopyrightText: Copyright (c) 2016 - 2025, the adamantine authors.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#ifndef THERMAL_OPERATOR_DEVICE_HH
#define THERMAL_OPERATOR_DEVICE_HH

#include <Boundary.hh>
#include <MaterialProperty.hh>
#include <ThermalOperatorBase.hh>

#include <deal.II/base/types.h>
#include <deal.II/matrix_free/cuda_matrix_free.h>

namespace adamantine
{
template <int dim, bool use_table, int p_order, int fe_degree,
          typename MaterialStates, typename MemorySpaceType>
class ThermalOperatorDevice final
    : public ThermalOperatorBase<dim, MemorySpaceType>
{
public:
  ThermalOperatorDevice(MPI_Comm const &communicator, Boundary const &boundary,
                        MaterialProperty<dim, p_order, MaterialStates,
                                         MemorySpaceType> &material_properties);

  void reinit(dealii::DoFHandler<dim> const &dof_handler,
              dealii::AffineConstraints<double> const &affine_constraints,
              dealii::hp::QCollection<1> const &q_collection) override;

  void compute_inverse_mass_matrix(
      dealii::DoFHandler<dim> const &dof_handler,
      dealii::AffineConstraints<double> const &affine_constraints) override;

  void clear() override;

  dealii::types::global_dof_index m() const override;

  dealii::types::global_dof_index n() const override;

  dealii::CUDAWrappers::MatrixFree<dim, double> const &get_matrix_free() const;

  void vmult(dealii::LA::distributed::Vector<double, MemorySpaceType> &dst,
             dealii::LA::distributed::Vector<double, MemorySpaceType> const
                 &src) const override;

  void vmult_add(dealii::LA::distributed::Vector<double, MemorySpaceType> &dst,
                 dealii::LA::distributed::Vector<double, MemorySpaceType> const
                     &src) const override;

  std::shared_ptr<dealii::LA::distributed::Vector<double, MemorySpaceType>>
  get_inverse_mass_matrix() const override;

  void initialize_dof_vector(
      dealii::LA::distributed::Vector<double, MemorySpaceType> &vector)
      const override;

  void update_boundary_material_properties(
      dealii::LA::distributed::Vector<double, MemorySpaceType> const &state);

  void get_state_from_material_properties() override;

  void set_state_to_material_properties() override;

  /**
   * Set the deposition cosine and sine angles and convert the data from
   * std::vector to Kokkos::View.
   */
  void set_material_deposition_orientation(
      std::vector<double> const &deposition_cos,
      std::vector<double> const &deposition_sin) override;

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
                 unsigned int q) const;

private:
  using kokkos_default = dealii::MemorySpace::Default::kokkos_space;

  /**
   * MPI communicator.
   */
  MPI_Comm const &_communicator;
  /**
   * Flag set to true if all the boundary conditions are adiabatic. It is set to
   * false otherwise.
   */
  bool _adiabatic_only_bc = true;
  dealii::types::global_dof_index _m;
  unsigned int _n_owned_cells;
  typename dealii::CUDAWrappers::MatrixFree<dim, double>::AdditionalData
      _matrix_free_data;
  /**
   * Material properties associated with the domain.
   */
  MaterialProperty<dim, p_order, MaterialStates, MemorySpaceType>
      &_material_properties;
  dealii::CUDAWrappers::MatrixFree<dim, double> _matrix_free;
  Kokkos::View<double *, kokkos_default> _liquid_ratio;
  Kokkos::View<double *, kokkos_default> _powder_ratio;
  Kokkos::View<dealii::types::material_id *, kokkos_default> _material_id;
  Kokkos::View<double *, kokkos_default> _inv_rho_cp;
  Kokkos::View<double *, kokkos_default> _deposition_cos;
  Kokkos::View<double *, kokkos_default> _deposition_sin;
  std::map<typename dealii::DoFHandler<dim>::cell_iterator,
           std::vector<unsigned int>>
      _cell_it_to_mf_pos;
  std::shared_ptr<dealii::LA::distributed::Vector<double, MemorySpaceType>>
      _inverse_mass_matrix;
  std::map<typename dealii::DoFHandler<dim>::cell_iterator, double>
      _inv_rho_cp_cells;
};

template <int dim, bool use_table, int p_order, int fe_degree,
          typename MaterialStates, typename MemorySpaceType>
inline dealii::types::global_dof_index
ThermalOperatorDevice<dim, use_table, p_order, fe_degree, MaterialStates,
                      MemorySpaceType>::m() const
{
  return _m;
}

template <int dim, bool use_table, int p_order, int fe_degree,
          typename MaterialStates, typename MemorySpaceType>
inline dealii::types::global_dof_index
ThermalOperatorDevice<dim, use_table, p_order, fe_degree, MaterialStates,
                      MemorySpaceType>::n() const
{
  // Operator must be square
  return _m;
}

template <int dim, bool use_table, int p_order, int fe_degree,
          typename MaterialStates, typename MemorySpaceType>
inline std::shared_ptr<dealii::LA::distributed::Vector<double, MemorySpaceType>>
ThermalOperatorDevice<dim, use_table, p_order, fe_degree, MaterialStates,
                      MemorySpaceType>::get_inverse_mass_matrix() const
{
  return _inverse_mass_matrix;
}

template <int dim, bool use_table, int p_order, int fe_degree,
          typename MaterialStates, typename MemorySpaceType>
inline dealii::CUDAWrappers::MatrixFree<dim, double> const &
ThermalOperatorDevice<dim, use_table, p_order, fe_degree, MaterialStates,
                      MemorySpaceType>::get_matrix_free() const
{
  return _matrix_free;
}

template <int dim, bool use_table, int p_order, int fe_degree,
          typename MaterialStates, typename MemorySpaceType>
inline double ThermalOperatorDevice<dim, use_table, p_order, fe_degree,
                                    MaterialStates, MemorySpaceType>::
    get_inv_rho_cp(typename dealii::DoFHandler<dim>::cell_iterator const &cell,
                   unsigned int) const
{
  auto inv_rho_cp = _inv_rho_cp_cells.find(cell);
  ASSERT(inv_rho_cp != _inv_rho_cp_cells.end(), "Internal error");

  return inv_rho_cp->second;
}

} // namespace adamantine

#endif
