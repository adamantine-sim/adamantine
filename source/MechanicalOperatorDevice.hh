/* SPDX-FileCopyrightText: Copyright (c) 2026, the adamantine authors.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#ifndef MECHANICAL_OPERATOR_DEVICE_HH
#define MECHANICAL_OPERATOR_DEVICE_HH

#include <BodyForce.hh>
#include <MaterialProperty.hh>

#include <deal.II/base/types.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/portable_matrix_free.h>

namespace adamantine
{
template <int dim, int fe_degree, int n_materials, int p_order, typename MaterialStates>
class MechanicalOperatorDevice
{
public:
  MechanicalOperatorDevice(
      MPI_Comm const &communicator,
      MaterialProperty<dim, n_materials, p_order, MaterialStates,
                       dealii::MemorySpace::Host> &material_properties);

  void reinit(dealii::DoFHandler<dim> const &dof_handler,
              dealii::AffineConstraints<double> const &affine_constraints);

  void vmult(dealii::LA::distributed::Vector<double,
                                             dealii::MemorySpace::Default> &dst,
             dealii::LA::distributed::Vector<
                 double, dealii::MemorySpace::Default> const &src) const;

  void vmult_add(
      dealii::LA::distributed::Vector<double, dealii::MemorySpace::Default>
          &dst,
      dealii::LA::distributed::Vector<
          double, dealii::MemorySpace::Default> const &src) const;

  void initialize_dof_vector(
      dealii::LA::distributed::Vector<double, dealii::MemorySpace::Default>
          &vector) const;

  dealii::Portable::MatrixFree<dim, double> const &get_matrix_free() const;

private:
  using kokkos_default = dealii::MemorySpace::Default::kokkos_space;

  MPI_Comm const &_communicator;
  MaterialProperty<dim, n_materials, p_order, MaterialStates,
                   dealii::MemorySpace::Host> &_material_properties;

  typename dealii::Portable::MatrixFree<dim, double>::AdditionalData
      _matrix_free_data;
  dealii::Portable::MatrixFree<dim, double> _matrix_free;

  Kokkos::View<double *, kokkos_default> _lambda;
  Kokkos::View<double *, kokkos_default> _mu;

  std::map<typename dealii::DoFHandler<dim>::cell_iterator,
           std::vector<unsigned int>>
      _cell_it_to_mf_pos;
  unsigned int _n_owned_cells = 0;
};

template <int dim, int fe_degree, int n_materials, int p_order, typename MaterialStates>
inline dealii::Portable::MatrixFree<dim, double> const &
MechanicalOperatorDevice<dim, fe_degree, n_materials, p_order,
                         MaterialStates>::get_matrix_free() const
{
  return _matrix_free;
}

} // namespace adamantine

#include <MechanicalOperatorDevice.templates.hh>

#endif
