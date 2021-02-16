/* Copyright (c) 2016 - 2021, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#ifndef THERMAL_OPERATOR_DEVICE_HH
#define THERMAL_OPERATOR_DEVICE_HH

#include <HeatSource.hh>
#include <MaterialProperty.hh>
#include <ThermalOperatorBase.hh>

#include <deal.II/lac/read_write_vector.h>
#include <deal.II/lac/cuda_vector.h>
#include <deal.II/matrix_free/cuda_matrix_free.h>

namespace adamantine
{
template <int dim, int fe_degree, typename MemorySpaceType>
class ThermalOperatorDevice final
    : public ThermalOperatorBase<dim, MemorySpaceType>
{
public:
  ThermalOperatorDevice(
      MPI_Comm const &communicator,
      std::shared_ptr<MaterialProperty<dim>> material_properties,
      std::vector<std::shared_ptr<HeatSource<dim>>> heat_sources);

  void reinit(dealii::DoFHandler<dim> const &dof_handler,
              dealii::AffineConstraints<double> const &affine_constraints,
              dealii::hp::QCollection<1> const &q_collection) override;

  void compute_inverse_mass_matrix(
      dealii::DoFHandler<dim> const &dof_handler,
      dealii::AffineConstraints<double> const &affine_constraints,
      dealii::hp::FECollection<dim> const &fe_collection) override;

  void clear();

  dealii::types::global_dof_index m() const override;

  dealii::types::global_dof_index n() const override;

  std::shared_ptr<dealii::LA::distributed::Vector<double, MemorySpaceType>>
  get_inverse_mass_matrix() const override;

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

  void initialize_dof_vector(
      dealii::LA::distributed::Vector<double, MemorySpaceType> &vector)
      const override;

  /*
    void evaluate_material_properties(
        dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host> const
            &state) override;
  */
  /**
   * Return the value of \f$ \frac{1}{\rho C_p} \f$ for a given cell.
   */
  /*
 double get_inv_rho_cp(
     typename dealii::DoFHandler<dim>::cell_iterator const &) const override;
*/
  /**
   * Extract the stateful properties from the _material_properties object and
   * populate new vectors with the correct order.
   */
  void extract_stateful_material_properties(
      dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host> const & vector)
      override;
  /**
   * Modify the stateful properties from the _material_properties object to
   * match the evolved values in ThermalOperator.
   */
  void sync_stateful_material_properties() override {};
  /**
   * Update the time and the current height, to be used to calculate the source
   * term.
   */
  void update_time_and_height(double time, double height) override;

private:
  MPI_Comm const &_communicator;
  dealii::types::global_dof_index _m;
  unsigned int _n_owned_cells;
  typename dealii::CUDAWrappers::MatrixFree<dim, double>::AdditionalData
      _matrix_free_data;
  std::shared_ptr<MaterialProperty<dim>> _material_properties;
  dealii::CUDAWrappers::MatrixFree<dim, double> _matrix_free;
  // dealii::LinearAlgebra::CUDAWrappers::Vector<double> _inv_rho_cp;
  // dealii::LinearAlgebra::CUDAWrappers::Vector<double> _thermal_conductivity;
  std::shared_ptr<dealii::LA::distributed::Vector<double, MemorySpaceType>>
      _inverse_mass_matrix;
  // std::map<typename dealii::DoFHandler<dim>::cell_iterator, double>
  //  _inv_rho_cp_cells;

  // New implementation
  double _time;
  double _current_height;
  std::vector<std::shared_ptr<HeatSource<dim>>> _heat_sources;
  dealii::LinearAlgebra::CUDAWrappers::Vector<double> _powder_ratio;
  //dealii::LinearAlgebra::CUDAWrappers::Vector<dealii::types::material_id>
  //    _material_id;
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

/*
template <int dim, int fe_degree, typename MemorySpaceType>
inline double
ThermalOperatorDevice<dim, fe_degree, MemorySpaceType>::get_inv_rho_cp(
    typename dealii::DoFHandler<dim>::cell_iterator const &cell) const
{
  // Temporary hard coding
  //auto inv_rho_cp = _inv_rho_cp_cells.find(cell);
  //ASSERT(inv_rho_cp != _inv_rho_cp_cells.end(), "Internal error");

  return 1.0; //inv_rho_cp->second;
}
*/

template <int dim, int fe_degree, typename MemorySpaceType>
inline void
ThermalOperatorDevice<dim, fe_degree, MemorySpaceType>::update_time_and_height(
    double time, double height)
{
  _time = time;
  _current_height = height;
}

} // namespace adamantine

#endif
