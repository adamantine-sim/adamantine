/* SPDX-FileCopyrightText: Copyright (c) 2026, the adamantine authors.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#ifndef MECHANICAL_OPERATOR_DEVICE_TEMPLATES_HH
#define MECHANICAL_OPERATOR_DEVICE_TEMPLATES_HH

#include <MechanicalOperatorDevice.hh>
#include <utils.hh>

#include <deal.II/base/vectorization.h>
#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/portable_fe_evaluation.h>
#include <deal.II/matrix_free/portable_matrix_free.h>

namespace adamantine
{
// Small wrapper functor type expected by MatrixFree::cell_loop
template <int dim, int fe_degree>
class LocalMechanicalOperatorDevice
{
public:
 static const unsigned int n_q_points =
      dealii::Utilities::pow(fe_degree + 1, dim);

  using kokkos_default = dealii::MemorySpace::Default::kokkos_space;
  LocalMechanicalOperatorDevice(
      Kokkos::View<double *, kokkos_default> lambda,
      Kokkos::View<double *, kokkos_default> mu)
      : _lambda(lambda), _mu(mu)
  {
  }

  KOKKOS_FUNCTION void operator()(
      typename dealii::Portable::MatrixFree<dim, double>::Data const *gpu_data,
      const dealii::Portable::DeviceVector<double> &src,
      dealii::Portable::DeviceVector<double> dst) const
  {
  dealii::Portable::FEEvaluation<dim, fe_degree, fe_degree + 1, dim, double>
      fe_eval(gpu_data);
  fe_eval.read_dof_values(src);
  fe_eval.evaluate(dealii::EvaluationFlags::gradients);

  const int cell = fe_eval.get_current_cell_index();
  
  gpu_data->for_each_quad_point([&](const int &q_point) { 
 auto const grad_u = fe_eval.get_gradient(q_point);
    auto const eps = (grad_u + dealii::transpose(grad_u)) * 0.5;
    double const trace_eps = dealii::trace(eps);

    unsigned int const pos = gpu_data->local_q_point_id(cell,
                                                         q_point);

    double const lambda = _lambda(pos);
    double const mu = _mu(pos);

    dealii::Tensor<2, dim, double> stress;
    for (unsigned int i = 0; i < dim; ++i)
      for (unsigned int j = 0; j < dim; ++j)
      {
        stress[i][j] = 2.0 * mu * eps[i][j];
        if (i == j)
          stress[i][j] += lambda * trace_eps;
      }

    fe_eval.submit_gradient(stress, q_point);
  });
  
  fe_eval.integrate(dealii::EvaluationFlags::gradients);
  fe_eval.distribute_local_to_global(dst);
  }

private:
  Kokkos::View<double *, kokkos_default> _lambda;
  Kokkos::View<double *, kokkos_default> _mu;
};

template <int dim, int n_materials, int fe_degree, typename MaterialStates>
MechanicalOperatorDevice<dim, n_materials, fe_degree, MaterialStates>::MechanicalOperatorDevice(
        MPI_Comm const &communicator,
        MaterialProperty<dim, n_materials, fe_degree, MaterialStates,
                         dealii::MemorySpace::Host> &material_properties)
    : _communicator(communicator), _material_properties(material_properties)
{
  _matrix_free_data.mapping_update_flags = dealii::update_gradients;
  //_matrix_free_data.tasks_parallel_scheme =
  //    dealii::MatrixFree<dim, double>::AdditionalData::partition_color;
}

template <int dim, int n_materials, int fe_degree, typename MaterialStates>
void MechanicalOperatorDevice<dim, n_materials, fe_degree, MaterialStates>::reinit(dealii::DoFHandler<dim> const &dof_handler,
           dealii::AffineConstraints<double> const &affine_constraints)
{
  dealii::IteratorFilters::ActiveFEIndexEqualTo filter(0, true);
  _matrix_free.reinit(dealii::StaticMappingQ1<dim>::mapping, dof_handler,
                      affine_constraints, dealii::QGauss<1>(fe_degree+1), filter,
                      _matrix_free_data);

  _n_owned_cells = dynamic_cast<dealii::parallel::DistributedTriangulationBase<dim> const *>(&
                          dof_handler.get_triangulation())
                          ->n_locally_owned_active_cells();

  // Build mapping between cells and mf positions
  _cell_it_to_mf_pos.clear();
  unsigned int constexpr n_dofs_1d = fe_degree + 1;
  unsigned int constexpr n_q_points_per_cell =
      dealii::Utilities::pow(n_dofs_1d, dim);

  auto graph = _matrix_free.get_colored_graph();
  unsigned int const n_colors = graph.size();
  for (unsigned int color = 0; color < n_colors; ++color)
  {
    auto gpu_data = _matrix_free.get_data(color);
    unsigned int const n_cells = gpu_data.n_cells;
    auto gpu_data_host = dealii::Portable::copy_mf_data_to_host<dim, double>(
        gpu_data, _matrix_free_data.mapping_update_flags);
    for (unsigned int cell_id = 0; cell_id < n_cells; ++cell_id)
    {
      auto cell = graph[color][cell_id];
      std::vector<unsigned int> quad_pos(n_q_points_per_cell);
      for (unsigned int i = 0; i < n_q_points_per_cell; ++i)
      {
        unsigned int const pos =
            gpu_data_host.local_q_point_id(cell_id, n_q_points_per_cell, i);
        quad_pos[i] = pos;
      }
      _cell_it_to_mf_pos[cell] = quad_pos;
    }
  }

  // Populate lambda and mu arrays on host then copy to device views
  unsigned int const n_coefs = n_q_points_per_cell * _n_owned_cells;
  std::vector<double> lambda_host(n_coefs);
  std::vector<double> mu_host(n_coefs);

  for (auto const &cell : dealii::filter_iterators(
           _matrix_free.get_dof_handler().active_cell_iterators(),
           dealii::IteratorFilters::LocallyOwnedCell(),
           dealii::IteratorFilters::ActiveFEIndexEqualTo(0)))
  {
    // Cast to triangulation cell to obtain material id
    typename dealii::Triangulation<dim>::active_cell_iterator cell_tria(cell);
    double const lambda_val = _material_properties.get_cell_value(
        cell_tria, StateProperty::lame_first_parameter);
    double const mu_val = _material_properties.get_cell_value(
        cell_tria, StateProperty::lame_second_parameter);

    for (unsigned int i = 0; i < n_q_points_per_cell; ++i)
    {
      unsigned int const pos = _cell_it_to_mf_pos[cell][i];
      lambda_host[pos] = lambda_val;
      mu_host[pos] = mu_val;
    }
  }

  using kokkos_default = dealii::MemorySpace::Default::kokkos_space;
  _lambda = Kokkos::View<double *, kokkos_default>(
      Kokkos::view_alloc("mech_lambda", Kokkos::WithoutInitializing),
      n_coefs);
  _mu = Kokkos::View<double *, kokkos_default>(
      Kokkos::view_alloc("mech_mu", Kokkos::WithoutInitializing), n_coefs);

  auto lambda_host_view = Kokkos::create_mirror_view(Kokkos::WithoutInitializing, _lambda);
  auto mu_host_view = Kokkos::create_mirror_view(Kokkos::WithoutInitializing, _mu);
  for (unsigned int i = 0; i < n_coefs; ++i)
  {
    lambda_host_view(i) = lambda_host[i];
    mu_host_view(i) = mu_host[i];
  }
  Kokkos::deep_copy(_lambda, lambda_host_view);
  Kokkos::deep_copy(_mu, mu_host_view);
}

template <int dim, int n_materials, int fe_degree, typename MaterialStates>
void MechanicalOperatorDevice<dim, n_materials, fe_degree, MaterialStates>::
    vmult(dealii::LA::distributed::Vector<double, dealii::MemorySpace::Default> &dst,
          dealii::LA::distributed::Vector<double, dealii::MemorySpace::Default> const &src)
    const
{
  dst = 0.;
  vmult_add(dst, src);
}

template <int dim, int n_materials, int fe_degree, typename MaterialStates>
void MechanicalOperatorDevice<dim, n_materials, fe_degree, MaterialStates>::vmult_add(dealii::LA::distributed::Vector<double, dealii::MemorySpace::Default> &dst,
              dealii::LA::distributed::Vector<double, dealii::MemorySpace::Default> const &src)
    const
{
  LocalMechanicalOperatorDevice<dim, fe_degree> local_operator(_lambda, _mu);
  _matrix_free.cell_loop(local_operator, src, dst);
  _matrix_free.copy_constrained_values(src, dst);
}

template <int dim, int n_materials, int fe_degree, typename MaterialStates>
void MechanicalOperatorDevice<dim, n_materials, fe_degree, MaterialStates>::initialize_dof_vector(
        dealii::LA::distributed::Vector<double, dealii::MemorySpace::Default> &vector)
    const
{
  _matrix_free.initialize_dof_vector(vector);
}

} // namespace adamantine

#endif
