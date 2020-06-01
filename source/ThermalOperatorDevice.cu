/* Copyright (c) 2016 - 2020, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#include <ThermalOperatorDevice.hh>
#include <instantiation.hh>

#include <deal.II/base/cuda_size.h>
#include <deal.II/matrix_free/cuda_fe_evaluation.h>

namespace
{
// Some form of this should be in deal.II
template <int dim, unsigned int n_dofs_1d, unsigned int n_q_points>
std::array<std::array<unsigned int, dim>, n_q_points> get_ijk()
{
  std::array<std::array<unsigned int, dim>, n_q_points> ijk;
  if (dim == 1)
  {
    for (unsigned int i = 0; i < n_q_points; ++i)
      ijk[i][0] = i % n_dofs_1d;
  }
  else if (dim == 2)
  {
    for (unsigned int i = 0; i < n_q_points; ++i)
    {
      ijk[i][0] = i % n_dofs_1d;
      ijk[i][1] = i / n_dofs_1d;
    }
  }
  else if (dim == 3)
  {
    for (unsigned int i = 0; i < n_q_points; ++i)
    {
      ijk[i][0] = i % n_dofs_1d;
      ijk[i][1] = (i / n_dofs_1d) % n_dofs_1d;
      ijk[i][2] = (i / n_dofs_1d) / n_dofs_1d;
    }
  }

  return ijk;
}

__global__ void invert_mass_matrix(double *values, unsigned int size)
{
  unsigned int i = threadIdx.x + blockIdx.x * gridDim.x;
  if (i < size)
  {
    if (values[i] > 1e-15)
      values[i] = 1. / values[i];
    else
      values[i] = 0.;
  }
}

template <int dim, int fe_degree>
class MassMatrixOperatorQuad
{
public:
  __device__ void
  operator()(dealii::CUDAWrappers::FEEvaluation<dim, fe_degree> *fe_eval) const;
};

template <int dim, int fe_degree>
__device__ void MassMatrixOperatorQuad<dim, fe_degree>::
operator()(dealii::CUDAWrappers::FEEvaluation<dim, fe_degree> *fe_eval) const
{
  fe_eval->submit_value(1.);
}

template <int dim, int fe_degree>
class LocalMassMarixOperator
{
public:
  __device__ void
  operator()(unsigned int const cell,
             typename dealii::CUDAWrappers::MatrixFree<dim, double>::Data const
                 *gpu_data,
             dealii::CUDAWrappers::SharedData<dim, double> *shared_data,
             double const *src, double *dst) const;

  static const unsigned int n_dofs_1d = fe_degree + 1;
  static const unsigned int n_local_dofs =
      dealii::Utilities::pow(fe_degree + 1, dim);
  static const unsigned int n_q_points =
      dealii::Utilities::pow(fe_degree + 1, dim);
};

template <int dim, int fe_degree>
__device__ void LocalMassMarixOperator<dim, fe_degree>::
operator()(unsigned int const cell,
           typename dealii::CUDAWrappers::MatrixFree<dim, double>::Data const
               *gpu_data,
           dealii::CUDAWrappers::SharedData<dim, double> *shared_data,
           double const * /*src*/, double *dst) const
{
  unsigned int const pos = dealii::CUDAWrappers::local_q_point_id<dim, double>(
      cell, gpu_data, n_dofs_1d, n_q_points);
  dealii::CUDAWrappers::FEEvaluation<dim, fe_degree, fe_degree + 1, 1, double>
      fe_eval(cell, gpu_data, shared_data);
  fe_eval.apply_for_each_quad_point(MassMatrixOperatorQuad<dim, fe_degree>());
  fe_eval.integrate(true, false);
  fe_eval.distribute_local_to_global(dst);
}

template <int dim, int fe_degree>
class ThermalOperatorQuad
{
public:
  __device__ ThermalOperatorQuad(double thermal_conductivity, double alpha)
      : _thermal_conductivity(thermal_conductivity), _alpha(alpha)
  {
  }

  __device__ void
  operator()(dealii::CUDAWrappers::FEEvaluation<dim, fe_degree> *fe_eval) const;

private:
  double _thermal_conductivity;
  // alpha = density * specifid heat
  double _alpha;
};

template <int dim, int fe_degree>
__device__ void ThermalOperatorQuad<dim, fe_degree>::
operator()(dealii::CUDAWrappers::FEEvaluation<dim, fe_degree> *fe_eval) const
{
  // coef = - thermal conductivity (grad * (density * specific heat))
  fe_eval->submit_gradient(-_thermal_conductivity *
                           (fe_eval->get_gradient() * _alpha));
}

template <int dim, int fe_degree>
class LocalThermalOperatorDevice
{
public:
  LocalThermalOperatorDevice(double *thermal_conductivity, double *alpha)
      : _thermal_conductivity(thermal_conductivity), _alpha(alpha)
  {
  }

  __device__ void
  operator()(unsigned int const cell,
             typename dealii::CUDAWrappers::MatrixFree<dim, double>::Data const
                 *gpu_data,
             dealii::CUDAWrappers::SharedData<dim, double> *shared_data,
             double const *src, double *dst) const;

  static const unsigned int n_dofs_1d = fe_degree + 1;
  static const unsigned int n_local_dofs =
      dealii::Utilities::pow(fe_degree + 1, dim);
  static const unsigned int n_q_points =
      dealii::Utilities::pow(fe_degree + 1, dim);

private:
  double *_thermal_conductivity;
  double *_alpha;
};

template <int dim, int fe_degree>
__device__ void LocalThermalOperatorDevice<dim, fe_degree>::
operator()(unsigned int const cell,
           typename dealii::CUDAWrappers::MatrixFree<dim, double>::Data const
               *gpu_data,
           dealii::CUDAWrappers::SharedData<dim, double> *shared_data,
           double const *src, double *dst) const
{
  unsigned int const pos = dealii::CUDAWrappers::local_q_point_id<dim, double>(
      cell, gpu_data, n_dofs_1d, n_q_points);
  dealii::CUDAWrappers::FEEvaluation<dim, fe_degree, fe_degree + 1, 1, double>
      fe_eval(cell, gpu_data, shared_data);
  fe_eval.read_dof_values(src);
  fe_eval.evaluate(false, true);
  fe_eval.apply_for_each_quad_point(ThermalOperatorQuad<dim, fe_degree>(
      _thermal_conductivity[pos], _alpha[pos]));
  fe_eval.integrate(false, true);
  fe_eval.distribute_local_to_global(dst);
}
} // namespace

namespace adamantine
{
template <int dim, int fe_degree, typename MemorySpaceType>
ThermalOperatorDevice<dim, fe_degree, MemorySpaceType>::ThermalOperatorDevice(
    MPI_Comm const &communicator,
    std::shared_ptr<MaterialProperty<dim>> material_properties)
    : _communicator(communicator), _m(0), _n_owned_cells(0),
      _material_properties(material_properties),
      _inverse_mass_matrix(
          new dealii::LA::distributed::Vector<double, MemorySpaceType>())
{
  _matrix_free_data.mapping_update_flags = dealii::update_gradients |
                                           dealii::update_JxW_values |
                                           dealii::update_quadrature_points;
}

template <int dim, int fe_degree, typename MemorySpaceType>
void ThermalOperatorDevice<dim, fe_degree, MemorySpaceType>::reinit(
    dealii::DoFHandler<dim> const &dof_handler,
    dealii::AffineConstraints<double> const &affine_constraints,
    dealii::QGaussLobatto<1> const &quad)
{
  _matrix_free.reinit(dof_handler, affine_constraints, quad, _matrix_free_data);
  _dof_handler = &dof_handler;
  dealii::LA::distributed::Vector<double, MemorySpaceType> tmp;
  _matrix_free.initialize_dof_vector(tmp);
  _m = tmp.size();
  _n_owned_cells =
      dynamic_cast<dealii::parallel::DistributedTriangulationBase<dim> const *>(
          &dof_handler.get_triangulation())
          ->n_locally_owned_active_cells();
}

template <int dim, int fe_degree, typename MemorySpaceType>
void ThermalOperatorDevice<dim, fe_degree, MemorySpaceType>::reinit(
    dealii::DoFHandler<dim> const &dof_handler,
    dealii::AffineConstraints<double> const &affine_constraints,
    dealii::QGauss<1> const &quad)
{
  _matrix_free.reinit(dof_handler, affine_constraints, quad, _matrix_free_data);
  _dof_handler = &dof_handler;
  dealii::LA::distributed::Vector<double, MemorySpaceType> tmp;
  _matrix_free.initialize_dof_vector(tmp);
  _m = tmp.size();
  _n_owned_cells =
      dynamic_cast<dealii::parallel::DistributedTriangulationBase<dim> const *>(
          &dof_handler.get_triangulation())
          ->n_locally_owned_active_cells();
}

template <int dim, int fe_degree, typename MemorySpaceType>
void ThermalOperatorDevice<dim, fe_degree, MemorySpaceType>::
    compute_inverse_mass_matrix(
        dealii::DoFHandler<dim> const &dof_handler,
        dealii::AffineConstraints<double> const &affine_constraints)
{
  // Compute the inverse of the mass matrix
  dealii::QGaussLobatto<1> mass_matrix_quad(fe_degree + 1);
  dealii::CUDAWrappers::MatrixFree<dim, double> mass_matrix_free;

  typename dealii::CUDAWrappers::MatrixFree<dim, double>::AdditionalData
      mf_data;
  // update_gradients is necessary because of a bug in deal.II
  mf_data.mapping_update_flags =
      dealii::update_values | dealii::update_gradients |
      dealii::update_JxW_values | dealii::update_quadrature_points;
  mass_matrix_free.reinit(dof_handler, affine_constraints, mass_matrix_quad,
                          mf_data);
  mass_matrix_free.initialize_dof_vector(*_inverse_mass_matrix);
  dealii::LA::distributed::Vector<double, MemorySpaceType> dummy;
  LocalMassMarixOperator<dim, fe_degree> local_operator;
  mass_matrix_free.cell_loop(local_operator, dummy, *_inverse_mass_matrix);
  _inverse_mass_matrix->compress(dealii::VectorOperation::add);
  unsigned int const local_size = _inverse_mass_matrix->local_size();
  const int n_blocks = 1 + local_size / dealii::CUDAWrappers::block_size;
  invert_mass_matrix<<<n_blocks, dealii::CUDAWrappers::block_size>>>(
      _inverse_mass_matrix->get_values(), local_size);
}

template <int dim, int fe_degree, typename MemorySpaceType>
void ThermalOperatorDevice<dim, fe_degree, MemorySpaceType>::clear()
{
  _matrix_free.free();
  _inverse_mass_matrix->reinit(0);
}

template <int dim, int fe_degree, typename MemorySpaceType>
void ThermalOperatorDevice<dim, fe_degree, MemorySpaceType>::vmult(
    dealii::LA::distributed::Vector<double, MemorySpaceType> &dst,
    dealii::LA::distributed::Vector<double, MemorySpaceType> const &src) const
{
  dst = 0.;
  vmult_add(dst, src);
}

template <int dim, int fe_degree, typename MemorySpaceType>
void ThermalOperatorDevice<dim, fe_degree, MemorySpaceType>::Tvmult(
    dealii::LA::distributed::Vector<double, MemorySpaceType> &dst,
    dealii::LA::distributed::Vector<double, MemorySpaceType> const &src) const
{
  dst = 0.;
  Tvmult_add(dst, src);
}

template <int dim, int fe_degree, typename MemorySpaceType>
void ThermalOperatorDevice<dim, fe_degree, MemorySpaceType>::vmult_add(
    dealii::LA::distributed::Vector<double, MemorySpaceType> &dst,
    dealii::LA::distributed::Vector<double, MemorySpaceType> const &src) const
{
  LocalThermalOperatorDevice<dim, fe_degree> local_operator(
      _thermal_conductivity.get_values(), _alpha.get_values());
  _matrix_free.cell_loop(local_operator, src, dst);
  _matrix_free.copy_constrained_values(src, dst);
}

template <int dim, int fe_degree, typename MemorySpaceType>
void ThermalOperatorDevice<dim, fe_degree, MemorySpaceType>::Tvmult_add(
    dealii::LA::distributed::Vector<double, MemorySpaceType> &dst,
    dealii::LA::distributed::Vector<double, MemorySpaceType> const &src) const
{
  // The system of equation is symmetric so we can use vmult_add
  vmult_add(dst, src);
}

template <int dim, int fe_degree, typename MemorySpaceType>
void ThermalOperatorDevice<dim, fe_degree, MemorySpaceType>::
    evaluate_material_properties(
        dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host> const
            &state)
{
  // Update the state of the materials
  _material_properties->update_state(*_dof_handler, state);

  unsigned int const n_coefs =
      dealii::Utilities::pow(fe_degree + 1, dim) * _n_owned_cells;
  _thermal_conductivity.reinit(n_coefs);
  _alpha.reinit(n_coefs);
  dealii::LA::ReadWriteVector<double> th_conductivity_host(n_coefs);
  dealii::LA::ReadWriteVector<double> alpha_host(n_coefs);

  unsigned int constexpr n_dofs_1d = fe_degree + 1;
  unsigned int constexpr n_q_points = dealii::Utilities::pow(n_dofs_1d, dim);
  auto ijk = get_ijk<dim, n_dofs_1d, n_q_points>();
  auto graph = _matrix_free.get_colored_graph();
  unsigned int const n_colors = graph.size();
  for (unsigned int color = 0; color < n_colors; ++color)
  {
    typename dealii::CUDAWrappers::MatrixFree<dim, double>::Data gpu_data =
        _matrix_free.get_data(color);
    unsigned int const n_cells = gpu_data.n_cells;
    for (unsigned int cell_id = 0; cell_id < n_cells; ++cell_id)
    {
      auto cell = graph[color][cell_id];
      double const cell_th_conductivity = _material_properties->get(
          cell, Property::thermal_conductivity, state);
      // Assume the material is solid
      double const cell_alpha =
          1. /
          (_material_properties->get(cell, Property::density, state) *
           _material_properties->get(cell, Property::specific_heat, state));
      for (unsigned int i = 0; i < n_q_points; ++i)
      {
        unsigned int const pos =
            dealii::CUDAWrappers::local_q_point_id_host<dim, double>(
                cell_id, gpu_data, n_dofs_1d, n_q_points, ijk[i]);
        th_conductivity_host[pos] = cell_th_conductivity;
        alpha_host[pos] = cell_alpha;
      }
    }
  }

  // Copy the coefficient to the host
  _thermal_conductivity.import(th_conductivity_host,
                               dealii::VectorOperation::insert);
  _alpha.import(alpha_host, dealii::VectorOperation::insert);
}

template <int dim, int fe_degree, typename MemorySpaceType>
void ThermalOperatorDevice<dim, fe_degree, MemorySpaceType>::
    initialize_dof_vector(
        dealii::LA::distributed::Vector<double, MemorySpaceType> &vector) const
{
  _matrix_free.initialize_dof_vector(vector);
}
} // namespace adamantine

// Instantiate class. Note that boost macro does not play well with nvcc so
// instantiate by hand
namespace adamantine
{
template class ThermalOperatorDevice<2, 1, dealii::MemorySpace::CUDA>;
template class ThermalOperatorDevice<2, 2, dealii::MemorySpace::CUDA>;
template class ThermalOperatorDevice<2, 3, dealii::MemorySpace::CUDA>;
template class ThermalOperatorDevice<2, 4, dealii::MemorySpace::CUDA>;
template class ThermalOperatorDevice<2, 5, dealii::MemorySpace::CUDA>;
template class ThermalOperatorDevice<2, 6, dealii::MemorySpace::CUDA>;
template class ThermalOperatorDevice<2, 7, dealii::MemorySpace::CUDA>;
template class ThermalOperatorDevice<2, 8, dealii::MemorySpace::CUDA>;
template class ThermalOperatorDevice<2, 9, dealii::MemorySpace::CUDA>;
template class ThermalOperatorDevice<2, 10, dealii::MemorySpace::CUDA>;

template class ThermalOperatorDevice<3, 1, dealii::MemorySpace::CUDA>;
template class ThermalOperatorDevice<3, 2, dealii::MemorySpace::CUDA>;
template class ThermalOperatorDevice<3, 3, dealii::MemorySpace::CUDA>;
template class ThermalOperatorDevice<3, 4, dealii::MemorySpace::CUDA>;
template class ThermalOperatorDevice<3, 5, dealii::MemorySpace::CUDA>;
template class ThermalOperatorDevice<3, 6, dealii::MemorySpace::CUDA>;
template class ThermalOperatorDevice<3, 7, dealii::MemorySpace::CUDA>;
template class ThermalOperatorDevice<3, 8, dealii::MemorySpace::CUDA>;
template class ThermalOperatorDevice<3, 9, dealii::MemorySpace::CUDA>;
template class ThermalOperatorDevice<3, 10, dealii::MemorySpace::CUDA>;
} // namespace adamantine
