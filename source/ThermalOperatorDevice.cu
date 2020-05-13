/* Copyright (c) 2016 - 2020, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#include <ThermalOperatorDevice.hh>
#include <instantiation.hh>

#include <deal.II/matrix_free/cuda_fe_evaluation.h>

namespace
{
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
      _material_properties(material_properties)
{
  _matrix_free_data.mapping_update_flags = dealii::update_gradients |
                                           dealii::update_JxW_values |
                                           dealii::update_quadrature_points;
}

template <int dim, int fe_degree, typename MemorySpaceType>
template <typename QuadratureType>
void ThermalOperatorDevice<dim, fe_degree, MemorySpaceType>::setup_dofs(
    dealii::DoFHandler<dim> const &dof_handler,
    dealii::AffineConstraints<double> const &affine_constraints,
    QuadratureType const &quad)
{
  _matrix_free.reinit(dof_handler, affine_constraints, quad, _matrix_free_data);
  _dof_handler = &dof_handler;
  if (_m == 0)
  {
    dealii::LA::distributed::Vector<double, MemorySpaceType> tmp;
    _matrix_free.initialize_dof_vector(tmp);
    _m = tmp.size();
    _n_owned_cells =
        dynamic_cast<
            dealii::parallel::DistributedTriangulationBase<dim> const *>(
            &dof_handler.get_triangulation())
            ->n_locally_owned_active_cells();
  }
}

template <int dim, int fe_degree, typename MemorySpaceType>
void ThermalOperatorDevice<dim, fe_degree, MemorySpaceType>::reinit(
    dealii::DoFHandler<dim> const &dof_handler,
    dealii::AffineConstraints<double> const &affine_constraints)
{
  std::ignore = dof_handler;
  std::ignore = affine_constraints;
}

template <int dim, int fe_degree, typename MemorySpaceType>
void ThermalOperatorDevice<dim, fe_degree, MemorySpaceType>::clear()
{
  _matrix_free.free();
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
  std::array<std::array<unsigned int, dim>, n_q_points> ijk;
  if (dim == 2)
  {
    for (unsigned int i = 0; i < n_q_points; ++i)
    {
      ijk[i][0] = i % n_dofs_1d;
      ijk[i][1] = i / n_dofs_1d;
    }
  }
  else if (dim == 2)
  {
    for (unsigned int i = 0; i < n_q_points; ++i)
    {
      ijk[i][0] = i % n_dofs_1d;
      ijk[i][1] = (i / n_dofs_1d) % n_dofs_1d;
      ijk[i][2] = (i / n_dofs_1d) / n_dofs_1d;
    }
  }

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

// Instantiate the function template.
namespace adamantine
{
template void
    ThermalOperatorDevice<2, 1, dealii::MemorySpace::CUDA>::setup_dofs<
        dealii::QGauss<1>>(dealii::DoFHandler<2> const &,
                           dealii::AffineConstraints<double> const &,
                           dealii::QGauss<1> const &);
template void
    ThermalOperatorDevice<2, 2, dealii::MemorySpace::CUDA>::setup_dofs<
        dealii::QGauss<1>>(dealii::DoFHandler<2> const &,
                           dealii::AffineConstraints<double> const &,
                           dealii::QGauss<1> const &);
template void
    ThermalOperatorDevice<2, 3, dealii::MemorySpace::CUDA>::setup_dofs<
        dealii::QGauss<1>>(dealii::DoFHandler<2> const &,
                           dealii::AffineConstraints<double> const &,
                           dealii::QGauss<1> const &);
template void
    ThermalOperatorDevice<2, 4, dealii::MemorySpace::CUDA>::setup_dofs<
        dealii::QGauss<1>>(dealii::DoFHandler<2> const &,
                           dealii::AffineConstraints<double> const &,
                           dealii::QGauss<1> const &);
template void
    ThermalOperatorDevice<2, 5, dealii::MemorySpace::CUDA>::setup_dofs<
        dealii::QGauss<1>>(dealii::DoFHandler<2> const &,
                           dealii::AffineConstraints<double> const &,
                           dealii::QGauss<1> const &);
template void
    ThermalOperatorDevice<2, 6, dealii::MemorySpace::CUDA>::setup_dofs<
        dealii::QGauss<1>>(dealii::DoFHandler<2> const &,
                           dealii::AffineConstraints<double> const &,
                           dealii::QGauss<1> const &);
template void
    ThermalOperatorDevice<2, 7, dealii::MemorySpace::CUDA>::setup_dofs<
        dealii::QGauss<1>>(dealii::DoFHandler<2> const &,
                           dealii::AffineConstraints<double> const &,
                           dealii::QGauss<1> const &);
template void
    ThermalOperatorDevice<2, 8, dealii::MemorySpace::CUDA>::setup_dofs<
        dealii::QGauss<1>>(dealii::DoFHandler<2> const &,
                           dealii::AffineConstraints<double> const &,
                           dealii::QGauss<1> const &);
template void
    ThermalOperatorDevice<2, 9, dealii::MemorySpace::CUDA>::setup_dofs<
        dealii::QGauss<1>>(dealii::DoFHandler<2> const &,
                           dealii::AffineConstraints<double> const &,
                           dealii::QGauss<1> const &);
template void
    ThermalOperatorDevice<2, 10, dealii::MemorySpace::CUDA>::setup_dofs<
        dealii::QGauss<1>>(dealii::DoFHandler<2> const &,
                           dealii::AffineConstraints<double> const &,
                           dealii::QGauss<1> const &);

template void
    ThermalOperatorDevice<2, 1, dealii::MemorySpace::CUDA>::setup_dofs<
        dealii::QGaussLobatto<1>>(dealii::DoFHandler<2> const &,
                                  dealii::AffineConstraints<double> const &,
                                  dealii::QGaussLobatto<1> const &);
template void
    ThermalOperatorDevice<2, 2, dealii::MemorySpace::CUDA>::setup_dofs<
        dealii::QGaussLobatto<1>>(dealii::DoFHandler<2> const &,
                                  dealii::AffineConstraints<double> const &,
                                  dealii::QGaussLobatto<1> const &);
template void
    ThermalOperatorDevice<2, 3, dealii::MemorySpace::CUDA>::setup_dofs<
        dealii::QGaussLobatto<1>>(dealii::DoFHandler<2> const &,
                                  dealii::AffineConstraints<double> const &,
                                  dealii::QGaussLobatto<1> const &);
template void
    ThermalOperatorDevice<2, 4, dealii::MemorySpace::CUDA>::setup_dofs<
        dealii::QGaussLobatto<1>>(dealii::DoFHandler<2> const &,
                                  dealii::AffineConstraints<double> const &,
                                  dealii::QGaussLobatto<1> const &);
template void
    ThermalOperatorDevice<2, 5, dealii::MemorySpace::CUDA>::setup_dofs<
        dealii::QGaussLobatto<1>>(dealii::DoFHandler<2> const &,
                                  dealii::AffineConstraints<double> const &,
                                  dealii::QGaussLobatto<1> const &);
template void
    ThermalOperatorDevice<2, 6, dealii::MemorySpace::CUDA>::setup_dofs<
        dealii::QGaussLobatto<1>>(dealii::DoFHandler<2> const &,
                                  dealii::AffineConstraints<double> const &,
                                  dealii::QGaussLobatto<1> const &);
template void
    ThermalOperatorDevice<2, 7, dealii::MemorySpace::CUDA>::setup_dofs<
        dealii::QGaussLobatto<1>>(dealii::DoFHandler<2> const &,
                                  dealii::AffineConstraints<double> const &,
                                  dealii::QGaussLobatto<1> const &);
template void
    ThermalOperatorDevice<2, 8, dealii::MemorySpace::CUDA>::setup_dofs<
        dealii::QGaussLobatto<1>>(dealii::DoFHandler<2> const &,
                                  dealii::AffineConstraints<double> const &,
                                  dealii::QGaussLobatto<1> const &);
template void
    ThermalOperatorDevice<2, 9, dealii::MemorySpace::CUDA>::setup_dofs<
        dealii::QGaussLobatto<1>>(dealii::DoFHandler<2> const &,
                                  dealii::AffineConstraints<double> const &,
                                  dealii::QGaussLobatto<1> const &);
template void
    ThermalOperatorDevice<2, 10, dealii::MemorySpace::CUDA>::setup_dofs<
        dealii::QGaussLobatto<1>>(dealii::DoFHandler<2> const &,
                                  dealii::AffineConstraints<double> const &,
                                  dealii::QGaussLobatto<1> const &);

template void
    ThermalOperatorDevice<3, 1, dealii::MemorySpace::CUDA>::setup_dofs<
        dealii::QGauss<1>>(dealii::DoFHandler<3> const &,
                           dealii::AffineConstraints<double> const &,
                           dealii::QGauss<1> const &);
template void
    ThermalOperatorDevice<3, 2, dealii::MemorySpace::CUDA>::setup_dofs<
        dealii::QGauss<1>>(dealii::DoFHandler<3> const &,
                           dealii::AffineConstraints<double> const &,
                           dealii::QGauss<1> const &);
template void
    ThermalOperatorDevice<3, 3, dealii::MemorySpace::CUDA>::setup_dofs<
        dealii::QGauss<1>>(dealii::DoFHandler<3> const &,
                           dealii::AffineConstraints<double> const &,
                           dealii::QGauss<1> const &);
template void
    ThermalOperatorDevice<3, 4, dealii::MemorySpace::CUDA>::setup_dofs<
        dealii::QGauss<1>>(dealii::DoFHandler<3> const &,
                           dealii::AffineConstraints<double> const &,
                           dealii::QGauss<1> const &);
template void
    ThermalOperatorDevice<3, 5, dealii::MemorySpace::CUDA>::setup_dofs<
        dealii::QGauss<1>>(dealii::DoFHandler<3> const &,
                           dealii::AffineConstraints<double> const &,
                           dealii::QGauss<1> const &);
template void
    ThermalOperatorDevice<3, 6, dealii::MemorySpace::CUDA>::setup_dofs<
        dealii::QGauss<1>>(dealii::DoFHandler<3> const &,
                           dealii::AffineConstraints<double> const &,
                           dealii::QGauss<1> const &);
template void
    ThermalOperatorDevice<3, 7, dealii::MemorySpace::CUDA>::setup_dofs<
        dealii::QGauss<1>>(dealii::DoFHandler<3> const &,
                           dealii::AffineConstraints<double> const &,
                           dealii::QGauss<1> const &);
template void
    ThermalOperatorDevice<3, 8, dealii::MemorySpace::CUDA>::setup_dofs<
        dealii::QGauss<1>>(dealii::DoFHandler<3> const &,
                           dealii::AffineConstraints<double> const &,
                           dealii::QGauss<1> const &);
template void
    ThermalOperatorDevice<3, 9, dealii::MemorySpace::CUDA>::setup_dofs<
        dealii::QGauss<1>>(dealii::DoFHandler<3> const &,
                           dealii::AffineConstraints<double> const &,
                           dealii::QGauss<1> const &);
template void
    ThermalOperatorDevice<3, 10, dealii::MemorySpace::CUDA>::setup_dofs<
        dealii::QGauss<1>>(dealii::DoFHandler<3> const &,
                           dealii::AffineConstraints<double> const &,
                           dealii::QGauss<1> const &);

template void
    ThermalOperatorDevice<3, 1, dealii::MemorySpace::CUDA>::setup_dofs<
        dealii::QGaussLobatto<1>>(dealii::DoFHandler<3> const &,
                                  dealii::AffineConstraints<double> const &,
                                  dealii::QGaussLobatto<1> const &);
template void
    ThermalOperatorDevice<3, 2, dealii::MemorySpace::CUDA>::setup_dofs<
        dealii::QGaussLobatto<1>>(dealii::DoFHandler<3> const &,
                                  dealii::AffineConstraints<double> const &,
                                  dealii::QGaussLobatto<1> const &);
template void
    ThermalOperatorDevice<3, 3, dealii::MemorySpace::CUDA>::setup_dofs<
        dealii::QGaussLobatto<1>>(dealii::DoFHandler<3> const &,
                                  dealii::AffineConstraints<double> const &,
                                  dealii::QGaussLobatto<1> const &);
template void
    ThermalOperatorDevice<3, 4, dealii::MemorySpace::CUDA>::setup_dofs<
        dealii::QGaussLobatto<1>>(dealii::DoFHandler<3> const &,
                                  dealii::AffineConstraints<double> const &,
                                  dealii::QGaussLobatto<1> const &);
template void
    ThermalOperatorDevice<3, 5, dealii::MemorySpace::CUDA>::setup_dofs<
        dealii::QGaussLobatto<1>>(dealii::DoFHandler<3> const &,
                                  dealii::AffineConstraints<double> const &,
                                  dealii::QGaussLobatto<1> const &);
template void
    ThermalOperatorDevice<3, 6, dealii::MemorySpace::CUDA>::setup_dofs<
        dealii::QGaussLobatto<1>>(dealii::DoFHandler<3> const &,
                                  dealii::AffineConstraints<double> const &,
                                  dealii::QGaussLobatto<1> const &);
template void
    ThermalOperatorDevice<3, 7, dealii::MemorySpace::CUDA>::setup_dofs<
        dealii::QGaussLobatto<1>>(dealii::DoFHandler<3> const &,
                                  dealii::AffineConstraints<double> const &,
                                  dealii::QGaussLobatto<1> const &);
template void
    ThermalOperatorDevice<3, 8, dealii::MemorySpace::CUDA>::setup_dofs<
        dealii::QGaussLobatto<1>>(dealii::DoFHandler<3> const &,
                                  dealii::AffineConstraints<double> const &,
                                  dealii::QGaussLobatto<1> const &);
template void
    ThermalOperatorDevice<3, 9, dealii::MemorySpace::CUDA>::setup_dofs<
        dealii::QGaussLobatto<1>>(dealii::DoFHandler<3> const &,
                                  dealii::AffineConstraints<double> const &,
                                  dealii::QGaussLobatto<1> const &);
template void
    ThermalOperatorDevice<3, 10, dealii::MemorySpace::CUDA>::setup_dofs<
        dealii::QGaussLobatto<1>>(dealii::DoFHandler<3> const &,
                                  dealii::AffineConstraints<double> const &,
                                  dealii::QGaussLobatto<1> const &);
} // namespace adamantine
