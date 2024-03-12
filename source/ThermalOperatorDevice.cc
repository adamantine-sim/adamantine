/* Copyright (c) 2016 - 2024, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#include <MaterialProperty.templates.hh>
#include <ThermalOperatorDevice.hh>
#include <instantiation.hh>
#include <types.hh>

#include <deal.II/base/config.h>
#include <deal.II/base/types.h>
#include <deal.II/fe/fe_update_flags.h>
#include <deal.II/matrix_free/cuda_fe_evaluation.h>
#include <deal.II/matrix_free/cuda_matrix_free.h>

namespace
{
template <int dim, int fe_degree>
class MassMatrixOperatorQuad
{
public:
  KOKKOS_FUNCTION void
  operator()(dealii::CUDAWrappers::FEEvaluation<dim, fe_degree> *fe_eval,
             int const q_point) const
  {
    fe_eval->submit_value(1., q_point);
  }
};

template <int dim, int fe_degree>
class LocalMassMatrixOperator
{
public:
  KOKKOS_FUNCTION void
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
KOKKOS_FUNCTION void LocalMassMatrixOperator<dim, fe_degree>::operator()(
    unsigned int const /*cell*/,
    typename dealii::CUDAWrappers::MatrixFree<dim, double>::Data const
        *gpu_data,
    dealii::CUDAWrappers::SharedData<dim, double> *shared_data,
    double const * /*src*/, double *dst) const
{
  dealii::CUDAWrappers::FEEvaluation<dim, fe_degree, fe_degree + 1, 1, double>
      fe_eval(gpu_data, shared_data);
  fe_eval.apply_for_each_quad_point(MassMatrixOperatorQuad<dim, fe_degree>());
  fe_eval.integrate(true, false);
  fe_eval.distribute_local_to_global(dst);
}

template <int dim, int p_order, int fe_degree>
class ThermalOperatorQuad
{
public:
  using kokkos_default = dealii::MemorySpace::Default::kokkos_space;

  KOKKOS_FUNCTION ThermalOperatorQuad(
      bool use_table, unsigned int cell,
      typename dealii::CUDAWrappers::MatrixFree<dim, double>::Data const
          *gpu_data,
      Kokkos::View<double *, kokkos_default> cos,
      Kokkos::View<double *, kokkos_default> sin,
      Kokkos::View<double *, kokkos_default> powder_ratio,
      Kokkos::View<double *, kokkos_default> liquid_ratio,
      Kokkos::View<dealii::types::material_id *, kokkos_default> material_id,
      Kokkos::View<double *, kokkos_default> inv_rho_cp,
      Kokkos::View<double **, kokkos_default> properties,
      Kokkos::View<double *****, kokkos_default> state_property_tables,
      Kokkos::View<double ****, kokkos_default> state_property_polynomials)
      : _use_table(use_table), _cell(cell), _gpu_data(gpu_data), _cos(cos),
        _sin(sin), _powder_ratio(powder_ratio), _liquid_ratio(liquid_ratio),
        _material_id(material_id), _inv_rho_cp(inv_rho_cp),
        _properties(properties), _state_property_tables(state_property_tables),
        _state_property_polynomials(state_property_polynomials)
  {
  }

  KOKKOS_FUNCTION void
  operator()(dealii::CUDAWrappers::FEEvaluation<dim, fe_degree> *fe_eval,
             int const q_point) const;

private:
  KOKKOS_FUNCTION void update_state_ratios(unsigned int pos, double temperature,
                                           double *state_ratios) const;

  KOKKOS_FUNCTION void get_inv_rho_cp(unsigned int pos, double *state_ratios,
                                      double temperature) const;

  KOKKOS_FUNCTION double
  compute_material_property(adamantine::StateProperty state_property,
                            dealii::types::material_id const material_id,
                            double const *state_ratios,
                            double temperature) const;

  bool _use_table;
  unsigned int _cell;
  static unsigned int constexpr _n_q_points =
      dealii::Utilities::pow(fe_degree + 1, dim);
  static unsigned int constexpr _n_material_states =
      static_cast<unsigned int>(adamantine::MaterialState::SIZE);
  typename dealii::CUDAWrappers::MatrixFree<dim, double>::Data const *_gpu_data;
  Kokkos::View<double *, kokkos_default> _cos;
  Kokkos::View<double *, kokkos_default> _sin;
  Kokkos::View<double *, kokkos_default> _powder_ratio;
  Kokkos::View<double *, kokkos_default> _liquid_ratio;
  Kokkos::View<dealii::types::material_id *, kokkos_default> _material_id;
  Kokkos::View<double *, kokkos_default> _inv_rho_cp;
  Kokkos::View<double **, kokkos_default> _properties;
  Kokkos::View<double *****, kokkos_default> _state_property_tables;
  Kokkos::View<double ****, kokkos_default> _state_property_polynomials;
};

template <int dim, int p_order, int fe_degree>
KOKKOS_FUNCTION void
ThermalOperatorQuad<dim, p_order, fe_degree>::update_state_ratios(
    unsigned int pos, double temperature, double *state_ratios) const
{
  unsigned int constexpr liquid =
      static_cast<unsigned int>(adamantine::MaterialState::liquid);
  unsigned int constexpr powder =
      static_cast<unsigned int>(adamantine::MaterialState::powder);
  unsigned int constexpr solid =
      static_cast<unsigned int>(adamantine::MaterialState::solid);

  // Get the material id at this point
  auto material_id = _material_id[pos];

  // Get the material thermodynamic properties
  double const solidus = _properties(
      material_id, static_cast<unsigned int>(adamantine::Property::solidus));
  double const liquidus = _properties(
      material_id, static_cast<unsigned int>(adamantine::Property::liquidus));

  // Update the state ratios
  state_ratios[powder] = _powder_ratio(pos);

  if (temperature < solidus)
    state_ratios[liquid] = 0.;
  else if (temperature > liquidus)
    state_ratios[liquid] = 1.;
  else
    state_ratios[liquid] = (temperature - solidus) / (liquidus - solidus);

  // Because the powder can only become liquid, the solid can only
  // become liquid, and the liquid can only become solid, the ratio of
  // powder can only decrease.
  state_ratios[powder] = (1. - state_ratios[liquid]) < state_ratios[powder]
                             ? (1. - state_ratios[liquid])
                             : state_ratios[powder];

  // Use max to make sure that we don't create matter because of
  // round-off.
  state_ratios[solid] = (1. - state_ratios[liquid] - state_ratios[powder]) > 0.
                            ? (1. - state_ratios[liquid] - state_ratios[powder])
                            : 0.;

  _powder_ratio(pos) = state_ratios[powder];
  _liquid_ratio(pos) = state_ratios[liquid];
}

template <int dim, int p_order, int fe_degree>
KOKKOS_FUNCTION void
ThermalOperatorQuad<dim, p_order, fe_degree>::get_inv_rho_cp(
    unsigned int pos, double *state_ratios, double temperature) const
{
  // Here we need the specific heat (including the latent heat contribution)
  // and the density

  auto material_id = _material_id(pos);
  // First, get the state-independent material properties
  double const solidus = _properties(
      material_id, static_cast<unsigned int>(adamantine::Property::solidus));
  double const liquidus = _properties(
      material_id, static_cast<unsigned int>(adamantine::Property::liquidus));
  double const latent_heat =
      _properties(material_id,
                  static_cast<unsigned int>(adamantine::Property::latent_heat));

  // Now compute the state-dependent properties
  double const density =
      compute_material_property(adamantine::StateProperty::density, material_id,
                                state_ratios, temperature);

  double specific_heat =
      compute_material_property(adamantine::StateProperty::specific_heat,
                                material_id, state_ratios, temperature);

  // Add in the latent heat contribution
  unsigned int constexpr liquid =
      static_cast<unsigned int>(adamantine::MaterialState::liquid);

  if (state_ratios[liquid] > 0.0 && (state_ratios[liquid] < 1.0))
  {
    specific_heat += latent_heat / (liquidus - solidus);
  }

  _inv_rho_cp(pos) = 1.0 / (density * specific_heat);
}

template <int dim, int p_order, int fe_degree>
KOKKOS_FUNCTION double
ThermalOperatorQuad<dim, p_order, fe_degree>::compute_material_property(
    adamantine::StateProperty state_property,
    dealii::types::material_id const material_id, double const *state_ratios,
    double temperature) const
{
  double value = 0.0;
  unsigned int const property_index = static_cast<unsigned int>(state_property);
  if (_use_table)
  {
    for (unsigned int material_state = 0; material_state < _n_material_states;
         ++material_state)
    {
      const dealii::types::material_id m_id = material_id;

      value += state_ratios[material_state] *
               adamantine::MaterialProperty<dim, p_order,
                                            dealii::MemorySpace::Default>::
                   compute_property_from_table(_state_property_tables, m_id,
                                               material_state, property_index,
                                               temperature);
    }
  }
  else
  {
    for (unsigned int material_state = 0; material_state < _n_material_states;
         ++material_state)
    {
      dealii::types::material_id m_id = material_id;

      for (unsigned int i = 0; i <= p_order; ++i)
      {
        value += state_ratios[material_state] *
                 _state_property_polynomials(m_id, material_state,
                                             property_index, i) *
                 std::pow(temperature, i);
      }
    }
  }

  return value;
}

template <int dim, int p_order, int fe_degree>
KOKKOS_FUNCTION void ThermalOperatorQuad<dim, p_order, fe_degree>::operator()(
    dealii::CUDAWrappers::FEEvaluation<dim, fe_degree> *fe_eval,
    int const q_point) const
{
  double temperature = fe_eval->get_value(q_point);

  double
      state_ratios[static_cast<unsigned int>(adamantine::MaterialState::SIZE)];
  unsigned int const pos =
      _gpu_data->local_q_point_id(_cell, _n_q_points, q_point);
  update_state_ratios(pos, temperature, state_ratios);
  get_inv_rho_cp(pos, state_ratios, temperature);
  auto material_id = _material_id[pos];
  auto const thermal_conductivity_x = compute_material_property(
      adamantine::StateProperty::thermal_conductivity_x, material_id,
      state_ratios, temperature);
  auto const thermal_conductivity_y = compute_material_property(
      adamantine::StateProperty::thermal_conductivity_y, material_id,
      state_ratios, temperature);
  auto const thermal_conductivity_z = compute_material_property(
      adamantine::StateProperty::thermal_conductivity_z, material_id,
      state_ratios, temperature);

  auto th_conductivity_grad = fe_eval->get_gradient(q_point);

  // In 2D we only use x and z, and there are no deposition angle
  if constexpr (dim == 2)
  {
    th_conductivity_grad[::adamantine::axis<dim>::x] *= thermal_conductivity_x;
    th_conductivity_grad[::adamantine::axis<dim>::z] *= thermal_conductivity_z;
  }

  if constexpr (dim == 3)
  {
    auto const th_conductivity_grad_x =
        th_conductivity_grad[::adamantine::axis<dim>::x];
    auto const th_conductivity_grad_y =
        th_conductivity_grad[::adamantine::axis<dim>::y];

    // The rotation is performed using the following formula
    //
    // (cos  -sin) (x  0) ( cos  sin)
    // (sin   cos) (0  y) (-sin  cos)
    // =
    // ((x*cos^2 + y*sin^2)  ((x-y) * (sin*cos)))
    // (((x-y) * (sin*cos))  (x*sin^2 + y*cos^2))
    th_conductivity_grad[::adamantine::axis<dim>::x] =
        (thermal_conductivity_x * _cos[pos] * _cos[pos] +
         thermal_conductivity_y * _sin[pos] * _sin[pos]) *
            th_conductivity_grad_x +
        ((thermal_conductivity_x - thermal_conductivity_y) * _sin[pos] *
         _cos[pos]) *
            th_conductivity_grad_y;
    th_conductivity_grad[::adamantine::axis<dim>::y] =
        ((thermal_conductivity_x - thermal_conductivity_y) * _sin[pos] *
         _cos[pos]) *
            th_conductivity_grad_x +
        (thermal_conductivity_x * _sin[pos] * _sin[pos] +
         thermal_conductivity_y * _cos[pos] * _cos[pos]) *
            th_conductivity_grad_y;

    // There is no deposition angle for the z axis
    th_conductivity_grad[::adamantine::axis<dim>::z] *= thermal_conductivity_z;
  }

  fe_eval->submit_gradient(-_inv_rho_cp[pos] * th_conductivity_grad, q_point);
}

template <int dim, int p_order, int fe_degree>
class LocalThermalOperatorDevice
{
public:
  using kokkos_default = dealii::MemorySpace::Default::kokkos_space;

  LocalThermalOperatorDevice(
      bool use_table, Kokkos::View<double *, kokkos_default> cos,
      Kokkos::View<double *, kokkos_default> sin,
      Kokkos::View<double *, kokkos_default> powder_ratio,
      Kokkos::View<double *, kokkos_default> liquid_ratio,
      Kokkos::View<dealii::types::material_id *, kokkos_default> material_id,
      Kokkos::View<double *, kokkos_default> inv_rho_cp,
      Kokkos::View<double **, kokkos_default> properties,
      Kokkos::View<double *****, kokkos_default> state_property_tables,
      Kokkos::View<double ****, kokkos_default> state_property_polynomials)
      : _use_table(use_table), _cos(cos), _sin(sin),
        _powder_ratio(powder_ratio), _liquid_ratio(liquid_ratio),
        _material_id(material_id), _inv_rho_cp(inv_rho_cp),
        _properties(properties), _state_property_tables(state_property_tables),
        _state_property_polynomials(state_property_polynomials)
  {
  }

  KOKKOS_FUNCTION void
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
  bool _use_table;
  static unsigned int constexpr _n_material_states =
      static_cast<unsigned int>(adamantine::MaterialState::SIZE);
  Kokkos::View<double *, kokkos_default> _cos;
  Kokkos::View<double *, kokkos_default> _sin;
  Kokkos::View<double *, kokkos_default> _powder_ratio;
  Kokkos::View<double *, kokkos_default> _liquid_ratio;
  Kokkos::View<dealii::types::material_id *, kokkos_default> _material_id;
  Kokkos::View<double *, kokkos_default> _inv_rho_cp;
  Kokkos::View<double **, kokkos_default> _properties;
  Kokkos::View<double *****, kokkos_default> _state_property_tables;
  Kokkos::View<double ****, kokkos_default> _state_property_polynomials;
};

template <int dim, int p_order, int fe_degree>
KOKKOS_FUNCTION void
LocalThermalOperatorDevice<dim, p_order, fe_degree>::operator()(
    unsigned int const cell,
    typename dealii::CUDAWrappers::MatrixFree<dim, double>::Data const
        *gpu_data,
    dealii::CUDAWrappers::SharedData<dim, double> *shared_data,
    double const *src, double *dst) const
{
  dealii::CUDAWrappers::FEEvaluation<dim, fe_degree, fe_degree + 1, 1, double>
      fe_eval(gpu_data, shared_data);
  fe_eval.read_dof_values(src);
  fe_eval.evaluate(/*values*/ true, /*gradients*/ true);

  fe_eval.apply_for_each_quad_point(
      ThermalOperatorQuad<dim, p_order, fe_degree>(
          _use_table, cell, gpu_data, _cos, _sin, _powder_ratio, _liquid_ratio,
          _material_id, _inv_rho_cp, _properties, _state_property_tables,
          _state_property_polynomials));

  fe_eval.integrate(/*values*/ false, /*gradients*/ true);
  fe_eval.distribute_local_to_global(dst);
}
} // namespace

namespace adamantine
{
template <int dim, int p_order, int fe_degree, typename MemorySpaceType>
ThermalOperatorDevice<dim, p_order, fe_degree, MemorySpaceType>::
    ThermalOperatorDevice(
        MPI_Comm const &communicator, BoundaryType boundary_type,
        MaterialProperty<dim, p_order, MemorySpaceType> &material_properties)
    : _communicator(communicator), _boundary_type(boundary_type), _m(0),
      _n_owned_cells(0), _material_properties(material_properties),
      _inverse_mass_matrix(
          new dealii::LA::distributed::Vector<double, MemorySpaceType>())
{
  _matrix_free_data.mapping_update_flags = dealii::update_gradients |
                                           dealii::update_JxW_values |
                                           dealii::update_quadrature_points;
}

template <int dim, int p_order, int fe_degree, typename MemorySpaceType>
void ThermalOperatorDevice<dim, p_order, fe_degree, MemorySpaceType>::reinit(
    dealii::DoFHandler<dim> const &dof_handler,
    dealii::AffineConstraints<double> const &affine_constraints,
    dealii::hp::QCollection<1> const &q_collection)
{
  dealii::IteratorFilters::ActiveFEIndexEqualTo filter(0, true);
  // deal.II does not support QCollection on GPU
  _matrix_free.reinit(dealii::StaticMappingQ1<dim>::mapping, dof_handler,
                      affine_constraints, q_collection[0], filter,
                      _matrix_free_data);
  dealii::LA::distributed::Vector<double, MemorySpaceType> tmp;
  _matrix_free.initialize_dof_vector(tmp);
  _m = tmp.size();
  _n_owned_cells =
      dynamic_cast<dealii::parallel::DistributedTriangulationBase<dim> const *>(
          &dof_handler.get_triangulation())
          ->n_locally_owned_active_cells();

  // Compute the mapping between DoFHandler cells and the access position in
  // MatrixFree
  _cell_it_to_mf_pos.clear();
  unsigned int constexpr n_dofs_1d = fe_degree + 1;
  unsigned int constexpr n_q_points_per_cell =
      dealii::Utilities::pow(n_dofs_1d, dim);
  auto graph = _matrix_free.get_colored_graph();
  unsigned int const n_colors = graph.size();
  for (unsigned int color = 0; color < n_colors; ++color)
  {
    typename dealii::CUDAWrappers::MatrixFree<dim, double>::Data gpu_data =
        _matrix_free.get_data(color);
    unsigned int const n_cells = gpu_data.n_cells;
    auto gpu_data_host =
        dealii::CUDAWrappers::copy_mf_data_to_host<dim, double>(
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
}

template <int dim, int p_order, int fe_degree, typename MemorySpaceType>
void ThermalOperatorDevice<dim, p_order, fe_degree, MemorySpaceType>::
    compute_inverse_mass_matrix(
        dealii::DoFHandler<dim> const &dof_handler,
        dealii::AffineConstraints<double> const &affine_constraints)
{
  // Compute the inverse of the mass matrix
  dealii::QGaussLobatto<1> mass_matrix_quad(fe_degree + 1);
  dealii::CUDAWrappers::MatrixFree<dim, double> mass_matrix_free;

  typename dealii::CUDAWrappers::MatrixFree<dim, double>::AdditionalData
      mf_data;
  // Due to a bug in deal.II 9.5 we need to update the quadrature points and the
  // gradients
#if (DEAL_II_VERSION_MAJOR == 9) && (DEAL_II_VERSION_MINOR == 5)
  mf_data.mapping_update_flags =
      dealii::update_quadrature_points | dealii::update_values |
      dealii::update_gradients | dealii::update_JxW_values;
#else
  mf_data.mapping_update_flags =
      dealii::update_values | dealii::update_JxW_values;
#endif
  dealii::IteratorFilters::ActiveFEIndexEqualTo filter(0, true);
  mass_matrix_free.reinit(dealii::StaticMappingQ1<dim>::mapping, dof_handler,
                          affine_constraints, mass_matrix_quad, filter,
                          mf_data);
  mass_matrix_free.initialize_dof_vector(*_inverse_mass_matrix);
  // We don't save memory by not allocating the vector. Instead this is done in
  // cell_loop by using a slower path
  dealii::LA::distributed::Vector<double, MemorySpaceType> dummy(
      _inverse_mass_matrix->get_partitioner());
  LocalMassMatrixOperator<dim, fe_degree> local_operator;
  mass_matrix_free.cell_loop(local_operator, dummy, *_inverse_mass_matrix);
  _inverse_mass_matrix->compress(dealii::VectorOperation::add);
  unsigned int const local_size = _inverse_mass_matrix->locally_owned_size();
  double *values = _inverse_mass_matrix->get_values();
  using ExecutionSpace = std::conditional_t<
      std::is_same_v<MemorySpaceType, dealii::MemorySpace::Host>,
      Kokkos::DefaultHostExecutionSpace, Kokkos::DefaultExecutionSpace>;
  Kokkos::parallel_for(
      "adamantine::compute_inverse_mass_matrix",
      Kokkos::RangePolicy<ExecutionSpace>(0, local_size), KOKKOS_LAMBDA(int i) {
        if (values[i] > 1e-15)
          values[i] = 1. / values[i];
        else
          values[i] = 0.;
      });
}

template <int dim, int p_order, int fe_degree, typename MemorySpaceType>
void ThermalOperatorDevice<dim, p_order, fe_degree, MemorySpaceType>::clear()
{
  _inverse_mass_matrix->reinit(0);
}

template <int dim, int p_order, int fe_degree, typename MemorySpaceType>
void ThermalOperatorDevice<dim, p_order, fe_degree, MemorySpaceType>::vmult(
    dealii::LA::distributed::Vector<double, MemorySpaceType> &dst,
    dealii::LA::distributed::Vector<double, MemorySpaceType> const &src) const
{
  dst = 0.;
  vmult_add(dst, src);
}

template <int dim, int p_order, int fe_degree, typename MemorySpaceType>
void ThermalOperatorDevice<dim, p_order, fe_degree, MemorySpaceType>::Tvmult(
    dealii::LA::distributed::Vector<double, MemorySpaceType> &dst,
    dealii::LA::distributed::Vector<double, MemorySpaceType> const &src) const
{
  dst = 0.;
  Tvmult_add(dst, src);
}

template <int dim, int p_order, int fe_degree, typename MemorySpaceType>
void ThermalOperatorDevice<dim, p_order, fe_degree, MemorySpaceType>::vmult_add(
    dealii::LA::distributed::Vector<double, MemorySpaceType> &dst,
    dealii::LA::distributed::Vector<double, MemorySpaceType> const &src) const
{
  ASSERT(_material_id.extent(0), "material_id has not been initialized");

  LocalThermalOperatorDevice<dim, p_order, fe_degree> local_operator(
      _material_properties.properties_use_table(), _deposition_cos,
      _deposition_sin, _powder_ratio, _liquid_ratio, _material_id, _inv_rho_cp,
      _material_properties.get_properties(),
      _material_properties.get_state_property_tables(),
      _material_properties.get_state_property_polynomials());
  _matrix_free.cell_loop(local_operator, src, dst);
  _matrix_free.copy_constrained_values(src, dst);
}

template <int dim, int p_order, int fe_degree, typename MemorySpaceType>
void ThermalOperatorDevice<dim, p_order, fe_degree, MemorySpaceType>::
    Tvmult_add(dealii::LA::distributed::Vector<double, MemorySpaceType> &dst,
               dealii::LA::distributed::Vector<double, MemorySpaceType> const
                   &src) const
{
  // The system of equation is symmetric so we can use vmult_add
  vmult_add(dst, src);
}

template <int dim, int p_order, int fe_degree, typename MemorySpaceType>
void ThermalOperatorDevice<dim, p_order, fe_degree, MemorySpaceType>::
    get_state_from_material_properties()
{
  unsigned int const n_coefs =
      dealii::Utilities::pow(fe_degree + 1, dim) * _n_owned_cells;
  _liquid_ratio = Kokkos::View<double *, kokkos_default>(
      Kokkos::view_alloc("liquid_ratio", Kokkos::WithoutInitializing), n_coefs);
  auto liquid_ratio_host =
      Kokkos::create_mirror_view(Kokkos::WithoutInitializing, _liquid_ratio);
  _powder_ratio = Kokkos::View<double *, kokkos_default>(
      Kokkos::view_alloc("powder_ratio", Kokkos::WithoutInitializing), n_coefs);
  auto powder_ratio_host =
      Kokkos::create_mirror_view(Kokkos::WithoutInitializing, _powder_ratio);
  _material_id = Kokkos::View<dealii::types::material_id *, kokkos_default>(
      Kokkos::view_alloc("material_id", Kokkos::WithoutInitializing), n_coefs);
  auto material_id_host =
      Kokkos::create_mirror_view(Kokkos::WithoutInitializing, _material_id);
  _inv_rho_cp = Kokkos::View<double *, kokkos_default>(
      Kokkos::view_alloc("inv_rho_cp", Kokkos::WithoutInitializing), n_coefs);

  unsigned int constexpr n_dofs_1d = fe_degree + 1;
  unsigned int constexpr n_q_points_per_cell =
      dealii::Utilities::pow(n_dofs_1d, dim);
  for (auto const &cell : dealii::filter_iterators(
           _matrix_free.get_dof_handler().active_cell_iterators(),
           dealii::IteratorFilters::LocallyOwnedCell(),
           dealii::IteratorFilters::ActiveFEIndexEqualTo(0)))
  {
    // Cast to Triangulation<dim>::cell_iterator to access the material_id
    typename dealii::Triangulation<dim>::active_cell_iterator cell_tria(cell);
    double const cell_liquid_ratio =
        _material_properties.get_state_ratio(cell_tria, MaterialState::liquid);
    double const cell_powder_ratio =
        _material_properties.get_state_ratio(cell_tria, MaterialState::powder);
    auto const cell_material_id = cell_tria->material_id();

    for (unsigned int i = 0; i < n_q_points_per_cell; ++i)
    {
      unsigned int const pos = _cell_it_to_mf_pos[cell][i];
      liquid_ratio_host(pos) = cell_liquid_ratio;
      powder_ratio_host(pos) = cell_powder_ratio;
      material_id_host(pos) = cell_material_id;
    }
  }

  // Move data to the device
  Kokkos::deep_copy(_liquid_ratio, liquid_ratio_host);
  Kokkos::deep_copy(_powder_ratio, powder_ratio_host);
  Kokkos::deep_copy(_material_id, material_id_host);
}

template <int dim, int p_order, int fe_degree, typename MemorySpaceType>
void ThermalOperatorDevice<dim, p_order, fe_degree,
                           MemorySpaceType>::set_state_to_material_properties()
{
  _material_properties.set_state_device(_liquid_ratio, _powder_ratio,
                                        _cell_it_to_mf_pos,
                                        _matrix_free.get_dof_handler());
}

template <int dim, int p_order, int fe_degree, typename MemorySpaceType>
void ThermalOperatorDevice<dim, p_order, fe_degree, MemorySpaceType>::
    update_boundary_material_properties(
        dealii::LA::distributed::Vector<double, MemorySpaceType> const
            &temperature)
{
  if (!(_boundary_type & BoundaryType::adiabatic))
    _material_properties.update_boundary_material_properties(
        _matrix_free.get_dof_handler(), temperature);
}

template <int dim, int p_order, int fe_degree, typename MemorySpaceType>
void ThermalOperatorDevice<dim, p_order, fe_degree, MemorySpaceType>::
    initialize_dof_vector(
        dealii::LA::distributed::Vector<double, MemorySpaceType> &vector) const
{
  _matrix_free.initialize_dof_vector(vector);
}

template <int dim, int p_order, int fe_degree, typename MemorySpaceType>
void ThermalOperatorDevice<dim, p_order, fe_degree, MemorySpaceType>::
    set_material_deposition_orientation(
        std::vector<double> const &deposition_cos,
        std::vector<double> const &deposition_sin)
{
  unsigned int const n_coefs =
      dealii::Utilities::pow(fe_degree + 1, dim) * _n_owned_cells;
  _deposition_cos = Kokkos::View<double *, kokkos_default>(
      Kokkos::view_alloc("deposition_cos", Kokkos::WithoutInitializing),
      n_coefs);
  auto deposition_cos_host =
      Kokkos::create_mirror_view(Kokkos::WithoutInitializing, _deposition_cos);
  _deposition_sin = Kokkos::View<double *, kokkos_default>(
      Kokkos::view_alloc("deposition_sin", Kokkos::WithoutInitializing),
      n_coefs);
  auto deposition_sin_host =
      Kokkos::create_mirror_view(Kokkos::WithoutInitializing, _deposition_sin);

  unsigned int constexpr n_dofs_1d = fe_degree + 1;
  unsigned int constexpr n_q_points_per_cell =
      dealii::Utilities::pow(n_dofs_1d, dim);
  unsigned int local_cell_id = 0;
  for (auto const &cell : dealii::filter_iterators(
           _matrix_free.get_dof_handler().active_cell_iterators(),
           dealii::IteratorFilters::LocallyOwnedCell(),
           dealii::IteratorFilters::ActiveFEIndexEqualTo(0)))
  {
    double const cos = deposition_cos[local_cell_id];
    double const sin = deposition_sin[local_cell_id];
    for (unsigned int i = 0; i < n_q_points_per_cell; ++i)
    {
      unsigned int const pos = _cell_it_to_mf_pos[cell][i];
      deposition_cos_host[pos] = cos;
      deposition_sin_host[pos] = sin;
    }
    ++local_cell_id;
  }

  // Copy the coefficient to the host
  Kokkos::deep_copy(_deposition_cos, deposition_cos_host);
  Kokkos::deep_copy(_deposition_sin, deposition_sin_host);
}

template <int dim, int p_order, int fe_degree, typename MemorySpaceType>
void ThermalOperatorDevice<dim, p_order, fe_degree,
                           MemorySpaceType>::update_inv_rho_cp_cell()
{
  auto inv_rho_cp_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, _inv_rho_cp);
  unsigned int constexpr n_dofs_1d = fe_degree + 1;
  unsigned int constexpr n_q_points_per_cell =
      dealii::Utilities::pow(n_dofs_1d, dim);
  auto graph = _matrix_free.get_colored_graph();
  unsigned int const n_colors = graph.size();
  for (unsigned int color = 0; color < n_colors; ++color)
  {
    typename dealii::CUDAWrappers::MatrixFree<dim, double>::Data gpu_data =
        _matrix_free.get_data(color);
    unsigned int const n_cells = gpu_data.n_cells;
    auto gpu_data_host =
        dealii::CUDAWrappers::copy_mf_data_to_host<dim, double>(
            gpu_data, _matrix_free_data.mapping_update_flags);
    for (unsigned int cell_id = 0; cell_id < n_cells; ++cell_id)
    {
      auto cell = graph[color][cell_id];
      // Need to compute the average
      double cell_inv_rho_cp = 0.;
      for (unsigned int i = 0; i < n_q_points_per_cell; ++i)
      {
        unsigned int const pos = _cell_it_to_mf_pos[cell][i];
        cell_inv_rho_cp += inv_rho_cp_host(pos);
      }
      cell_inv_rho_cp /= n_q_points_per_cell;

      _inv_rho_cp_cells[cell] = cell_inv_rho_cp;
    }
  }
}
} // namespace adamantine

INSTANTIATE_DIM_PORDER_FEDEGREE_DEVICE(TUPLE(ThermalOperatorDevice))
