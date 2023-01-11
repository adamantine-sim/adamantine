/* Copyright (c) 2016 - 2022, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#ifndef THERMAL_PHYSICS_HH
#define THERMAL_PHYSICS_HH

#include <Geometry.hh>
#include <HeatSource.hh>
#include <ImplicitOperator.hh>
#include <ThermalOperatorBase.hh>
#include <ThermalPhysicsInterface.hh>

#include <deal.II/base/time_stepping.h>
#include <deal.II/base/time_stepping.templates.h>
#include <deal.II/distributed/cell_weights.h>
#include <deal.II/hp/fe_collection.h>

#include <boost/property_tree/ptree.hpp>

#include <memory>

namespace adamantine
{
/**
 * This class takes care of building the linear operator and the
 * right-hand-side. Also used to evolve the system in time.
 */
template <int dim, int fe_degree, typename MemorySpaceType,
          typename QuadratureType>
class ThermalPhysics : public ThermalPhysicsInterface<dim, MemorySpaceType>
{
public:
  /**
   * Constructor.
   */
  ThermalPhysics(MPI_Comm const &communicator,
                 boost::property_tree::ptree const &database,
                 Geometry<dim> &geometry,
                 MaterialProperty<dim, MemorySpaceType> &material_properties);

  void setup_dofs() override;

  void compute_inverse_mass_matrix() override;

  void add_material(
      std::vector<std::vector<
          typename dealii::DoFHandler<dim>::active_cell_iterator>> const
          &elements_to_activate,
      std::vector<double> const &new_deposition_cos,
      std::vector<double> const &new_deposition_sin,
      std::vector<double> const &new_has_melted_indicator,
      unsigned int const activation_start, unsigned int const activation_end,
      double const initial_temperature,
      dealii::LA::distributed::Vector<double, MemorySpaceType> &solution)
      override;

  /**
   * For ThermalPhysics, update_physics_parameters is used to modify the heat
   * sources in the middle of a simulation, e.g. for data assimilation with an
   * augmented ensemble involving heat source parameters.
   */
  void update_physics_parameters(
      boost::property_tree::ptree const &heat_source_database) override;

  double evolve_one_time_step(
      double t, double delta_t,
      dealii::LA::distributed::Vector<double, MemorySpaceType> &solution,
      std::vector<Timer> &timers) override;

  double get_delta_t_guess() const override;

  void initialize_dof_vector(
      dealii::LA::distributed::Vector<double, MemorySpaceType> &vector)
      const override;

  /**
   * Initialize the given vector. The value is assumed to be a temperature.
   */
  void
  initialize_dof_vector(double const value,
                        dealii::LA::distributed::Vector<double, MemorySpaceType>
                            &vector) const override;

  void get_state_from_material_properties() override;

  void set_state_to_material_properties() override;

  void update_material_deposition_orientation() override;

  void set_material_deposition_orientation(
      std::vector<double> const &deposition_cos,
      std::vector<double> const &deposition_sin) override;

  double get_deposition_cos(unsigned int const i) const override;

  double get_deposition_sin(unsigned int const i) const override;

  void mark_cells_above_temperature(
      const double threshold_temperature,
      dealii::LA::distributed::Vector<double, MemorySpaceType> const
          temperature) override;

  std::vector<double> get_has_melted_indicator() const override;

  dealii::DoFHandler<dim> &get_dof_handler() override;

  dealii::AffineConstraints<double> &get_affine_constraints() override;

  std::vector<std::shared_ptr<HeatSource<dim>>> &get_heat_sources() override;

  unsigned int get_fe_degree() const override;

  /**
   * Return the current height of the heat source.
   */
  double get_current_source_height() const;

private:
  using LA_Vector =
      typename dealii::LA::distributed::Vector<double, MemorySpaceType>;

  /**
   * Compute the right-hand side and apply the TermalOperator.
   */
  LA_Vector evaluate_thermal_physics(double const t, LA_Vector const &y,
                                     std::vector<Timer> &timers) const;

  /**
   * Compute the inverse of the ImplicitOperator.
   */
  LA_Vector id_minus_tau_J_inverse(double const t, double const tau,
                                   LA_Vector const &y,
                                   std::vector<Timer> &timers) const;

  /**
   * This flag is true if the time stepping method is embedded.
   */
  bool _embedded_method = false;
  /**
   * This flag is true if the time stepping method is implicit.
   */
  bool _implicit_method = false;
  /**
   * This flag is true if right preconditioning is used to invert the
   * ImplicitOperator.
   */
  bool _right_preconditioning;
  /**
   * Maximum number of iterations to invert the ImplicitOperator.
   */
  unsigned int _max_iter;
  /**
   * Maximum number of temporary vectors when inverting the ImplicitOperator.
   */
  unsigned int _max_n_tmp_vectors;
  /**
   * Guess of the next time step.
   */
  double _delta_t_guess;
  /**
   * Tolerance to inverte the ImplicitOperator.
   */
  double _tolerance;
  /**
   * Current height of the object.
   */
  double _current_source_height = 0.;
  /**
   * Type of boundary.
   */
  BoundaryType _boundary_type;
  /**
   * Associated geometry.
   */
  Geometry<dim> &_geometry;
  /**
   * Associated Lagrange finite elements.
   */
  dealii::hp::FECollection<dim> _fe_collection;
  /**
   * Associated DoFHandler.
   */
  dealii::DoFHandler<dim> _dof_handler;
  /**
   * Associated AffineConstraints<double>.
   */
  dealii::AffineConstraints<double> _affine_constraints;
  /**
   * Associated quadature, either Gauss or Gauss-Lobatto.
   */
  dealii::hp::QCollection<1> _q_collection;
  /**
   * Object used to attach to each cell, a weight (used for load balancing)
   * equal to the number of degrees of freedom associated with the cell.
   */
  dealii::parallel::CellWeights<dim> _cell_weights;
  /**
   * Cosine of the material deposition angles.
   */
  std::vector<double> _deposition_cos;
  /**
   * Sine of the material deposition angles.
   */
  std::vector<double> _deposition_sin;
  /**
   * Indicator variable for whether a point has ever been above the solidus. The
   * value is 0 for material that has not yet melted and 1 for material that has
   * melted. Due to how the state is transferred for remeshing, this has to be
   * stored as a double.
   */
  std::vector<double> _has_melted_indicator;
  /**
   * Associated material properties.
   */
  MaterialProperty<dim, MemorySpaceType> &_material_properties;
  /**
   * Vector of heat sources.
   */
  std::vector<std::shared_ptr<HeatSource<dim>>> _heat_sources;
  /**
   * Shared pointer to the underlying ThermalOperator.
   */
  std::shared_ptr<ThermalOperatorBase<dim, MemorySpaceType>> _thermal_operator;
  /**
   * Unique pointer to the underlying ImplicitOperator.
   */
  std::unique_ptr<ImplicitOperator<MemorySpaceType>> _implicit_operator;
  /**
   * Shared pointer to the underlying time stepping scheme.
   */
  std::unique_ptr<dealii::TimeStepping::RungeKutta<LA_Vector>> _time_stepping;
};

template <int dim, int fe_degree, typename MemorySpaceType,
          typename QuadratureType>
inline double ThermalPhysics<dim, fe_degree, MemorySpaceType,
                             QuadratureType>::get_delta_t_guess() const
{
  return _delta_t_guess;
}

template <int dim, int fe_degree, typename MemorySpaceType,
          typename QuadratureType>
inline void
ThermalPhysics<dim, fe_degree, MemorySpaceType,
               QuadratureType>::update_material_deposition_orientation()
{
  _thermal_operator->set_material_deposition_orientation(_deposition_cos,
                                                         _deposition_sin);
}

template <int dim, int fe_degree, typename MemorySpaceType,
          typename QuadratureType>
inline void ThermalPhysics<dim, fe_degree, MemorySpaceType, QuadratureType>::
    set_material_deposition_orientation(
        std::vector<double> const &deposition_cos,
        std::vector<double> const &deposition_sin)
{
  _deposition_cos = deposition_cos;
  _deposition_sin = deposition_sin;
  update_material_deposition_orientation();
}

template <int dim, int fe_degree, typename MemorySpaceType,
          typename QuadratureType>
inline double
ThermalPhysics<dim, fe_degree, MemorySpaceType,
               QuadratureType>::get_deposition_cos(unsigned int const i) const
{
  return _deposition_cos[i];
}

template <int dim, int fe_degree, typename MemorySpaceType,
          typename QuadratureType>
inline double
ThermalPhysics<dim, fe_degree, MemorySpaceType,
               QuadratureType>::get_deposition_sin(unsigned int const i) const
{
  return _deposition_sin[i];
}

template <int dim, int fe_degree, typename MemorySpaceType,
          typename QuadratureType>
inline std::vector<double>
ThermalPhysics<dim, fe_degree, MemorySpaceType,
               QuadratureType>::get_has_melted_indicator() const
{
  return _has_melted_indicator;
}

template <int dim, int fe_degree, typename MemorySpaceType,
          typename QuadratureType>
inline dealii::DoFHandler<dim> &
ThermalPhysics<dim, fe_degree, MemorySpaceType,
               QuadratureType>::get_dof_handler()
{
  return _dof_handler;
}

template <int dim, int fe_degree, typename MemorySpaceType,
          typename QuadratureType>
inline dealii::AffineConstraints<double> &
ThermalPhysics<dim, fe_degree, MemorySpaceType,
               QuadratureType>::get_affine_constraints()
{
  return _affine_constraints;
}

template <int dim, int fe_degree, typename MemorySpaceType,
          typename QuadratureType>
inline std::vector<std::shared_ptr<HeatSource<dim>>> &
ThermalPhysics<dim, fe_degree, MemorySpaceType,
               QuadratureType>::get_heat_sources()
{
  return _heat_sources;
}

template <int dim, int fe_degree, typename MemorySpaceType,
          typename QuadratureType>
inline unsigned int
ThermalPhysics<dim, fe_degree, MemorySpaceType, QuadratureType>::get_fe_degree()
    const
{
  return fe_degree;
}

template <int dim, int fe_degree, typename MemorySpaceType,
          typename QuadratureType>
inline double ThermalPhysics<dim, fe_degree, MemorySpaceType,
                             QuadratureType>::get_current_source_height() const
{
  return _current_source_height;
}
} // namespace adamantine

#endif
