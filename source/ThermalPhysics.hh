/* Copyright (c) 2016 - 2021, the adamantine authors.
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
#include <Physics.hh>
#include <ThermalOperatorBase.hh>

#include <deal.II/base/time_stepping.h>
#include <deal.II/base/time_stepping.templates.h>
#include <deal.II/hp/fe_collection.h>

#include <boost/property_tree/ptree.hpp>

namespace adamantine
{
/**
 * This class takes care of building the linear operator and the
 * right-hand-side. Also used to evolve the system in time.
 */
template <int dim, int fe_degree, typename MemorySpaceType,
          typename QuadratureType>
class ThermalPhysics : public Physics<dim, MemorySpaceType>
{
public:
  /**
   * Constructor.
   * \param[in] database requires the following entries:
   *   - <B>materials</B>: property tree
   *   - <B>sources</B>: property tree
   *   - <B>sources.n_beams</B>: unsigned int in \f$[0,\infty)\f$
   *   - <B>sources.beam_X</B>: property tree with X the number associated to
   *   the heat source
   *   - <B>time_stepping</B>: property tree
   *   - <B>time_stepping.method</B>: string
   *   - <B>time_stepping.coarsening_parameter</B>: double in \f$[0,\infty)\f$
   *   [optional, default value of 1.2]
   *   - <B>time_stepping.refining_parameter</B>: double in \f$(0,1)\f$
   *   [optional, default value of 0.8]
   *   - <B>time_stepping.min_time_step</B>: double in \f$[0,\infty)\f$
   *   [optional, default value of 1e-14]
   *   - <B>time_stepping.max_time_step</B>: double in \f$(0,\infty)\f$
   *   [optional, default value of 1e100]
   *   - <B>time_stepping.refining_tolerance</B>: double in \f$(0,\infty)\f$
   *   [optional, default value of 1e-8]
   *   - <B>time_stepping.coarsening_tolerance</B>: double in \f$(0, \infty)\f$
   *   [optional, default value of 1e-12]
   *   - <B>time_stepping.max_iteration</B>: unsigned int \f$[0,\infty)\f$
   *   [optional, default value of 1000]
   *   - <B>time_stepping.right_preconditioning</B>: boolean [optional, default
   *   value is false]
   *   - <B>time_stepping.n_tmp_vectors</B>: unsigned int in \f$[0,\infty)\f$
   *   [optional, default of 30]
   *   - <B>time_stepping.newton_max_iteration</B>: unsigned int in
   *   \f$[0,\infty)\f$ [optional, default valie of 100]
   *   - <B>time_stepping.newton_tolerance</B>: double in \f$(0,\infty)\f$
   *   [optional, default of 1e-6]
   *   - <B>time_stepping.jfnk</B>: boolean [optional, default value of false]
   */
  ThermalPhysics(MPI_Comm const &communicator,
                 boost::property_tree::ptree const &database,
                 Geometry<dim> &geometry);

  void setup_dofs() override;

  void compute_inverse_mass_matrix() override;

  void add_material(
      std::vector<typename dealii::DoFHandler<dim>::active_cell_iterator> const
          &elements_to_activate,
      double initial_temperature,
      dealii::LA::distributed::Vector<double, MemorySpaceType> &solution)
      override;

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

  /**
   * Populate the stateful material properties in the Physics object from the
   * MaterialProperty object.
   */
  void extract_stateful_material_properties() override;

  dealii::DoFHandler<dim> &get_dof_handler() override;

  dealii::AffineConstraints<double> &get_affine_constraints() override;

  std::shared_ptr<MaterialProperty<dim>> get_material_property() override;

  /**
   * Return the heat sources.
   */
  std::vector<std::shared_ptr<HeatSource<dim>>> &get_heat_sources();

  /**
   * Return the current height of the object.
   */
  double get_current_height() const;

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
  double _current_height = 0.;
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
   * Shared pointer to the material properties associated to the domain.
   */
  std::shared_ptr<MaterialProperty<dim>> _material_properties;
  /**
   * Property tree for the material properties from the user input.
   */
  boost::property_tree::ptree _material_database;
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
inline std::shared_ptr<MaterialProperty<dim>>
ThermalPhysics<dim, fe_degree, MemorySpaceType,
               QuadratureType>::get_material_property()
{
  _thermal_operator->sync_stateful_material_properties();
  return _material_properties;
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
inline double ThermalPhysics<dim, fe_degree, MemorySpaceType,
                             QuadratureType>::get_current_height() const
{
  return _current_height;
}
} // namespace adamantine

#endif
