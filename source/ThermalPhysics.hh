/* SPDX-FileCopyrightText: Copyright (c) 2016 - 2025, the adamantine authors.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#ifndef THERMAL_PHYSICS_HH
#define THERMAL_PHYSICS_HH

#include <Boundary.hh>
#include <Geometry.hh>
#include <HeatSource.hh>
#include <ThermalOperatorBase.hh>
#include <ThermalPhysicsInterface.hh>

#include <deal.II/base/time_stepping.h>
#include <deal.II/base/time_stepping.templates.h>
#include <deal.II/distributed/cell_data_transfer.templates.h>
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
template <int dim, int p_order, int fe_degree, typename MaterialStates,
          typename MemorySpaceType, typename QuadratureType>
class ThermalPhysics : public ThermalPhysicsInterface<dim, MemorySpaceType>
{
public:
  /**
   * Constructor.
   */
  ThermalPhysics(MPI_Comm const &communicator,
                 boost::property_tree::ptree const &database,
                 Geometry<dim> &geometry, Boundary const &boundary,
                 MaterialProperty<dim, p_order, MaterialStates, MemorySpaceType>
                     &material_properties);

  void setup() override;

  void setup_dofs() override;

  void compute_inverse_mass_matrix() override;

  void add_material_start(
      std::vector<std::vector<
          typename dealii::DoFHandler<dim>::active_cell_iterator>> const
          &elements_to_activate,
      std::vector<double> const &new_deposition_cos,
      std::vector<double> const &new_deposition_sin,
      std::vector<bool> &new_has_melted, unsigned int const activation_start,
      unsigned int const activation_end,
      dealii::LA::distributed::Vector<double, MemorySpaceType> &solution)
      override;

  void add_material_end(double const new_material_temperature,
                        dealii::LA::distributed::Vector<double, MemorySpaceType>
                            &solution) override;

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

  void
  initialize_dof_vector(double const value,
                        dealii::LA::distributed::Vector<double, MemorySpaceType>
                            &vector) const override;

  void get_state_from_material_properties() override;

  void set_state_to_material_properties() override;

  void load_checkpoint(std::string const &filename,
                       dealii::LA::distributed::Vector<double, MemorySpaceType>
                           &temperature) override;

  void save_checkpoint(std::string const &filename,
                       dealii::LA::distributed::Vector<double, MemorySpaceType>
                           &temperature) override;

  void set_material_deposition_orientation(
      std::vector<double> const &deposition_cos,
      std::vector<double> const &deposition_sin) override;

  double get_deposition_cos(unsigned int const i) const override;

  double get_deposition_sin(unsigned int const i) const override;

  void mark_has_melted(double const threshold_temperature,
                       dealii::LA::distributed::Vector<double, MemorySpaceType>
                           &temperature) override;

  std::vector<bool> get_has_melted_vector() const override;

  void set_has_melted_vector(std::vector<bool> const &has_melted) override;

  bool get_has_melted(unsigned int const i) const override;

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
   * Update the depostion cosine and sine from the Physics object to the
   * operator object.
   */
  void update_material_deposition_orientation();

  /**
   * Compute the right-hand side and apply the TermalOperator.
   */
  LA_Vector evaluate_thermal_physics(double const t, LA_Vector const &y,
                                     std::vector<Timer> &timers) const;

  /**
   * This flag is true if the time stepping method is forward euler.
   */
  bool _forward_euler = false;
  /**
   * Current height of the object.
   */
  double _current_source_height = 0.;
  /**
   * Associated geometry.
   */
  Geometry<dim> &_geometry;
  /**
   * Associated boundary.
   */
  Boundary _boundary;
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
   * value is false for material that has not yet melted and true for material
   * that has melted.
   */
  std::vector<bool> _has_melted;
  /**
   * Associated material properties.
   */
  MaterialProperty<dim, p_order, MaterialStates, MemorySpaceType>
      &_material_properties;
  /**
   * Vector of heat sources.
   */
  std::vector<std::shared_ptr<HeatSource<dim>>> _heat_sources;
  /**
   * Shared pointer to the underlying ThermalOperator.
   */
  std::shared_ptr<ThermalOperatorBase<dim, MemorySpaceType>> _thermal_operator;
  /**
   * Shared pointer to the underlying time stepping scheme.
   */
  std::unique_ptr<dealii::TimeStepping::ExplicitRungeKutta<LA_Vector>>
      _time_stepping;
  /**
   * Cell data transfer object used for updating _solution, _has_melted,
   * _deposition_cos, _deposition_sin, and state of _material_properties when
   * the triangulation is updated when adding material
   */
  std::unique_ptr<dealii::parallel::distributed::CellDataTransfer<
      dim, dim, std::vector<std::vector<double>>>>
      _cell_data_trans;

  /**
   * Temporary data used in _cell_data_trans for _solution
   */
  dealii::Vector<double> _cell_solution;

  /**
   * Temporary data used in _cell_data_trans for transfer
   */
  std::vector<std::vector<double>> _data_to_transfer;
};

template <int dim, int p_order, int fe_degree, typename MaterialStates,
          typename MemorySpaceType, typename QuadratureType>
inline void
ThermalPhysics<dim, p_order, fe_degree, MaterialStates, MemorySpaceType,
               QuadratureType>::update_material_deposition_orientation()
{
  _thermal_operator->set_material_deposition_orientation(_deposition_cos,
                                                         _deposition_sin);
}

template <int dim, int p_order, int fe_degree, typename MaterialStates,
          typename MemorySpaceType, typename QuadratureType>
inline void ThermalPhysics<dim, p_order, fe_degree, MaterialStates,
                           MemorySpaceType, QuadratureType>::
    set_material_deposition_orientation(
        std::vector<double> const &deposition_cos,
        std::vector<double> const &deposition_sin)
{
  _deposition_cos = deposition_cos;
  _deposition_sin = deposition_sin;
  update_material_deposition_orientation();
}

template <int dim, int p_order, int fe_degree, typename MaterialStates,
          typename MemorySpaceType, typename QuadratureType>
inline double
ThermalPhysics<dim, p_order, fe_degree, MaterialStates, MemorySpaceType,
               QuadratureType>::get_deposition_cos(unsigned int const i) const
{
  return _deposition_cos[i];
}

template <int dim, int p_order, int fe_degree, typename MaterialStates,
          typename MemorySpaceType, typename QuadratureType>
inline double
ThermalPhysics<dim, p_order, fe_degree, MaterialStates, MemorySpaceType,
               QuadratureType>::get_deposition_sin(unsigned int const i) const
{
  return _deposition_sin[i];
}

template <int dim, int p_order, int fe_degree, typename MaterialStates,
          typename MemorySpaceType, typename QuadratureType>
inline std::vector<bool>
ThermalPhysics<dim, p_order, fe_degree, MaterialStates, MemorySpaceType,
               QuadratureType>::get_has_melted_vector() const
{
  return _has_melted;
}

template <int dim, int p_order, int fe_degree, typename MaterialStates,
          typename MemorySpaceType, typename QuadratureType>
inline void
ThermalPhysics<dim, p_order, fe_degree, MaterialStates, MemorySpaceType,
               QuadratureType>::set_has_melted_vector(std::vector<bool> const
                                                          &has_melted)
{
  _has_melted = has_melted;
}

template <int dim, int p_order, int fe_degree, typename MaterialStates,
          typename MemorySpaceType, typename QuadratureType>
inline bool
ThermalPhysics<dim, p_order, fe_degree, MaterialStates, MemorySpaceType,
               QuadratureType>::get_has_melted(unsigned int const i) const
{
  return _has_melted[i];
}

template <int dim, int p_order, int fe_degree, typename MaterialStates,
          typename MemorySpaceType, typename QuadratureType>
inline dealii::DoFHandler<dim> &
ThermalPhysics<dim, p_order, fe_degree, MaterialStates, MemorySpaceType,
               QuadratureType>::get_dof_handler()
{
  return _dof_handler;
}

template <int dim, int p_order, int fe_degree, typename MaterialStates,
          typename MemorySpaceType, typename QuadratureType>
inline dealii::AffineConstraints<double> &
ThermalPhysics<dim, p_order, fe_degree, MaterialStates, MemorySpaceType,
               QuadratureType>::get_affine_constraints()
{
  return _affine_constraints;
}

template <int dim, int p_order, int fe_degree, typename MaterialStates,
          typename MemorySpaceType, typename QuadratureType>
inline std::vector<std::shared_ptr<HeatSource<dim>>> &
ThermalPhysics<dim, p_order, fe_degree, MaterialStates, MemorySpaceType,
               QuadratureType>::get_heat_sources()
{
  return _heat_sources;
}

template <int dim, int p_order, int fe_degree, typename MaterialStates,
          typename MemorySpaceType, typename QuadratureType>
inline unsigned int
ThermalPhysics<dim, p_order, fe_degree, MaterialStates, MemorySpaceType,
               QuadratureType>::get_fe_degree() const
{
  return fe_degree;
}

template <int dim, int p_order, int fe_degree, typename MaterialStates,
          typename MemorySpaceType, typename QuadratureType>
inline double
ThermalPhysics<dim, p_order, fe_degree, MaterialStates, MemorySpaceType,
               QuadratureType>::get_current_source_height() const
{
  return _current_source_height;
}
} // namespace adamantine

#endif
