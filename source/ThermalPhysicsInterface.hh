/* Copyright (c) 2016 - 2024, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#ifndef THERMAL_PHYSICS_INTERFACE_HH
#define THERMAL_PHYSICS_INTERFACE_HH

#include <types.hh>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>

namespace adamantine
{
// Forward declarations
class Timer;

template <int dim>
class HeatSource;

/**
 * This class defines the interface for ThermalPhysics used in run(). The
 * objective of this class is to simplify code in run() by reducing the number
 * of template parameters from four to two.
 */
template <int dim, typename MemorySpaceType>
class ThermalPhysicsInterface
{
public:
  ThermalPhysicsInterface() = default;

  virtual ~ThermalPhysicsInterface() = default;

  /**
   * Set up and initialize the data structure.
   */
  virtual void setup() = 0;

  /**
   * Associate the AffineConstraints<double> and the MatrixFree objects to the
   * underlying Triangulation.
   */
  virtual void setup_dofs() = 0;

  /**
   * Compute the inverse of the mass matrix associated to the Physics.
   */
  virtual void compute_inverse_mass_matrix() = 0;

  /**
   * Activate more elements of the mesh and interpolate the solution to the new
   * domain.
   */
  virtual void add_material(
      std::vector<std::vector<
          typename dealii::DoFHandler<dim>::active_cell_iterator>> const
          &elements_to_activate,
      std::vector<double> const &new_deposition_cos,
      std::vector<double> const &new_deposition_sin,
      std::vector<bool> &new_has_melted, unsigned int const activation_start,
      unsigned int const activation_end, double const initial_temperature,
      dealii::LA::distributed::Vector<double, MemorySpaceType> &solution) = 0;

  /**
   * Public interface for modifying the private state of the Physics object. One
   * use of this is to modify nominally constant parameters in the middle of a
   * simulation based on data assimilation with an augmented state.
   */
  virtual void
  update_physics_parameters(boost::property_tree::ptree const &database) = 0;

  /**
   * Evolve the physics from time t to time t+delta_t. solution first contains
   * the field at time t and after execution of the function, the field at time
   * t+delta_t.
   */
  virtual double evolve_one_time_step(
      double t, double delta_t,
      dealii::LA::distributed::Vector<double, MemorySpaceType> &solution,
      std::vector<Timer> &timers) = 0;

  /**
   * Initialize the given vector with the given value.
   */
  virtual void
  initialize_dof_vector(double const value,
                        dealii::LA::distributed::Vector<double, MemorySpaceType>
                            &vector) const = 0;

  /**
   * Populate the state of the materials in the Physics object from the
   * MaterialProperty object.
   */
  virtual void get_state_from_material_properties() = 0;

  /**
   * Populate the state of the materials in the MaterialProperty object from the
   * Physics object.
   */
  virtual void set_state_to_material_properties() = 0;

  /**
   * Load the state of the simulation from files.
   */
  virtual void
  load_checkpoint(std::string const &filename,
                  dealii::LA::distributed::Vector<double, MemorySpaceType>
                      &temperature) = 0;

  /**
   * Write the current state of the simulation on the filesystem.
   */
  virtual void
  save_checkpoint(std::string const &filename,
                  dealii::LA::distributed::Vector<double, MemorySpaceType>
                      &temperature) = 0;

  /**
   * Set the deposition cosine and sine and call
   * update_material_deposition_orientation.
   */
  virtual void set_material_deposition_orientation(
      std::vector<double> const &deposition_cos,
      std::vector<double> const &deposition_sin) = 0;

  /**
   * Return the cosine of the material deposition angle for the activated cell
   * @p i.
   */
  virtual double get_deposition_cos(unsigned int const i) const = 0;

  /**
   * Return the sine of the material deposition angle for the activated cell
   * @p i.
   */
  virtual double get_deposition_sin(unsigned int const i) const = 0;

  /**
   * Mark cells above the given temperature to denote that they have melted
   */
  virtual void
  mark_has_melted(double const threshold_temperature,
                  dealii::LA::distributed::Vector<double, MemorySpaceType>
                      &temperature) = 0;

  /**
   * Returns _has_melted
   */
  virtual std::vector<bool> get_has_melted_vector() const = 0;

  /**
   * Sets _has_melted
   */
  virtual void set_has_melted_vector(std::vector<bool> const &has_melted) = 0;

  /**
   * Returns one entry of _has_melted
   */
  virtual bool get_has_melted(const unsigned int) const = 0;

  /**
   * Return the DoFHandler.
   */
  virtual dealii::DoFHandler<dim> &get_dof_handler() = 0;

  /**
   * Return the AffineConstraints<double>.
   */
  virtual dealii::AffineConstraints<double> &get_affine_constraints() = 0;

  /**
   * Return the heat sources.
   */
  virtual HeatSources<MemorySpaceType, dim> &get_heat_sources() = 0;

  /**
   * Return the degree of the finite element.
   */
  virtual unsigned int get_fe_degree() const = 0;
};
} // namespace adamantine
#endif
