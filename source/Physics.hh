/* Copyright (c) 2016 - 2019, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#ifndef PHYSICS_HH
#define PHYSICS_HH

#include <MaterialProperty.hh>
#include <Timer.hh>
#include <types.hh>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>

namespace adamantine
{
/**
 * This class defines the inteface that every physics needs to implement.
 */
template <int dim>
class Physics
{
public:
  Physics() = default;

  virtual ~Physics() = default;

  /**
   * Associate the AffineConstraints<double> and the MatrixFree objects to the
   * underlying Triangulation.
   */
  virtual void setup_dofs() = 0;

  /**
   * Reinitialize the physics and the associated operator.
   */
  virtual void reinit() = 0;

  /**
   * Evolve the physics from time t to time t+delta_t. solution first contains
   * the field at time t and after execution of the function, the field at time
   * t+delta_t.
   */
  virtual double
  evolve_one_time_step(double t, double delta_t,
                       dealii::LA::distributed::Vector<double> &solution,
                       std::vector<Timer> &timers) = 0;

  /**
   * Return a guess of what should be the nex time step.
   */
  virtual double get_delta_t_guess() const = 0;

  /**
   * Initialize the given vector.
   */
  virtual void initialize_dof_vector(
      dealii::LA::distributed::Vector<double> &vector) const = 0;

  /**
   * Initialize the given vector with the given value.
   */
  virtual void initialize_dof_vector(
      double const value,
      dealii::LA::distributed::Vector<double> &vector) const = 0;

  /**
   * Return the DoFHandler.
   */
  virtual dealii::DoFHandler<dim> &get_dof_handler() = 0;

  /**
   * Return the AffineConstraints<double>.
   */
  virtual dealii::AffineConstraints<double> &get_affine_constraints() = 0;

  /**
   * Return a shared pointer of the MaterialProperty.
   */
  virtual std::shared_ptr<MaterialProperty<dim>> get_material_property() = 0;
};
} // namespace adamantine
#endif
