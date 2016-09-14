/* Copyright (c) 2016, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#ifndef _PHYSICS_HH_
#define _PHYSICS_HH_

#include "types.hh"
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/la_parallel_vector.h>

namespace adamantine
{
/**
 * This class defines the inteface that every physics needs to implement.
 */
template <int dim, typename NumberType>
class Physics
{
public:
  Physics() = default;

  virtual ~Physics() = default;

  /**
   * Associate the ConstraintMatrix and the MatrixFree objects to the
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
  virtual double evolve_one_time_step(
      double t, double delta_t,
      dealii::LA::distributed::Vector<NumberType> &solution) = 0;

  /**
   * Return a guess of what should be the nex time step.
   */
  virtual double get_delta_t_guess() const = 0;

  /**
   * Initialize the given vector.
   */
  virtual void initialize_dof_vector(
      dealii::LA::distributed::Vector<NumberType> &vector) const = 0;

  /**
   * Return the DoFHandler.
   */
  virtual dealii::DoFHandler<dim> &get_dof_handler() = 0;

  /**
   * Return the ConstraintMatrix.
   */
  virtual dealii::ConstraintMatrix &get_constraint_matrix() = 0;
};
}
#endif
