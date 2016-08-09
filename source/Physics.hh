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
#include <deal.II/lac/la_parallel_vector.h>

namespace adamantine
{

template <int dim, typename NumberType>
class Physics
{
public:
  Physics() = default;

  virtual ~Physics() = default;

  virtual void reinit() = 0;

  virtual double evolve_one_time_step(
      double t, double delta_t,
      dealii::LA::distributed::Vector<NumberType> &solution) = 0;

  virtual double get_delta_t_guess() const = 0;

  virtual void initialize_dof_vector(
      dealii::LA::distributed::Vector<NumberType> &vector) const = 0;

  virtual dealii::DoFHandler<dim> &get_dof_handler() = 0;
};
}
#endif
