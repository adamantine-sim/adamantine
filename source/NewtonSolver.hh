/* Copyright (c) 2016 - 2017, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#ifndef NEWTON_SOLVER_HH
#define NEWTON_SOLVER_HH

#include <types.hh>

#include <deal.II/lac/la_parallel_vector.h>

#include <functional>

namespace adamantine
{

class NewtonSolver
{
  /**
   * This class implements a Newton solver with basic line search capabilities.
   */
public:
  /**
   * Constructor. \p max_it is the maximal number of Newton iteration and \p
   * tolerance
   * is the tolerance on the solution.
   */
  NewtonSolver(unsigned int max_it, double tolerance);

  /**
   * Solve non-linear problem.
   * \param[in] compute_residual: this function must return the residual for a
   * given vector.
   * \param[in] compute_inv_jacobian: this function must return the inverse of
   * the Jacobian for a given vector.
   * \param[in] y is the initial guess and the solution of the problem.
   */
  void solve(std::function<dealii::LA::distributed::Vector<double>(
                 dealii::LA::distributed::Vector<double> const &)> const
                 &compute_residual,
             std::function<dealii::LA::distributed::Vector<double>(
                 dealii::LA::distributed::Vector<double> const &)> const
                 &compute_inv_jacobian,
             dealii::LA::distributed::Vector<double> &y);

private:
  /**
   * Maximum number of iteration.
   */
  unsigned int _max_it;
  /**
   * Tolerance.
   */
  double _tolerance;
};
} // namespace adamantine

#endif
