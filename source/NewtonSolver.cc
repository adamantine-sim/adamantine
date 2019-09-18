/* Copyright (c) 2016, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#include "NewtonSolver.hh"

namespace adamantine
{
NewtonSolver::NewtonSolver(unsigned int max_it, double tolerance)
    : _max_it(max_it), _tolerance(tolerance)
{
}

void NewtonSolver::solve(
    std::function<dealii::LA::distributed::Vector<double>(
        dealii::LA::distributed::Vector<double> const &)> const
        &compute_residual,
    std::function<dealii::LA::distributed::Vector<double>(
        dealii::LA::distributed::Vector<double> const &)> const
        &compute_inv_jacobian,
    dealii::LA::distributed::Vector<double> &y)
{

  dealii::LA::distributed::Vector<double> y_old = y;
  dealii::LA::distributed::Vector<double> residual = compute_residual(y);
  unsigned int i = 0;
  double residual_norm_old = residual.l2_norm();
  double residual_norm = residual_norm_old;
  while (i < _max_it)
  {
    dealii::LA::distributed::Vector<double> newton_step =
        compute_inv_jacobian(y);
    newton_step.scale(residual);
    // alpha is used for line search.
    double alpha = 1.0;
    while (residual_norm >= residual_norm_old)
    {
      y.sadd(1.0, -alpha, newton_step);
      residual = compute_residual(y);
      residual_norm = residual.l2_norm();
      alpha /= 2.;
      // Break if the line search is falling to improve the solution.
      if (alpha < 1e-6)
        break;
    }
    if (residual_norm < _tolerance)
      break;

    y_old = y;
    residual_norm_old = residual_norm;

    ++i;
  }
}
} // namespace adamantine
