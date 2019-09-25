/* Copyright (c) 2016, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#define BOOST_TEST_MODULE Newton_Solver

#include "NewtonSolver.hh"

#include "main.cc"

dealii::LA::distributed::Vector<double>
compute_residual(dealii::LA::distributed::Vector<double> const &x)
{
  dealii::LA::distributed::Vector<double> res(2);
  res[0] = std::pow(x[0], 4) - 1.;
  res[1] = std::pow(x[1], 6) - 1.;

  return res;
}

dealii::LA::distributed::Vector<double>
compute_inv_jacobian(dealii::LA::distributed::Vector<double> const &x)
{
  dealii::LA::distributed::Vector<double> inv_jacobian(2);
  inv_jacobian[0] = 1. / (4. * std::pow(x[0], 3));
  inv_jacobian[1] = 1. / (6. * std::pow(x[1], 5));

  return inv_jacobian;
}

BOOST_AUTO_TEST_CASE(newton_solver)
{
  dealii::LA::distributed::Vector<double> src(2);
  src[0] = 2.;
  src[1] = 2.;

  adamantine::NewtonSolver newton_solver(10, 1e-7);
  newton_solver.solve(&compute_residual, &compute_inv_jacobian, src);

  double const tolerance = 1e-5;
  BOOST_CHECK_CLOSE(1., src[0], tolerance);
  BOOST_CHECK_CLOSE(1., src[1], tolerance);
}
