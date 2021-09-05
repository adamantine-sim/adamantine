/* Copyright (c) 2016 - 2021, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#define BOOST_TEST_MODULE HeatSource

#include <DataAssimilator.hh>

#include "main.cc"

namespace adamantine
{
BOOST_AUTO_TEST_CASE(data_assimilator)
{
  double tolerance = 1.0e-8;
  double var = 1.0;

  DataAssimilator<dealii::MemorySpace::Host> da;

  BOOST_CHECK_CLOSE(var, 0.0, tolerance);
}
} // namespace adamantine
