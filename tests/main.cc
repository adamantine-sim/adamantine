/* SPDX-FileCopyrightText: Copyright (c) 2016 - 2024, the adamantine authors.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#define BOOST_TEST_NO_MAIN
#include <deal.II/base/mpi.h>

#include <boost/test/unit_test.hpp>

#include <cfenv>

bool init_function() { return true; }

int main(int argc, char *argv[])
{
#ifndef __APPLE__
  feenableexcept(FE_INVALID);
#endif
  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(
      argc, argv, dealii::numbers::invalid_unsigned_int);

  return boost::unit_test::unit_test_main(&init_function, argc, argv);
}
