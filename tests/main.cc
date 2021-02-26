/* Copyright (c) 2016 - 2021, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#define BOOST_TEST_NO_MAIN
#include <deal.II/base/mpi.h>

#include <boost/test/unit_test.hpp>

#include <Kokkos_Core.hpp>

bool init_function() { return true; }

int main(int argc, char *argv[])
{
  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(
      argc, argv, dealii::numbers::invalid_unsigned_int);
  Kokkos::ScopeGuard guard(argc, argv);

  return boost::unit_test::unit_test_main(&init_function, argc, argv);
}
