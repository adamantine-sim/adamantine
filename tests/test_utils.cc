/* SPDX-FileCopyrightText: Copyright (c) 2017 - 2024, the adamantine authors.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#define BOOST_TEST_MODULE utils

#include <utils.hh>

#include "main.cc"

BOOST_AUTO_TEST_CASE(utils)
{
  // Do not check ASSERT because it is not checked in release mode

  BOOST_CHECK_THROW(adamantine::ASSERT_THROW(false, "test"),
                    std::runtime_error);

  BOOST_CHECK_THROW(adamantine::ASSERT_THROW_NOT_IMPLEMENTED(),
                    adamantine::NotImplementedExc);
}
