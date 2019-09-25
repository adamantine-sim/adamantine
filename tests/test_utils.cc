/* Copyright (c) 2017, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
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
