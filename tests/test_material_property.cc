/* SPDX-FileCopyrightText: Copyright (c) 2016 - 2024, the adamantine authors.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#define BOOST_TEST_MODULE MaterialProperty

// clang-format off
#include "main.cc"

#include "test_material_property.hh"
// clang-format on

BOOST_AUTO_TEST_CASE(material_property_host)
{
  material_property<dealii::MemorySpace::Host>();
}

BOOST_AUTO_TEST_CASE(ratios_host) { ratios<dealii::MemorySpace::Host>(); }

BOOST_AUTO_TEST_CASE(material_property_table_host)
{
  material_property_table<dealii::MemorySpace::Host>();
}

BOOST_AUTO_TEST_CASE(material_property_polynomials_host)
{
  material_property_polynomials<dealii::MemorySpace::Host>();
}
