/* SPDX-FileCopyrightText: Copyright (c) 2021 - 2024, the adamantine authors.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#define BOOST_TEST_MODULE MaterialPropertyDevice

// clang-format off
#include "main.cc"

#include "test_material_property.hh"
// clang-format on

BOOST_AUTO_TEST_CASE(material_property_device)
{
  material_property<dealii::MemorySpace::Default>();
}

BOOST_AUTO_TEST_CASE(ratios_device) { ratios<dealii::MemorySpace::Default>(); }

BOOST_AUTO_TEST_CASE(material_property_table_device)
{
  material_property_table<dealii::MemorySpace::Default>();
}

BOOST_AUTO_TEST_CASE(material_property_polynomials_device)
{
  material_property_polynomials<dealii::MemorySpace::Default>();
}
