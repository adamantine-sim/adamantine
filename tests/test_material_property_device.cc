/* Copyright (c) 2021, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
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
