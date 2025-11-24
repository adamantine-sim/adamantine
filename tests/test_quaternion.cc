/* SPDX-FileCopyrightText: Copyright (c) 202%, the adamantine authors.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#define BOOST_TEST_MODULE Quaternion

#include <Quaternion.hh>

#include "main.cc"

BOOST_AUTO_TEST_CASE(rotation)
{
  // Rotation of 2*Pi/3
  adamantine::Quaternion quaternion(0.5, 0.5, 0.5, 0.5);
  dealii::Point<3, dealii::VectorizedArray<double>> p1(1., 0., 0.);

  auto rotated_p1 = quaternion.rotate(p1);
  dealii::Point<3, dealii::VectorizedArray<double>> ref_1(0., 1., 0.);
  BOOST_TEST(rotated_p1 == ref_1);

  auto rotated2x_p1 = quaternion.rotate(rotated_p1);
  dealii::Point<3, dealii::VectorizedArray<double>> ref_2(0., 0., 1.);
  BOOST_TEST(rotated2x_p1 == ref_2);

  auto rotated3x_p1 = quaternion.rotate(rotated2x_p1);
  BOOST_TEST(rotated3x_p1 == p1);

  auto inv_rotated_p1 = quaternion.inv_rotate(rotated_p1);
  BOOST_TEST(inv_rotated_p1 == p1);
}
