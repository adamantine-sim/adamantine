/* Copyright (c) 2021, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#define BOOST_TEST_MODULE ArrayMD

#include <ArrayMD.hh>

#include "main.cc"

BOOST_AUTO_TEST_CASE(array2d)
{
  int constexpr dim_0 = 2;
  int constexpr dim_1 = 3;

  adamantine::Array2D<dim_0, dim_1, dealii::MemorySpace::Host> array2d_template;
  adamantine::Array2D<-1, dim_1, dealii::MemorySpace::Host>
      array2d_partial_template(dim_0);

  for (int i = 0; i < dim_0; ++i)
    for (int j = 0; j < dim_1; ++j)
    {
      array2d_template(i, j) = i + j;
      array2d_partial_template(i, j) = i + j;
    }

  for (int i = 0; i < dim_0; ++i)
    for (int j = 0; j < dim_1; ++j)
    {
      BOOST_CHECK(array2d_template(i, j) == i + j);
      BOOST_CHECK(array2d_partial_template(i, j) == i + j);
    }
}

BOOST_AUTO_TEST_CASE(array4d)
{
  int constexpr dim_0 = 2;
  int constexpr dim_1 = 3;
  int constexpr dim_2 = 4;
  int constexpr dim_3 = 5;

  adamantine::Array4D<dim_0, dim_1, dim_2, dim_3, dealii::MemorySpace::Host>
      array4d_template;
  adamantine::Array4D<dim_0, dim_1, dim_2, -1, dealii::MemorySpace::Host>
      array4d_partial_template_1(dim_3);
  adamantine::Array4D<-1, dim_1, dim_2, -1, dealii::MemorySpace::Host>
      array4d_partial_template_2(dim_0, dim_3);

  for (int i = 0; i < dim_0; ++i)
    for (int j = 0; j < dim_1; ++j)
      for (int k = 0; k < dim_2; ++k)
        for (int l = 0; l < dim_3; ++l)
        {
          array4d_template(i, j, k, l) = i + j + k + l;
          array4d_partial_template_1(i, j, k, l) = i + j + k + l;
          array4d_partial_template_2(i, j, k, l) = i + j + k + l;
        }

  for (int i = 0; i < dim_0; ++i)
    for (int j = 0; j < dim_1; ++j)
      for (int k = 0; k < dim_2; ++k)
        for (int l = 0; l < dim_3; ++l)
        {
          BOOST_CHECK(array4d_template(i, j, k, l) == i + j + k + l);
          BOOST_CHECK(array4d_partial_template_1(i, j, k, l) == i + j + k + l);
          BOOST_CHECK(array4d_partial_template_2(i, j, k, l) == i + j + k + l);
        }
}

BOOST_AUTO_TEST_CASE(array5d)
{
  int constexpr dim_0 = 2;
  int constexpr dim_1 = 3;
  int constexpr dim_2 = 4;
  int constexpr dim_3 = 5;
  int constexpr dim_4 = 6;

  adamantine::Array5D<dim_0, dim_1, dim_2, dim_3, dim_4,
                      dealii::MemorySpace::Host>
      array5d_template;
  adamantine::Array5D<-1, dim_1, dim_2, -1, dim_4, dealii::MemorySpace::Host>
      array5d_partial_template(dim_0, dim_3);

  for (int i = 0; i < dim_0; ++i)
    for (int j = 0; j < dim_1; ++j)
      for (int k = 0; k < dim_2; ++k)
        for (int l = 0; l < dim_3; ++l)
          for (int m = 0; m < dim_4; ++m)
          {
            array5d_template(i, j, k, l, m) = i + j + k + l + m;
            array5d_partial_template(i, j, k, l, m) = i + j + k + l + m;
          }

  for (int i = 0; i < dim_0; ++i)
    for (int j = 0; j < dim_1; ++j)
      for (int k = 0; k < dim_2; ++k)
        for (int l = 0; l < dim_3; ++l)
          for (int m = 0; m < dim_4; ++m)
          {
            BOOST_CHECK(array5d_template(i, j, k, l, m) == i + j + k + l + m);
            BOOST_CHECK(array5d_partial_template(i, j, k, l, m) ==
                        i + j + k + l + m);
          }
}
