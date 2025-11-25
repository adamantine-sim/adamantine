/* SPDX-FileCopyrightText: Copyright (c) 2025, the adamantine authors.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#ifndef QUATERNION_HH
#define QUATERNION_HH

#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/vectorization.h>

namespace adamantine
{
/**
 * This class defines rotation using quaternion. Unit quaternions, known as
 * versors, provide a convenient mathematical notation for representing spatial
 * orientations and rotations of elements in three dimensional space.
 * Specifically, they encode information about an axis-angle rotation about an
 * arbitrary axis.
 */
class Quaternion
{
public:
  /**
   * Default constructor. The object is initialized in an invalid state. It can
   * only be used after calling reinit().
   */
  Quaternion();

  /**
   * Constructor. The quaternion should be normed.
   */
  Quaternion(double const r, double const i, double const j, double const k);

  /**
   * Reinitialize the quaternion. The norm should be equal to one.
   */
  void reinit(double const r, double const i, double const j, double const k);

  /**
   * Rotate the given points.
   */
  dealii::Point<3, dealii::VectorizedArray<double>>
  rotate(dealii::Point<3, dealii::VectorizedArray<double>> const &points) const;

  /**
   * Rotate the given points by the inverse of the quaternion.
   */
  dealii::Point<3, dealii::VectorizedArray<double>> inv_rotate(
      dealii::Point<3, dealii::VectorizedArray<double>> const &points) const;

private:
  /**
   * Rotation matrix defined by the quaternion.
   */
  dealii::Tensor<2, 3, dealii::VectorizedArray<double>> _rotation_matrix;
  /**
   * Rotation matrix defined by the inverse quaternion.
   */
  dealii::Tensor<2, 3, dealii::VectorizedArray<double>> _inv_rotation_matrix;
};

inline dealii::Point<3, dealii::VectorizedArray<double>> Quaternion::rotate(
    dealii::Point<3, dealii::VectorizedArray<double>> const &points) const
{
  return dealii::Point<3, dealii::VectorizedArray<double>>(_rotation_matrix *
                                                           points);
}

inline dealii::Point<3, dealii::VectorizedArray<double>> Quaternion::inv_rotate(
    dealii::Point<3, dealii::VectorizedArray<double>> const &points) const
{
  return dealii::Point<3, dealii::VectorizedArray<double>>(
      _inv_rotation_matrix * points);
}
} // namespace adamantine

#endif
