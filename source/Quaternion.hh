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
   * Return true if the quaternion has been initialized.
   */
  bool is_valid() const;

  /**
   * Rotate the given point.
   */
  dealii::Point<3, double> rotate(dealii::Point<3, double> const &point) const;

  /**
   * Rotate the given points.
   */
  dealii::Point<3, dealii::VectorizedArray<double>>
  rotate(dealii::Point<3, dealii::VectorizedArray<double>> const &points) const;

  /**
   * Rotate the given point by the inverse of the quaternion.
   */
  dealii::Point<3, double>
  inv_rotate(dealii::Point<3, double> const &point) const;

  /**
   * Rotate the given points by the inverse of the quaternion.
   */
  dealii::Point<3, dealii::VectorizedArray<double>> inv_rotate(
      dealii::Point<3, dealii::VectorizedArray<double>> const &points) const;

  /**
   * Test for equality of two quaternions.
   */
  bool operator==(Quaternion const &other) const;

  /**
   * Multiply two quaternions together using the Hamilton product. The resulting
   * quaternion describes the single combined rotation that results from
   * applying the two rotations.
   */
  Quaternion &operator*=(Quaternion const &other);

  /**
   * Multiply the quaternions using the inverse of @p other.
   */
  Quaternion &operator/=(Quaternion const &other);

  /**
   * Compute the power to the @p exp of the quaternion. If @p exp is less than
   * one, this corresponds to a partial rotation.
   */
  void pow(double const exp);

private:
  /**
   * Build the different rotation matrices.
   */
  void build_rotation_matrices();

  /**
   * Flag is false if the default constructor was called and reinit() has not
   * been called yet.
   */
  bool _is_valid = false;
  /**
   * Store the value of the quaternion.
   */
  std::array<double, 4> _quaternion;
  /**
   * Rotation matrix defined by the quaternion.
   */
  dealii::Tensor<2, 3, double> _rotation_matrix;
  /**
   * Rotation matrix defined by the inverse quaternion.
   */
  dealii::Tensor<2, 3, double> _inv_rotation_matrix;
  /**
   * Vectorized rotation matrix defined by the quaternion.
   */
  dealii::Tensor<2, 3, dealii::VectorizedArray<double>> _vec_rotation_matrix;
  /**
   * Vectorized rotation matrix defined by the inverse quaternion.
   */
  dealii::Tensor<2, 3, dealii::VectorizedArray<double>>
      _vec_inv_rotation_matrix;
};

inline dealii::Point<3, double>
Quaternion::rotate(dealii::Point<3, double> const &point) const
{
  return dealii::Point<3, double>(_rotation_matrix * point);
}

inline dealii::Point<3, dealii::VectorizedArray<double>> Quaternion::rotate(
    dealii::Point<3, dealii::VectorizedArray<double>> const &points) const
{
  return dealii::Point<3, dealii::VectorizedArray<double>>(
      _vec_rotation_matrix * points);
}

inline dealii::Point<3, double>
Quaternion::inv_rotate(dealii::Point<3, double> const &point) const
{
  return dealii::Point<3, double>(_inv_rotation_matrix * point);
}

inline dealii::Point<3, dealii::VectorizedArray<double>> Quaternion::inv_rotate(
    dealii::Point<3, dealii::VectorizedArray<double>> const &points) const
{
  return dealii::Point<3, dealii::VectorizedArray<double>>(
      _vec_inv_rotation_matrix * points);
}
} // namespace adamantine

#endif
