/* SPDX-FileCopyrightText: Copyright (c) 2025 - 2026, the adamantine authors.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include <Quaternion.hh>
#include <utils.hh>

#include <deal.II/base/signaling_nan.h>

#include <cmath>

namespace adamantine
{
Quaternion::Quaternion()
{
  _quaternion = {{std::numeric_limits<double>::signaling_NaN(),
                  std::numeric_limits<double>::signaling_NaN(),
                  std::numeric_limits<double>::signaling_NaN(),
                  std::numeric_limits<double>::signaling_NaN()}};
  _rotation_matrix =
      dealii::numbers::signaling_nan<dealii::Tensor<2, 3, double>>();
  _inv_rotation_matrix =
      dealii::numbers::signaling_nan<dealii::Tensor<2, 3, double>>();
  _vec_rotation_matrix = dealii::numbers::signaling_nan<
      dealii::Tensor<2, 3, dealii::VectorizedArray<double>>>();
  _vec_inv_rotation_matrix = dealii::numbers::signaling_nan<
      dealii::Tensor<2, 3, dealii::VectorizedArray<double>>>();
}

Quaternion::Quaternion(double const r, double const i, double const j,
                       double const k)
{
  reinit(r, i, j, k);
}

void Quaternion::reinit(double const r, double const i, double const j,
                        double const k)
{
  ASSERT(std::abs(r * r + i * i + j * j + k * k - 1.0) < 1e-10,
         "The quaternion is not normed.");

  _is_valid = true;

  // Store the quaternion
  _quaternion = {{r, i, j, k}};

  build_rotation_matrices();
}

bool Quaternion::is_valid() const { return _is_valid; }

bool Quaternion::operator==(Quaternion const &other) const
{
  ASSERT(_is_valid, "");
  ASSERT(other.is_valid(), "");

  return _quaternion == other._quaternion;
}

Quaternion &Quaternion::operator*=(Quaternion const &other)
{
  ASSERT(_is_valid, "");
  ASSERT(other.is_valid(), "");

  double r = _quaternion[0];
  double i = _quaternion[1];
  double j = _quaternion[2];
  double k = _quaternion[3];
  double other_r = other._quaternion[0];
  double other_i = other._quaternion[1];
  double other_j = other._quaternion[2];
  double other_k = other._quaternion[3];

  _quaternion[0] = r * other_r - i * other_i - j * other_j - k * other_k;
  _quaternion[1] = r * other_i + i * other_r + j * other_k - k * other_j;
  _quaternion[2] = r * other_j - i * other_k + j * other_r + k * other_i;
  _quaternion[3] = r * other_k + i * other_j - j * other_i + k * other_r;

  build_rotation_matrices();

  return *this;
}

Quaternion &Quaternion::operator/=(Quaternion const &other)
{
  ASSERT(_is_valid, "");
  ASSERT(other.is_valid(), "");

  // Invert the other quaternion
  Quaternion inv_other(other._quaternion[0], -other._quaternion[1],
                       -other._quaternion[2], -other._quaternion[3]);

  return (*this) *= inv_other;
}

void Quaternion::pow(double const exp)
{
  ASSERT(_is_valid, "");

  double const partial_norm = std::sqrt(_quaternion[1] * _quaternion[1] +
                                        _quaternion[2] * _quaternion[2] +
                                        _quaternion[3] * _quaternion[3]);
  std::array<double, 3> const unit_vector = {{_quaternion[1] / partial_norm,
                                              _quaternion[2] / partial_norm,
                                              _quaternion[3] / partial_norm}};
  double theta = std::atan2(partial_norm, _quaternion[0]);

  _quaternion[0] = std::cos(exp * theta);
  _quaternion[1] = unit_vector[0] * std::sin(exp * theta);
  _quaternion[2] = unit_vector[1] * std::sin(exp * theta);
  _quaternion[3] = unit_vector[2] * std::sin(exp * theta);

  build_rotation_matrices();
}

void Quaternion::build_rotation_matrices()
{
  double r = _quaternion[0];
  double i = _quaternion[1];
  double j = _quaternion[2];
  double k = _quaternion[3];

  // Build the rotation matrix
  _rotation_matrix[0][0] = 1.0 - 2.0 * (j * j + k * k);
  _rotation_matrix[0][1] = 2.0 * (i * j - k * r);
  _rotation_matrix[0][2] = 2.0 * (i * k + j * r);
  _rotation_matrix[1][0] = 2.0 * (i * j + k * r);
  _rotation_matrix[1][1] = 1.0 - 2.0 * (i * i + k * k);
  _rotation_matrix[1][2] = 2.0 * (j * k - i * r);
  _rotation_matrix[2][0] = 2.0 * (i * k - j * r);
  _rotation_matrix[2][1] = 2.0 * (j * k + i * r);
  _rotation_matrix[2][2] = 1.0 - 2.0 * (i * i + j * j);

  // Build the vectorized rotation matrix
  for (int i = 0; i < 3; ++i)
  {
    for (int j = 0; j < 3; ++j)
    {
      _vec_rotation_matrix[i][j] = _rotation_matrix[i][j];
    }
  }

  // Build the inverse rotation matrix. The inverse of a unit quaternion is
  // obtained by changing the sign of its imaginay components, i.e., the inverse
  // of (r,i,j,k) is (r,-i,-j,-k).
  _inv_rotation_matrix[0][0] = 1.0 - 2.0 * (j * j + k * k);
  _inv_rotation_matrix[0][1] = 2.0 * (i * j + k * r);
  _inv_rotation_matrix[0][2] = 2.0 * (i * k - j * r);
  _inv_rotation_matrix[1][0] = 2.0 * (i * j - k * r);
  _inv_rotation_matrix[1][1] = 1.0 - 2.0 * (i * i + k * k);
  _inv_rotation_matrix[1][2] = 2.0 * (j * k + i * r);
  _inv_rotation_matrix[2][0] = 2.0 * (i * k + j * r);
  _inv_rotation_matrix[2][1] = 2.0 * (j * k - i * r);
  _inv_rotation_matrix[2][2] = 1.0 - 2.0 * (i * i + j * j);

  // Build the vectorized inverse rotation matrix
  for (int i = 0; i < 3; ++i)
  {
    for (int j = 0; j < 3; ++j)
    {
      _vec_inv_rotation_matrix[i][j] = _inv_rotation_matrix[i][j];
    }
  }
}
} // namespace adamantine
