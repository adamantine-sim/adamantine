/* SPDX-FileCopyrightText: Copyright (c) 2025, the adamantine authors.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include <Quaternion.hh>
#include <utils.hh>

#include <iostream>

namespace adamantine
{
Quaternion::Quaternion()
{
  _rotation_matrix = std::numeric_limits<double>::signaling_NaN();
  _inv_rotation_matrix = std::numeric_limits<double>::signaling_NaN();
}

Quaternion::Quaternion(double const r, double const i, double const j,
                       double const k)
{
  reinit(r, i, j, k);
}

void Quaternion::reinit(double const r, double const i, double const j,
                        double const k)
{
  ASSERT(std::abs(r * r + i * i + j * j + k * k - 1.0) < 1e-12,
         "The norm of the quaternion is not normed.");

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
}
} // namespace adamantine
