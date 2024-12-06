/* SPDX-FileCopyrightText: Copyright (c) 2023 - 2024, the adamantine authors.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#ifndef EXPERIMENTAL_DATA_HH
#define EXPERIMENTAL_DATA_HH

#include <experimental_data_utils.hh>

namespace adamantine
{
/**
 * Base class that describes the interfaces of classes that manipulate
 * experimental data.
 */
template <int dim>
class ExperimentalData
{
public:
  virtual ~ExperimentalData() = default;

  /**
   * Read data from the next frame and return the frame ID.
   */
  virtual unsigned int read_next_frame() = 0;

  /**
   * Return the Points and their associated value (temperature).
   */
  virtual PointsValues<dim> get_points_values() = 0;
};

} // namespace adamantine

#endif
