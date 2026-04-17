/* SPDX-FileCopyrightText: Copyright (c) 2020 - 2026, the adamantine authors.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#ifndef GAUSSIAN_HEAT_SOURCE_HH
#define GAUSSIAN_HEAT_SOURCE_HH

#include <HeatSource.hh>
#include <Quaternion.hh>

#include <limits>

namespace adamantine
{
/**
 * A derived class from HeatSource for the Gaussian model of a laser heat source.
 */
template <int dim>
class GaussianHeatSource final : public HeatSource<dim>
{
public:
  /**
   * Constructor.
   * \param[in] beam_database requires the following entries:
   *   - <B>absorption_efficiency</B>: double in \f$[0,1]\f$
   *   - <B>depth</B>: double in \f$[0,\infty)\f$
   *   - <B>diameter</B>: double in \f$[0,\infty)\f$
   *   - <B>max_power</B>: double in \f$[0, \infty)\f$
   *   - <B>scan_path_file</B>: name of the file that contains the scan path
   *     segments
   *   - <B>scan_path_file_format</B>: format of the scan path file
   *   - <B>A</B>: double, empirical coefficient for the model
   *   - <B>B</B>: double, empirical coefficient for the model
   * \param[in] units_optional_database may contain the following entries:
   *   - <B>heat_source.dimension</B>
   *   - <B>heat_source.power</B>
   */
  GaussianHeatSource(
      boost::property_tree::ptree const &beam_database,
      boost::optional<boost::property_tree::ptree const &> const
          &units_optional_database);

  /**
   * Set the time variable. This updates the beam position and recalculates
   * the shape factor (_k) and normalization constant (_alpha).
   */
  void update_time(double time) final;

  /**
   * Returns the value of a Gaussian heat source at a specified point.
   */
  double value(dealii::Point<dim> const &point) const final;

  /**
   * Same function as above but it uses vectorized data.
   */
  dealii::VectorizedArray<double>
  value(dealii::Point<dim, dealii::VectorizedArray<double>> const &points)
      const final;

  /**
   * Returns the bounding box of the heat source at a given time.
   */
  dealii::BoundingBox<dim>
  get_bounding_box(double time, double const scaling_factor) const final;

private:
  double _A;
  double _B;

  bool const _five_axis;
  Quaternion _quaternion;

  dealii::Point<3, dealii::VectorizedArray<double>> _beam_center;
  dealii::VectorizedArray<double> _depth =
      std::numeric_limits<double>::signaling_NaN();
  dealii::VectorizedArray<double> _radius_squared =
      std::numeric_limits<double>::signaling_NaN();

  dealii::VectorizedArray<double> _alpha =
      std::numeric_limits<double>::signaling_NaN();

  double _k = std::numeric_limits<double>::signaling_NaN();
};
} // namespace adamantine

#endif
