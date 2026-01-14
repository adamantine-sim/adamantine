/* SPDX-FileCopyrightText: Copyright (c) 2020 - 2026, the adamantine authors.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#ifndef GOLDAK_HEAT_SOURCE_HH
#define GOLDAK_HEAT_SOURCE_HH

#include <HeatSource.hh>
#include <Quaternion.hh>

#include <limits>

namespace adamantine
{
/**
 * A derived class from HeatSource for the Goldak model of a laser heat source.
 * The form of the heat source model is taken from the following reference:
 * Coleman et al, Journal of Heat Transfer, (in press, 2020).
 */
template <int dim>
class GoldakHeatSource final : public HeatSource<dim>
{
public:
  /**
   * Constructor.
   * \param[in] beam_database requires the following entries:
   *   - <B>absorption_efficiency</B>: double in \f$[0,1]\f$
   *   - <B>depth</B>: double in \f$[0,\infty)\f$
   *   - <B>diameter</B>: double in \f$[0,\infty)\f$
   *   - <B>max_power</B>: double in \f$[0, \infty)\f$
   *   - <B>input_file</B>: name of the file that contains the scan path
   *     segments
   * \param[in] units_optional_database may contain the following entries:
   *   - <B>heat_source.dimension</B>
   *   - <B>heat_source.power</B>
   */
  GoldakHeatSource(boost::property_tree::ptree const &beam_database,
                   boost::optional<boost::property_tree::ptree const &> const
                       &units_optional_database);

  /**
   * Set the time variable.
   */
  void update_time(double time) final;

  /**
   * Returns the value of a Goldak heat source at a specified point and
   * time.
   */
  double value(dealii::Point<dim> const &point) const final;

  /**
   * Same function as above but it uses vectorized data.
   */
  dealii::VectorizedArray<double>
  value(dealii::Point<dim, dealii::VectorizedArray<double>> const &points)
      const final;

  dealii::BoundingBox<dim>
  get_bounding_box(double time, double const scaling_factor) const final;

private:
  bool const _five_axis;
  Quaternion _quaternion;
  dealii::Point<3, dealii::VectorizedArray<double>> _beam_center;
  dealii::VectorizedArray<double> _alpha =
      std::numeric_limits<double>::signaling_NaN();
  dealii::VectorizedArray<double> _depth =
      std::numeric_limits<double>::signaling_NaN();
  dealii::VectorizedArray<double> _radius_squared =
      std::numeric_limits<double>::signaling_NaN();
  double const _pi_over_3_to_1p5 = std::pow(dealii::numbers::PI / 3.0, 1.5);
};
} // namespace adamantine

#endif
