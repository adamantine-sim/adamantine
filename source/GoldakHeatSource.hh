/* SPDX-FileCopyrightText: Copyright (c) 2020 - 2024, the adamantine authors.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#ifndef GOLDAK_HEAT_SOURCE_HH
#define GOLDAK_HEAT_SOURCE_HH

#include <HeatSource.hh>

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
  double value(dealii::Point<dim> const &point,
               double const height) const final;

  dealii::BoundingBox<dim>
  get_bounding_box(double const scaling_factor) const final;

private:
  dealii::Point<3> _beam_center;
  double _alpha = std::numeric_limits<double>::signaling_NaN();
  double const _pi_over_3_to_1p5 = std::pow(dealii::numbers::PI / 3.0, 1.5);
};
} // namespace adamantine

#endif
