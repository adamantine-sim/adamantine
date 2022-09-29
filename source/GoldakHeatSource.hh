/* Copyright (c) 2020 - 2022, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#ifndef GOLDAK_HEAT_SOURCE_HH
#define GOLDAK_HEAT_SOURCE_HH

#include <HeatSource.hh>

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
   * \param[in] database requires the following entries:
   *   - <B>absorption_efficiency</B>: double in \f$[0,1]\f$
   *   - <B>depth</B>: double in \f$[0,\infty)\f$
   *   - <B>diameter</B>: double in \f$[0,\infty)\f$
   *   - <B>max_power</B>: double in \f$[0, \infty)\f$
   *   - <B>input_file</B>: name of the file that contains the scan path
   *     segments
   */
  GoldakHeatSource(boost::property_tree::ptree const &database);

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

private:
  dealii::Point<3> _beam_center;
  double _alpha;
  double const _pi_over_3_to_1p5 = std::pow(dealii::numbers::PI / 3.0, 1.5);
};
} // namespace adamantine

#endif
