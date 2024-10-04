/* Copyright (c) 2020 - 2024, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#ifndef ELECTRON_BEAM_HEAT_SOURCE_HH
#define ELECTRON_BEAM_HEAT_SOURCE_HH

#include <HeatSource.hh>

#include <limits>

namespace adamantine
{
/**
 * A derived class from HeatSource for a model of an electron beam heat source.
 * The form of the heat source model is taken from the following reference:
 * Raghavan et al, Acta Materilia, 112, 2016, pp 303-314.
 */
template <int dim>
class ElectronBeamHeatSource final : public HeatSource<dim>
{
public:
  /**
   * Constructor.
   *  \param[in] beam_database requires the following entries:
   *    - <B>absorption_efficiency</B>: double in \f$[0,1]\f$
   *    - <B>depth</B>: double in \f$[0,\infty)\f$
   *    - <B>diameter</B>: double in \f$[0,\infty)\f$
   *    - <B>max_power</B>: double in \f$[0, \infty)\f$
   *    - <B>input_file</B>: name of the file that contains the scan path
   *      segments
   *  \param[in] units_optional_database may have the following entries:
   *    - <B>heat_source.dimension</B>
   *    - <B>heat_source.power</B>
   */
  ElectronBeamHeatSource(
      boost::property_tree::ptree const &beam_database,
      boost::optional<boost::property_tree::ptree const &> const
          &units_optional_database);

  /**
   * Set the time variable.
   */
  void update_time(double time) final;

  /**
   * Returns the value of an electron beam heat source at a specified point and
   * time.
   */
  double value(dealii::Point<dim> const &point,
               double const height) const final;

private:
  dealii::Point<3> _beam_center;
  double _alpha = std::numeric_limits<double>::signaling_NaN();
  double const _log_01 = std::log(0.1);
};
} // namespace adamantine

#endif
