/* Copyright (c) 2016 - 2020, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#ifndef BEAM_HEAT_SOURCE_PROPERTIES_HH
#define BEAM_HEAT_SOURCE_PROPERTIES_HH

#include <boost/property_tree/ptree.hpp>

#include <cmath>

namespace adamantine
{
/**
 * This structure stores all the physical properties necessary to define an
 * beam heat source.
 */
struct BeamHeatSourceProperties
{
public:
  BeamHeatSourceProperties() = default;

  /**
   * Constructor.
   * \param[in] database requires the following entries:
   *   - <B>absorption_efficiency</B>: double in \f$[0,1]\f$
   *   - <B>depth</B>: double in \f$[0,\infty)\f$
   *   - <B>diameter</B>: double in \f$[0,\infty)\f$
   *   - <B>max_power</B>: double in \f$[0, \infty)\f$
   */
  BeamHeatSourceProperties(boost::property_tree::ptree const &database)
  {
    // PropertyTreeInput sources.beam_X.depth
    depth = database.get<double>("depth");
    // PropertyTreeInput sources.beam_X.absorption_efficiency
    absorption_efficiency = database.get<double>("absorption_efficiency");
    // PropertyTreeInput sources.beam_X.diameter
    radius_squared = std::pow(database.get<double>("diameter") / 2.0, 2);
    // PropertyTreeInput sources.beam_X.max_power
    max_power = database.get<double>("max_power");
  }

  /**
   * A metric of the depth of the heat source into the material. The specific
   * definition of the depth may differ across heat source types.
   */
  double depth = 0.;
  /**
   * Energy conversion efficiency on the surface.
   */
  double absorption_efficiency = 0.;

  /**
   * Square of the beam radius.
   */
  double radius_squared = 0.;
  /**
   * Maximum power of the beam.
   */
  double max_power = 0.;
};
} // namespace adamantine

#endif
