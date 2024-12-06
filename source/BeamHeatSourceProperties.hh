/* SPDX-FileCopyrightText: Copyright (c) 2016 - 2024, the adamantine authors.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#ifndef BEAM_HEAT_SOURCE_PROPERTIES_HH
#define BEAM_HEAT_SOURCE_PROPERTIES_HH

#include <types.hh>

#include <boost/property_tree/ptree.hpp>

#include <cmath>

namespace adamantine
{
/**
 * This class stores all the physical properties necessary to define an
 * beam heat source.
 */
class BeamHeatSourceProperties
{
public:
  BeamHeatSourceProperties() = default;

  /**
   * Constructor.
   * \param[in] beam_database requires the following entries:
   *   - <B>absorption_efficiency</B>: double in \f$[0,1]\f$
   *   - <B>depth</B>: double in \f$[0,\infty)\f$
   *   - <B>diameter</B>: double in \f$[0,\infty)\f$
   *   - <B>max_power</B>: double in \f$[0, \infty)\f$
   * \param[in] units_optional_database can have the following entries:
   *   - <B>heat_source.dimension</B>
   *   - <B>heat_source.power</B>
   */
  BeamHeatSourceProperties(
      boost::property_tree::ptree const &beam_database,
      boost::optional<boost::property_tree::ptree const &> const
          &units_optional_database)
  {
    if (units_optional_database)
    {
      auto unit_database = units_optional_database.get();
      // PropertyTreeInput units.heat_source.dimension
      std::string unit = unit_database.get("heat_source.dimension", "meter");
      _dimension_scaling = g_unit_scaling_factor[unit];
      // PropertyTreeInput units.heat_source.power
      unit = unit_database.get("heat_source.power", "watt");
      _power_scaling = g_unit_scaling_factor[unit];
    }

    set_from_database(beam_database);
  }

  void set_from_database(boost::property_tree::ptree const &database)
  {
    // PropertyTreeInput sources.beam_X.depth
    depth = database.get<double>("depth") * _dimension_scaling;
    // PropertyTreeInput sources.beam_X.absorption_efficiency
    absorption_efficiency = database.get<double>("absorption_efficiency");
    // PropertyTreeInput sources.beam_X.diameter
    radius_squared = std::pow(
        database.get<double>("diameter") * _dimension_scaling / 2.0, 2);
    // PropertyTreeInput sources.beam_X.max_power
    max_power = database.get<double>("max_power") * _power_scaling;
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

private:
  /**
   * Scaling factor for the dimension of the heat source.
   */
  double _dimension_scaling = 1.;
  /**
   * Scaling factor for the power of the heat source.
   */
  double _power_scaling = 1.;
};
} // namespace adamantine

#endif
