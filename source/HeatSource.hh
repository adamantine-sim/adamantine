/* Copyright (c) 2016 - 2020, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#ifndef HEAT_SOURCE_HH
#define HEAT_SOURCE_HH

#include <BeamHeatSourceProperties.hh>
#include <ScanPath.hh>

#include <deal.II/base/point.h>

namespace adamantine
{
/**
 * This is the base class for describing the functional form of a heat
 * source. It has a pure virtual "value" method that needs to be implemented in
 * a derived class.
 * NOTE: The coordinate system in this class is different than
 * for the finite element mesh. In this class, the first two components of a
 * dealii::Point<3> describe the position along the surface of the part. The
 * last component is the height through the thickness of the part from the base
 * plate. This is in opposition to the finite element mesh where the first and
 * last components of a dealii::Point<3> describe the position along the surface
 * of the part, and the second component is the thickness. That is, the last two
 * components are swapped between the two coordinate systems.
 */
template <int dim>
class HeatSource
{
public:
  /**
   * Default constructor. This constructor should only be used for non-beam heat
   * source.
   */
  HeatSource() = default;

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
  HeatSource(boost::property_tree::ptree const &database)
      : _beam(database),
        // PropertyTreeInput sources.beam_X.scan_path_file
        // PropertyTreeInput sources.beam_X.scan_path_format
        _scan_path(database.get<std::string>("scan_path_file"),
                   database.get<std::string>("scan_path_file_format"))
  {
  }

  /**
   * Destructor.
   */
  virtual ~HeatSource() = default;

  /**
   * Compute the heat source at a given point at a given time given the current
   * height of the object being manufactured.
   */
  virtual double value(dealii::Point<dim> const &point, double const time,
                       double const height) const = 0;

protected:
  /**
   * Structure of the physical properties of the beam heat source.
   */
  BeamHeatSourceProperties _beam;

  /**
   * The scan path for the heat source.
   */
  ScanPath _scan_path;
};

} // namespace adamantine

#endif
