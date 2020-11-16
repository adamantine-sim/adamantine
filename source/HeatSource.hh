/* Copyright (c) 2016 - 2020, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#ifndef HEAT_SOURCE_HH
#define HEAT_SOURCE_HH

#include <ScanPath.hh>

#include <deal.II/base/point.h>

#include <boost/property_tree/ptree.hpp>

#include <vector>

namespace adamantine
{
/**
 * This structure stores all the physical properties necessary to define an
 * heat source.
 */
struct HeatSourceProperties
{
public:
  /**
   * A metric of the depth of the heat source into the material. The specific
   * definition of the depth may differ across heat source types.
   */
  double depth;
  /**
   * Energy conversion efficiency on the surface.
   */
  double absorption_efficiency;

  /**
   * Square of the beam radius.
   */
  double radius_squared;
  /**
   * Maximum power of the beam.
   */
  double max_power;
};

/**
 * This is the base class for describing the functional form of a heat source.
 * It has a pure virtual "value" method that needs to be implemented in a
 * derived class.
 * NOTE: The coordinate system in this class is different than for the finite
 * element mesh. In this class, the first two components of a dealii::Point<3>
 * describe the position along the surface of the part. The last component is
 * the height through the thickness of the part from the base plate. This is in
 * opposition to the finite element mesh where the first and last components of
 * a dealii::Point<3> describe the position along the surface of the part, and
 * the second component is the thickness. That is, the last two components are
 * swapped between the two coordinate systems.
 */
template <int dim>
class HeatSource
{
public:
  /**
   * Constructor.
   * \param[in] database requires the following entries:
   *   - <B>absorption_efficiency</B>: double in \f$[0,1]\f$
   *   - <B>depth</B>: double in \f$[0,\infty)\f$
   *   - <B>diameter</B>: double in \f$[0,\infty)\f$
   *   - <B>max_power</B>: double in \f$[0, \infty)\f$
   *   - <B>scan_path_file</B>: name of the file that contains the scan path
   *     segments
   */
  HeatSource(boost::property_tree::ptree const &database);

  /**
   * Destructor.
   */
  virtual ~HeatSource() = default;

  /**
   * Set the maximum height of the domain. This is the height at which the
   * heat source penetrates the material.
   */
  void set_max_height(double height);

  /**
   * Compute the heat source at a given point at a given time.
   */
  virtual double value(dealii::Point<dim> const &point,
                       double const time) const = 0;

protected:
  /**
   * Height of the domain.
   */
  double _max_height;
  /**
   * Structure of the physical properties of the heat source.
   */
  HeatSourceProperties _beam;

  /**
   * The scan path for the heat source.
   */
  ScanPath _scan_path;
};

template <int dim>
inline void HeatSource<dim>::set_max_height(double height)
{
  _max_height = height;
}

} // namespace adamantine

#endif
