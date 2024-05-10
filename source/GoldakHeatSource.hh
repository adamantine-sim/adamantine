/* Copyright (c) 2020 - 2022, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#ifndef GOLDAK_HEAT_SOURCE_HH
#define GOLDAK_HEAT_SOURCE_HH

#include <BeamHeatSourceProperties.hh>
#include <ScanPath.hh>

namespace adamantine
{
/**
 * Goldak model of a laser heat source.
 * The form of the heat source model is taken from the following reference:
 * Coleman et al, Journal of Heat Transfer, (in press, 2020).
 */
template <int dim, typename MemorySpaceType>
class GoldakHeatSource
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
  GoldakHeatSource(BeamHeatSourceProperties const &beam,
                   ScanPath<MemorySpaceType> const &scan_path);

  /**
   * Set the time variable.
   */
  void update_time(double time);

  /**
   * Returns the value of a Goldak heat source at a specified point and
   * time.
   */
  double value(dealii::Point<dim> const &point, double const height) const;

  ScanPath<MemorySpaceType> const &get_scan_path() const { return _scan_path; }

  double get_current_height(double const time) const
  {
    return _scan_path.value(time)[2];
  }

  void set_beam_properties(boost::property_tree::ptree const &database)
  {
    _beam.set_from_database(database);
  }

  BeamHeatSourceProperties get_beam_properties() const { return _beam; }

private:
  dealii::Point<3> _beam_center;
  double _alpha;
  /**
   * Structure of the physical properties of the beam heat source.
   */
  BeamHeatSourceProperties _beam;

  /**
   * The scan path for the heat source.
   */
  ScanPath<MemorySpaceType> _scan_path;
};
} // namespace adamantine

#endif
