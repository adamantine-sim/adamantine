/* Copyright (c) 2020 - 2024, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#ifndef ELECTRON_BEAM_HEAT_SOURCE_HH
#define ELECTRON_BEAM_HEAT_SOURCE_HH

#include <BeamHeatSourceProperties.hh>
#include <ScanPath.hh>

#include <limits>

namespace adamantine
{
/**
 * A model of an electron beam heat source.
 * The form of the heat source model is taken from the following reference:
 * Raghavan et al, Acta Materilia, 112, 2016, pp 303-314.
 */
template <int dim, typename MemorySpaceType>
class ElectronBeamHeatSource
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
  ElectronBeamHeatSource(BeamHeatSourceProperties const &beam,
                         ScanPath<MemorySpaceType> const &scan_path);

  /**
   * Set the time variable.
   */
  void update_time(double time);

  /**
   * Returns the value of an electron beam heat source at a specified point and
   * time.
   */
  double value(dealii::Point<dim> const &point, double const height) const;

  /**
   * Return the scan path.
   */
  ScanPath<MemorySpaceType> const &get_scan_path() const;

  void set_scan_path(ScanPath<MemorySpaceType> const scan_path)
  {
    _scan_path = scan_path;
  }

  /**
   * Compute the current height of the where the heat source meets the material
   * (i.e. the current scan path height).
   */
  double get_current_height(double const time) const;

  /**
   * (Re)set the BeamHeatSourceProperties member variable, necessary if the
   * beam parameters vary in time (e.g. due to data assimilation).
   */
  void set_beam_properties(boost::property_tree::ptree const &database);

  /**
   * Return the beam properties.
   */
  BeamHeatSourceProperties get_beam_properties() const;

private:
  dealii::Point<3> _beam_center;
  double _alpha = std::numeric_limits<double>::signaling_NaN();
  BeamHeatSourceProperties _beam;
  ScanPath<MemorySpaceType> _scan_path;
};

template <int dim, typename MemorySpaceType>
ScanPath<MemorySpaceType> const &
ElectronBeamHeatSource<dim, MemorySpaceType>::get_scan_path() const
{
  return _scan_path;
}

template <int dim, typename MemorySpaceType>
double ElectronBeamHeatSource<dim, MemorySpaceType>::get_current_height(
    double const time) const
{
  return _scan_path.value(time)[2];
}

template <int dim, typename MemorySpaceType>
void ElectronBeamHeatSource<dim, MemorySpaceType>::set_beam_properties(
    boost::property_tree::ptree const &database)
{
  _beam.set_from_database(database);
}
template <int dim, typename MemorySpaceType>
BeamHeatSourceProperties
ElectronBeamHeatSource<dim, MemorySpaceType>::get_beam_properties() const
{
  return _beam;
}

} // namespace adamantine

#endif
