/* SPDX-FileCopyrightText: Copyright (c) 2020 - 2024, the adamantine authors.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#ifndef HEAT_SOURCE_HH
#define HEAT_SOURCE_HH

#include <BeamHeatSourceProperties.hh>
#include <ScanPath.hh>
#include <types.hh>

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
  HeatSource(boost::property_tree::ptree const &beam_database,
             boost::optional<boost::property_tree::ptree const &> const
                 &units_optional_database)
      : _beam(beam_database, units_optional_database),
        // PropertyTreeInput sources.beam_X.scan_path_file
        // PropertyTreeInput sources.beam_X.scan_path_format
        _scan_path(beam_database.get<std::string>("scan_path_file"),
                   beam_database.get<std::string>("scan_path_file_format"),
                   units_optional_database)
  {
  }

  /**
   * Destructor.
   */
  virtual ~HeatSource() = default;

  /**
   * Set the time variable.
   */
  virtual void update_time(double time) = 0;

  /**
   * Compute the heat source at a given point at a given time given the current
   * height of the object being manufactured.
   */
  virtual double value(dealii::Point<dim> const &point,
                       double const height) const = 0;
  /**
   * Return the scan path for the heat source.
   */
  virtual ScanPath &get_scan_path();

  /**
   * Compute the current height of the where the heat source meets the material
   * (i.e. the current scan path height).
   */
  virtual double get_current_height(double const time) const;

  /**
   * (Re)sets the BeamHeatSourceProperties member variable, necessary if the
   * beam parameters vary in time (e.g. due to data assimilation).
   */
  virtual void set_beam_properties(boost::property_tree::ptree const &database);

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

template <int dim>
inline ScanPath &HeatSource<dim>::get_scan_path()
{
  return _scan_path;
}

template <int dim>
inline double HeatSource<dim>::get_current_height(double const time) const
{
  return _scan_path.value(time)[2];
}

template <int dim>
inline void HeatSource<dim>::set_beam_properties(
    boost::property_tree::ptree const &database)
{
  _beam.set_from_database(database);
}

} // namespace adamantine

#endif
