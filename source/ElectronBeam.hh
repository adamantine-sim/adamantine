/* Copyright (c) 2016 - 2017, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#ifndef ELECTRON_BEAM_HH
#define ELECTRON_BEAM_HH

#include <deal.II/base/function_parser.h>

#include <boost/property_tree/ptree.hpp>

namespace adamantine
{
/**
 * This structure stores all the physical properties necessary to define an
 * electron beam.
 */
struct BeamProperty
{
public:
  /**
   * Absolute penetration of the electron beam into the material where 99% of
   * the beam energy is absorbed.
   */
  double depth;
  /**
   * Energy conversion efficiency on the surface.
   */
  double energy_conversion_eff;
  /**
   * Efficiency of beam control.
   */
  double control_eff;
  /**
   * Square of the beam diameter.
   */
  double diameter_squared;
  /**
   * Maximum power of the beam.
   */
  double max_power;
};

/**
 * This class describes the evolution of an electron beam source.
 */
template <int dim>
class ElectronBeam : public dealii::Function<dim>
{
public:
  /**
   * Constructor.
   * \param[in] database requires the following entries:
   *   - <B>energy_conversion_efficiency</B>: double in \f$[0,1]\f$
   *   - <B>control_efficiency</b>: double in \f$[0,1]\f$
   *   - <B>depth</B>: double in \f$[0,\infty)\f$
   *   - <B>diameter</B>: double in \f$[0,\infty)\f$
   *   - <B>max_power</B>: double in \f$[0, \infty)\f$ [optional: if not
   *   defined, <i>current</i> and <i>voltage</i> need to be defined]
   *   - <B>current</B>: double in \f$[0, \infty)\f$ [optional: if defined
   *   <i>voltage</i> should be defined too, if not defined <i>max_power</i>
   *   should be defined]
   *   - <B>voltage</B>: double in \f$[0,\infty)\f$ [optional: if defined
   *   <i>current</i> should be defined too, if not defined <i>max_power</i>
   *   should be defined]
   *   - <B>input_file</B>: name of the csv file that contains the successive
   *   position of the electron beam [optional: if not defined then
   *   <i>abscissa</i> and, in 3D, <i>ordinate</i> need to be defined]
   *   - <B>delimiter</B>: delimiting character used in <i>input_file</i>
   *   [required if <i>input_file</i> is defined]
   *   - <B>abscissa</B>: string, abscissa of the beam as a function of time
   *   (e.g. "(t-1) * (t-2)") [optional: need to be defined if <i>input_file</i>
   *   is not defined]
   *   - <B>ordinate</B>: string, ordinate of the beam as a function of time
   *   [required only for three dimensional calculation and if <i>input_file</i>
   *   is not defined]
   */
  ElectronBeam(boost::property_tree::ptree const &database);

  /**
   * Set the maximum height of the domain. This is the height at which the
   * electron beam penetrate the material.
   */
  void set_max_height(double height);

  /**
   * Compute the heat source at a given point at the current time.
   */
  double value(dealii::Point<dim> const &point,
               unsigned int const component = 0) const override;

  /**
   * Reset the current time and the position to the last saved state.
   */
  void rewind_time();

  /**
   * Save the current time and the position in the list of successive positions
   * of the beam.
   */
  void save_time();

private:
  /**
   * Flag is true if the beam is a point source of which the position is read
   * from a
   * file.
   */
  bool _is_point_source;

  /**
   * Height of the domain.
   */
  double _max_height;
  /**
   * Structure of the physical properties of the electron beam.
   */
  BeamProperty _beam;
  /**
   * Function that describes the position of the beam on the surface.
   */
  std::array<std::unique_ptr<dealii::Function<1>>, dim - 1> _position;
};

template <int dim>
inline void ElectronBeam<dim>::set_max_height(double height)
{
  _max_height = height;
}
} // namespace adamantine

#endif
