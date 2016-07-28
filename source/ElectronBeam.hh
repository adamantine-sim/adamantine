/* Copyright (c) 2016, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#ifndef _ELECTRON_BEAM_HH_
#define _ELECTRON_BEAM_HH_

#include <deal.II/base/function_parser.h>
#include <boost/property_tree/ptree.hpp>

namespace adamantine
{
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

template <int dim>
class ElectronBeam : public dealii::Function<dim>
{
public:
  ElectronBeam(boost::property_tree::ptree const &database);

  void set_max_height(double height);

  /**
   * Compute the heat source at a given point at the time _current_time.
   */
  double value(dealii::Point<dim> const &point,
               unsigned int const component = 0) const override;

private:
  /**
   * Height of the domain.
   */
  double _max_height;
  BeamProperty _beam;
  /**
   * Function that describes the position of the beam on the surface.
   */
  std::array<dealii::FunctionParser<1>, dim - 1> _position;
};

template <int dim>
inline void ElectronBeam<dim>::set_max_height(double height)
{
  _max_height = height;
}
}

#endif
