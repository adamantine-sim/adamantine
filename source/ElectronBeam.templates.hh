/* Copyright (c) 2016, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#ifndef _ELECTRON_BEAM_TEMPLATES_HH_
#define _ELECTRON_BEAM_TEMPLATES_HH_

#include "ElectronBeam.hh"

using std::pow;

namespace adamantine
{
template <int dim>
ElectronBeam<dim>::ElectronBeam(boost::property_tree::ptree const &database)
    : dealii::Function<dim>(), _max_height(0.)
{
  // Set the properties of the electron beam.
  _beam.depth = database.get<double>("depth");
  _beam.energy_conversion_eff =
      database.get<double>("energy_conversion_efficiency");
  _beam.control_eff = database.get<double>("control_efficiency");
  _beam.diameter_squared = pow(database.get<double>("diameter"), 2);
  boost::optional<double> max_power =
      database.get_optional<double>("max_power");
  if (max_power)
    _beam.max_power = max_power.get();
  else
  {
    double const current = database.get<double>("current");
    double const voltage = database.get<double>("voltage");
    _beam.max_power = current * voltage;
  }

  // The only variable that can be used to define the position is the time t.
  std::string variable = "t";
  // Predefined constants
  std::map<std::string, double> constants;
  constants["pi"] = dealii::numbers::PI;

  std::array<std::string, 2> position_expression = {{"abscissa", "ordinate"}};
  for (unsigned int i = 0; i < dim - 1; ++i)
  {
    std::string expression = database.get<std::string>(position_expression[i]);
    _position[i].initialize(variable, expression, constants);
  }
}

template <int dim>
double ElectronBeam<dim>::value(dealii::Point<dim> const &point,
                                unsigned int const /*component*/) const
{
  double const z = point[1] - _max_height;
  if ((z + _beam.depth) < 0.)
    return 0.;
  else
  {
    double const distribution_z =
        -3. * pow(z / _beam.depth, 2) - 2. * (z / _beam.depth) + 1.;

    dealii::Point<1> time;
    time[0] = this->get_time();
    double const beam_center_x = _position[0].value(time);
    double xpy_squared = pow(point[0] - beam_center_x, 2);
    if (dim == 3)
    {
      double const beam_center_y = _position[1].value(time);
      xpy_squared += pow(point[2] - beam_center_y, 2);
    }

    double constexpr four_ln_pone = 4. * std::log(0.1);
    double heat_source = 0.;
    heat_source =
        -_beam.energy_conversion_eff * _beam.control_eff * _beam.max_power *
        four_ln_pone /
        (dealii::numbers::PI * _beam.diameter_squared * _beam.depth) *
        std::exp(four_ln_pone * xpy_squared / _beam.diameter_squared) *
        distribution_z;

    return heat_source;
  }
}
}

#endif
