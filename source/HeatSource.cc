/* Copyright (c) 2016 - 2020, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#include <HeatSource.hh>
#include <instantiation.hh>

namespace adamantine
{

template <int dim>
HeatSource<dim>::HeatSource(boost::property_tree::ptree const &database)
    : _max_height(0.), _scan_path(database.get<std::string>("input_file"))
{
  // Set the properties of the heat source.
  _beam.depth = database.get<double>("depth");
  _beam.absorption_efficiency = database.get<double>("absorption_efficiency");
  _beam.radius_squared = std::pow(database.get("diameter", 2e-3) / 2.0, 2);
  _beam.max_power = database.get<double>("max_power");
}

} // namespace adamantine

INSTANTIATE_DIM(HeatSource)
