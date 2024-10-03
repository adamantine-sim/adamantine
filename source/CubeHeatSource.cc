/* Copyright (c) 2020 - 2024, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#include <CubeHeatSource.hh>
#include <instantiation.hh>
#include <types.hh>

namespace adamantine
{
template <int dim>
CubeHeatSource<dim>::CubeHeatSource(
    boost::property_tree::ptree const &source_database,
    boost::optional<boost::property_tree::ptree const &> const
        &units_optional_database)
    : HeatSource<dim>()
{
  double dimension_scaling = 1.;
  double power_scaling = 1.;
  if (units_optional_database)
  {
    auto database = units_optional_database.get();
    // PropertyTreeInput units.scan_path_distance
    std::string unit = database.get("heat_source.dimension", "meter");
    dimension_scaling = g_unit_scaling_factor[unit];
    // PropertyTreeInput units.heat_source.power
    unit = database.get("heat_source.power", "watt");
    power_scaling = g_unit_scaling_factor[unit];
  }

  _start_time = source_database.get<double>("start_time");
  _end_time = source_database.get<double>("end_time");
  _value = source_database.get<double>("value") * power_scaling;
  _min_point[0] = source_database.get<double>("min_x") * dimension_scaling;
  _max_point[0] = source_database.get<double>("max_x") * dimension_scaling;
  _min_point[1] = source_database.get<double>("min_y") * dimension_scaling;
  _max_point[1] = source_database.get<double>("max_y") * dimension_scaling;
  if (dim == 3)
  {
    _min_point[2] = source_database.get<double>("min_z") * dimension_scaling;
    _max_point[2] = source_database.get<double>("max_z") * dimension_scaling;
  }
}

template <int dim>
void CubeHeatSource<dim>::update_time(double time)
{
  _source_on = ((time > _start_time) && (time < _end_time));
}

template <int dim>
double CubeHeatSource<dim>::value(dealii::Point<dim> const &point,
                                  double const /*height*/) const
{
  if (_source_on)
  {
    bool in_source = true;
    for (int i = 0; i < dim; ++i)
    {
      if ((point[i] < _min_point[i]) || (point[i] > _max_point[i]))
      {
        in_source = false;
        break;
      }
    }

    if (in_source)
      return _value;
  }

  return 0.;
}

template <int dim>
double CubeHeatSource<dim>::get_current_height(double const /*time*/) const
{
  return _max_point[axis<dim>::z];
}

} // namespace adamantine

INSTANTIATE_DIM(CubeHeatSource)
