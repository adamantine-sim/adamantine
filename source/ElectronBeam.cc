/* Copyright (c) 2016 - 2020, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#ifndef ELECTRON_BEAM_TEMPLATES_HH
#define ELECTRON_BEAM_TEMPLATES_HH

#include <ElectronBeam.hh>
#include <instantiation.hh>
#include <utils.hh>

#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>

#include <cstdlib>

using std::pow;

namespace adamantine
{
namespace internal
{
class PointSource : public dealii::Function<1>
{
public:
  PointSource(std::vector<double> const &position);

  double value(dealii::Point<1> const &time,
               unsigned int const component = 0) const override;

  void rewind_time();

  void save_time();

private:
  mutable unsigned int _current_pos;
  unsigned int _saved_pos;
  mutable dealii::Point<1> _current_time;
  dealii::Point<1> _saved_time;
  std::vector<double> _position;
};

PointSource::PointSource(std::vector<double> const &position)
    : _current_pos(-1), _saved_pos(-1), _position(position)
{
  _current_time[0] = -1.;
}

double PointSource::value(dealii::Point<1> const &time,
                          unsigned int const) const
{
  // If the time is greater than the current one, we use the next entry in the
  // vector.
  if (time[0] > _current_time[0])
  {
    ++_current_pos;
    _current_time[0] = time[0];
  }

  return _position[_current_pos];
}

void PointSource::rewind_time()
{
  _current_pos = _saved_pos;
  _current_time = _saved_time;
}

void PointSource::save_time()
{
  _saved_pos = _current_pos;
  _saved_time = _current_time;
}
} // namespace internal

template <int dim>
ElectronBeam<dim>::ElectronBeam(boost::property_tree::ptree const &database)
    : dealii::Function<dim>(), _is_point_source(false), _max_height(0.)
{
  // Set the properties of the electron beam.
  _beam.depth = database.get<double>("depth");
  _beam.energy_conversion_eff =
      database.get<double>("energy_conversion_efficiency");
  _beam.control_eff = database.get<double>("control_efficiency");
  _beam.diameter_squared = pow(database.get("diameter", 2e-3), 2);
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

  boost::optional<std::string> input_file =
      database.get_optional<std::string>("input_file");
  if (input_file)
  {
    std::array<std::vector<double>, dim - 1> points;
    std::string delimiter = database.get<std::string>("delimiter");
    ASSERT_THROW(boost::filesystem::exists(input_file.get()) == true,
                 "The file " + input_file.get() + " does not exist.");
    std::ifstream file(input_file.get());
    std::string line;
    // Read the file line by line, split the strings, and convert the strings
    // into double.
    if (file)
    {
      while (std::getline(file, line))
      {
        std::vector<std::string> split_strings;
        boost::algorithm::split(split_strings, line,
                                boost::is_any_of(delimiter),
                                boost::algorithm::token_compress_on);
        ASSERT_THROW(split_strings.size() == dim - 1,
                     "Problem parsing the source input file.");
        for (unsigned int i = 0; i < dim - 1; ++i)
          points[i].push_back(std::atof(split_strings[i].c_str()));
      }
      file.close();
    }

    // Create the point source.
    for (unsigned int i = 0; i < dim - 1; ++i)
      _position[i].reset(new internal::PointSource(points[i]));
    _is_point_source = true;
  }
  else
  {
    std::array<std::string, 2> position_expression = {{"abscissa", "ordinate"}};
    for (unsigned int i = 0; i < dim - 1; ++i)
    {
      std::string expression =
          database.get<std::string>(position_expression[i]);
      _position[i].reset(new dealii::FunctionParser<1>());
      static_cast<dealii::FunctionParser<1> *>(_position[i].get())
          ->initialize(variable, expression, constants);
    }
  }
}

template <int dim>
void ElectronBeam<dim>::rewind_time()
{
  if (_is_point_source)
    for (unsigned int i = 0; i < dim - 1; ++i)
      static_cast<internal::PointSource *>(_position[i].get())->rewind_time();
}

template <int dim>
void ElectronBeam<dim>::save_time()
{
  if (_is_point_source)
    for (unsigned int i = 0; i < dim - 1; ++i)
      static_cast<internal::PointSource *>(_position[i].get())->save_time();
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

    double const beam_center_x = 0.0 + 1.0e6 * time[0]; //_position[0]->value(time);


    double xpy_squared = pow(point[0] - beam_center_x, 2);
    if (dim == 3)
    {
      double const beam_center_y = _position[1]->value(time);
      xpy_squared += pow(point[2] - beam_center_y, 2);
    }
    /*
    double const four_ln_pone = 4. * std::log(0.1);
    double heat_source = 0.;
    heat_source =
        -_beam.energy_conversion_eff * _beam.control_eff * _beam.max_power *
        four_ln_pone /
        (dealii::numbers::PI * _beam.diameter_squared * _beam.depth) *
        std::exp(four_ln_pone * xpy_squared / _beam.diameter_squared) *
        distribution_z;
    */

    double heat_source = 0.;
    heat_source =
        -_beam.energy_conversion_eff * _beam.control_eff * _beam.max_power *
        (4. * std::log(0.1)) /
        (dealii::numbers::PI * _beam.diameter_squared * _beam.depth) *
        std::exp((4. * std::log(0.1)) * xpy_squared / _beam.diameter_squared) *
        distribution_z;


    return heat_source;
  }
}
} // namespace adamantine

INSTANTIATE_DIM(ElectronBeam)

#endif
