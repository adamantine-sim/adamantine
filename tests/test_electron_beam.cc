/* Copyright (c) 2016, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#define BOOST_TEST_MODULE ElectronBeam

#include "main.cc"

#include "ElectronBeam.hh"

BOOST_AUTO_TEST_CASE(beam_2d)
{
  boost::property_tree::ptree database;

  database.put("depth", 0.1);
  database.put("energy_conversion_efficiency", 0.1);
  database.put("control_efficiency", 1.0);
  database.put("diameter", 1.0);
  database.put("max_power", 10.);
  database.put("abscissa", "t");

  adamantine::ElectronBeam<2> beam(database);
  beam.set_time(1.0);
  beam.set_max_height(0.2);

  dealii::Point<2> point;
  point[0] = 1.;
  point[1] = 0.2;

  double const tolerance = 1e-5;
  double value = beam.value(point);
  BOOST_CHECK_CLOSE(value, 29.317423, tolerance);

  point[0] = 0.9;
  point[1] = 0.15;
  value = beam.value(point);
  BOOST_CHECK_CLOSE(value, 33.422260, tolerance);

  database.put("depth", 1e100);
  database.put("energy_conversion_efficiency", 0.1);
  database.put("control_efficiency", 1.0);
  database.put("diameter", 1e100);
  database.put("max_power", 1e300);
  database.put("abscissa", "t");

  adamantine::ElectronBeam<2> beam_2(database);
  beam_2.set_time(1.0);
  beam_2.set_max_height(0.2);
  value = beam_2.value(point);
  BOOST_CHECK_CLOSE(value, 0.29317423955177113, tolerance);

  point[0] = 1.;
  point[1] = 0.2;
  value = beam_2.value(point);
  BOOST_CHECK_CLOSE(value, 0.29317423955177113, tolerance);
}

BOOST_AUTO_TEST_CASE(beam_3d)
{
  boost::property_tree::ptree database;

  database.put("depth", 0.1);
  database.put("energy_conversion_efficiency", 0.1);
  database.put("control_efficiency", 0.1);
  database.put("diameter", 1.0);
  database.put("max_power", 100.);
  database.put("abscissa", "t");
  database.put("ordinate", "2*t");

  adamantine::ElectronBeam<3> beam(database);
  beam.set_time(0.5);
  beam.set_max_height(0.2);

  dealii::Point<3> point;
  point[0] = 0.5;
  point[1] = 0.2;
  point[2] = 1.0;

  double const tolerance = 1e-5;
  double value = beam.value(point);
  BOOST_CHECK_CLOSE(value, 29.317423, tolerance);

  point[0] = 0.9;
  point[1] = 0.15;
  point[2] = 1.1;
  value = beam.value(point);
  BOOST_CHECK_CLOSE(value, 7.65659755, tolerance);
}
