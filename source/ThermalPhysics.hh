/* Copyright (c) 2016, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#ifndef _THERMAL_PHYSICS_HH_
#define _THERMAL_PHYSICS_HH_

#include <boost/property_tree/ptree.hpp>

namespace adamantine
{

/**
 * This class takes care of building the linear operator and the
 * right-hand-side.
 */
class ThermalPhysics
{
public:
  TermalPhysics(boost::property_tree::tree const &database);
};
}

#endif
