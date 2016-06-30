/* Copyright (c) 2016, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#ifndef _GEOMETRY_HH_
#define _GEOMETRY_HH_

#include <boost/property_tree/ptree.hpp>

namespace adamantine
{

template <int dim>
Geometry
{
  public:
    Geometry(boost::property_tree::ptree const &database)
};

}

#endif
