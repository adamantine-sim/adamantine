/* Copyright (c) 2016, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#ifndef _GEOMETRY_HH_
#define _GEOMETRY_HH_

#include <boost/mpi/communicator.hpp>
#include <boost/property_tree/ptree.hpp>
#include <deal.II/distributed/tria.h>

namespace adamantine
{

enum Material {powder, solid, liquid};

template <int dim>
class Geometry
{
  public:
    Geometry(boost::mpi::communicator const &communicator,
        boost::property_tree::ptree const &database);

  private:
      dealii::parallel::distributed::Triangulation<dim> _triangulation;
};

}

#endif
