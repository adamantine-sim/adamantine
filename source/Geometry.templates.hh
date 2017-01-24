/* Copyright (c) 2016, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#ifndef _GEOMETRY_TEMPLATES_HH_
#define _GEOMETRY_TEMPLATES_HH_

#include "Geometry.hh"
#include "types.hh"
#include "utils.hh"
#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/grid/grid_generator.h>

namespace adamantine
{

template <int dim>
Geometry<dim>::Geometry(boost::mpi::communicator const &communicator,
                        boost::property_tree::ptree const &database)
    : _triangulation(communicator)
{
  std::vector<unsigned int> repetitions(dim);
  repetitions[0] = database.get("length_divisions", 10);
  repetitions[1] = database.get("height_divisions", 10);
  if (dim == 3)
    repetitions[2] = database.get("width_divisions", 10);

  dealii::Point<dim> p1;
  dealii::Point<dim> p2;
  p2[0] = database.get<double>("length");
  p2[1] = database.get<double>("height");
  _max_height = p2[1];
  if (dim == 3)
    p2[2] = database.get<double>("width");

  // For now we assume that the geometry is very simple.
  dealii::GridGenerator::subdivided_hyper_rectangle(_triangulation, repetitions,
                                                    p1, p2, true);

  // Assign the MaterialID and the MaterialState.
  dealii::types::boundary_id const top_boundary = 3;
  for (auto cell : _triangulation.active_cell_iterators())
  {
    cell->set_material_id(0);
    if (cell->at_boundary())
    {
      bool is_powder = false;
      for (unsigned int i = 0; i < dealii::GeometryInfo<dim>::faces_per_cell;
           ++i)
      {
        if ((cell->face(i)->at_boundary()) &&
            (cell->face(i)->boundary_id() == top_boundary))
        {
          cell->set_user_index(static_cast<int>(MaterialState::powder));
          is_powder = true;
          break;
        }
      }
      if (is_powder == false)
        cell->set_user_index(static_cast<int>(MaterialState::solid));
    }
    else
      cell->set_user_index(static_cast<int>(MaterialState::solid));
  }
}
}

#endif
