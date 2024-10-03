/* Copyright (c) 2016 - 2021, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#ifndef GEOMETRY_HH
#define GEOMETRY_HH

#include <deal.II/distributed/tria.h>

#include <boost/property_tree/ptree.hpp>

namespace adamantine
{
/**
 * This class generates and stores a Triangulation given a database.
 */
template <int dim>
class Geometry
{
public:
  /**
   * Constructor.
   */
  Geometry(MPI_Comm const &communicator,
           boost::property_tree::ptree const &database,
           boost::optional<boost::property_tree::ptree const &> const
               &units_optional_database);

  /**
   * Return the underlying Triangulation.
   */
  dealii::parallel::distributed::Triangulation<dim> &get_triangulation();

private:
  /**
   * Triangulation of the domain.
   */
  dealii::parallel::distributed::Triangulation<dim> _triangulation;

  /**
   * Assign the material state to the mesh.
   */
  void assign_material_state(boost::property_tree::ptree const &database);
};

template <int dim>
inline dealii::parallel::distributed::Triangulation<dim> &
Geometry<dim>::get_triangulation()
{
  return _triangulation;
}
} // namespace adamantine

#endif
