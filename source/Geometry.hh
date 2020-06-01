/* Copyright (c) 2016 - 2020, the adamantine authors.
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
   * \param[in] database requires the following entries:
   *   - <B>length_divisions</B>: unsigned int in \f$[1,\infty)\f$ [optional:
   *   default value is 10]
   *   - <B>height_divisions</B>: unsigned int in \f$[1,\infty)\f$ [optional:
   *   default value is 10]
   *   - <B>width_divisions</B>: unsigned int in \f$[1,\infty)\f$ [optional:
   *   default value is 10, only used in three dimensional calculation]
   *   - <B>length</B>: double in \f$(0,\infty)\f$
   *   - <B>height</B>: double in \f$(0,\infty)\f$
   *   - <B>width</B>: double in \f$(0,\infty)\f$ [only used in three
   *   dimensional calculation]
   */
  Geometry(MPI_Comm const &communicator,
           boost::property_tree::ptree const &database);

  /**
   * Return the underlying Triangulation.
   */
  dealii::parallel::distributed::Triangulation<dim> &get_triangulation();

  /**
   * Return the maximum height of the domain.
   */
  double get_max_height() const;

private:
  /**
   * Maximum height of the domain.
   */
  double _max_height;
  /**
   * Shared pointer to the underlying Triangulation.
   */
  dealii::parallel::distributed::Triangulation<dim> _triangulation;
};

template <int dim>
inline double Geometry<dim>::get_max_height() const
{
  return _max_height;
}

template <int dim>
inline dealii::parallel::distributed::Triangulation<dim> &
Geometry<dim>::get_triangulation()
{
  return _triangulation;
}
} // namespace adamantine

#endif
