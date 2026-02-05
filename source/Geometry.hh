/* SPDX-FileCopyrightText: Copyright (c) 2016 - 2026, the adamantine authors.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#ifndef GEOMETRY_HH
#define GEOMETRY_HH

#include <deal.II/distributed/tria.h>

#include <boost/property_tree/ptree.hpp>

#include <ArborX_Config.hpp>
#if ARBORX_VERSION_MAJOR >= 2
#include <ArborX_Triangle.hpp>
#else
#include <ArborX_HyperTriangle.hpp>
#endif

namespace adamantine
{
/**
 * This class generates and stores a Triangulation given a database.
 */
template <int dim>
class Geometry
{
#if ARBORX_VERSION_MAJOR >= 2
  using Point = ArborX::Point<3, double>;
  using Triangle = ArborX::Triangle<3, double>;
#else
  using Point = ArborX::ExperimentalHyperGeometry::Point<3, double>;
  using Triangle = ArborX::ExperimentalHyperGeometry::Triangle<3, double>;
#endif

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

  /**
   * Return the triangles from the STL file.
   */
  Kokkos::View<Triangle *, Kokkos::HostSpace> get_stl_triangles();

private:
  /**
   * Assign the material state to the mesh.
   */
  void assign_material_state(boost::property_tree::ptree const &database);

  /**
   *  Read the given binary STL file and apply a scaling factor to all
   *  coordinates read from the STL file.
   */
  void read_stl(std::string const &filename, double const stl_scaling);

  /**
   * Triangulation of the domain.
   */
  dealii::parallel::distributed::Triangulation<dim> _triangulation;
  /**
   * View of the triangles from the STL file.
   */
  Kokkos::View<Triangle *, Kokkos::HostSpace> _stl_triangles;
};

template <int dim>
inline dealii::parallel::distributed::Triangulation<dim> &
Geometry<dim>::get_triangulation()
{
  return _triangulation;
}

template <int dim>
inline Kokkos::View<typename Geometry<dim>::Triangle *, Kokkos::HostSpace>
Geometry<dim>::get_stl_triangles()
{
  return _stl_triangles;
}
} // namespace adamantine

#endif
