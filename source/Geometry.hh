/* SPDX-FileCopyrightText: Copyright (c) 2016 - 2026, the adamantine authors.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#ifndef GEOMETRY_HH
#define GEOMETRY_HH

#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_handler.h>

#include <boost/property_tree/ptree.hpp>

#include <ArborX.hpp>
#include <ArborX_Config.hpp>
#if ARBORX_VERSION_MAJOR >= 2
#include <ArborX_Triangle.hpp>
#else
#include <ArborX_HyperTriangle.hpp>
#endif

#include <memory>

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
   * Return true if the file is defined by an STL file and false otherwise.
   */
  bool use_stl() const;

  /**
   * Return the underlying Triangulation.
   */
  dealii::parallel::distributed::Triangulation<dim> &get_triangulation();

  /**
   * Return the triangles from the STL file.
   */
  Kokkos::View<Triangle *, Kokkos::HostSpace> get_stl_triangles();

#if ARBORX_VERSION_MAJOR >= 2
  /**
   * Return true if a cell is fully included inside the domain defined by the
   * STL. Return false if the cell is outside the domain or if it intersects it.
   *
   * @note This function assumes that the STL triangle normals are all pointing
   * outwards from the enclosed volume.
   */
  bool is_within_stl(
      typename dealii::DoFHandler<dim>::active_cell_iterator const &cell) const;
#endif

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
   * Flag is true if the domain is defined by a STL file.
   */
  bool _use_stl = false;
  /**
   * Triangulation of the domain.
   */
  dealii::parallel::distributed::Triangulation<dim> _triangulation;
  /**
   * View of the triangles from the STL file.
   */
  Kokkos::View<Triangle *, Kokkos::HostSpace> _stl_triangles;

#if ARBORX_VERSION_MAJOR >= 2
  /**
   * BVH tree with STL triangles as leaf nodes.
   */
  std::unique_ptr<ArborX::BVH<Kokkos::HostSpace, Triangle>> _bvh;
#endif
};

template <int dim>
inline bool Geometry<dim>::use_stl() const
{
  return _use_stl;
}

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
