/* Copyright (c) 2016 - 2021, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#ifndef GEOMETRY_TEMPLATES_HH
#define GEOMETRY_TEMPLATES_HH

#include <Geometry.hh>
#include <instantiation.hh>
#include <types.hh>
#include <utils.hh>

#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_tools.h>

namespace adamantine
{
template <int dim>
Geometry<dim>::Geometry(MPI_Comm const &communicator,
                        boost::property_tree::ptree const &database)
    : _triangulation(communicator)
{
  // PropertyTreeInput geometry.import_mesh
  bool import_mesh = database.get<bool>("import_mesh");

  if (import_mesh == true)
  {
    // PropertyTreeInput geometry.mesh_file
    std::string mesh_file = database.get<std::string>("mesh_file");
    // PropertyTreeInput geometry.mesh_format
    std::string mesh_format = database.get<std::string>("mesh_format");
    dealii::Triangulation<dim> serial_triangulation;
    dealii::GridIn<dim> grid_in;
    grid_in.attach_triangulation(serial_triangulation);
    typename dealii::GridIn<dim>::Format grid_in_format;
    if (mesh_format == "abaqus")
    {
      grid_in_format = dealii::GridIn<dim>::Format::abaqus;
    }
    else if (mesh_format == "assimp")
    {
      grid_in_format = dealii::GridIn<dim>::Format::assimp;
    }
    else if (mesh_format == "unv")
    {
      grid_in_format = dealii::GridIn<dim>::Format::unv;
    }
    else if (mesh_format == "ucd")
    {
      grid_in_format = dealii::GridIn<dim>::Format::ucd;
    }
    else if (mesh_format == "dbmesh")
    {
      grid_in_format = dealii::GridIn<dim>::Format::dbmesh;
    }
    else if (mesh_format == "gmsh")
    {
      grid_in_format = dealii::GridIn<dim>::Format::msh;
    }
    else if (mesh_format == "tecplot")
    {
      grid_in_format = dealii::GridIn<dim>::Format::tecplot;
    }
    else if (mesh_format == "xda")
    {
      grid_in_format = dealii::GridIn<dim>::Format::xda;
    }
    else if (mesh_format == "vtk")
    {
      grid_in_format = dealii::GridIn<dim>::Format::vtk;
    }
    else if (mesh_format == "vtu")
    {
      grid_in_format = dealii::GridIn<dim>::Format::vtu;
    }
    else if (mesh_format == "exodusii")
    {
      grid_in_format = dealii::GridIn<dim>::Format::exodusii;
    }
    else
    {
      grid_in_format = dealii::GridIn<dim>::Format::Default;
    }

    grid_in.read(mesh_file, grid_in_format);
    _triangulation.copy_triangulation(serial_triangulation);

    // Apply user-specified scaling to the mesh
    // PropertyTreeInput geometry.mesh_scale_factor
    auto mesh_scaling = database.get("mesh_scale_factor", 1.0);
    dealii::GridTools::scale(mesh_scaling, _triangulation);

    // Set the mesh material id to 0 if specified in the input
    // PropertyTreeInput geometry.reset_material_id
    auto reset_material_id = database.get("reset_material_id", false);
    if (reset_material_id)
    {
      for (auto cell : _triangulation.active_cell_iterators())
      {
        cell->set_material_id(0);
      }
    }
  }
  else
  {
    std::vector<unsigned int> repetitions(dim);
    // PropertyTreeInput geometry.length_divisions
    repetitions[axis<dim>::x] = database.get("length_divisions", 10);
    // PropertyTreeInput geometry.height_divisions
    repetitions[axis<dim>::z] = database.get("height_divisions", 10);
    // PropertyTreeInput geometry.width_divisions
    if (dim == 3)
      repetitions[axis<dim>::y] = database.get("width_divisions", 10);

    dealii::Point<dim> p1;
    dealii::Point<dim> p2;
    // PropertyTreeInput geometry.length
    p2[axis<dim>::x] = database.get<double>("length");
    // PropertyTreeInput geometry.length_origin
    p1[axis<dim>::x] = database.get("length_origin", 0.0);
    // PropertyTreeInput geometry.height
    p2[axis<dim>::z] = database.get<double>("height");
    // PropertyTreeInput geometry.height_origin
    p1[axis<dim>::z] = database.get("height_origin", 0.0);
    if (dim == 3)
    {
      // PropertyTreeInput geometry.width
      p2[axis<dim>::y] = database.get<double>("width");
      // PropertyTreeInput geometry.width_origin
      p1[axis<dim>::y] = database.get("width_origin", 0.0);
    }

    p2 = p2 + p1;

    // For now we assume that the geometry is very simple.
    dealii::GridGenerator::subdivided_hyper_rectangle(
        _triangulation, repetitions, p1, p2, true);

    // Assign the MaterialID.
    for (auto cell : _triangulation.active_cell_iterators())
    {
      cell->set_material_id(0);
    }
  }

  assign_material_state(database);
}

template <int dim>
void Geometry<dim>::assign_material_state(
    boost::property_tree::ptree const &database)
{
  // PropertyTreeInput geometry.material_height
  double const material_height = database.get("material_height", 1e9);
  // PropertyTreeInput geometry.use_powder
  bool const use_powder = database.get("use_powder", false);

  if (use_powder)
  {
    // PropertyTreeInput geometry.powder_layer
    double const powder_layer = database.get<double>("powder_layer");
    double const solid_height = material_height - powder_layer;

    for (auto cell : _triangulation.active_cell_iterators())
    {
      if (cell->center()[axis<dim>::z] < solid_height)
      {
        cell->set_user_index(static_cast<int>(MaterialState::solid));
      }
      else
      {
        cell->set_user_index(static_cast<int>(MaterialState::powder));
      }
    }
  }
  else
  {
    // Everything is made of solid material
    for (auto cell : _triangulation.active_cell_iterators())
    {
      cell->set_user_index(static_cast<int>(MaterialState::solid));
    }
  }
}
} // namespace adamantine

INSTANTIATE_DIM(Geometry)

#endif
