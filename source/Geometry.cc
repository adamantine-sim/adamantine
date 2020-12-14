/* Copyright (c) 2016 - 2020, the adamantine authors.
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
    dealii::GridIn<dim> grid_in;
    grid_in.attach_triangulation(_triangulation);
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

    if (grid_in_format != dealii::GridIn<dim>::Format::assimp)
    {
      grid_in.read(mesh_file, grid_in_format);
    }

    // PropertyTreeInput geometry.top_boundary_id
    dealii::types::boundary_id const top_boundary =
        database.get<int>("top_boundary_id");
    assign_material_state(top_boundary);
  }
  else
  {
    std::vector<unsigned int> repetitions(dim);
    // PropertyTreeInput geometry.length_divisions
    repetitions[0] = database.get("length_divisions", 10);
    // PropertyTreeInput geometry.height_divisions
    repetitions[1] = database.get("height_divisions", 10);
    // PropertyTreeInput geometry.width_divisions
    if (dim == 3)
      repetitions[2] = database.get("width_divisions", 10);

    dealii::Point<dim> p1;
    dealii::Point<dim> p2;
    // PropertyTreeInput geometry.length
    p2[0] = database.get<double>("length");
    // PropertyTreeInput geometry.height
    p2[1] = database.get<double>("height");
    // PropertyTreeInput geometry.width
    if (dim == 3)
      p2[2] = database.get<double>("width");

    // For now we assume that the geometry is very simple.
    dealii::GridGenerator::subdivided_hyper_rectangle(
        _triangulation, repetitions, p1, p2, true);

    // Assign the MaterialID.
    for (auto cell : _triangulation.active_cell_iterators())
    {
      cell->set_material_id(0);
    }

    // Assign the MaterialState.
    dealii::types::boundary_id const top_boundary = 3;
    assign_material_state(top_boundary);
  }
}

template <int dim>
void Geometry<dim>::assign_material_state(
    dealii::types::boundary_id top_boundary)
{
  for (auto cell : _triangulation.active_cell_iterators())
  {
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
} // namespace adamantine

INSTANTIATE_DIM(Geometry)

#endif
