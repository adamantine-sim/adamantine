/* Copyright (c) 2021, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#ifndef MATERIAL_DEPOSITION_HH
#define MATERIAL_DEPOSITION_HH

#include <HeatSources.hh>

#include <deal.II/base/bounding_box.h>
#include <deal.II/dofs/dof_handler.h>

#include <boost/property_tree/ptree.hpp>

namespace adamantine
{
/**
 * Return the bounding boxes, the deposition times, the cosine of the deposition
 * angles, and the sine of the deposition angles.
 */
template <int dim>
std::tuple<std::vector<dealii::BoundingBox<dim>>, std::vector<double>,
           std::vector<double>, std::vector<double>>
create_material_deposition_boxes(
    boost::property_tree::ptree const &geometry_database,
    HeatSources<dim, dealii::MemorySpace::Host> &heat_sources);
/**
 * Read the material deposition file and return the bounding boxes, the
 * deposition times, the cosine of the deposition angles, and the sine of the
 * deposition angles.
 */
template <int dim>
std::tuple<std::vector<dealii::BoundingBox<dim>>, std::vector<double>,
           std::vector<double>, std::vector<double>>
read_material_deposition(boost::property_tree::ptree const &geometry_database);
/**
 * Return the bounding boxes, the deposition times, the cosine of the deposition
 * angles, and the sine of deposition angles based on the scan path.
 */
template <int dim>
std::tuple<std::vector<dealii::BoundingBox<dim>>, std::vector<double>,
           std::vector<double>, std::vector<double>>
deposition_along_scan_path(
    boost::property_tree::ptree const &geometry_database,
    ScanPath<dealii::MemorySpace::Host> const &scan_path);
/**
 * Merge a vector of tuple of bounding boxes, deposition times, cosine of
 * deposition angles, and sine of deposition angles into a
 * single tuple of vectors, sorted by deposition time.
 */
template <int dim>
std::tuple<std::vector<dealii::BoundingBox<dim>>, std::vector<double>,
           std::vector<double>, std::vector<double>>
merge_deposition_paths(
    std::vector<std::tuple<std::vector<dealii::BoundingBox<dim>>,
                           std::vector<double>, std::vector<double>,
                           std::vector<double>>> const &bounding_box_lists);
/**
 * Return a vector of cells to activate for each time deposition.
 */
template <int dim>
std::vector<std::vector<typename dealii::DoFHandler<dim>::active_cell_iterator>>
get_elements_to_activate(
    dealii::DoFHandler<dim> const &dof_handler,
    std::vector<dealii::BoundingBox<dim>> const &material_deposition_boxes);
} // namespace adamantine

#endif
