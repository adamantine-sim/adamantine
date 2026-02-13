/* SPDX-FileCopyrightText: Copyright (c) 2021 - 2026, the adamantine authors.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#ifndef MATERIAL_DEPOSITION_HH
#define MATERIAL_DEPOSITION_HH

#include <Geometry.hh>
#include <HeatSource.hh>

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
    std::vector<std::shared_ptr<HeatSource<dim>>> &heat_sources);
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
deposition_along_scan_path(boost::property_tree::ptree const &geometry_database,
                           ScanPath const &scan_path);
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
    Geometry<dim> const &geometry, dealii::DoFHandler<dim> const &dof_handler,
    std::vector<dealii::BoundingBox<dim>> const &material_deposition_boxes);
} // namespace adamantine

#endif
