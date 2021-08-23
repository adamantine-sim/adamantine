/* Copyright (c) 2021, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#ifndef MATERIAL_DEPOSITION_HH
#define MATERIAL_DEPOSITION_HH

#include <HeatSource.hh>

#include <deal.II/base/bounding_box.h>
#include <deal.II/dofs/dof_handler.h>

#include <boost/property_tree/ptree.hpp>

namespace adamantine
{
/**
 * Return a vector of bounding boxes and a vector of deposition times
 */
template <int dim>
std::pair<std::vector<dealii::BoundingBox<dim>>, std::vector<double>>
create_material_deposition_boxes(
    boost::property_tree::ptree const &geometry_database,
    std::vector<std::shared_ptr<HeatSource<dim>>> &heat_sources);
/**
 * Read the material deposition file and return a vector of bounding boxes
 * and a vector of deposition times.
 */
template <int dim>
std::pair<std::vector<dealii::BoundingBox<dim>>, std::vector<double>>
read_material_deposition(boost::property_tree::ptree const &geometry_database);
/**
 * Return a vector of bounding boxes and a vector of deposition times based on
 * the scan path.
 */
template <int dim>
std::pair<std::vector<dealii::BoundingBox<dim>>, std::vector<double>>
deposition_along_scan_path(boost::property_tree::ptree const &geometry_database,
                           ScanPath const &scan_path);
/**
 * Merge a vector of pairs of bounding boxes and a deposition times into a
 * single pair of vectors, sorted by deposition time.
 */
template <int dim>
std::pair<std::vector<dealii::BoundingBox<dim>>, std::vector<double>>
merge_bounding_box_lists(
    std::vector<
        std::pair<std::vector<dealii::BoundingBox<dim>>, std::vector<double>>>
        bounding_box_lists);
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
