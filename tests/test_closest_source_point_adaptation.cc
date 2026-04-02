/* Copyright (c) 2016 - 2024, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#define BOOST_TEST_MODULE ClosestSourcePointAdaptation

#include "../source/ClosestSourcePointAdaptation.hh"

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/distributed/cell_data_transfer.templates.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/grid/grid_generator.h>

#include <random>
#include <string>

#include "main.cc"

template <int dim, int spacedim>
void test()
{
  dealii::parallel::distributed::Triangulation<dim, spacedim> tria(
      MPI_COMM_WORLD);

  dealii::GridGenerator::subdivided_hyper_cube(tria, 2);
  tria.refine_global(3);

  dealii::QGauss<dim> quadrature(2);
  const unsigned int n_q_points = quadrature.size();
  const unsigned int n_active_cells = tria.n_global_active_cells();

  std::vector<std::vector<double>> cell_quad_point_values(
      n_active_cells,
      std::vector<double>(n_q_points,
                          std::numeric_limits<double>::signaling_NaN()));

  std::random_device rnd_device;
  std::mt19937 mersenne_engine{rnd_device()};
  std::uniform_real_distribution<double> dist{1., 2.};

  for (auto &cell : tria.active_cell_iterators())
    if (cell->is_locally_owned())
    {
      cell->set_refine_flag();
      std::generate(cell_quad_point_values[cell->active_cell_index()].begin(),
                    cell_quad_point_values[cell->active_cell_index()].end(),
                    [&]() { return dist(mersenne_engine); });
    }

  adamantine::ClosestQuadPointAdaptation<dim, spacedim, double>
      closest_quad_point_adaptation(quadrature);
  dealii::parallel::distributed::CellDataTransfer<
      dim, spacedim, std::vector<std::vector<double>>>
      cell_data_transfer(
          tria, false,
          [&](const typename dealii::Triangulation<dim, spacedim>::cell_iterator
                  &parent,
              const std::vector<double> parent_values)
          {
            return closest_quad_point_adaptation.coarse_to_fine(parent,
                                                                parent_values);
          },
          [&](const typename dealii::Triangulation<dim, spacedim>::cell_iterator
                  &parent,
              const std::vector<std::vector<double>> child_values)
          {
            return closest_quad_point_adaptation.fine_to_coarse(parent,
                                                                child_values);
          });

  cell_data_transfer.prepare_for_coarsening_and_refinement(
      cell_quad_point_values);
  tria.execute_coarsening_and_refinement();

  const unsigned int n_active_cells_new = tria.n_global_active_cells();

  std::vector<std::vector<double>> cell_quad_point_values_new(
      n_active_cells_new,
      std::vector<double>(n_q_points,
                          std::numeric_limits<double>::signaling_NaN()));
  cell_data_transfer.unpack(cell_quad_point_values_new);

  for (auto &cell : tria.active_cell_iterators())
    if (cell->is_locally_owned())
    {
      cell->set_coarsen_flag();
      for (unsigned int i = 0; i < n_q_points; ++i)
      {
        const double quad_point_value =
            cell_quad_point_values_new[cell->active_cell_index()][i];
        BOOST_TEST(quad_point_value >= 1.);
        BOOST_TEST(quad_point_value <= 2.);
      }
    }

  cell_data_transfer.prepare_for_coarsening_and_refinement(
      cell_quad_point_values_new);
  tria.execute_coarsening_and_refinement();

  BOOST_TEST(tria.n_global_active_cells() == n_active_cells);
  std::vector<std::vector<double>> cell_quad_point_values_final(
      n_active_cells,
      std::vector<double>(n_q_points,
                          std::numeric_limits<double>::signaling_NaN()));

  cell_data_transfer.unpack(cell_quad_point_values_final);

  for (auto &cell : tria.active_cell_iterators())
    if (cell->is_locally_owned())
    {
      cell->set_refine_flag();
      for (unsigned int i = 0; i < n_q_points; ++i)
      {
        BOOST_TEST(cell_quad_point_values_final[cell->active_cell_index()][i] ==
                   cell_quad_point_values[cell->active_cell_index()][i]);
      }
    }
}

BOOST_AUTO_TEST_CASE(closest_source_point_adaptation)
{
  test<2, 2>();
  test<3, 3>();
}
