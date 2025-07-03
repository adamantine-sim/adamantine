/* Copyright (c) 2016 - 2024, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#ifndef CLOSEST_SOURCE_POINT_ADAPTATION_HH
#define CLOSEST_SOURCE_POINT_ADAPTATION_HH

#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_values.h>

#include <random>

namespace adamantine
{

template <int dim, int spacedim, typename quad_point_value_type>
class ClosestQuadPointAdaptation
{
public:
  template <class Quadrature>
  ClosestQuadPointAdaptation(const Quadrature &quadrature)
      : m_fe_values_coarse(m_fe_nothing, quadrature,
                           dealii::update_quadrature_points),
        m_fe_values_fine(m_fe_nothing, quadrature,
                         dealii::update_quadrature_points)
  {
  }

  std::vector<std::vector<quad_point_value_type>> coarse_to_fine(
      const typename dealii::Triangulation<dim, spacedim>::cell_iterator
          &parent,
      const std::vector<quad_point_value_type> parent_values)
  {
    m_fe_values_coarse.reinit(parent);
    const std::vector<dealii::Point<spacedim>> &coarse_quad_points =
        m_fe_values_coarse.get_quadrature_points();

    std::vector<std::vector<quad_point_value_type>> child_values(
        parent->n_children(),
        std::vector<quad_point_value_type>(coarse_quad_points.size()));
    for (unsigned int i = 0; i < parent->n_children(); ++i)
    {
      m_fe_values_fine.reinit(parent->child(i));
      const std::vector<dealii::Point<spacedim>> &fine_quad_points =
          m_fe_values_fine.get_quadrature_points();
      for (unsigned int j = 0; j < fine_quad_points.size(); ++j)
      {
        std::pair min_index_distance{-1, std::numeric_limits<double>::max()};
        for (unsigned int k = 0; k < coarse_quad_points.size(); ++k)
        {
          double candidate_distance =
              (fine_quad_points[j] - coarse_quad_points[k]).norm_square();
          if (candidate_distance < min_index_distance.second)
          {
            min_index_distance.first = k;
            min_index_distance.second = candidate_distance;
          }
        }
        child_values[i][j] = parent_values[min_index_distance.first];
      }
    }
    return child_values;
  }

  std::vector<quad_point_value_type> fine_to_coarse(
      const typename dealii::Triangulation<dim, spacedim>::cell_iterator
          &parent,
      const std::vector<std::vector<quad_point_value_type>> &children_values)
  {
    m_fe_values_coarse.reinit(parent);
    const std::vector<dealii::Point<spacedim>> &coarse_quad_points =
        m_fe_values_coarse.get_quadrature_points();

    std::vector<std::pair<std::pair<int, int>, double>> min_index_distances{
        coarse_quad_points.size(),
        std::pair{std::pair{-1, -1}, std::numeric_limits<double>::max()}};
    for (unsigned int i = 0; i < parent->n_children(); ++i)
    {
      m_fe_values_fine.reinit(parent->child(i));
      const std::vector<dealii::Point<spacedim>> &fine_quad_points =
          m_fe_values_fine.get_quadrature_points();
      for (unsigned int j = 0; j < fine_quad_points.size(); ++j)
      {
        for (unsigned int k = 0; k < coarse_quad_points.size(); ++k)
        {
          double candidate_distance =
              (fine_quad_points[j] - coarse_quad_points[k]).norm_square();
          if (candidate_distance < min_index_distances[k].second)
          {
            min_index_distances[k].first = std::pair{i, j};
            min_index_distances[k].second = candidate_distance;
          }
        }
      }
    }
    std::vector<quad_point_value_type> parent_values(coarse_quad_points.size());
    for (unsigned int k = 0; k < coarse_quad_points.size(); ++k)
    {
      parent_values[k] = children_values[min_index_distances[k].first.first]
                                        [min_index_distances[k].first.second];
    }

    return parent_values;
  }

private:
  dealii::FE_Nothing<dim, spacedim> m_fe_nothing;
  dealii::FEValues<dim, spacedim> m_fe_values_coarse;
  dealii::FEValues<dim, spacedim> m_fe_values_fine;
};

} // namespace adamantine

#endif
