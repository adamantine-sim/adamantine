/* Copyright (c) 2022, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#define BOOST_TEST_MODULE MechanicalPhysics

#include <Geometry.hh>
#include <MechanicalPhysics.hh>

#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <iostream>

#include "main.cc"

namespace tt = boost::test_tools;

class ElastoStaticity
{
public:
  ElastoStaticity();
  void setup_system();
  void assemble_system();
  dealii::Vector<double> solve();

private:
  dealii::Triangulation<3> _triangulation;
  dealii::DoFHandler<3> _dof_handler;

  dealii::FESystem<3> _fe;

  dealii::AffineConstraints<double> _constraints;

  dealii::SparsityPattern _sparsity_pattern;
  dealii::SparseMatrix<double> _system_matrix;

  dealii::Vector<double> _system_rhs;
};

void right_hand_side(const std::vector<dealii::Point<3>> &points,
                     std::vector<dealii::Tensor<1, 3>> &values)
{
  for (unsigned int point_n = 0; point_n < points.size(); ++point_n)
  {
    values[point_n][2] = -10.0;
  }
}

ElastoStaticity::ElastoStaticity()
    : _dof_handler(_triangulation), _fe(dealii::FE_Q<3>(1), 3)
{
  std::vector<unsigned int> repetitions = {6, 3, 3};
  dealii::GridGenerator::subdivided_hyper_rectangle(
      _triangulation, repetitions, dealii::Point<3>(0, 0, 0),
      dealii::Point<3>(12, 6, 6), true);
}

void ElastoStaticity::setup_system()
{
  _dof_handler.distribute_dofs(_fe);
  _system_rhs.reinit(_dof_handler.n_dofs());

  _constraints.clear();
  dealii::DoFTools::make_hanging_node_constraints(_dof_handler, _constraints);
  dealii::VectorTools::interpolate_boundary_values(
      _dof_handler, 0, dealii::Functions::ZeroFunction<3>(3), _constraints);
  _constraints.close();

  dealii::DynamicSparsityPattern dsp(_dof_handler.n_dofs(),
                                     _dof_handler.n_dofs());
  dealii::DoFTools::make_sparsity_pattern(_dof_handler, dsp, _constraints,
                                          /*keep_constrained_dofs = */ false);
  _sparsity_pattern.copy_from(dsp);

  _system_matrix.reinit(_sparsity_pattern);
}

void ElastoStaticity::assemble_system()
{
  dealii::QGauss<3> quadrature_formula(_fe.degree + 1);

  dealii::FEValues<3> fe_values(
      _fe, quadrature_formula,
      dealii::update_values | dealii::update_gradients |
          dealii::update_quadrature_points | dealii::update_JxW_values);

  unsigned int const dofs_per_cell = _fe.n_dofs_per_cell();
  unsigned int const n_q_points = quadrature_formula.size();

  dealii::FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  dealii::Vector<double> cell_rhs(dofs_per_cell);

  std::vector<dealii::types::global_dof_index> local_dof_indices(dofs_per_cell);

  std::vector<double> lambda_values(n_q_points);
  std::vector<double> mu_values(n_q_points);

  dealii::Functions::ConstantFunction<3> lambda(2.);
  dealii::Functions::ConstantFunction<3> mu(3.);

  std::vector<dealii::Tensor<1, 3>> rhs_values(n_q_points);

  for (auto const &cell : _dof_handler.active_cell_iterators())
  {
    cell_matrix = 0;
    cell_rhs = 0;

    fe_values.reinit(cell);

    lambda.value_list(fe_values.get_quadrature_points(), lambda_values);
    mu.value_list(fe_values.get_quadrature_points(), mu_values);
    right_hand_side(fe_values.get_quadrature_points(), rhs_values);

    for (unsigned int i : fe_values.dof_indices())
    {
      unsigned int const component_i = _fe.system_to_component_index(i).first;

      for (unsigned int j : fe_values.dof_indices())
      {
        unsigned int const component_j = _fe.system_to_component_index(j).first;

        for (unsigned int q_point : fe_values.quadrature_point_indices())
        {
          cell_matrix(i, j) +=
              ((fe_values.shape_grad(i, q_point)[component_i] *
                fe_values.shape_grad(j, q_point)[component_j] *
                lambda_values[q_point]) +
               (fe_values.shape_grad(i, q_point)[component_j] *
                fe_values.shape_grad(j, q_point)[component_i] *
                mu_values[q_point]) +
               ((component_i == component_j)
                    ? (fe_values.shape_grad(i, q_point) *
                       fe_values.shape_grad(j, q_point) * mu_values[q_point])
                    : 0)) *
              fe_values.JxW(q_point);
        }
      }
    }

    for (unsigned int i : fe_values.dof_indices())
    {
      unsigned int const component_i = _fe.system_to_component_index(i).first;

      for (unsigned int q_point : fe_values.quadrature_point_indices())
        cell_rhs(i) += fe_values.shape_value(i, q_point) *
                       rhs_values[q_point][component_i] *
                       fe_values.JxW(q_point);
    }

    cell->get_dof_indices(local_dof_indices);
    _constraints.distribute_local_to_global(
        cell_matrix, cell_rhs, local_dof_indices, _system_matrix, _system_rhs);
  }
}

dealii::Vector<double> ElastoStaticity::solve()
{
  dealii::SolverControl solver_control(1000, 1e-12);
  dealii::SolverCG<dealii::Vector<double>> cg(solver_control);

  dealii::PreconditionSSOR<dealii::SparseMatrix<double>> preconditioner;
  preconditioner.initialize(_system_matrix, 1.2);

  dealii::Vector<double> solution(_dof_handler.n_dofs());
  cg.solve(_system_matrix, solution, _system_rhs, preconditioner);

  _constraints.distribute(solution);

  return solution;
}

BOOST_AUTO_TEST_CASE(elastostatic)
{
  MPI_Comm communicator = MPI_COMM_WORLD;

  // Geometry database
  boost::property_tree::ptree geometry_database;
  geometry_database.put("import_mesh", false);
  geometry_database.put("length", 12);
  geometry_database.put("length_divisions", 6);
  geometry_database.put("height", 6);
  geometry_database.put("height_divisions", 3);
  geometry_database.put("width", 6);
  geometry_database.put("width_divisions", 3);
  // Build Geometry
  adamantine::Geometry<3> geometry(communicator, geometry_database);
  // Mechanical database
  boost::property_tree::ptree mechanical_database;
  mechanical_database.put("fe_degree", 1);
  mechanical_database.put("lame_first_param", 2);
  mechanical_database.put("lame_second_param", 3);
  // Build MechanicalPhysics
  adamantine::MechanicalPhysics<3> mechanical_physics(
      communicator, mechanical_database, geometry);
  mechanical_physics.setup_dofs();
  auto solution = mechanical_physics.solve();

  // Reference computation
  ElastoStaticity elasto_staticity;
  elasto_staticity.setup_system();
  elasto_staticity.assemble_system();
  auto reference_solution = elasto_staticity.solve();

  double const tolerance = 1e-9;
  BOOST_TEST(solution.size() == reference_solution.size());
  for (unsigned int i = 0; i < reference_solution.size(); ++i)
    BOOST_TEST(solution[i] == reference_solution[i], tt::tolerance(tolerance));
}
