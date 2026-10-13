/* SPDX-FileCopyrightText: Copyright (c) 2022 - 2026, the adamantine authors.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#define BOOST_TEST_MODULE MechanicalOperator

#include <Geometry.hh>
#include <MechanicalOperator.hh>

#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/hp/fe_values.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/numerics/vector_tools.h>

#include <boost/property_tree/ptree.hpp>

#include <algorithm>

#include "main.cc"

namespace utf = boost::unit_test;

template <int dim>
class AffineDisplacement : public dealii::Function<dim>
{
public:
  explicit AffineDisplacement(unsigned int component)
      : dealii::Function<dim>(dim), _component(component)
  {
  }

  double value(dealii::Point<dim> const &point,
               unsigned int component = 0) const override
  {
    return component == _component ? point[0] : 0.;
  }

private:
  unsigned int _component;
};

template <int dim>
void right_hand_side(const std::vector<dealii::Point<dim>> &points,
                     std::vector<dealii::Tensor<1, dim>> &values)
{
  dealii::Point<dim> point_1, point_2;
  point_1(0) = 0.5;
  point_2(0) = -0.5;

  for (unsigned int point_n = 0; point_n < points.size(); ++point_n)
  {
    if (((points[point_n] - point_1).norm_square() < 0.2 * 0.2) ||
        ((points[point_n] - point_2).norm_square() < 0.2 * 0.2))
      values[point_n][0] = 1.0;
    else
      values[point_n][0] = 0.0;

    if (points[point_n].norm_square() < 0.2 * 0.2)
      values[point_n][1] = 1.0;
    else
      values[point_n][1] = 0.0;
  }
}

BOOST_AUTO_TEST_CASE(elastostatic, *utf::tolerance(1e-12))
{
  MPI_Comm communicator = MPI_COMM_WORLD;
  int constexpr dim = 3;

  // Create the Geometry
  boost::property_tree::ptree geometry_database;
  geometry_database.put("import_mesh", false);
  geometry_database.put("length", 6);
  geometry_database.put("length_divisions", 3);
  geometry_database.put("height", 6);
  geometry_database.put("height_divisions", 3);
  geometry_database.put("width", 6);
  geometry_database.put("width_divisions", 3);
  boost::optional<boost::property_tree::ptree const &> units_optional_database;
  adamantine::Geometry<dim> geometry(communicator, geometry_database,
                                     units_optional_database);
  auto const &triangulation = geometry.get_triangulation();
  for (auto cell : triangulation.cell_iterators())
  {
    cell->set_material_id(0);
    cell->set_user_index(
        static_cast<int>(adamantine::SolidLiquidPowder::State::solid));
  }
  // Create the MaterialProperty
  boost::property_tree::ptree material_database;
  material_database.put("property_format", "polynomial");
  material_database.put("n_materials", 1);
  double const lame_first = 2.;
  double const lame_second = 3.;
  material_database.put("material_0.solid.density", 1.);
  material_database.put("material_0.solid.lame_first_parameter", lame_first);
  material_database.put("material_0.solid.lame_second_parameter", lame_second);
  adamantine::MaterialProperty<dim, 1, 4, adamantine::SolidLiquidPowder,
                               dealii::MemorySpace::Host>
      material_properties(communicator, triangulation, material_database);
  // Create the DoFHandler
  dealii::hp::FECollection<dim> fe_collection;
  fe_collection.push_back(dealii::FESystem<dim>(dealii::FE_Q<dim>(2) ^ dim));
  fe_collection.push_back(
      dealii::FESystem<dim>(dealii::FE_Nothing<dim>() ^ dim));
  dealii::DoFHandler<dim> dof_handler(geometry.get_triangulation());
  dof_handler.distribute_dofs(fe_collection);
  dealii::AffineConstraints<double> affine_constraints;
  dealii::DoFTools::make_hanging_node_constraints(dof_handler,
                                                  affine_constraints);
  dealii::VectorTools::interpolate_boundary_values(
      dof_handler, 0, dealii::Functions::ZeroFunction<dim>(dim),
      affine_constraints);
  affine_constraints.close();
  dealii::hp::QCollection<dim> q_collection;
  q_collection.push_back(dealii::QGauss<dim>(3));
  q_collection.push_back(dealii::QGauss<dim>(1));

  adamantine::MechanicalOperator<dim, 1, 4, adamantine::SolidLiquidPowder,
                                 dealii::MemorySpace::Host>
      mechanical_operator(communicator, material_properties);
  std::vector<std::shared_ptr<adamantine::BodyForce<dim>>> body_forces;
  auto gravity_force = std::make_shared<adamantine::GravityForce<
      dim, 1, 4, adamantine::SolidLiquidPowder, dealii::MemorySpace::Host>>(
      material_properties);
  body_forces.push_back(gravity_force);
  mechanical_operator.reinit(dof_handler, affine_constraints, q_collection,
                             body_forces);

  // deal.II reference implementation
  dealii::SparsityPattern sparsity_pattern;
  dealii::SparseMatrix<double> system_matrix;
  {
    dealii::DoFHandler<dim> ref_dof_handler(geometry.get_triangulation());
    dealii::FESystem<dim> fe(dealii::FE_Q<dim>(2), dim);
    ref_dof_handler.distribute_dofs(fe);

    dealii::Vector<double> system_rhs(ref_dof_handler.n_dofs());

    dealii::AffineConstraints<double> constraints;
    dealii::DoFTools::make_hanging_node_constraints(ref_dof_handler,
                                                    constraints);
    dealii::VectorTools::interpolate_boundary_values(
        ref_dof_handler, 0, dealii::Functions::ZeroFunction<dim>(dim),
        constraints);
    constraints.close();

    dealii::DynamicSparsityPattern dsp(ref_dof_handler.n_dofs(),
                                       ref_dof_handler.n_dofs());
    dealii::DoFTools::make_sparsity_pattern(ref_dof_handler, dsp, constraints,
                                            /*keep_constrained_dofs = */ false);
    sparsity_pattern.copy_from(dsp);

    system_matrix.reinit(sparsity_pattern);

    dealii::QGauss<dim> quadrature_formula(3);
    dealii::FEValues<dim> fe_values(
        fe, quadrature_formula,
        dealii::update_values | dealii::update_gradients |
            dealii::update_quadrature_points | dealii::update_JxW_values);

    unsigned int const dofs_per_cell = fe.n_dofs_per_cell();
    unsigned int const n_q_points = quadrature_formula.size();

    dealii::FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    dealii::Vector<double> cell_rhs(dofs_per_cell);

    std::vector<dealii::types::global_dof_index> local_dof_indices(
        dofs_per_cell);

    std::vector<double> lambda_values(n_q_points);
    std::vector<double> mu_values(n_q_points);

    dealii::Functions::ConstantFunction<dim> lambda(lame_first),
        mu(lame_second);

    std::vector<dealii::Tensor<1, dim>> rhs_values(n_q_points);

    for (const auto &cell : ref_dof_handler.active_cell_iterators())
    {
      cell_matrix = 0;
      cell_rhs = 0;

      fe_values.reinit(cell);

      lambda.value_list(fe_values.get_quadrature_points(), lambda_values);
      mu.value_list(fe_values.get_quadrature_points(), mu_values);
      right_hand_side(fe_values.get_quadrature_points(), rhs_values);

      for (const unsigned int i : fe_values.dof_indices())
      {
        const unsigned int component_i = fe.system_to_component_index(i).first;

        for (const unsigned int j : fe_values.dof_indices())
        {
          const unsigned int component_j =
              fe.system_to_component_index(j).first;

          for (const unsigned int q_point :
               fe_values.quadrature_point_indices())
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

      for (const unsigned int i : fe_values.dof_indices())
      {
        const unsigned int component_i = fe.system_to_component_index(i).first;

        for (const unsigned int q_point : fe_values.quadrature_point_indices())
          cell_rhs(i) += fe_values.shape_value(i, q_point) *
                         rhs_values[q_point][component_i] *
                         fe_values.JxW(q_point);
      }

      cell->get_dof_indices(local_dof_indices);
      constraints.distribute_local_to_global(cell_matrix, local_dof_indices,
                                             system_matrix);
    }
  }
  // Compare the results
  unsigned int const n_dofs = system_matrix.m();
  dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host> src(
      n_dofs);
  dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host> dst_1(
      n_dofs);
  dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host> dst_2(
      n_dofs);

  dealii::IndexSet index_set = dealii::complete_index_set(n_dofs);
#if DEAL_II_VERSION_GTE(9, 7, 0) && defined(DEAL_II_TRILINOS_WITH_TPETRA)
  dealii::LinearAlgebra::TpetraWrappers::Vector<double,
                                                dealii::MemorySpace::Default>
      src_device(index_set, MPI_COMM_SELF);
  dealii::LinearAlgebra::TpetraWrappers::Vector<double,
                                                dealii::MemorySpace::Default>
      dst_device(index_set, MPI_COMM_SELF);
#else
  dealii::TrilinosWrappers::MPI::Vector src_device(index_set, MPI_COMM_SELF);
  dealii::TrilinosWrappers::MPI::Vector dst_device(index_set, MPI_COMM_SELF);
#endif
  dealii::LinearAlgebra::ReadWriteVector<double> rw_vector(index_set);

  for (unsigned int i = 0; i < n_dofs; ++i)
  {
    src = 0.;
    src[i] = 1;

    rw_vector.import_elements(src, dealii::VectorOperation::insert);
    src_device.import_elements(rw_vector, dealii::VectorOperation::insert);
    mechanical_operator.system_matrix().vmult(dst_device, src_device);
    rw_vector.import_elements(dst_device, dealii::VectorOperation::insert);
    dst_1.import_elements(rw_vector, dealii::VectorOperation::insert);

    system_matrix.vmult(dst_2, src);
    for (unsigned int j = 0; j < n_dofs; ++j)
      BOOST_TEST(dst_1[j] - dst_2[j] == 0.);
  }

  // Linear displacement: u = (x, 0, 0) gives
  // a(u,u) = (lambda + 2 mu) |Omega|.
  dealii::VectorTools::interpolate(dof_handler, AffineDisplacement<dim>(0),
                                   src);
  affine_constraints.distribute(src);
  rw_vector.import_elements(src, dealii::VectorOperation::insert);
  src_device.import_elements(rw_vector, dealii::VectorOperation::insert);
  mechanical_operator.system_matrix().vmult(dst_device, src_device);
  rw_vector.import_elements(dst_device, dealii::VectorOperation::insert);
  dst_1.import_elements(rw_vector, dealii::VectorOperation::insert);
  double constexpr volume = 6. * 6. * 6.;
  double const elastic_energy = src * dst_1;
  BOOST_TEST(elastic_energy == (lame_first + 2. * lame_second) * volume);

  // The virtual field v = (0, 0, x) gives the exact gravity work
  // integral_Omega rho g . v = -rho g |Omega| L / 2.
  dealii::VectorTools::interpolate(dof_handler, AffineDisplacement<dim>(2),
                                   src);
  affine_constraints.distribute(src);
  double const gravity_work = src * mechanical_operator.rhs();
  BOOST_TEST(gravity_work == -9.80665 * volume * 6. / 2.);
}

BOOST_AUTO_TEST_CASE(thermoelastic, *utf::tolerance(1e-12))
{
  MPI_Comm communicator = MPI_COMM_WORLD;
  int constexpr dim = 3;

  // Create the Geometry
  boost::property_tree::ptree geometry_database;
  geometry_database.put("import_mesh", false);
  geometry_database.put("length", 6);
  geometry_database.put("length_divisions", 3);
  geometry_database.put("height", 6);
  geometry_database.put("height_divisions", 3);
  geometry_database.put("width", 6);
  geometry_database.put("width_divisions", 3);
  boost::optional<boost::property_tree::ptree const &> units_optional_database;
  adamantine::Geometry<dim> geometry(communicator, geometry_database,
                                     units_optional_database);
  auto const &triangulation = geometry.get_triangulation();
  for (auto cell : triangulation.cell_iterators())
  {
    cell->set_material_id(0);
    cell->set_user_index(
        static_cast<int>(adamantine::SolidLiquidPowder::State::solid));
  }
  // Create the MaterialProperty
  boost::property_tree::ptree material_database;
  material_database.put("property_format", "polynomial");
  material_database.put("n_materials", 1);
  double const lame_first = 2.;
  double const lame_second = 3.;
  double const thermal_expansion = 0.25;
  material_database.put("material_0.solid.lame_first_parameter", lame_first);
  material_database.put("material_0.solid.lame_second_parameter", lame_second);
  material_database.put("material_0.solid.thermal_expansion_coef",
                        thermal_expansion);
  adamantine::MaterialProperty<dim, -1, 4, adamantine::SolidLiquidPowder,
                               dealii::MemorySpace::Host>
      material_properties(communicator, triangulation, material_database);
  // Create the mechanical DoFHandler
  dealii::hp::FECollection<dim> mechanical_fe_collection;
  mechanical_fe_collection.push_back(
      dealii::FESystem<dim>(dealii::FE_Q<dim>(2) ^ dim));
  mechanical_fe_collection.push_back(
      dealii::FESystem<dim>(dealii::FE_Nothing<dim>() ^ dim));
  dealii::DoFHandler<dim> mechanical_dof_handler(geometry.get_triangulation());
  mechanical_dof_handler.distribute_dofs(mechanical_fe_collection);
  dealii::AffineConstraints<double> mechanical_affine_constraints;
  dealii::DoFTools::make_hanging_node_constraints(
      mechanical_dof_handler, mechanical_affine_constraints);
  dealii::VectorTools::interpolate_boundary_values(
      mechanical_dof_handler, 0, dealii::Functions::ZeroFunction<dim>(dim),
      mechanical_affine_constraints);
  mechanical_affine_constraints.close();
  dealii::hp::QCollection<dim> mechanical_q_collection;
  mechanical_q_collection.push_back(dealii::QGauss<dim>(3));
  mechanical_q_collection.push_back(dealii::QGauss<dim>(1));
  // Create the MechanicalOperator
  adamantine::MechanicalOperator<dim, -1, 4, adamantine::SolidLiquidPowder,
                                 dealii::MemorySpace::Host>
      mechanical_operator(communicator, material_properties);
  std::vector<std::shared_ptr<adamantine::BodyForce<dim>>> body_forces;
  auto gravity_force = std::make_shared<adamantine::GravityForce<
      dim, -1, 4, adamantine::SolidLiquidPowder, dealii::MemorySpace::Host>>(
      material_properties);
  body_forces.push_back(gravity_force);
  mechanical_operator.reinit(mechanical_dof_handler,
                             mechanical_affine_constraints,
                             mechanical_q_collection, body_forces);

  dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host>
      virtual_displacement(mechanical_dof_handler.locally_owned_dofs(),
                           communicator);
  dealii::VectorTools::interpolate(
      mechanical_dof_handler, AffineDisplacement<dim>(0), virtual_displacement);
  mechanical_affine_constraints.distribute(virtual_displacement);

  // The direct thermal-strain formulation produces a compressive stress for
  // zero displacement. Its internal-force contribution is the negative of the
  // equivalent thermal RHS.
  double constexpr volume = 6. * 6. * 6.;
  double const beta = (3. * lame_first + 2. * lame_second) * thermal_expansion;
  std::vector<std::vector<dealii::SymmetricTensor<2, dim>>> thermal_stress(
      triangulation.n_active_cells(),
      std::vector<dealii::SymmetricTensor<2, dim>>(
          mechanical_q_collection.max_n_quadrature_points(),
          -beta * dealii::unit_symmetric_tensor<dim>()));
  auto const residual = mechanical_operator.assemble_residual(thermal_stress);
  double const virtual_work = virtual_displacement * residual;
  BOOST_TEST(virtual_work == -beta * volume);
}

BOOST_AUTO_TEST_CASE(jacobian, *utf::tolerance(1e-10))
{
  MPI_Comm communicator = MPI_COMM_WORLD;
  int constexpr dim = 3;

  boost::property_tree::ptree geometry_database;
  geometry_database.put("import_mesh", false);
  geometry_database.put("length", 1.);
  geometry_database.put("length_divisions", 1);
  geometry_database.put("height", 1.);
  geometry_database.put("height_divisions", 1);
  geometry_database.put("width", 1.);
  geometry_database.put("width_divisions", 1);
  boost::optional<boost::property_tree::ptree const &> units_optional_database;
  adamantine::Geometry<dim> geometry(communicator, geometry_database,
                                     units_optional_database);
  auto const &triangulation = geometry.get_triangulation();
  for (auto cell : triangulation.cell_iterators())
  {
    cell->set_material_id(0);
    cell->set_user_index(
        static_cast<int>(adamantine::SolidLiquidPowder::State::solid));
  }

  double constexpr lambda = 2.;
  double constexpr mu = 3.;
  boost::property_tree::ptree material_database;
  material_database.put("property_format", "polynomial");
  material_database.put("n_materials", 1);
  material_database.put("material_0.solid.lame_first_parameter", lambda);
  material_database.put("material_0.solid.lame_second_parameter", mu);
  adamantine::MaterialProperty<dim, 1, 4, adamantine::SolidLiquidPowder,
                               dealii::MemorySpace::Host>
      material_properties(communicator, triangulation, material_database);

  dealii::hp::FECollection<dim> fe_collection;
  fe_collection.push_back(dealii::FESystem<dim>(dealii::FE_Q<dim>(1) ^ dim));
  fe_collection.push_back(
      dealii::FESystem<dim>(dealii::FE_Nothing<dim>() ^ dim));
  dealii::DoFHandler<dim> dof_handler(triangulation);
  dof_handler.distribute_dofs(fe_collection);

  dealii::AffineConstraints<double> affine_constraints;
  dealii::DoFTools::make_hanging_node_constraints(dof_handler,
                                                  affine_constraints);
  affine_constraints.close();

  dealii::hp::QCollection<dim> q_collection;
  q_collection.push_back(dealii::QGauss<dim>(2));
  q_collection.push_back(dealii::QGauss<dim>(1));

  adamantine::MechanicalOperator<dim, 1, 4, adamantine::SolidLiquidPowder,
                                 dealii::MemorySpace::Host>
      mechanical_operator(communicator, material_properties);
  mechanical_operator.reinit(dof_handler, affine_constraints, q_collection, {});

  auto const constitutive_tangent =
      lambda * dealii::outer_product(dealii::unit_symmetric_tensor<dim>(),
                                     dealii::unit_symmetric_tensor<dim>()) +
      2. * mu * dealii::identity_tensor<dim>();
  std::vector<std::vector<dealii::SymmetricTensor<4, dim>>> tangent(
      triangulation.n_active_cells(),
      std::vector<dealii::SymmetricTensor<4, dim>>(
          q_collection.max_n_quadrature_points(), constitutive_tangent));
  mechanical_operator.assemble_jacobian(tangent);

  using VectorType =
      dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host>;
  VectorType displacement(dof_handler.locally_owned_dofs(), communicator);
  VectorType direction(dof_handler.locally_owned_dofs(), communicator);
  dealii::VectorTools::interpolate(dof_handler, AffineDisplacement<dim>(0),
                                   displacement);
  dealii::VectorTools::interpolate(dof_handler, AffineDisplacement<dim>(1),
                                   direction);
  affine_constraints.distribute(displacement);
  affine_constraints.distribute(direction);

  auto compute_stress = [&](VectorType const &input)
  {
    std::vector<std::vector<dealii::SymmetricTensor<2, dim>>> stress(
        triangulation.n_active_cells(),
        std::vector<dealii::SymmetricTensor<2, dim>>(
            q_collection.max_n_quadrature_points()));
    auto input_with_ghosts = input;
    input_with_ghosts.update_ghost_values();
    dealii::hp::FEValues<dim> hp_fe_values(fe_collection, q_collection,
                                           dealii::update_gradients);
    dealii::FEValuesExtractors::Vector const displacements(0);
    std::vector<dealii::SymmetricTensor<2, dim>> strain(
        q_collection.max_n_quadrature_points());
    unsigned int cell_id = 0;
    for (auto const &cell : dof_handler.active_cell_iterators())
    {
      if (cell->is_locally_owned() && cell->active_fe_index() == 0)
      {
        hp_fe_values.reinit(cell);
        auto const &fe_values = hp_fe_values.get_present_fe_values();
        fe_values[displacements].get_function_symmetric_gradients(
            input_with_ghosts, strain);
        for (auto const q : fe_values.quadrature_point_indices())
          stress[cell_id][q] = constitutive_tangent * strain[q];
      }
      ++cell_id;
    }
    return stress;
  };

  auto const residual =
      mechanical_operator.assemble_residual(compute_stress(displacement));
  double constexpr h = 1.e-5;
  auto perturbed_displacement = displacement;
  perturbed_displacement.add(h, direction);
  auto finite_difference = mechanical_operator.assemble_residual(
      compute_stress(perturbed_displacement));
  finite_difference -= residual;
  finite_difference /= h;

  dealii::IndexSet const locally_owned_dofs = dof_handler.locally_owned_dofs();
#if DEAL_II_VERSION_GTE(9, 7, 0) && defined(DEAL_II_TRILINOS_WITH_TPETRA)
  using TrilinosVectorType = dealii::LinearAlgebra::TpetraWrappers::Vector<
      double, dealii::MemorySpace::Default>;
#else
  using TrilinosVectorType = dealii::TrilinosWrappers::MPI::Vector;
#endif
  auto matrix_action = [&](auto const &matrix)
  {
    TrilinosVectorType direction_device(locally_owned_dofs, communicator);
    TrilinosVectorType matrix_action_device(locally_owned_dofs, communicator);
    dealii::LinearAlgebra::ReadWriteVector<double> rw_vector(
        locally_owned_dofs);
    rw_vector.import_elements(direction, dealii::VectorOperation::insert);
    direction_device.import_elements(rw_vector,
                                     dealii::VectorOperation::insert);
    matrix.vmult(matrix_action_device, direction_device);
    rw_vector.import_elements(matrix_action_device,
                              dealii::VectorOperation::insert);
    VectorType action(locally_owned_dofs, communicator);
    action.import_elements(rw_vector, dealii::VectorOperation::insert);
    return action;
  };
  auto const jacobian_action =
      matrix_action(mechanical_operator.system_matrix());

  finite_difference -= jacobian_action;
  BOOST_CHECK_SMALL(finite_difference.l2_norm(), 1.e-9);

  // A nonlinear solve overwrites the shared matrix with its current
  // Jacobian. Ensure that the elastic matrix can be restored before the next
  // nonlinear solve.
  auto scaled_tangent = tangent;
  for (auto &cell_tangent : scaled_tangent)
    for (auto &q_point_tangent : cell_tangent)
      q_point_tangent *= 0.5;
  mechanical_operator.assemble_jacobian(scaled_tangent);
  auto scaled_jacobian_action =
      matrix_action(mechanical_operator.system_matrix());
  scaled_jacobian_action -= jacobian_action;
  BOOST_CHECK(scaled_jacobian_action.l2_norm() > 1.e-10);

  mechanical_operator.ensure_elastic_matrix();
  auto restored_elastic_action =
      matrix_action(mechanical_operator.system_matrix());
  restored_elastic_action -= jacobian_action;
  BOOST_CHECK_SMALL(restored_elastic_action.l2_norm(), 1.e-12);
}
