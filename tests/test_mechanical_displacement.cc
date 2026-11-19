/* SPDX-FileCopyrightText: Copyright (c) 2026, the adamantine authors.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#define BOOST_TEST_MODULE MechanicalDisplacement

#include <Geometry.hh>
#include <MaterialStates.hh>
#include <MechanicalPhysics.hh>
#include <types.hh>

#include <deal.II/base/function_parser.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/numerics/vector_tools.h>

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "main.cc"

/** Transfer displacement/history through a local refinement or coarsening. */
template <typename Mechanics, typename Properties, typename Triangulation>
void adapt_displacement_patch(Mechanics &mechanics, Properties &properties,
                              Triangulation &triangulation, bool refine)
{
  auto const cells_before = triangulation.n_global_active_cells();
  for (auto const &cell : triangulation.active_cell_iterators())
    if (cell->is_locally_owned())
    {
      if (refine)
      {
        cell->set_refine_flag();
        break;
      }
      if (cell->level() > 0)
        cell->set_coarsen_flag();
    }
  mechanics.prepare_transfer_mpi();
  triangulation.execute_coarsening_and_refinement();
  if (refine)
    BOOST_CHECK(triangulation.n_global_active_cells() > cells_before);
  else
    BOOST_CHECK(triangulation.n_global_active_cells() < cells_before);
  properties.reinit_dofs();
  properties.set_initial_state();
  mechanics.complete_transfer_mpi();
  mechanics.setup_dofs();
}

template <int dim>
void displacement_patch(bool normal_component_only = false,
                        bool adapt_mesh = false)
{
  std::vector<std::string> const coordinate_names =
      dim == 2 ? std::vector<std::string>{"x", "z"}
               : std::vector<std::string>{"x", "y", "z"};
  auto const variables = dim == 2 ? "x,z,t" : "x,y,z,t";
  MPI_Comm const communicator = MPI_COMM_WORLD;
  boost::property_tree::ptree geometry_database;
  geometry_database.put("import_mesh", false);
  geometry_database.put("length", 1.);
  geometry_database.put("height", 1.);
  geometry_database.put("width", 1.);
  geometry_database.put("length_divisions", 3);
  geometry_database.put("height_divisions", 3);
  geometry_database.put("width_divisions", 3);
  boost::optional<boost::property_tree::ptree const &> units;
  adamantine::Geometry<dim> geometry(communicator, geometry_database, units);
  for (auto cell : geometry.get_triangulation().active_cell_iterators())
  {
    cell->set_material_id(0);
    cell->set_user_index(
        static_cast<int>(adamantine::SolidLiquidPowder::State::solid));
  }
  boost::property_tree::ptree materials;
  materials.put("property_format", "polynomial");
  materials.put("n_materials", 1);
  materials.put("material_0.solid.lame_first_parameter", 2.);
  materials.put("material_0.solid.lame_second_parameter", 3.);
  adamantine::MaterialProperty<dim, 1, 4, adamantine::SolidLiquidPowder,
                               dealii::MemorySpace::Host>
      properties(communicator, geometry.get_triangulation(), materials);
  boost::property_tree::ptree boundaries;
  for (auto id : geometry.get_triangulation().get_boundary_ids())
  {
    auto const prefix = "boundary_" + std::to_string(id);
    boundaries.put(prefix + ".type", "displacement");
    for (unsigned int d = 0; d < dim; ++d)
      if (!normal_component_only || d == id / 2)
        boundaries.put(prefix + ".displacement." + coordinate_names[d],
                       "0.001*t*" + coordinate_names[d]);
  }
  adamantine::Boundary boundary(
      boundaries, geometry.get_triangulation().get_boundary_ids());
  adamantine::MechanicalPhysics<dim, 1, 4, adamantine::SolidLiquidPowder,
                                dealii::MemorySpace::Host>
      mechanics(communicator, 1, geometry, boundary, properties, {});
  mechanics.setup_dofs();
  // Affine dilation has constant strain 0.001*t*I and stress
  // (dim*lambda+2*mu)*0.001*t*I, including at unconstrained interior DoFs.
  bool coarsened = false;
  for (double time : {0., 1., 2., 2., 0.5})
  {
    auto solution = mechanics.solve(time);
    auto expected = solution;
    dealii::FunctionParser<dim> function(dim);
    std::vector<std::string> expressions;
    for (unsigned int d = 0; d < dim; ++d)
      expressions.push_back("0.001*t*" + coordinate_names[d]);
    function.initialize(variables, expressions, {}, true);
    function.set_time(time);
    dealii::VectorTools::interpolate(mechanics.get_dof_handler(), function,
                                     expected);
    solution -= expected;
    BOOST_CHECK_SMALL(solution.l2_norm(), 1.e-10);
    auto const stress =
        (dim * 2. + 6.) * 0.001 * time * dealii::unit_symmetric_tensor<dim>();
    auto const stresses = mechanics.get_stress_tensor();
    for (auto const &cell : mechanics.get_dof_handler().active_cell_iterators())
      if (cell->is_locally_owned())
        for (auto const &value : stresses[cell->active_cell_index()])
          BOOST_CHECK_SMALL((value - stress).norm(), 1.e-10);
    if (adapt_mesh && (time == 1. || (time == 2. && !coarsened)))
    {
      bool const refine = time == 1.;
      // The plastic test repeats t=1 to check idempotence; refine only once.
      if (refine && geometry.get_triangulation().n_global_levels() > 1)
        continue;
      adapt_displacement_patch(mechanics, properties,
                               geometry.get_triangulation(), refine);
      coarsened = !refine;
    }
  }
}

BOOST_AUTO_TEST_CASE(elastic_displacement_patch_2d) { displacement_patch<2>(); }
BOOST_AUTO_TEST_CASE(elastic_displacement_components)
{
  displacement_patch<3>(true);
}

BOOST_AUTO_TEST_CASE(elastic_displacement_patch_3d) { displacement_patch<3>(); }

/** Verify prescribed small-strain shear against the combined-hardening law. */
template <int dim>
void plastic_displacement_patch(bool adapt_mesh = false)
{
  MPI_Comm const communicator = MPI_COMM_WORLD;
  boost::property_tree::ptree geometry_database;
  geometry_database.put("import_mesh", false);
  geometry_database.put("length", 1.);
  geometry_database.put("height", 1.);
  geometry_database.put("width", 1.);
  geometry_database.put("length_divisions", 6);
  geometry_database.put("height_divisions", 6);
  geometry_database.put("width_divisions", 6);
  boost::optional<boost::property_tree::ptree const &> units;
  adamantine::Geometry<dim> geometry(communicator, geometry_database, units);
  for (auto cell : geometry.get_triangulation().active_cell_iterators())
  {
    cell->set_material_id(0);
    cell->set_user_index(
        static_cast<int>(adamantine::SolidLiquidPowder::State::solid));
  }
  boost::property_tree::ptree materials;
  materials.put("property_format", "polynomial");
  materials.put("n_materials", 1);
  materials.put("material_0.solid.lame_first_parameter", 2.);
  materials.put("material_0.solid.lame_second_parameter", 3.);
  materials.put("material_0.solid.elastic_limit", 0.01);
  materials.put("material_0.solid.plastic_modulus", 1.5);
  materials.put("material_0.solid.isotropic_hardening", 0.25);
  adamantine::MaterialProperty<dim, 1, 4, adamantine::SolidLiquidPowder,
                               dealii::MemorySpace::Host>
      properties(communicator, geometry.get_triangulation(), materials);

  boost::property_tree::ptree boundaries;
  for (auto id : geometry.get_triangulation().get_boundary_ids())
  {
    auto const prefix = "boundary_" + std::to_string(id);
    boundaries.put(prefix + ".type", "displacement");
    boundaries.put(prefix + ".displacement.x", "0.002*t*z");
    boundaries.put(prefix + ".displacement.z", "0.002*t*x");
    if constexpr (dim == 3)
      boundaries.put(prefix + ".displacement.y", "0");
  }
  adamantine::Boundary boundary(
      boundaries, geometry.get_triangulation().get_boundary_ids());
  adamantine::MechanicalPhysics<dim, 1, 4, adamantine::SolidLiquidPowder,
                                dealii::MemorySpace::Host>
      mechanics(communicator, 1, geometry, boundary, properties, {});
  mechanics.setup_dofs();
  dealii::SymmetricTensor<2, dim> direction;
  direction[adamantine::axis<dim>::x][adamantine::axis<dim>::z] =
      1. / std::sqrt(2.);
  dealii::FunctionParser<dim> exact(dim);
  std::vector<std::string> expressions(dim, "0");
  expressions[adamantine::axis<dim>::x] = "0.002*t*z";
  expressions[adamantine::axis<dim>::z] = "0.002*t*x";
  exact.initialize(dim == 2 ? "x,z,t" : "x,y,z,t", expressions, {}, true);

  // eps_xz = 0.002*t. Before yielding the stress norm is
  // 2*mu*sqrt(2)*eps_xz. During monotonic plastic loading its excess over
  // the initial yield threshold is divided between elasticity and hardening:
  // p = (2*mu*sqrt(2)*eps_xz - yield)/(2*mu + H).
  // The final small unload is elastic, so p stays at its peak-load value.
  double peak_time = 0.;
  bool coarsened = false;
  for (double time : {0., 0.25, 1., 1., 2., 1.9, 1.9})
  {
    peak_time = std::max(peak_time, time);
    double const plastic_amplitude =
        std::max(0., (6. * std::sqrt(2.) * 0.002 * peak_time - 0.01) / 7.5);
    auto const expected_plastic_strain = plastic_amplitude * direction;
    auto const expected_back_stress =
        0.75 * 1.5 * plastic_amplitude * direction;
    double const expected_yield = 0.01 + 0.25 * 1.5 * plastic_amplitude;
    auto const expected_stress =
        (6. * std::sqrt(2.) * 0.002 * time - 6. * plastic_amplitude) *
        direction;

    auto displacement = mechanics.solve(time);
    auto expected = displacement;
    exact.set_time(time);
    dealii::VectorTools::interpolate(mechanics.get_dof_handler(), exact,
                                     expected);
    displacement -= expected;
    BOOST_CHECK_SMALL(displacement.l2_norm(), 1.e-9);
    auto const stresses = mechanics.get_stress_tensor();
    auto const &states = mechanics.get_constitutive_state();
    for (auto const &cell : mechanics.get_dof_handler().active_cell_iterators())
      if (cell->is_locally_owned())
      {
        auto const index = cell->active_cell_index();
        for (auto const &stress : stresses[index])
          BOOST_CHECK_SMALL((stress - expected_stress).norm(), 1.e-9);
        for (auto const &state : states[index])
        {
          BOOST_CHECK_SMALL(
              (state.plastic_strain - expected_plastic_strain).norm(), 1.e-9);
          BOOST_CHECK_SMALL((state.back_stress - expected_back_stress).norm(),
                            1.e-9);
          BOOST_CHECK_SMALL(state.plastic_internal_variable - expected_yield,
                            1.e-9);
        }
      }
    // The committed state must balance the free-DoF residual. Repeated solves
    // at unchanged time must not apply the boundary motion or plastic flow
    // again.
    auto zero_increment = expected;
    zero_increment = 0.;
    zero_increment.update_ghost_values();
    BOOST_CHECK_SMALL(mechanics.evaluate_residual(zero_increment).l2_norm(),
                      1.e-9);
    if (adapt_mesh && (time == 1. || (time == 2. && !coarsened)))
    {
      bool const refine = time == 1.;
      // The plastic test repeats t=1 to check idempotence; refine only once.
      if (refine && geometry.get_triangulation().n_global_levels() > 1)
        continue;
      adapt_displacement_patch(mechanics, properties,
                               geometry.get_triangulation(), refine);
      coarsened = !refine;
    }
  }
}

BOOST_AUTO_TEST_CASE(plastic_displacement_patch_2d)
{
  plastic_displacement_patch<2>();
}
BOOST_AUTO_TEST_CASE(plastic_displacement_patch_3d)
{
  plastic_displacement_patch<3>();
}

BOOST_AUTO_TEST_CASE(elastic_displacement_adaptation)
{
  displacement_patch<3>(false, true);
}
BOOST_AUTO_TEST_CASE(plastic_displacement_adaptation)
{
  plastic_displacement_patch<3>(true);
}
