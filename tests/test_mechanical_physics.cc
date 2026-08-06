/* SPDX-FileCopyrightText: Copyright (c) 2022 - 2026, the adamantine authors.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include "MaterialStates.hh"
#define BOOST_TEST_MODULE MechanicalPhysics

#include <Geometry.hh>
#include <MechanicalPhysics.hh>
#include <PostProcessor.hh>
#include <ThermalPhysics.hh>

#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_nothing.h>
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

#include <cmath>
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
    values[point_n][2] = -9.80665;
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
      _dof_handler, 4, dealii::Functions::ZeroFunction<3>(3), _constraints);
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
  boost::optional<boost::property_tree::ptree const &> units_optional_database;
  // Build Geometry
  adamantine::Geometry<3> geometry(communicator, geometry_database,
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
  material_database.put("material_0.solid.density", 1.);
  material_database.put("material_0.solid.lame_first_parameter", 2.);
  material_database.put("material_0.solid.lame_second_parameter", 3.);
  adamantine::MaterialProperty<3, 1, 4, adamantine::SolidLiquidPowder,
                               dealii::MemorySpace::Host>
      material_properties(communicator, triangulation, material_database);
  // Create the Boundary
  boost::property_tree::ptree boundary_database;
  boundary_database.put("boundary_4.type", "clamped");
  adamantine::Boundary boundary(
      boundary_database, geometry.get_triangulation().get_boundary_ids());
  // Build MechanicalPhysics
  unsigned int const fe_degree = 1;
  std::vector<double> empty_vector;
  adamantine::MechanicalPhysics<3, 1, 4, adamantine::SolidLiquidPowder,
                                dealii::MemorySpace::Host>
      mechanical_physics(communicator, fe_degree, geometry, boundary,
                         material_properties, empty_vector);
  std::vector<std::shared_ptr<adamantine::BodyForce<3>>> body_forces;
  auto gravity_force = std::make_shared<adamantine::GravityForce<
      3, 1, 4, adamantine::SolidLiquidPowder, dealii::MemorySpace::Host>>(
      material_properties);
  body_forces.push_back(gravity_force);
  mechanical_physics.setup_dofs(body_forces);
  auto solution = mechanical_physics.solve();

  // Reference computation
  ElastoStaticity elasto_staticity;
  elasto_staticity.setup_system();
  elasto_staticity.assemble_system();
  auto reference_solution = elasto_staticity.solve();

  double const tolerance = 2e-9;
  BOOST_TEST(solution.size() == reference_solution.size());

  // Use BOOST_CHECK_SMALL so that minor deviations from zero related to finite
  // solver tolerances don't trigger failures. The largest solution values are
  // O(1), so the tolerance is strict enough to catch meaningful differences.
  for (unsigned int i = 0; i < reference_solution.size(); ++i)
    BOOST_CHECK_SMALL(solution[i] - reference_solution[i], tolerance);
}

BOOST_AUTO_TEST_CASE(fe_nothing)
{
  MPI_Comm communicator = MPI_COMM_WORLD;

  // Geometry database
  boost::property_tree::ptree geometry_database;
  geometry_database.put("import_mesh", false);
  geometry_database.put("length", 12);
  geometry_database.put("length_divisions", 6);
  geometry_database.put("height", 8);
  geometry_database.put("height_divisions", 4);
  geometry_database.put("width", 6);
  geometry_database.put("width_divisions", 3);
  boost::optional<boost::property_tree::ptree const &> units_optional_database;
  // Build Geometry
  adamantine::Geometry<3> geometry(communicator, geometry_database,
                                   units_optional_database);
  auto const &triangulation = geometry.get_triangulation();
  for (auto cell : triangulation.cell_iterators())
  {
    cell->set_material_id(0);
    if (cell->center()[2] < 6.)
    {
      cell->set_user_index(
          static_cast<int>(adamantine::SolidLiquidPowder::State::solid));
    }
    else
    {
      cell->set_user_index(
          static_cast<int>(adamantine::SolidLiquidPowder::State::powder));
    }
  }
  // Create the MaterialProperty
  boost::property_tree::ptree material_database;
  material_database.put("property_format", "polynomial");
  material_database.put("n_materials", 1);
  material_database.put("material_0.solid.density", 1.);
  material_database.put("material_0.solid.lame_first_parameter", 2.);
  material_database.put("material_0.solid.lame_second_parameter", 3.);
  adamantine::MaterialProperty<3, 1, 2, adamantine::SolidLiquidPowder,
                               dealii::MemorySpace::Host>
      material_properties(communicator, triangulation, material_database);
  // Create the Boundary
  boost::property_tree::ptree boundary_database;
  boundary_database.put("boundary_4.type", "clamped");
  adamantine::Boundary boundary(
      boundary_database, geometry.get_triangulation().get_boundary_ids());
  // Build MechanicalPhysics
  unsigned int const fe_degree = 1;
  std::vector<double> empty_vector;
  adamantine::MechanicalPhysics<3, 1, 2, adamantine::SolidLiquidPowder,
                                dealii::MemorySpace::Host>
      mechanical_physics(communicator, fe_degree, geometry, boundary,
                         material_properties, empty_vector);
  std::vector<std::shared_ptr<adamantine::BodyForce<3>>> body_forces;
  auto gravity_force = std::make_shared<adamantine::GravityForce<
      3, 1, 2, adamantine::SolidLiquidPowder, dealii::MemorySpace::Host>>(
      material_properties);
  body_forces.push_back(gravity_force);
  mechanical_physics.setup_dofs(body_forces);
  auto solution = mechanical_physics.solve();

  // Reference computation
  ElastoStaticity elasto_staticity;
  elasto_staticity.setup_system();
  elasto_staticity.assemble_system();
  auto reference_solution = elasto_staticity.solve();

  double const tolerance = 2e-9;
  BOOST_TEST(solution.size() == reference_solution.size());

  // Use BOOST_CHECK_SMALL so that minor deviations from zero related to finite
  // solver tolerances don't trigger failures. The largest solution values are
  // O(1), so the tolerance is strict enough to catch meaningful differences.
  for (unsigned int i = 0; i < reference_solution.size(); ++i)
    BOOST_CHECK_SMALL(solution[i] - reference_solution[i], tolerance);
}

template <int dim>
class InitialValueT : public dealii::Function<dim>
{
public:
  double value(const dealii::Point<dim> &p,
               const unsigned int /*component = 0*/) const override
  {
    dealii::Point<dim> center = {2.0e-5, 2.0e-5, 2.0e-5};
    const double a = 1.0e-6;
    const double T0 = 3.0;
    double dist = center.distance(p);
    if (dist < a)
    {
      return T0;
    }
    else
    {
      return 2.0;
    }
  }
};

namespace utf = boost::unit_test;

/*
 * This test uses Eshelby's analytical solution for a spherical inclusion with
 * uniform isotropic eigenstrain. For equal elastic properties inside and
 * outside the sphere, u = A r inside and u = A a^3 r/|r|^3 outside, where
 * A = 3 K alpha DeltaT / (3 K + 4 mu).
 *
 * Reference: J. D. Eshelby, "The Determination of the Elastic Field of an
 * Ellipsoidal Inclusion, and Related Problems", Proc. Royal Soc. A 241
 * (1957), 376-396, DOI: 10.1098/rspa.1957.0133.
 *
 * The finite domain is clamped on one face and kept deliberately coarse, so
 * the two sampled displacements use a loose tolerance. Less than 5% deviation
 * is obtained with five refinement cycles.
 */
template <unsigned int dim>
std::vector<dealii::Vector<double>>
run_eshelby(std::vector<dealii::Point<dim>> pts, unsigned int refinement_cycles)
{
  MPI_Comm communicator = MPI_COMM_WORLD;

  // Create the Geometry
  boost::property_tree::ptree geometry_database;
  geometry_database.put("import_mesh", false);
  geometry_database.put("length", 4.0e-5); // m
  geometry_database.put("length_divisions", 16);
  geometry_database.put("height", 4.0e-5); // m
  geometry_database.put("height_divisions", 16);
  geometry_database.put("width", 4.0e-5); // m
  geometry_database.put("width_divisions", 16);
  boost::optional<boost::property_tree::ptree const &> units_optional_database;
  adamantine::Geometry<dim> geometry(communicator, geometry_database,
                                     units_optional_database);
  auto &triangulation = geometry.get_triangulation();

  const dealii::Point<dim> center = {2.0e-5, 2.0e-5, 2.0e-5};

  for (unsigned int cycle = 0; cycle < refinement_cycles; ++cycle)
  {
    for (auto cell :
         dealii::filter_iterators(triangulation.active_cell_iterators(),
                                  dealii::IteratorFilters::LocallyOwnedCell()))
    {
      cell->set_material_id(0);
      cell->set_user_index(
          static_cast<int>(adamantine::SolidLiquidPowder::State::solid));
      auto dist_from_center = center.distance(cell->center());
      auto rad = 3.0e-6;
      if (cycle == 0)
      {
        rad = 4.0e-6;
      }

      if (dist_from_center < rad)
      {
        cell->set_refine_flag();
      }
    }
    triangulation.prepare_coarsening_and_refinement();
    triangulation.execute_coarsening_and_refinement();
  }

  // Create the Boundary
  boost::property_tree::ptree boundary_database;
  boundary_database.put("type", "adiabatic");
  boundary_database.put("boundary_4.type", "adiabatic,clamped");
  adamantine::Boundary boundary(
      boundary_database, geometry.get_triangulation().get_boundary_ids());

  // Create the MaterialProperty
  boost::property_tree::ptree material_database;
  material_database.put("property_format", "polynomial");
  material_database.put("n_materials", 1);

  double const bulk_modulus = 160.0e9; // Pa
  double const shear_modulus = 79.0e9; // Pa

  double const lame_first = bulk_modulus - 2. / 3. * shear_modulus;
  double const lame_second = shear_modulus;

  material_database.put("material_0.solid.lame_first_parameter", lame_first);
  material_database.put("material_0.solid.lame_second_parameter", lame_second);

  double const alpha = 0.01;
  material_database.put("material_0.solid.thermal_expansion_coef", alpha);
  adamantine::MaterialProperty<dim, -1, 3, adamantine::SolidLiquidPowder,
                               dealii::MemorySpace::Host>
      material_properties(communicator, triangulation, material_database);

  // Build ThermalPhysics
  boost::property_tree::ptree database;
  database.put("time_stepping.method", "forward_euler");
  database.put("time_stepping.max_iteration", 100);
  database.put("time_stepping.tolerance", 1e-6);
  database.put("time_stepping.n_tmp_vectors", 100);
  database.put("sources.beam_0.scan_path_file",
               "scan_path_test_thermal_physics.txt");
  database.put("sources.beam_0.type", "electron_beam");
  database.put("sources.beam_0.scan_path_file_format", "segment");
  database.put("sources.n_beams", 1);
  database.put("sources.beam_0.depth", 1e100);
  database.put("sources.beam_0.diameter", 1e100);
  database.put("sources.beam_0.max_power", 1e300);
  database.put("sources.beam_0.absorption_efficiency", 0.1);
  database.put("sources.beam_0.type", "electron_beam");
  database.put("sources.beam_0.scan_path_file",
               "scan_path_test_thermal_physics.txt");
  database.put("sources.beam_0.scan_path_file_format", "segment");
  adamantine::ThermalPhysics<dim, -1, 3, 1, adamantine::SolidLiquidPowder,
                             dealii::MemorySpace::Host, dealii::QGauss<1>>
      thermal_physics(communicator, database, geometry, boundary,
                      material_properties);
  thermal_physics.setup();

  dealii::LinearAlgebra::distributed::Vector<double> temperature;
  thermal_physics.initialize_dof_vector(100.0, temperature);

  dealii::VectorTools::interpolate(thermal_physics.get_dof_handler(),
                                   InitialValueT<3>(), temperature);

  // Build MechanicalPhysics
  unsigned int const fe_degree = 1;
  std::vector<double> initial_temperature = {2.0};
  adamantine::MechanicalPhysics<3, -1, 3, adamantine::SolidLiquidPowder,
                                dealii::MemorySpace::Host>
      mechanical_physics(communicator, fe_degree, geometry, boundary,
                         material_properties, initial_temperature);

  boost::property_tree::ptree post_processor_database;
  post_processor_database.put("filename_prefix", "mech_phys_test");
  post_processor_database.put("thermal_output", true);
  post_processor_database.put("mechanical_output", true);

  adamantine::PostProcessor<dim> post_processor(
      communicator, post_processor_database, thermal_physics.get_dof_handler(),
      mechanical_physics.get_dof_handler());

  std::vector<bool> has_melted(triangulation.n_active_cells(), false);

  mechanical_physics.setup_dofs(thermal_physics.get_dof_handler(), temperature,
                                has_melted, true);

  auto solution = mechanical_physics.solve();

  // Output (for debugging)
  /*
  mechanical_physics.get_affine_constraints().distribute(solution);
  post_processor.write_output(0, 0, 0, temperature, solution,
                              material_properties.get_state(),
                              material_properties.get_dofs_map(),
                              material_properties.get_dof_handler());
  */

  std::vector<dealii::Vector<double>> pt_values;

  for (auto pt : pts)
  {
    dealii::Vector<double> displacement_value(3);
    dealii::VectorTools::point_value(mechanical_physics.get_dof_handler(),
                                     solution, pt, displacement_value);
    pt_values.push_back(displacement_value);
  }

  return pt_values;
};

BOOST_AUTO_TEST_CASE(thermoelastic_eshelby, *utf::tolerance(0.16))
{

  int constexpr dim = 3;

  const dealii::Point<dim> pt1 = {2.08e-5, 2.0e-5, 2.0e-5};
  const dealii::Point<dim> pt2 = {2.3e-5, 2.2e-5, 1.9e-5};
  std::vector<dealii::Point<dim>> pts = {pt1, pt2};

  unsigned int refinement_cyles = 3;

  auto pt_values = run_eshelby<dim>(pts, refinement_cyles);

  std::vector<double> ref_u_pt1 = {4.8241206e-09, 0.0, 0.0};
  std::vector<double> ref_u_pt2 = {3.45348338e-10, 2.30232226e-10,
                                   -1.15116113e-10};

  for (unsigned int i = 0; i < dim; ++i)
  {
    BOOST_TEST(pt_values[0][i] == ref_u_pt1[i]);
  }

  for (unsigned int i = 0; i < dim; ++i)
  {
    BOOST_TEST(pt_values[1][i] == ref_u_pt2[i]);
  }
}

BOOST_AUTO_TEST_CASE(elastoplastic_radial_return)
{
  MPI_Comm communicator = MPI_COMM_WORLD;

  boost::property_tree::ptree geometry_database;
  geometry_database.put("import_mesh", false);
  geometry_database.put("length", 1.);
  geometry_database.put("length_divisions", 2);
  geometry_database.put("height", 1.);
  geometry_database.put("height_divisions", 2);
  geometry_database.put("width", 1.);
  geometry_database.put("width_divisions", 2);
  boost::optional<boost::property_tree::ptree const &> units_optional_database;
  adamantine::Geometry<3> geometry(communicator, geometry_database,
                                   units_optional_database);
  auto const &triangulation = geometry.get_triangulation();
  for (auto cell : triangulation.cell_iterators())
  {
    cell->set_material_id(0);
    cell->set_user_index(
        static_cast<int>(adamantine::SolidLiquidPowder::State::solid));
  }

  double constexpr mu = 3.;
  double constexpr plastic_modulus = 1.5;
  double constexpr isotropic_hardening = 0.25;
  boost::property_tree::ptree material_database;
  material_database.put("property_format", "polynomial");
  material_database.put("n_materials", 1);
  material_database.put("material_0.solid.lame_first_parameter", 2.);
  material_database.put("material_0.solid.lame_second_parameter", mu);
  material_database.put("material_0.solid.plastic_modulus", plastic_modulus);
  material_database.put("material_0.solid.isotropic_hardening",
                        isotropic_hardening);
  material_database.put("material_0.solid.elastic_limit", 0.);
  adamantine::MaterialProperty<3, 1, 4, adamantine::SolidLiquidPowder,
                               dealii::MemorySpace::Host>
      material_properties(communicator, triangulation, material_database);

  boost::property_tree::ptree boundary_database;
  for (unsigned int id = 0; id < 6; ++id)
    boundary_database.put("boundary_" + std::to_string(id) + ".type",
                          "clamped");
  adamantine::Boundary boundary(
      boundary_database, geometry.get_triangulation().get_boundary_ids());

  std::vector<double> no_reference_temperatures;
  adamantine::MechanicalPhysics<3, 1, 4, adamantine::SolidLiquidPowder,
                                dealii::MemorySpace::Host>
      mechanical_physics(communicator, 1, geometry, boundary,
                         material_properties, no_reference_temperatures);
  mechanical_physics.setup_dofs();

  // The documented elastic branch is chi <= kappa. In particular, chi =
  // kappa = 0 must not enter the plastic branch and form the undefined 0/0
  // flow direction.
  mechanical_physics.solve();
  auto &stress = mechanical_physics.get_stress_tensor();
  for (auto const &cell_stress : stress)
    for (auto const &value : cell_stress)
    {
      BOOST_CHECK_SMALL(value.norm(), 1.e-14);
      for (unsigned int i = 0; i < value.n_independent_components; ++i)
        BOOST_TEST(std::isfinite(value.access_raw_entry(i)));
    }

  // Combined isotropic-kinematic hardening radial return, following R. I.
  // Borja, Plasticity: Modeling & Computation, Springer, 2013, Chapter 3,
  // DOI: 10.1007/978-3-642-38547-6.
  dealii::SymmetricTensor<2, 3> trial_stress;
  trial_stress[0][0] = 1.;
  trial_stress[1][1] = -1.;
  double const trial_norm = trial_stress.norm();
  auto const flow_direction = trial_stress / trial_norm;
  for (auto &cell_stress : stress)
    for (auto &value : cell_stress)
      value = trial_stress;

  mechanical_physics.solve();
  double const plastic_increment_1 = trial_norm / (2. * mu + plastic_modulus);
  double const returned_stress_norm =
      trial_norm - 2. * mu * plastic_increment_1;
  auto const expected_stress_1 = returned_stress_norm * flow_direction;
  for (auto const &cell_stress : stress)
    for (auto const &value : cell_stress)
      BOOST_CHECK_SMALL((value - expected_stress_1).norm(), 1.e-12);

  // A second collinear increment exercises the stored back stress. This is a
  // regression for accidentally multiplying the kinematic update by H twice
  // and omitting Delta eta.
  double constexpr stress_increment = 0.2;
  for (auto &cell_stress : stress)
    for (auto &value : cell_stress)
      value += stress_increment * flow_direction;

  mechanical_physics.solve();
  double const plastic_increment_2 =
      stress_increment / (2. * mu + plastic_modulus);
  double const expected_stress_norm =
      returned_stress_norm + stress_increment - 2. * mu * plastic_increment_2;
  auto const expected_stress_2 = expected_stress_norm * flow_direction;
  for (auto const &cell_stress : stress)
    for (auto const &value : cell_stress)
      BOOST_CHECK_SMALL((value - expected_stress_2).norm(), 1.e-12);
}

BOOST_AUTO_TEST_CASE(cell_data_transfer_refine_coarsen)
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
  boost::optional<boost::property_tree::ptree const &> units_optional_database;
  // Build Geometry
  adamantine::Geometry<3> geometry(communicator, geometry_database,
                                   units_optional_database);
  auto &triangulation = geometry.get_triangulation();
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
  material_database.put("material_0.solid.density", 1.);
  material_database.put("material_0.solid.lame_first_parameter", 2.);
  material_database.put("material_0.solid.lame_second_parameter", 3.);
  material_database.put("material_0.solid.elastic_limit", 0.1);
  adamantine::MaterialProperty<3, 1, 4, adamantine::SolidLiquidPowder,
                               dealii::MemorySpace::Host>
      material_properties(communicator, triangulation, material_database);
  // Create the Boundary
  boost::property_tree::ptree boundary_database;
  boundary_database.put("boundary_4.type", "clamped");
  adamantine::Boundary boundary(
      boundary_database, geometry.get_triangulation().get_boundary_ids());
  // Build MechanicalPhysics
  unsigned int const fe_degree = 1;
  std::vector<double> empty_vector;
  adamantine::MechanicalPhysics<3, 1, 4, adamantine::SolidLiquidPowder,
                                dealii::MemorySpace::Host>
      mechanical_physics(communicator, fe_degree, geometry, boundary,
                         material_properties, empty_vector);
  std::vector<std::shared_ptr<adamantine::BodyForce<3>>> body_forces;
  auto gravity_force = std::make_shared<adamantine::GravityForce<
      3, 1, 4, adamantine::SolidLiquidPowder, dealii::MemorySpace::Host>>(
      material_properties);
  body_forces.push_back(gravity_force);
  mechanical_physics.setup_dofs(body_forces);
  mechanical_physics.solve();

  // Record initial stress values before any mesh adaptation
  std::vector<std::vector<dealii::SymmetricTensor<2, 3>>> initial_stress_copy =
      mechanical_physics.get_stress_tensor();

  // --- Refine one cell per processor ---
  for (auto const &cell :
       dealii::filter_iterators(triangulation.active_cell_iterators(),
                                dealii::IteratorFilters::LocallyOwnedCell()))
  {
    cell->set_refine_flag();
    break;
  }
  mechanical_physics.prepare_transfer_mpi();
  triangulation.execute_coarsening_and_refinement();
  mechanical_physics.complete_transfer_mpi();
  mechanical_physics.setup_dofs(body_forces);

  // --- Coarsen back: flag all refined children ---
  for (auto const &cell :
       dealii::filter_iterators(triangulation.active_cell_iterators(),
                                dealii::IteratorFilters::LocallyOwnedCell()))
  {
    if (cell->level() > 0)
      cell->set_coarsen_flag();
  }
  mechanical_physics.prepare_transfer_mpi();
  triangulation.execute_coarsening_and_refinement();
  mechanical_physics.complete_transfer_mpi();
  mechanical_physics.setup_dofs(body_forces);

  // The mesh should be back to its original size
  auto const &final_stress = mechanical_physics.get_stress_tensor();
  BOOST_TEST(final_stress.size() == initial_stress_copy.size());

  // Stress values on every cell/quadrature point should be recovered exactly
  // (up to floating-point round-off) after the refine-then-coarsen round trip.
  for (unsigned int cell_id = 0; cell_id < final_stress.size(); ++cell_id)
  {
    for (unsigned int q = 0; q < final_stress[cell_id].size(); ++q)
    {
      double const tolerance = 1e-10 * initial_stress_copy[cell_id][q].norm();

      for (unsigned int i = 0;
           i < dealii::SymmetricTensor<2, 3>::n_independent_components; ++i)
      {
        BOOST_CHECK_SMALL(
            final_stress[cell_id][q].access_raw_entry(i) -
                initial_stress_copy[cell_id][q].access_raw_entry(i),
            tolerance);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE(thermoelastic_stress_uniform_heating)
{
  MPI_Comm communicator = MPI_COMM_WORLD;

  boost::property_tree::ptree geometry_database;
  geometry_database.put("import_mesh", false);
  geometry_database.put("length", 1.);
  geometry_database.put("length_divisions", 2);
  geometry_database.put("height", 1.);
  geometry_database.put("height_divisions", 2);
  geometry_database.put("width", 1.);
  geometry_database.put("width_divisions", 2);
  boost::optional<boost::property_tree::ptree const &> units_optional_database;
  adamantine::Geometry<3> geometry(communicator, geometry_database,
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
  double constexpr alpha = 0.01;
  boost::property_tree::ptree material_database;
  material_database.put("property_format", "polynomial");
  material_database.put("n_materials", 1);
  material_database.put("material_0.solid.lame_first_parameter", lambda);
  material_database.put("material_0.solid.lame_second_parameter", mu);
  material_database.put("material_0.solid.thermal_expansion_coef", alpha);
  adamantine::MaterialProperty<3, 1, 4, adamantine::SolidLiquidPowder,
                               dealii::MemorySpace::Host>
      material_properties(communicator, triangulation, material_database);

  boost::property_tree::ptree boundary_database;
  for (unsigned int id = 0; id < 6; ++id)
    boundary_database.put("boundary_" + std::to_string(id) + ".type",
                          "clamped");
  adamantine::Boundary boundary(
      boundary_database, geometry.get_triangulation().get_boundary_ids());

  dealii::hp::FECollection<3> thermal_fe_collection;
  thermal_fe_collection.push_back(dealii::FE_Q<3>(1));
  thermal_fe_collection.push_back(dealii::FE_Nothing<3>());
  dealii::DoFHandler<3> thermal_dof_handler(geometry.get_triangulation());
  thermal_dof_handler.distribute_dofs(thermal_fe_collection);
  dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host>
      temperature(thermal_dof_handler.locally_owned_dofs(), communicator);
  temperature = 350.;

  // For an unmelted substrate the last entry is the reference temperature.
  std::vector<double> reference_temperatures = {1000., 300.};
  std::vector<bool> has_melted(triangulation.n_active_cells(), false);
  adamantine::MechanicalPhysics<3, 1, 4, adamantine::SolidLiquidPowder,
                                dealii::MemorySpace::Host>
      mechanical_physics(communicator, 1, geometry, boundary,
                         material_properties, reference_temperatures);
  mechanical_physics.setup_dofs(thermal_dof_handler, temperature, has_melted,
                                true);
  auto displacement = mechanical_physics.solve();
  BOOST_CHECK_SMALL(displacement.l2_norm(), 1.e-12);

  // A uniformly heated, fully constrained isotropic solid has u = 0 and
  // sigma = -(3 lambda + 2 mu) alpha (T-T_ref) I. See Y. C. Fung and
  // Pin Tong, Classical and Computational Solid Mechanics, World Scientific,
  // 2001, Chapter 14, DOI: 10.1142/4362.
  auto check_stress = [&](double temperature_value)
  {
    double const expected_normal_stress =
        -(3. * lambda + 2. * mu) * alpha * (temperature_value - 300.);
    auto const expected_stress =
        expected_normal_stress * dealii::unit_symmetric_tensor<3>();
    for (auto const &cell_stress : mechanical_physics.get_stress_tensor())
      for (auto const &value : cell_stress)
        BOOST_CHECK_SMALL((value - expected_stress).norm(), 1.e-12);
  };
  check_stress(350.);

  // Updating the temperature must apply only the thermal stress increment, not
  // the total thermal stress a second time.
  temperature = 360.;
  mechanical_physics.update_rhs(thermal_dof_handler, temperature, has_melted);
  displacement = mechanical_physics.solve();
  BOOST_CHECK_SMALL(displacement.l2_norm(), 1.e-12);
  check_stress(360.);
}
