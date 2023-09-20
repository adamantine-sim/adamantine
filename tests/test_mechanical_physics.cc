/* Copyright (c) 2022, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

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
  // Build Geometry
  adamantine::Geometry<3> geometry(communicator, geometry_database);
  auto const &triangulation = geometry.get_triangulation();
  for (auto cell : triangulation.cell_iterators())
  {
    cell->set_material_id(0);
    cell->set_user_index(static_cast<int>(adamantine::MaterialState::solid));
  }
  // Create the MaterialProperty
  boost::property_tree::ptree material_database;
  material_database.put("property_format", "polynomial");
  material_database.put("n_materials", 1);
  material_database.put("material_0.solid.density", 1.);
  material_database.put("material_0.solid.lame_first_parameter", 2.);
  material_database.put("material_0.solid.lame_second_parameter", 3.);
  adamantine::MaterialProperty<3, dealii::MemorySpace::Host>
      material_properties(communicator, triangulation, material_database);
  // Build MechanicalPhysics
  unsigned int const fe_degree = 1;
  std::vector<double> empty_vector;
  adamantine::MechanicalPhysics<3, dealii::MemorySpace::Host>
      mechanical_physics(communicator, fe_degree, geometry, material_properties,
                         empty_vector);
  std::vector<std::shared_ptr<adamantine::BodyForce<3>>> body_forces;
  auto gravity_force =
      std::make_shared<adamantine::GravityForce<3, dealii::MemorySpace::Host>>(
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
  // Build Geometry
  adamantine::Geometry<3> geometry(communicator, geometry_database);
  auto const &triangulation = geometry.get_triangulation();
  for (auto cell : triangulation.cell_iterators())
  {
    cell->set_material_id(0);
    if (cell->center()[2] < 6.)
    {
      cell->set_user_index(static_cast<int>(adamantine::MaterialState::solid));
    }
    else
    {
      cell->set_user_index(static_cast<int>(adamantine::MaterialState::powder));
    }
  }
  // Create the MaterialProperty
  boost::property_tree::ptree material_database;
  material_database.put("property_format", "polynomial");
  material_database.put("n_materials", 1);
  material_database.put("material_0.solid.density", 1.);
  material_database.put("material_0.solid.lame_first_parameter", 2.);
  material_database.put("material_0.solid.lame_second_parameter", 3.);
  adamantine::MaterialProperty<3, dealii::MemorySpace::Host>
      material_properties(communicator, triangulation, material_database);
  // Build MechanicalPhysics
  unsigned int const fe_degree = 1;
  std::vector<double> empty_vector;
  adamantine::MechanicalPhysics<3, dealii::MemorySpace::Host>
      mechanical_physics(communicator, fe_degree, geometry, material_properties,
                         empty_vector);
  std::vector<std::shared_ptr<adamantine::BodyForce<3>>> body_forces;
  auto gravity_force =
      std::make_shared<adamantine::GravityForce<3, dealii::MemorySpace::Host>>(
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
 * This test case uses the analytic solution for the displacement around a
 * spherical inclusion as a test case. As a result of the boundary conditions
 * (fixed on one face) and the need to keep the test computationally
 * inexpensive, a loose tolerance has been chosen. Also for simplicity, the
 * analytic solution is being calculated externally and the values at only two
 * points are being checked.
 *
 * Less than 5% deviation from analytic solution achieved with
 * refinement_cyles=5.
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
  adamantine::Geometry<dim> geometry(communicator, geometry_database);
  auto &triangulation = geometry.get_triangulation();

  const dealii::Point<dim> center = {2.0e-5, 2.0e-5, 2.0e-5};

  for (unsigned int cycle = 0; cycle < refinement_cycles; ++cycle)
  {
    for (auto cell :
         dealii::filter_iterators(triangulation.active_cell_iterators(),
                                  dealii::IteratorFilters::LocallyOwnedCell()))
    {
      cell->set_material_id(0);
      cell->set_user_index(static_cast<int>(adamantine::MaterialState::solid));
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
  adamantine::MaterialProperty<dim, dealii::MemorySpace::Host>
      material_properties(communicator, triangulation, material_database);

  // Build ThermalPhysics
  boost::property_tree::ptree database;
  database.put("time_stepping.method", "backward_euler");
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
  database.put("boundary.type", "adiabatic");
  adamantine::ThermalPhysics<dim, 1, dealii::MemorySpace::Host,
                             dealii::QGauss<1>>
      thermal_physics(communicator, database, geometry, material_properties);
  thermal_physics.setup_dofs();
  thermal_physics.update_material_deposition_orientation();
  thermal_physics.compute_inverse_mass_matrix();
  thermal_physics.get_state_from_material_properties();

  dealii::LinearAlgebra::distributed::Vector<double> temperature;
  thermal_physics.initialize_dof_vector(100.0, temperature);

  dealii::VectorTools::interpolate(thermal_physics.get_dof_handler(),
                                   InitialValueT<3>(), temperature);

  // Build MechanicalPhysics
  unsigned int const fe_degree = 1;
  std::vector<double> initial_temperature = {2.0};
  adamantine::MechanicalPhysics<3, dealii::MemorySpace::Host>
      mechanical_physics(communicator, fe_degree, geometry, material_properties,
                         initial_temperature);

  boost::property_tree::ptree post_processor_database;
  post_processor_database.put("filename_prefix", "mech_phys_test");
  post_processor_database.put("thermal_output", true);
  post_processor_database.put("mechanical_output", true);

  adamantine::PostProcessor<dim> post_processor(
      communicator, post_processor_database, thermal_physics.get_dof_handler(),
      mechanical_physics.get_dof_handler());

  std::vector<bool> has_melted(triangulation.n_active_cells(), false);

  mechanical_physics.setup_dofs(thermal_physics.get_dof_handler(), temperature,
                                has_melted);

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
