/* SPDX-FileCopyrightText: Copyright (c) 2016 - 2025, the adamantine authors.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include "MaterialStates.hh"

#include <deal.II/lac/full_matrix.h>
#define BOOST_TEST_MODULE ThermalOperator

#include <Geometry.hh>
#include <GoldakHeatSource.hh>
#include <ThermalOperator.hh>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/matrix_free/fe_point_evaluation.h>
#include <deal.II/numerics/matrix_tools.h>

#include <boost/property_tree/ptree.hpp>

#include "main.cc"

namespace tt = boost::test_tools;
namespace utf = boost::unit_test;

class LaplaceCoefficient : public dealii::Function<2>
{
public:
  double value(const dealii::Point<2> &, const unsigned int = 0) const final
  {
    return -1;
  }
};

BOOST_AUTO_TEST_CASE(thermal_operator, *utf::tolerance(1e-15))
{
  MPI_Comm communicator = MPI_COMM_WORLD;

  // Create the Geometry
  boost::property_tree::ptree geometry_database;
  geometry_database.put("import_mesh", false);
  geometry_database.put("length", 12);
  geometry_database.put("length_divisions", 4);
  geometry_database.put("height", 6);
  geometry_database.put("height_divisions", 5);
  boost::optional<boost::property_tree::ptree const &> units_optional_database;
  adamantine::Geometry<2> geometry(communicator, geometry_database,
                                   units_optional_database);

  // Create the Boundary
  boost::property_tree::ptree boundary_database;
  boundary_database.put("type", "adiabatic");
  adamantine::Boundary boundary(
      boundary_database, geometry.get_triangulation().get_boundary_ids());

  // Create the DoFHandler
  dealii::hp::FECollection<2> fe_collection;
  fe_collection.push_back(dealii::FE_Q<2>(2));
  fe_collection.push_back(dealii::FE_Nothing<2>());
  dealii::DoFHandler<2> dof_handler(geometry.get_triangulation());
  dof_handler.distribute_dofs(fe_collection);
  dealii::AffineConstraints<double> affine_constraints;
  affine_constraints.close();
  dealii::hp::QCollection<1> q_collection;
  q_collection.push_back(dealii::QGauss<1>(3));
  q_collection.push_back(dealii::QGauss<1>(1));

  // Create the MaterialProperty
  boost::property_tree::ptree mat_prop_database;
  mat_prop_database.put("property_format", "polynomial");
  mat_prop_database.put("n_materials", 1);
  mat_prop_database.put("material_0.solid.density", 1.);
  mat_prop_database.put("material_0.powder.density", 1.);
  mat_prop_database.put("material_0.liquid.density", 1.);
  mat_prop_database.put("material_0.solid.specific_heat", 1.);
  mat_prop_database.put("material_0.powder.specific_heat", 1.);
  mat_prop_database.put("material_0.liquid.specific_heat", 1.);
  mat_prop_database.put("material_0.solid.thermal_conductivity_x", 10.);
  mat_prop_database.put("material_0.solid.thermal_conductivity_z", 10.);
  mat_prop_database.put("material_0.powder.thermal_conductivity_x", 10.);
  mat_prop_database.put("material_0.powder.thermal_conductivity_z", 10.);
  mat_prop_database.put("material_0.liquid.thermal_conductivity_x", 10.);
  mat_prop_database.put("material_0.liquid.thermal_conductivity_z", 10.);
  adamantine::MaterialProperty<2, 1, adamantine::SolidLiquidPowder,
                               dealii::MemorySpace::Host>
      mat_properties(communicator, geometry.get_triangulation(),
                     mat_prop_database);

  // Create the heat sources
  boost::property_tree::ptree beam_database;
  beam_database.put("depth", 0.1);
  beam_database.put("absorption_efficiency", 0.1);
  beam_database.put("diameter", 1.0);
  beam_database.put("max_power", 0.);
  beam_database.put("scan_path_file", "scan_path.txt");
  beam_database.put("scan_path_file_format", "segment");
  std::vector<std::shared_ptr<adamantine::HeatSource<2>>> heat_sources;
  heat_sources.resize(1);
  heat_sources[0] = std::make_shared<adamantine::GoldakHeatSource<2>>(
      beam_database, units_optional_database);
  heat_sources[0]->update_time(0.);

  // Initialize the ThermalOperator
  adamantine::ThermalOperator<2, false, 1, 2, adamantine::SolidLiquidPowder,
                              dealii::MemorySpace::Host>
      thermal_operator(communicator, boundary, mat_properties, heat_sources);
  std::vector<double> deposition_cos(
      geometry.get_triangulation().n_locally_owned_active_cells(), 1.);
  std::vector<double> deposition_sin(
      geometry.get_triangulation().n_locally_owned_active_cells(), 0.);
  thermal_operator.reinit(dof_handler, affine_constraints, q_collection);
  thermal_operator.set_material_deposition_orientation(deposition_cos,
                                                       deposition_sin);
  thermal_operator.compute_inverse_mass_matrix(dof_handler, affine_constraints);
  thermal_operator.get_state_from_material_properties();

  BOOST_TEST(thermal_operator.m() == 99);
  BOOST_TEST(thermal_operator.m() == thermal_operator.n());

  // Check matrix-vector multiplications
  dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host> src;
  dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host> dst_1;
  dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host> dst_2;

  dealii::MatrixFree<2, double> const &matrix_free =
      thermal_operator.get_matrix_free();
  matrix_free.initialize_dof_vector(src);
  matrix_free.initialize_dof_vector(dst_1);
  matrix_free.initialize_dof_vector(dst_2);

  src = 1.;
  thermal_operator.vmult(dst_1, src);
  BOOST_TEST(dst_1.l1_norm() == 0.);

  thermal_operator.Tvmult(dst_2, src);
  BOOST_TEST(dst_2.l1_norm() == dst_1.l1_norm());

  dst_2 = 1.;
  thermal_operator.vmult_add(dst_2, src);
  thermal_operator.vmult(dst_1, src);
  dst_1 += src;
  BOOST_TEST(dst_1.l1_norm() == dst_2.l1_norm());

  dst_1 = 1.;
  thermal_operator.Tvmult_add(dst_1, src);
  BOOST_TEST(dst_1.l1_norm() == dst_2.l1_norm());
}

BOOST_AUTO_TEST_CASE(spmv, *utf::tolerance(1e-12))
{
  MPI_Comm communicator = MPI_COMM_WORLD;

  // Create the Geometry
  boost::property_tree::ptree geometry_database;
  geometry_database.put("import_mesh", false);
  geometry_database.put("length", 12);
  geometry_database.put("length_divisions", 4);
  geometry_database.put("height", 6);
  geometry_database.put("height_divisions", 5);
  boost::optional<boost::property_tree::ptree const &> units_optional_database;
  adamantine::Geometry<2> geometry(communicator, geometry_database,
                                   units_optional_database);
  // Create the Boundary
  boost::property_tree::ptree boundary_database;
  boundary_database.put("type", "adiabatic");
  adamantine::Boundary boundary(
      boundary_database, geometry.get_triangulation().get_boundary_ids());

  // Create the DoFHandler
  dealii::hp::FECollection<2> fe_collection;
  fe_collection.push_back(dealii::FE_Q<2>(2));
  fe_collection.push_back(dealii::FE_Nothing<2>());
  dealii::DoFHandler<2> dof_handler(geometry.get_triangulation());
  dof_handler.distribute_dofs(fe_collection);
  dealii::AffineConstraints<double> affine_constraints;
  affine_constraints.close();
  dealii::hp::QCollection<1> q_collection;
  q_collection.push_back(dealii::QGauss<1>(3));
  q_collection.push_back(dealii::QGauss<1>(1));

  // Create the MaterialProperty
  boost::property_tree::ptree mat_prop_database;
  mat_prop_database.put("property_format", "polynomial");
  mat_prop_database.put("n_materials", 1);
  mat_prop_database.put("material_0.solid.density", 1.);
  mat_prop_database.put("material_0.powder.density", 1.);
  mat_prop_database.put("material_0.liquid.density", 1.);
  mat_prop_database.put("material_0.solid.specific_heat", 1.);
  mat_prop_database.put("material_0.powder.specific_heat", 1.);
  mat_prop_database.put("material_0.liquid.specific_heat", 1.);
  mat_prop_database.put("material_0.solid.thermal_conductivity_x", 1.);
  mat_prop_database.put("material_0.solid.thermal_conductivity_z", 1.);
  mat_prop_database.put("material_0.powder.thermal_conductivity_x", 1.);
  mat_prop_database.put("material_0.powder.thermal_conductivity_z", 1.);
  mat_prop_database.put("material_0.liquid.thermal_conductivity_x", 1.);
  mat_prop_database.put("material_0.liquid.thermal_conductivity_z", 1.);
  adamantine::MaterialProperty<2, 2, adamantine::SolidLiquidPowder,
                               dealii::MemorySpace::Host>
      mat_properties(communicator, geometry.get_triangulation(),
                     mat_prop_database);

  // Create the heat sources
  boost::property_tree::ptree beam_database;
  beam_database.put("depth", 0.1);
  beam_database.put("absorption_efficiency", 0.1);
  beam_database.put("diameter", 1.0);
  beam_database.put("max_power", 0.);
  beam_database.put("scan_path_file", "scan_path.txt");
  beam_database.put("scan_path_file_format", "segment");
  std::vector<std::shared_ptr<adamantine::HeatSource<2>>> heat_sources;
  heat_sources.resize(1);
  heat_sources[0] = std::make_shared<adamantine::GoldakHeatSource<2>>(
      beam_database, units_optional_database);
  heat_sources[0]->update_time(0.);

  // Initialize the ThermalOperator
  adamantine::ThermalOperator<2, false, 2, 2, adamantine::SolidLiquidPowder,
                              dealii::MemorySpace::Host>
      thermal_operator(communicator, boundary, mat_properties, heat_sources);
  std::vector<double> deposition_cos(
      geometry.get_triangulation().n_locally_owned_active_cells(), 1.);
  std::vector<double> deposition_sin(
      geometry.get_triangulation().n_locally_owned_active_cells(), 0.);
  thermal_operator.reinit(dof_handler, affine_constraints, q_collection);
  thermal_operator.set_material_deposition_orientation(deposition_cos,
                                                       deposition_sin);
  thermal_operator.compute_inverse_mass_matrix(dof_handler, affine_constraints);
  thermal_operator.get_state_from_material_properties();
  BOOST_TEST(thermal_operator.m() == 99);
  BOOST_TEST(thermal_operator.m() == thermal_operator.n());

  // Build the matrix. This only works in serial.
  dealii::DynamicSparsityPattern dsp(dof_handler.n_dofs());
  dealii::DoFTools::make_sparsity_pattern(dof_handler, dsp, affine_constraints);
  dealii::SparsityPattern sparsity_pattern;
  sparsity_pattern.copy_from(dsp);
  dealii::SparseMatrix<double> sparse_matrix(sparsity_pattern);
  dealii::MatrixCreator::create_laplace_matrix(
      dof_handler, dealii::QGauss<2>(3), sparse_matrix);

  // Compare vmult using matrix free and building the matrix
  dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host> src_1;
  dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host> src_2;
  dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host> dst_1;
  dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host> dst_2;

  dealii::MatrixFree<2, double> const &matrix_free =
      thermal_operator.get_matrix_free();
  matrix_free.initialize_dof_vector(src_1);
  matrix_free.initialize_dof_vector(src_2);
  matrix_free.initialize_dof_vector(dst_1);
  matrix_free.initialize_dof_vector(dst_2);

  for (unsigned int i = 0; i < thermal_operator.m(); ++i)
  {
    src_1 = 0.;
    src_2 = 0.;
    src_1[i] = 1;
    src_2[i] = -1;
    thermal_operator.vmult(dst_1, src_1);
    sparse_matrix.vmult(dst_2, src_2);
    BOOST_TEST(dst_1 == dst_2, tt::per_element());
  }
}

BOOST_AUTO_TEST_CASE(spmv_anisotropic, *utf::tolerance(1e-12))
{
  MPI_Comm communicator = MPI_COMM_WORLD;

  // Create the Geometry
  boost::property_tree::ptree geometry_database;
  geometry_database.put("import_mesh", false);
  geometry_database.put("length", 12);
  geometry_database.put("length_divisions", 4);
  geometry_database.put("height", 6);
  geometry_database.put("height_divisions", 5);
  boost::optional<boost::property_tree::ptree const &> units_optional_database;
  adamantine::Geometry<2> geometry(communicator, geometry_database,
                                   units_optional_database);

  // Create the Boundary
  boost::property_tree::ptree boundary_database;
  boundary_database.put("type", "adiabatic");
  adamantine::Boundary boundary(
      boundary_database, geometry.get_triangulation().get_boundary_ids());

  // Create the DoFHandler
  dealii::hp::FECollection<2> fe_collection;
  fe_collection.push_back(dealii::FE_Q<2>(2));
  fe_collection.push_back(dealii::FE_Nothing<2>());
  dealii::DoFHandler<2> dof_handler(geometry.get_triangulation());
  dof_handler.distribute_dofs(fe_collection);
  dealii::AffineConstraints<double> affine_constraints;
  affine_constraints.close();
  dealii::hp::QCollection<1> q_collection;
  q_collection.push_back(dealii::QGauss<1>(3));
  q_collection.push_back(dealii::QGauss<1>(1));

  // Create the MaterialProperty
  boost::property_tree::ptree mat_prop_database;
  mat_prop_database.put("property_format", "polynomial");
  mat_prop_database.put("n_materials", 1);
  mat_prop_database.put("material_0.solid.density", 1.);
  mat_prop_database.put("material_0.powder.density", 1.);
  mat_prop_database.put("material_0.liquid.density", 1.);
  mat_prop_database.put("material_0.solid.specific_heat", 1.);
  mat_prop_database.put("material_0.powder.specific_heat", 1.);
  mat_prop_database.put("material_0.liquid.specific_heat", 1.);
  mat_prop_database.put("material_0.solid.thermal_conductivity_x", 1.);
  mat_prop_database.put("material_0.solid.thermal_conductivity_z", 0.);
  mat_prop_database.put("material_0.powder.thermal_conductivity_x", 1.);
  mat_prop_database.put("material_0.powder.thermal_conductivity_z", 0.);
  mat_prop_database.put("material_0.liquid.thermal_conductivity_x", 1.);
  mat_prop_database.put("material_0.liquid.thermal_conductivity_z", 0.);
  adamantine::MaterialProperty<2, 2, adamantine::SolidLiquidPowder,
                               dealii::MemorySpace::Host>
      mat_properties(communicator, geometry.get_triangulation(),
                     mat_prop_database);

  // Create the heat sources
  boost::property_tree::ptree beam_database;
  beam_database.put("depth", 0.1);
  beam_database.put("absorption_efficiency", 0.1);
  beam_database.put("diameter", 1.0);
  beam_database.put("max_power", 0.);
  beam_database.put("scan_path_file", "scan_path.txt");
  beam_database.put("scan_path_file_format", "segment");
  std::vector<std::shared_ptr<adamantine::HeatSource<2>>> heat_sources;
  heat_sources.resize(1);
  heat_sources[0] = std::make_shared<adamantine::GoldakHeatSource<2>>(
      beam_database, units_optional_database);
  heat_sources[0]->update_time(0.);

  // Initialize the ThermalOperator
  adamantine::ThermalOperator<2, false, 2, 2, adamantine::SolidLiquidPowder,
                              dealii::MemorySpace::Host>
      thermal_operator(communicator, boundary, mat_properties, heat_sources);
  std::vector<double> deposition_cos(
      geometry.get_triangulation().n_locally_owned_active_cells(), 1.);
  std::vector<double> deposition_sin(
      geometry.get_triangulation().n_locally_owned_active_cells(), 0.);
  thermal_operator.reinit(dof_handler, affine_constraints, q_collection);
  thermal_operator.set_material_deposition_orientation(deposition_cos,
                                                       deposition_sin);
  thermal_operator.compute_inverse_mass_matrix(dof_handler, affine_constraints);
  thermal_operator.get_state_from_material_properties();
  BOOST_TEST(thermal_operator.m() == 99);
  BOOST_TEST(thermal_operator.m() == thermal_operator.n());

  // Build the matrix. This only works in serial.
  dealii::DynamicSparsityPattern dsp(dof_handler.n_dofs());
  dealii::DoFTools::make_sparsity_pattern(dof_handler, dsp, affine_constraints);
  dealii::SparsityPattern sparsity_pattern;
  sparsity_pattern.copy_from(dsp);
  dealii::SparseMatrix<double> sparse_matrix(sparsity_pattern);
  // Assemble the anisotropic matrix
  {
    auto &fe = dof_handler.get_fe();
    dealii::QGauss<2> quadrature_formula(3);
    dealii::FEValues<2> fe_values(fe, quadrature_formula,
                                  dealii::update_gradients |
                                      dealii::update_quadrature_points |
                                      dealii::update_JxW_values);
    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    dealii::FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    std::vector<dealii::types::global_dof_index> local_dof_indices(
        dofs_per_cell);
    for (auto const &cell : dof_handler.active_cell_iterators())
    {
      cell_matrix = 0;
      fe_values.reinit(cell);
      for (unsigned int const q_index : fe_values.quadrature_point_indices())
      {
        for (unsigned int const i : fe_values.dof_indices())
        {
          // Compute (coef_x grad_x, coef_y grad_y) with coef_x = 1 and coef_y =
          // 0
          auto coef_grad = fe_values.shape_grad(i, q_index);
          coef_grad[1] = 0.;
          for (unsigned int const j : fe_values.dof_indices())
            cell_matrix(i, j) += coef_grad * fe_values.shape_grad(j, q_index) *
                                 fe_values.JxW(q_index);
        }
      }
      cell->get_dof_indices(local_dof_indices);
      affine_constraints.distribute_local_to_global(
          cell_matrix, local_dof_indices, sparse_matrix);
    }
  }

  // Compare vmult using matrix free and building the matrix
  dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host> src_1;
  dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host> src_2;
  dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host> dst_1;
  dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host> dst_2;

  dealii::MatrixFree<2, double> const &matrix_free =
      thermal_operator.get_matrix_free();
  matrix_free.initialize_dof_vector(src_1);
  matrix_free.initialize_dof_vector(src_2);
  matrix_free.initialize_dof_vector(dst_1);
  matrix_free.initialize_dof_vector(dst_2);

  for (unsigned int i = 0; i < thermal_operator.m(); ++i)
  {
    src_1 = 0.;
    src_2 = 0.;
    src_1[i] = 1;
    src_2[i] = -1;
    thermal_operator.vmult(dst_1, src_1);
    sparse_matrix.vmult(dst_2, src_2);
    BOOST_TEST(dst_1 == dst_2, tt::per_element());
  }
}

BOOST_AUTO_TEST_CASE(spmv_anisotropic_angle, *utf::tolerance(1e-10))
{
  MPI_Comm communicator = MPI_COMM_WORLD;

  // Create the Geometry
  boost::property_tree::ptree geometry_database;
  geometry_database.put("import_mesh", false);
  geometry_database.put("length", 12);
  geometry_database.put("length_divisions", 4);
  geometry_database.put("height", 6);
  geometry_database.put("height_divisions", 2);
  geometry_database.put("width", 6);
  geometry_database.put("width_divisions", 2);
  boost::optional<boost::property_tree::ptree const &> units_optional_database;
  adamantine::Geometry<3> geometry(communicator, geometry_database,
                                   units_optional_database);

  // Create the Boundary
  boost::property_tree::ptree boundary_database;
  boundary_database.put("type", "adiabatic");
  adamantine::Boundary boundary(
      boundary_database, geometry.get_triangulation().get_boundary_ids());

  // Create the DoFHandler
  dealii::hp::FECollection<3> fe_collection;
  fe_collection.push_back(dealii::FE_Q<3>(2));
  fe_collection.push_back(dealii::FE_Nothing<3>());
  dealii::DoFHandler<3> dof_handler(geometry.get_triangulation());
  dof_handler.distribute_dofs(fe_collection);
  dealii::AffineConstraints<double> affine_constraints;
  affine_constraints.close();
  dealii::hp::QCollection<1> q_collection;
  q_collection.push_back(dealii::QGauss<1>(3));
  q_collection.push_back(dealii::QGauss<1>(1));

  // Create the MaterialProperty
  boost::property_tree::ptree mat_prop_database;
  double const th_cond_x = 1.;
  double const th_cond_y = 0.8;
  double const th_cond_z = 0.8;
  mat_prop_database.put("property_format", "polynomial");
  mat_prop_database.put("n_materials", 1);
  mat_prop_database.put("material_0.solid.density", 1.);
  mat_prop_database.put("material_0.powder.density", 1.);
  mat_prop_database.put("material_0.liquid.density", 1.);
  mat_prop_database.put("material_0.solid.specific_heat", 1.);
  mat_prop_database.put("material_0.powder.specific_heat", 1.);
  mat_prop_database.put("material_0.liquid.specific_heat", 1.);
  mat_prop_database.put("material_0.solid.thermal_conductivity_x", th_cond_x);
  mat_prop_database.put("material_0.solid.thermal_conductivity_y", th_cond_y);
  mat_prop_database.put("material_0.solid.thermal_conductivity_z", th_cond_z);
  mat_prop_database.put("material_0.powder.thermal_conductivity_x", th_cond_x);
  mat_prop_database.put("material_0.powder.thermal_conductivity_y", th_cond_y);
  mat_prop_database.put("material_0.powder.thermal_conductivity_z", th_cond_z);
  mat_prop_database.put("material_0.liquid.thermal_conductivity_x", th_cond_x);
  mat_prop_database.put("material_0.liquid.thermal_conductivity_y", th_cond_y);
  mat_prop_database.put("material_0.liquid.thermal_conductivity_z", th_cond_z);
  adamantine::MaterialProperty<3, 1, adamantine::SolidLiquidPowder,
                               dealii::MemorySpace::Host>
      mat_properties(communicator, geometry.get_triangulation(),
                     mat_prop_database);

  // Create the heat sources
  std::vector<std::shared_ptr<adamantine::HeatSource<3>>> heat_sources;

  // Initialize the ThermalOperator
  adamantine::ThermalOperator<3, false, 1, 2, adamantine::SolidLiquidPowder,
                              dealii::MemorySpace::Host>
      thermal_operator(communicator, boundary, mat_properties, heat_sources);
  double constexpr deposition_angle = M_PI / 6.;
  std::vector<double> deposition_cos(
      geometry.get_triangulation().n_locally_owned_active_cells(),
      std::cos(deposition_angle));
  std::vector<double> deposition_sin(
      geometry.get_triangulation().n_locally_owned_active_cells(),
      std::sin(deposition_angle));
  thermal_operator.reinit(dof_handler, affine_constraints, q_collection);
  thermal_operator.set_material_deposition_orientation(deposition_cos,
                                                       deposition_sin);
  thermal_operator.compute_inverse_mass_matrix(dof_handler, affine_constraints);
  thermal_operator.get_state_from_material_properties();
  BOOST_TEST(thermal_operator.m() == thermal_operator.n());

  // Build the matrix. This only works in serial.
  dealii::DynamicSparsityPattern dsp(dof_handler.n_dofs());
  dealii::DoFTools::make_sparsity_pattern(dof_handler, dsp, affine_constraints);
  dealii::SparsityPattern sparsity_pattern;
  sparsity_pattern.copy_from(dsp);
  dealii::SparseMatrix<double> sparse_matrix(sparsity_pattern);
  // Assemble the anisotropic matrix
  {
    auto &fe = dof_handler.get_fe();
    dealii::QGauss<3> quadrature_formula(3);
    dealii::FEValues<3> fe_values(fe, quadrature_formula,
                                  dealii::update_gradients |
                                      dealii::update_quadrature_points |
                                      dealii::update_JxW_values);
    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    dealii::FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    std::vector<dealii::types::global_dof_index> local_dof_indices(
        dofs_per_cell);
    dealii::FullMatrix<double> coef(2);
    coef(0, 0) = th_cond_x;
    coef(1, 1) = th_cond_y;
    dealii::FullMatrix<double> rotation(2);
    rotation(0, 0) = std::cos(deposition_angle);
    rotation(0, 1) = -std::sin(deposition_angle);
    rotation(1, 0) = std::sin(deposition_angle);
    rotation(1, 1) = std::cos(deposition_angle);
    dealii::FullMatrix<double> rotated_coef(2);
    rotated_coef.triple_product(coef, rotation, rotation,
                                /*transpose first rotation matrix*/ false,
                                /*transpose second rotation matrix*/ true);

    for (auto const &cell : dof_handler.active_cell_iterators())
    {
      cell_matrix = 0;
      fe_values.reinit(cell);
      for (unsigned int const q_index : fe_values.quadrature_point_indices())
      {
        for (unsigned int const i : fe_values.dof_indices())
        {
          auto grad = fe_values.shape_grad(i, q_index);
          dealii::Tensor<1, 3> coef_grad;
          coef_grad[0] =
              rotated_coef(0, 0) * grad[0] + rotated_coef(0, 1) * grad[1];
          coef_grad[1] =
              rotated_coef(1, 0) * grad[0] + rotated_coef(1, 1) * grad[1];
          coef_grad[2] = th_cond_z * grad[2];
          for (unsigned int const j : fe_values.dof_indices())
            cell_matrix(i, j) += coef_grad * fe_values.shape_grad(j, q_index) *
                                 fe_values.JxW(q_index);
        }
      }
      cell->get_dof_indices(local_dof_indices);
      affine_constraints.distribute_local_to_global(
          cell_matrix, local_dof_indices, sparse_matrix);
    }
  }

  // Compare vmult using matrix free and building the matrix
  dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host> src_1;
  dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host> src_2;
  dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host> dst_1;
  dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host> dst_2;

  dealii::MatrixFree<3, double> const &matrix_free =
      thermal_operator.get_matrix_free();
  matrix_free.initialize_dof_vector(src_1);
  matrix_free.initialize_dof_vector(src_2);
  matrix_free.initialize_dof_vector(dst_1);
  matrix_free.initialize_dof_vector(dst_2);

  for (unsigned int i = 0; i < thermal_operator.m(); ++i)
  {
    src_1 = 0.;
    src_2 = 0.;
    src_1[i] = 1;
    src_2[i] = -1;
    thermal_operator.vmult(dst_1, src_1);
    sparse_matrix.vmult(dst_2, src_2);
    BOOST_TEST(dst_1 == dst_2, tt::per_element());
  }
}

BOOST_AUTO_TEST_CASE(spmv_rad, *utf::tolerance(1e-12))
{
  MPI_Comm communicator = MPI_COMM_WORLD;

  // Create the Geometry
  boost::property_tree::ptree geometry_database;
  geometry_database.put("import_mesh", false);
  geometry_database.put("length", 12);
  geometry_database.put("length_divisions", 4);
  geometry_database.put("height", 6);
  geometry_database.put("height_divisions", 5);
  boost::optional<boost::property_tree::ptree const &> units_optional_database;
  adamantine::Geometry<2> geometry(communicator, geometry_database,
                                   units_optional_database);

  // Create the Boundary
  boost::property_tree::ptree boundary_database;
  boundary_database.put("type", "radiative");
  adamantine::Boundary boundary(
      boundary_database, geometry.get_triangulation().get_boundary_ids());

  // Create the DoFHandler
  dealii::hp::FECollection<2> fe_collection;
  fe_collection.push_back(dealii::FE_Q<2>(2));
  fe_collection.push_back(dealii::FE_Nothing<2>());
  dealii::DoFHandler<2> dof_handler(geometry.get_triangulation());
  dof_handler.distribute_dofs(fe_collection);
  dealii::AffineConstraints<double> affine_constraints;
  affine_constraints.close();
  dealii::hp::QCollection<1> q_collection;
  q_collection.push_back(dealii::QGauss<1>(3));
  q_collection.push_back(dealii::QGauss<1>(1));

  // Create the MaterialProperty
  double const emissivity = 1e5;
  boost::property_tree::ptree mat_prop_database;
  mat_prop_database.put("property_format", "polynomial");
  mat_prop_database.put("n_materials", 1);
  mat_prop_database.put("material_0.solid.density", 1.);
  mat_prop_database.put("material_0.powder.density", 1.);
  mat_prop_database.put("material_0.liquid.density", 1.);
  mat_prop_database.put("material_0.solid.specific_heat", 1.);
  mat_prop_database.put("material_0.powder.specific_heat", 1.);
  mat_prop_database.put("material_0.liquid.specific_heat", 1.);
  mat_prop_database.put("material_0.solid.thermal_conductivity_x", 1.);
  mat_prop_database.put("material_0.solid.thermal_conductivity_z", 1.);
  mat_prop_database.put("material_0.powder.thermal_conductivity_x", 1.);
  mat_prop_database.put("material_0.powder.thermal_conductivity_z", 1.);
  mat_prop_database.put("material_0.liquid.thermal_conductivity_x", 1.);
  mat_prop_database.put("material_0.liquid.thermal_conductivity_z", 1.);
  mat_prop_database.put("material_0.solid.emissivity", emissivity);
  mat_prop_database.put("material_0.powder.emissivity", emissivity);
  mat_prop_database.put("material_0.liquid.emissivity", emissivity);
  mat_prop_database.put("material_0.solid.convection_heat_transfer_coef", 1.);
  mat_prop_database.put("material_0.powder.convection_heat_transfer_coef", 1.);
  mat_prop_database.put("material_0.liquid.convection_heat_transfer_coef", 1.);
  mat_prop_database.put("material_0.radiation_temperature_infty", 0.0);
  mat_prop_database.put("material_0.convection_temperature_infty", 0.0);
  adamantine::MaterialProperty<2, 1, adamantine::SolidLiquidPowder,
                               dealii::MemorySpace::Host>
      mat_properties(communicator, geometry.get_triangulation(),
                     mat_prop_database);

  // Create the heat sources
  boost::property_tree::ptree beam_database;
  beam_database.put("depth", 0.1);
  beam_database.put("absorption_efficiency", 0.1);
  beam_database.put("diameter", 1.0);
  beam_database.put("max_power", 0.);
  beam_database.put("scan_path_file", "scan_path.txt");
  beam_database.put("scan_path_file_format", "segment");
  std::vector<std::shared_ptr<adamantine::HeatSource<2>>> heat_sources;
  heat_sources.resize(1);
  heat_sources[0] = std::make_shared<adamantine::GoldakHeatSource<2>>(
      beam_database, units_optional_database);
  heat_sources[0]->update_time(0.);

  // Initialize the ThermalOperator
  adamantine::ThermalOperator<2, false, 1, 2, adamantine::SolidLiquidPowder,
                              dealii::MemorySpace::Host>
      thermal_operator(communicator, boundary, mat_properties, heat_sources);
  std::vector<double> deposition_cos(
      geometry.get_triangulation().n_locally_owned_active_cells(), 1.);
  std::vector<double> deposition_sin(
      geometry.get_triangulation().n_locally_owned_active_cells(), 0.);
  thermal_operator.reinit(dof_handler, affine_constraints, q_collection);
  thermal_operator.set_material_deposition_orientation(deposition_cos,
                                                       deposition_sin);
  thermal_operator.compute_inverse_mass_matrix(dof_handler, affine_constraints);
  dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host>
      temperature(thermal_operator.m());
  for (unsigned int i = 0; i < temperature.locally_owned_size(); ++i)
  {
    temperature.local_element(i) = 1.;
  }
  thermal_operator.get_state_from_material_properties();
  BOOST_TEST(thermal_operator.m() == 99);
  BOOST_TEST(thermal_operator.m() == thermal_operator.n());

  // Compare vmult using matrix free and building the matrix
  dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host> src;
  dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host> dst_1;
  dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host> dst_2;

  dealii::MatrixFree<2, double> const &matrix_free =
      thermal_operator.get_matrix_free();
  matrix_free.initialize_dof_vector(src);
  matrix_free.initialize_dof_vector(dst_1);
  matrix_free.initialize_dof_vector(dst_2);
  LaplaceCoefficient coefficient;

  for (unsigned int i = 0; i < thermal_operator.m(); ++i)
  {
    src = 0.;
    src[i] = 1;

    // Build the matrix. This only works in serial.
    dealii::DynamicSparsityPattern dsp(dof_handler.n_dofs());
    dealii::DoFTools::make_sparsity_pattern(dof_handler, dsp,
                                            affine_constraints);
    dealii::SparsityPattern sparsity_pattern;
    sparsity_pattern.copy_from(dsp);
    dealii::SparseMatrix<double> sparse_matrix(sparsity_pattern);
    dealii::MatrixCreator::create_laplace_matrix(
        dof_handler, dealii::QGauss<2>(3), sparse_matrix, &coefficient);
    // Take care of the boundary condition
    unsigned int const fe_degree = 2;
    dealii::FE_Q<2> fe(fe_degree);
    dealii::QGauss<1> face_quadrature_formula(fe_degree + 1);
    unsigned int const n_face_q_points = face_quadrature_formula.size();
    dealii::FEFaceValues<2> fe_face_values(
        fe, face_quadrature_formula,
        dealii::update_values | dealii::update_quadrature_points |
            dealii::update_JxW_values);
    unsigned int const dofs_per_cell = fe.n_dofs_per_cell();
    dealii::FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    std::vector<dealii::types::global_dof_index> local_dof_indices(
        dofs_per_cell);
    dealii::FEPointEvaluation<1, 2> fe_pt_evaluator(
        dealii::StaticMappingQ1<2>::mapping, fe,
        dealii::UpdateFlags::update_values);
    dealii::Vector<double> src_values(fe.n_dofs_per_cell());
    for (auto const &cell : dof_handler.active_cell_iterators())
    {
      if (cell->at_boundary())
      {
        cell_matrix = 0.;
        cell->get_dof_values(src, src_values);
        for (auto const &face : cell->face_iterators())
        {
          if (face->at_boundary())
          {
            fe_face_values.reinit(cell, face);
            std::vector<dealii::Point<2>> face_quad_pts;
            for (unsigned int q = 0; q < n_face_q_points; ++q)
            {
              auto q_point = fe_face_values.quadrature_point(q);
              face_quad_pts.push_back(
                  dealii::StaticMappingQ1<2>::mapping
                      .transform_real_to_unit_cell(cell, q_point));
            }
            fe_pt_evaluator.reinit(cell, face_quad_pts);
            fe_pt_evaluator.evaluate(dealii::make_array_view(src_values),
                                     dealii::EvaluationFlags::values);
            for (unsigned int q = 0; q < n_face_q_points; ++q)
            {
              double T = fe_pt_evaluator.get_value(q);
              double const heat_transfer_coeff =
                  emissivity * adamantine::Constant::stefan_boltzmann * T *
                  (T * T);
              for (unsigned i = 0; i < dofs_per_cell; ++i)
                for (unsigned j = 0; j < dofs_per_cell; ++j)
                {
                  cell_matrix(i, j) -=
                      heat_transfer_coeff * fe_face_values.shape_value(i, q) *
                      fe_face_values.shape_value(j, q) * fe_face_values.JxW(q);
                }
            }
          }
        }
        cell->get_dof_indices(local_dof_indices);
        affine_constraints.distribute_local_to_global(
            cell_matrix, local_dof_indices, sparse_matrix);
      }
    }

    thermal_operator.vmult(dst_1, src);
    sparse_matrix.vmult(dst_2, src);
    BOOST_TEST(dst_1 == dst_2, tt::per_element());
  }
}

BOOST_AUTO_TEST_CASE(spmv_conv, *utf::tolerance(1e-12))
{
  MPI_Comm communicator = MPI_COMM_WORLD;

  // Create the Geometry
  boost::property_tree::ptree geometry_database;
  geometry_database.put("import_mesh", false);
  geometry_database.put("length", 12);
  geometry_database.put("length_divisions", 4);
  geometry_database.put("height", 6);
  geometry_database.put("height_divisions", 5);
  boost::optional<boost::property_tree::ptree const &> units_optional_database;
  adamantine::Geometry<2> geometry(communicator, geometry_database,
                                   units_optional_database);

  // Create the Boundary
  boost::property_tree::ptree boundary_database;
  boundary_database.put("type", "convective");
  adamantine::Boundary boundary(
      boundary_database, geometry.get_triangulation().get_boundary_ids());

  // Create the DoFHandler
  dealii::hp::FECollection<2> fe_collection;
  fe_collection.push_back(dealii::FE_Q<2>(2));
  fe_collection.push_back(dealii::FE_Nothing<2>());
  dealii::DoFHandler<2> dof_handler(geometry.get_triangulation());
  dof_handler.distribute_dofs(fe_collection);
  dealii::AffineConstraints<double> affine_constraints;
  affine_constraints.close();
  dealii::hp::QCollection<1> q_collection;
  q_collection.push_back(dealii::QGauss<1>(3));
  q_collection.push_back(dealii::QGauss<1>(1));

  // Create the MaterialProperty
  double const emissivity = 1e5;
  boost::property_tree::ptree mat_prop_database;
  mat_prop_database.put("property_format", "polynomial");
  mat_prop_database.put("n_materials", 1);
  mat_prop_database.put("material_0.solid.density", 1.);
  mat_prop_database.put("material_0.powder.density", 1.);
  mat_prop_database.put("material_0.liquid.density", 1.);
  mat_prop_database.put("material_0.solid.specific_heat", 1.);
  mat_prop_database.put("material_0.powder.specific_heat", 1.);
  mat_prop_database.put("material_0.liquid.specific_heat", 1.);
  mat_prop_database.put("material_0.solid.thermal_conductivity_x", 1.);
  mat_prop_database.put("material_0.solid.thermal_conductivity_z", 1.);
  mat_prop_database.put("material_0.powder.thermal_conductivity_x", 1.);
  mat_prop_database.put("material_0.powder.thermal_conductivity_z", 1.);
  mat_prop_database.put("material_0.liquid.thermal_conductivity_x", 1.);
  mat_prop_database.put("material_0.liquid.thermal_conductivity_z", 1.);
  mat_prop_database.put("material_0.solid.emissivity", emissivity);
  mat_prop_database.put("material_0.powder.emissivity", emissivity);
  mat_prop_database.put("material_0.liquid.emissivity", emissivity);
  mat_prop_database.put("material_0.solid.convection_heat_transfer_coef", 1.);
  mat_prop_database.put("material_0.powder.convection_heat_transfer_coef", 1.);
  mat_prop_database.put("material_0.liquid.convection_heat_transfer_coef", 1.);
  mat_prop_database.put("material_0.radiation_temperature_infty", 0.0);
  mat_prop_database.put("material_0.convection_temperature_infty", 0.0);
  adamantine::MaterialProperty<2, 1, adamantine::SolidLiquidPowder,
                               dealii::MemorySpace::Host>
      mat_properties(communicator, geometry.get_triangulation(),
                     mat_prop_database);

  // Create the heat sources
  boost::property_tree::ptree beam_database;
  beam_database.put("depth", 0.1);
  beam_database.put("absorption_efficiency", 0.1);
  beam_database.put("diameter", 1.0);
  beam_database.put("max_power", 0.);
  beam_database.put("scan_path_file", "scan_path.txt");
  beam_database.put("scan_path_file_format", "segment");
  std::vector<std::shared_ptr<adamantine::HeatSource<2>>> heat_sources;
  heat_sources.resize(1);
  heat_sources[0] = std::make_shared<adamantine::GoldakHeatSource<2>>(
      beam_database, units_optional_database);
  heat_sources[0]->update_time(0.);

  // Initialize the ThermalOperator
  adamantine::ThermalOperator<2, false, 1, 2, adamantine::SolidLiquidPowder,
                              dealii::MemorySpace::Host>
      thermal_operator(communicator, boundary, mat_properties, heat_sources);
  std::vector<double> deposition_cos(
      geometry.get_triangulation().n_locally_owned_active_cells(), 1.);
  std::vector<double> deposition_sin(
      geometry.get_triangulation().n_locally_owned_active_cells(), 0.);
  thermal_operator.reinit(dof_handler, affine_constraints, q_collection);
  thermal_operator.set_material_deposition_orientation(deposition_cos,
                                                       deposition_sin);
  thermal_operator.compute_inverse_mass_matrix(dof_handler, affine_constraints);
  dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host>
      temperature(thermal_operator.m());
  for (unsigned int i = 0; i < temperature.locally_owned_size(); ++i)
  {
    temperature.local_element(i) = 1.;
  }
  thermal_operator.get_state_from_material_properties();
  BOOST_TEST(thermal_operator.m() == 99);
  BOOST_TEST(thermal_operator.m() == thermal_operator.n());

  // Compare vmult using matrix free and building the matrix
  dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host> src;
  dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host> dst_1;
  dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host> dst_2;

  dealii::MatrixFree<2, double> const &matrix_free =
      thermal_operator.get_matrix_free();
  matrix_free.initialize_dof_vector(src);
  matrix_free.initialize_dof_vector(dst_1);
  matrix_free.initialize_dof_vector(dst_2);
  LaplaceCoefficient coefficient;

  for (unsigned int i = 0; i < thermal_operator.m(); ++i)
  {
    src = 0.;
    src[i] = 1;

    // Build the matrix. This only works in serial.
    dealii::DynamicSparsityPattern dsp(dof_handler.n_dofs());
    dealii::DoFTools::make_sparsity_pattern(dof_handler, dsp,
                                            affine_constraints);
    dealii::SparsityPattern sparsity_pattern;
    sparsity_pattern.copy_from(dsp);
    dealii::SparseMatrix<double> sparse_matrix(sparsity_pattern);
    dealii::MatrixCreator::create_laplace_matrix(
        dof_handler, dealii::QGauss<2>(3), sparse_matrix, &coefficient);
    // Take care of the boundary condition
    unsigned int const fe_degree = 2;
    dealii::FE_Q<2> fe(fe_degree);
    dealii::QGauss<1> face_quadrature_formula(fe_degree + 1);
    unsigned int const n_face_q_points = face_quadrature_formula.size();
    dealii::FEFaceValues<2> fe_face_values(
        fe, face_quadrature_formula,
        dealii::update_values | dealii::update_quadrature_points |
            dealii::update_JxW_values);
    unsigned int const dofs_per_cell = fe.n_dofs_per_cell();
    dealii::FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    std::vector<dealii::types::global_dof_index> local_dof_indices(
        dofs_per_cell);
    dealii::Vector<double> src_values(fe.n_dofs_per_cell());
    for (auto const &cell : dof_handler.active_cell_iterators())
    {
      if (cell->at_boundary())
      {
        cell_matrix = 0.;
        cell->get_dof_values(src, src_values);
        for (auto const &face : cell->face_iterators())
        {
          if (face->at_boundary())
          {
            fe_face_values.reinit(cell, face);
            std::vector<dealii::Point<2>> face_quad_pts;
            for (unsigned int q = 0; q < n_face_q_points; ++q)
            {
              auto q_point = fe_face_values.quadrature_point(q);
              face_quad_pts.push_back(
                  dealii::StaticMappingQ1<2>::mapping
                      .transform_real_to_unit_cell(cell, q_point));
            }
            for (unsigned int q = 0; q < n_face_q_points; ++q)
            {
              for (unsigned i = 0; i < dofs_per_cell; ++i)
                for (unsigned j = 0; j < dofs_per_cell; ++j)
                {
                  cell_matrix(i, j) -= fe_face_values.shape_value(i, q) *
                                       fe_face_values.shape_value(j, q) *
                                       fe_face_values.JxW(q);
                }
            }
          }
        }
        cell->get_dof_indices(local_dof_indices);
        affine_constraints.distribute_local_to_global(
            cell_matrix, local_dof_indices, sparse_matrix);
      }
    }

    thermal_operator.vmult(dst_1, src);
    sparse_matrix.vmult(dst_2, src);
    BOOST_TEST(dst_1 == dst_2, tt::per_element());
  }
}
