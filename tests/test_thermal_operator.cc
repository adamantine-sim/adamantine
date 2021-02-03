/* Copyright (c) 2016 - 2020, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#define BOOST_TEST_MODULE ThermalOperator

#include <Geometry.hh>
#include <GoldakHeatSource.hh>
#include <ThermalOperator.hh>

#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/numerics/matrix_tools.h>

#include <boost/property_tree/ptree.hpp>

#include "main.cc"

BOOST_AUTO_TEST_CASE(thermal_operator)
{
  MPI_Comm communicator = MPI_COMM_WORLD;

  // Create the Geometry
  boost::property_tree::ptree geometry_database;
  geometry_database.put("import_mesh", false);
  geometry_database.put("length", 12);
  geometry_database.put("length_divisions", 4);
  geometry_database.put("height", 6);
  geometry_database.put("height_divisions", 5);
  adamantine::Geometry<2> geometry(communicator, geometry_database);
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
  mat_prop_database.put("material_0.solid.thermal_conductivity", 10.);
  mat_prop_database.put("material_0.powder.thermal_conductivity", 10.);
  mat_prop_database.put("material_0.liquid.thermal_conductivity", 10.);
  std::shared_ptr<adamantine::MaterialProperty<2>> mat_properties(
      new adamantine::MaterialProperty<2>(
          communicator, geometry.get_triangulation(), mat_prop_database));

  boost::property_tree::ptree beam_database;
  beam_database.put("depth", 0.1);
  beam_database.put("absorption_efficiency", 0.1);
  beam_database.put("diameter", 1.0);
  beam_database.put("max_power", 10.);
  beam_database.put("scan_path_file", "scan_path.txt");
  beam_database.put("scan_path_file_format", "segment");
  std::vector<std::shared_ptr<adamantine::HeatSource<2>>> heat_sources;
  heat_sources.resize(1);
  heat_sources[0] =
      std::make_shared<adamantine::GoldakHeatSource<2>>(beam_database);

  // Initialize the ThermalOperator
  adamantine::ThermalOperator<2, 2, dealii::MemorySpace::Host> thermal_operator(
      communicator, mat_properties, heat_sources);
  thermal_operator.reinit(dof_handler, affine_constraints, q_collection);
  thermal_operator.compute_inverse_mass_matrix(dof_handler, affine_constraints,
                                               fe_collection);
  dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host> dummy(
      thermal_operator.m());
  thermal_operator.evaluate_material_properties(dummy);
  BOOST_CHECK(thermal_operator.m() == 99);
  BOOST_CHECK(thermal_operator.m() == thermal_operator.n());

  // Check matrix-vector multiplications
  double const tolerance = 1e-15;
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
  BOOST_CHECK_CLOSE(dst_1.l1_norm(), 0., tolerance);

  thermal_operator.Tvmult(dst_2, src);
  BOOST_CHECK_CLOSE(dst_2.l1_norm(), dst_1.l1_norm(), tolerance);

  dst_2 = 1.;
  thermal_operator.vmult_add(dst_2, src);
  thermal_operator.vmult(dst_1, src);
  dst_1 += src;
  BOOST_CHECK_CLOSE(dst_1.l1_norm(), dst_2.l1_norm(), tolerance);

  dst_1 = 1.;
  thermal_operator.Tvmult_add(dst_1, src);
  BOOST_CHECK_CLOSE(dst_1.l1_norm(), dst_2.l1_norm(), tolerance);
}

BOOST_AUTO_TEST_CASE(spmv)
{
  MPI_Comm communicator = MPI_COMM_WORLD;

  // Create the Geometry
  boost::property_tree::ptree geometry_database;
  geometry_database.put("import_mesh", false);
  geometry_database.put("length", 12);
  geometry_database.put("length_divisions", 4);
  geometry_database.put("height", 6);
  geometry_database.put("height_divisions", 5);
  adamantine::Geometry<2> geometry(communicator, geometry_database);
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
  mat_prop_database.put("material_0.solid.thermal_conductivity", 1.);
  mat_prop_database.put("material_0.powder.thermal_conductivity", 1.);
  mat_prop_database.put("material_0.liquid.thermal_conductivity", 1.);
  mat_prop_database.put("material_0.solidus", 400.);
  mat_prop_database.put("material_0.liquidus", 500.);
  mat_prop_database.put("material_0.latent_heat", 0.);
  std::shared_ptr<adamantine::MaterialProperty<2>> mat_properties(
      new adamantine::MaterialProperty<2>(
          communicator, geometry.get_triangulation(), mat_prop_database));

  boost::property_tree::ptree beam_database;
  beam_database.put("depth", 0.1);
  beam_database.put("absorption_efficiency", 0.1);
  beam_database.put("diameter", 1.0);
  beam_database.put("max_power", 0.0);
  beam_database.put("scan_path_file", "scan_path.txt");
  beam_database.put("scan_path_file_format", "segment");
  std::vector<std::shared_ptr<adamantine::HeatSource<2>>> heat_sources;
  heat_sources.resize(1);
  heat_sources[0] =
      std::make_shared<adamantine::GoldakHeatSource<2>>(beam_database);

  // Initialize the ThermalOperator
  adamantine::ThermalOperator<2, 2, dealii::MemorySpace::Host> thermal_operator(
      communicator, mat_properties, heat_sources);
  thermal_operator.reinit(dof_handler, affine_constraints, q_collection);
  thermal_operator.compute_inverse_mass_matrix(dof_handler, affine_constraints,
                                               fe_collection);
  dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host> dummy(
      thermal_operator.m());
  // thermal_operator.evaluate_material_properties(dummy);
  thermal_operator.update_time_and_height(0.0, 6.0);
  thermal_operator.extract_stateful_material_properties(dummy);
  BOOST_CHECK(thermal_operator.m() == 99);
  BOOST_CHECK(thermal_operator.m() == thermal_operator.n());

  // Build the matrix. This only works in serial.
  dealii::DynamicSparsityPattern dsp(dof_handler.n_dofs());
  dealii::DoFTools::make_sparsity_pattern(dof_handler, dsp, affine_constraints);
  dealii::SparsityPattern sparsity_pattern;
  sparsity_pattern.copy_from(dsp);
  dealii::SparseMatrix<double> sparse_matrix(sparsity_pattern);
  dealii::MatrixCreator::create_laplace_matrix(
      dof_handler, dealii::QGauss<2>(3), sparse_matrix);

  // Compare vmult using matrix free and building the matrix
  double const tolerance = 1e-12;
  dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host> src;
  dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host> dst_1;
  dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host> dst_2;

  dealii::MatrixFree<2, double> const &matrix_free =
      thermal_operator.get_matrix_free();
  matrix_free.initialize_dof_vector(src);
  matrix_free.initialize_dof_vector(dst_1);
  matrix_free.initialize_dof_vector(dst_2);

  for (unsigned int i = 0; i < thermal_operator.m(); ++i)
  {
    src = 0.;
    src[i] = 1.;
    thermal_operator.vmult(dst_1, src);
    sparse_matrix.vmult(dst_2, src);
    for (unsigned int j = 0; j < thermal_operator.m(); ++j)
      BOOST_CHECK_CLOSE(dst_1[j], -dst_2[j], tolerance);
  }
}
