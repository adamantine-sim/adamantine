/* Copyright (c) 2016 - 2020, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#define BOOST_TEST_MODULE ThermalOperator

#include <Geometry.hh>
#include <ThermalOperator.hh>
#include <ThermalOperatorDevice.hh>

#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/read_write_vector.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/numerics/matrix_tools.h>

#include <boost/property_tree/ptree.hpp>

#include "main.cc"

BOOST_AUTO_TEST_CASE(thermal_operator_dev)
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

  // Initialize the ThermalOperator
  adamantine::ThermalOperatorDevice<2, 2, dealii::MemorySpace::CUDA>
      thermal_operator_dev(communicator, mat_properties);
  thermal_operator_dev.compute_inverse_mass_matrix(
      dof_handler, affine_constraints, fe_collection);
  thermal_operator_dev.reinit(dof_handler, affine_constraints, q_collection);
  dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host> dummy(
      thermal_operator_dev.m());
  thermal_operator_dev.evaluate_material_properties(dummy);
  BOOST_CHECK(thermal_operator_dev.m() == 99);
  BOOST_CHECK(thermal_operator_dev.m() == thermal_operator_dev.n());

  // Check matrix-vector multiplications
  double const tolerance = 1e-10;
  dealii::LA::distributed::Vector<double, dealii::MemorySpace::CUDA> src;
  dealii::LA::distributed::Vector<double, dealii::MemorySpace::CUDA> dst_1;
  dealii::LA::distributed::Vector<double, dealii::MemorySpace::CUDA> dst_2;

  auto const &matrix_free = thermal_operator_dev.get_matrix_free();
  matrix_free.initialize_dof_vector(src);
  matrix_free.initialize_dof_vector(dst_1);
  matrix_free.initialize_dof_vector(dst_2);

  src = 1.;
  thermal_operator_dev.vmult(dst_1, src);
  BOOST_CHECK_CLOSE(dst_1.l1_norm() + 1, 1., tolerance);

  thermal_operator_dev.Tvmult(dst_2, src);
  BOOST_CHECK_CLOSE(dst_2.l1_norm(), dst_1.l1_norm(), tolerance);

  dst_2 = 1.;
  thermal_operator_dev.vmult_add(dst_2, src);
  thermal_operator_dev.vmult(dst_1, src);
  dst_1 += src;
  BOOST_CHECK_CLOSE(dst_1.l1_norm(), dst_2.l1_norm(), tolerance);

  dst_1 = 1.;
  thermal_operator_dev.Tvmult_add(dst_1, src);
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
  std::shared_ptr<adamantine::MaterialProperty<2>> mat_properties(
      new adamantine::MaterialProperty<2>(
          communicator, geometry.get_triangulation(), mat_prop_database));

  // Initialize the ThermalOperator
  adamantine::ThermalOperatorDevice<2, 2, dealii::MemorySpace::CUDA>
      thermal_operator_dev(communicator, mat_properties);
  thermal_operator_dev.compute_inverse_mass_matrix(
      dof_handler, affine_constraints, fe_collection);
  thermal_operator_dev.reinit(dof_handler, affine_constraints, q_collection);
  dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host> dummy(
      thermal_operator_dev.m());
  thermal_operator_dev.evaluate_material_properties(dummy);
  BOOST_CHECK(thermal_operator_dev.m() == 99);
  BOOST_CHECK(thermal_operator_dev.m() == thermal_operator_dev.n());

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
  dealii::LA::distributed::Vector<double, dealii::MemorySpace::CUDA> src_dev;
  dealii::LA::distributed::Vector<double, dealii::MemorySpace::CUDA> dst_dev;

  dealii::CUDAWrappers::MatrixFree<2, double> const &matrix_free =
      thermal_operator_dev.get_matrix_free();
  matrix_free.initialize_dof_vector(src_dev);
  matrix_free.initialize_dof_vector(dst_dev);

  dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host> src_host(
      src_dev.get_partitioner());
  dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host> dst_host(
      dst_dev.get_partitioner());

  for (unsigned int i = 0; i < thermal_operator_dev.m(); ++i)
  {
    dealii::LinearAlgebra::ReadWriteVector<double> rw_vector(
        thermal_operator_dev.m());
    rw_vector = 0.;
    rw_vector[i] = 1.;
    src_dev.import(rw_vector, dealii::VectorOperation::insert);

    thermal_operator_dev.vmult(dst_dev, src_dev);
    rw_vector.import(dst_dev, dealii::VectorOperation::insert);
    src_host = 0.;
    src_host[i] = 1.;
    sparse_matrix.vmult(dst_host, src_host);
    for (unsigned int j = 0; j < thermal_operator_dev.m(); ++j)
      BOOST_CHECK_CLOSE(rw_vector[j], -dst_host[j], tolerance);
  }
}

BOOST_AUTO_TEST_CASE(mf_spmv)
{
  MPI_Comm communicator = MPI_COMM_WORLD;

  // Create the Geometry
  boost::property_tree::ptree geometry_database;
  geometry_database.put("length", 2e-2);
  geometry_database.put("import_mesh", false);
  geometry_database.put("length_divisions", 10);
  geometry_database.put("height", 1e-2);
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
  mat_prop_database.put("material_0.solid.density", 7541.);
  mat_prop_database.put("material_0.powder.density", 7541.);
  mat_prop_database.put("material_0.liquid.density", 7541.);
  mat_prop_database.put("material_0.solid.specific_heat", 600.);
  mat_prop_database.put("material_0.powder.specific_heat", 600.);
  mat_prop_database.put("material_0.liquid.specific_heat", 600.);
  mat_prop_database.put("material_0.solid.thermal_conductivity", 0.266);
  mat_prop_database.put("material_0.powder.thermal_conductivity", 0.266);
  mat_prop_database.put("material_0.liquid.thermal_conductivity", 0.266);
  std::shared_ptr<adamantine::MaterialProperty<2>> mat_properties(
      new adamantine::MaterialProperty<2>(
          communicator, geometry.get_triangulation(), mat_prop_database));

  // Initialize the ThermalOperator
  adamantine::ThermalOperatorDevice<2, 2, dealii::MemorySpace::CUDA>
      thermal_operator_dev(communicator, mat_properties);
  thermal_operator_dev.compute_inverse_mass_matrix(
      dof_handler, affine_constraints, fe_collection);
  thermal_operator_dev.reinit(dof_handler, affine_constraints, q_collection);
  dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host> dummy(
      thermal_operator_dev.m());
  thermal_operator_dev.evaluate_material_properties(dummy);
  // BOOST_CHECK(thermal_operator_dev.m() == 99);
  BOOST_CHECK(thermal_operator_dev.m() == thermal_operator_dev.n());

  adamantine::ThermalOperator<2, 2, dealii::MemorySpace::Host>
      thermal_operator_host(communicator, mat_properties);
  thermal_operator_host.compute_inverse_mass_matrix(
      dof_handler, affine_constraints, fe_collection);
  thermal_operator_host.reinit(dof_handler, affine_constraints, q_collection);
  thermal_operator_host.evaluate_material_properties(dummy);

  // Compare vmult using matrix free and building the matrix
  double const tolerance = 1e-12;
  dealii::LA::distributed::Vector<double, dealii::MemorySpace::CUDA> src_dev;
  dealii::LA::distributed::Vector<double, dealii::MemorySpace::CUDA> dst_dev;

  dealii::CUDAWrappers::MatrixFree<2, double> const &matrix_free =
      thermal_operator_dev.get_matrix_free();
  matrix_free.initialize_dof_vector(src_dev);
  matrix_free.initialize_dof_vector(dst_dev);

  dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host> src_host(
      src_dev.get_partitioner());
  dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host> dst_host(
      dst_dev.get_partitioner());

  for (unsigned int i = 0; i < thermal_operator_dev.m(); ++i)
  {
    src_host = 0.;
    src_host[i] = 1.;
    dst_host = 0.;
    dst_host[i] = 1.;
    src_dev.import(src_host, dealii::VectorOperation::insert);
    dst_dev.import(dst_host, dealii::VectorOperation::insert);

    thermal_operator_host.vmult_add(dst_host, src_host);
    thermal_operator_dev.vmult_add(dst_dev, src_dev);

    dealii::LinearAlgebra::ReadWriteVector<double> rw_vector(
        thermal_operator_dev.m());
    rw_vector.import(dst_dev, dealii::VectorOperation::insert);
    for (unsigned int j = 0; j < thermal_operator_dev.m(); ++j)
    {
      double rw_value = std::abs(rw_vector[j]) > 1e-15 ? rw_vector[j] : 0.;
      double dst_host_value = std::abs(dst_host[j]) > 1e-15 ? dst_host[j] : 0.;
      BOOST_CHECK_CLOSE(rw_value, dst_host_value, tolerance);
    }
  }
}
