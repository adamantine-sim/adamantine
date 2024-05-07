/* Copyright (c) 2016 - 2024, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#include "MaterialStates.hh"
#include "ScanPath.hh"
#define BOOST_TEST_MODULE ThermalOperatorDevice

#include <Geometry.hh>
#include <GoldakHeatSource.hh>
#include <ThermalOperator.hh>
#include <ThermalOperatorDevice.hh>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/read_write_vector.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/numerics/matrix_tools.h>

#include <boost/property_tree/ptree.hpp>

#include "main.cc"

namespace tt = boost::test_tools;
namespace utf = boost::unit_test;

BOOST_AUTO_TEST_CASE(thermal_operator_dev, *utf::tolerance(1e-10))
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
  mat_prop_database.put("material_0.solid.thermal_conductivity_x", 10.);
  mat_prop_database.put("material_0.solid.thermal_conductivity_z", 10.);
  mat_prop_database.put("material_0.powder.thermal_conductivity_x", 10.);
  mat_prop_database.put("material_0.powder.thermal_conductivity_z", 10.);
  mat_prop_database.put("material_0.liquid.thermal_conductivity_x", 10.);
  mat_prop_database.put("material_0.liquid.thermal_conductivity_z", 10.);
  adamantine::MaterialProperty<2, 0, adamantine::SolidLiquidPowder,
                               dealii::MemorySpace::Default>
      mat_properties(communicator, geometry.get_triangulation(),
                     mat_prop_database);

  // Initialize the ThermalOperator
  adamantine::ThermalOperatorDevice<2, false, 0, 2,
                                    adamantine::SolidLiquidPowder,
                                    dealii::MemorySpace::Default>
      thermal_operator_dev(communicator, adamantine::BoundaryType::adiabatic,
                           mat_properties);
  thermal_operator_dev.compute_inverse_mass_matrix(dof_handler,
                                                   affine_constraints);
  std::vector<double> deposition_cos(
      geometry.get_triangulation().n_locally_owned_active_cells(), 1.);
  std::vector<double> deposition_sin(
      geometry.get_triangulation().n_locally_owned_active_cells(), 0.);
  thermal_operator_dev.reinit(dof_handler, affine_constraints, q_collection);
  thermal_operator_dev.set_material_deposition_orientation(deposition_cos,
                                                           deposition_sin);
  thermal_operator_dev.get_state_from_material_properties();
  BOOST_TEST(thermal_operator_dev.m() == 99);
  BOOST_TEST(thermal_operator_dev.m() == thermal_operator_dev.n());

  // Check matrix-vector multiplications
  dealii::LA::distributed::Vector<double, dealii::MemorySpace::Default> src;
  dealii::LA::distributed::Vector<double, dealii::MemorySpace::Default> dst_1;
  dealii::LA::distributed::Vector<double, dealii::MemorySpace::Default> dst_2;

  auto const &matrix_free = thermal_operator_dev.get_matrix_free();
  matrix_free.initialize_dof_vector(src);
  matrix_free.initialize_dof_vector(dst_1);
  matrix_free.initialize_dof_vector(dst_2);

  src = 1.;
  thermal_operator_dev.vmult(dst_1, src);
  BOOST_TEST(dst_1.l1_norm() == 0);

  thermal_operator_dev.Tvmult(dst_2, src);
  BOOST_TEST(dst_2.l1_norm() == dst_1.l1_norm());

  dst_2 = 1.;
  thermal_operator_dev.vmult_add(dst_2, src);
  thermal_operator_dev.vmult(dst_1, src);
  dst_1 += src;
  BOOST_TEST(dst_1.l1_norm() == dst_2.l1_norm());

  dst_1 = 1.;
  thermal_operator_dev.Tvmult_add(dst_1, src);
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
  mat_prop_database.put("material_0.solid.thermal_conductivity_x", 1.);
  mat_prop_database.put("material_0.solid.thermal_conductivity_z", 1.);
  mat_prop_database.put("material_0.powder.thermal_conductivity_x", 1.);
  mat_prop_database.put("material_0.powder.thermal_conductivity_z", 1.);
  mat_prop_database.put("material_0.liquid.thermal_conductivity_x", 1.);
  mat_prop_database.put("material_0.liquid.thermal_conductivity_z", 1.);
  adamantine::MaterialProperty<2, 3, adamantine::SolidLiquidPowder,
                               dealii::MemorySpace::Default>
      mat_properties(communicator, geometry.get_triangulation(),
                     mat_prop_database);

  // Initialize the ThermalOperator
  adamantine::ThermalOperatorDevice<2, false, 3, 2,
                                    adamantine::SolidLiquidPowder,
                                    dealii::MemorySpace::Default>
      thermal_operator_dev(communicator, adamantine::BoundaryType::adiabatic,
                           mat_properties);
  thermal_operator_dev.compute_inverse_mass_matrix(dof_handler,
                                                   affine_constraints);
  std::vector<double> deposition_cos(
      geometry.get_triangulation().n_locally_owned_active_cells(), 1.);
  std::vector<double> deposition_sin(
      geometry.get_triangulation().n_locally_owned_active_cells(), 0.);
  thermal_operator_dev.reinit(dof_handler, affine_constraints, q_collection);
  thermal_operator_dev.set_material_deposition_orientation(deposition_cos,
                                                           deposition_sin);
  thermal_operator_dev.get_state_from_material_properties();
  BOOST_TEST(thermal_operator_dev.m() == 99);
  BOOST_TEST(thermal_operator_dev.m() == thermal_operator_dev.n());

  // Build the matrix. This only works in serial.
  dealii::DynamicSparsityPattern dsp(dof_handler.n_dofs());
  dealii::DoFTools::make_sparsity_pattern(dof_handler, dsp, affine_constraints);
  dealii::SparsityPattern sparsity_pattern;
  sparsity_pattern.copy_from(dsp);
  dealii::SparseMatrix<double> sparse_matrix(sparsity_pattern);
  dealii::MatrixCreator::create_laplace_matrix(
      dof_handler, dealii::QGauss<2>(3), sparse_matrix);

  // Compare vmult using matrix free and building the matrix
  dealii::LA::distributed::Vector<double, dealii::MemorySpace::Default> src_dev;
  dealii::LA::distributed::Vector<double, dealii::MemorySpace::Default> dst_dev;

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
      BOOST_TEST(rw_vector[j] == -dst_host[j]);
  }
}

BOOST_AUTO_TEST_CASE(mf_spmv, *utf::tolerance(1.5e-12))
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
  mat_prop_database.put("material_0.solid.thermal_conductivity_x", 0.266);
  mat_prop_database.put("material_0.solid.thermal_conductivity_z", 0.266);
  mat_prop_database.put("material_0.powder.thermal_conductivity_x", 0.266);
  mat_prop_database.put("material_0.powder.thermal_conductivity_z", 0.266);
  mat_prop_database.put("material_0.liquid.thermal_conductivity_x", 0.266);
  mat_prop_database.put("material_0.liquid.thermal_conductivity_z", 0.266);
  adamantine::MaterialProperty<2, 4, adamantine::SolidLiquidPowder,
                               dealii::MemorySpace::Host>
      mat_properties_host(communicator, geometry.get_triangulation(),
                          mat_prop_database);
  adamantine::MaterialProperty<2, 4, adamantine::SolidLiquidPowder,
                               dealii::MemorySpace::Default>
      mat_properties(communicator, geometry.get_triangulation(),
                     mat_prop_database);

  // Create the heat sources
  boost::property_tree::ptree beam_database;
  beam_database.put("depth", 0.1);
  beam_database.put("absorption_efficiency", 0.1);
  beam_database.put("diameter", 1.0);
  beam_database.put("max_power", 10.);
  beam_database.put("scan_path_file", "scan_path.txt");
  beam_database.put("scan_path_file_format", "segment");
  adamantine::HeatSources<dealii::MemorySpace::Host, 2> heat_sources(
      beam_database);

  // Initialize the ThermalOperator
  adamantine::ThermalOperatorDevice<2, false, 4, 2,
                                    adamantine::SolidLiquidPowder,
                                    dealii::MemorySpace::Default>
      thermal_operator_dev(communicator, adamantine::BoundaryType::adiabatic,
                           mat_properties);
  thermal_operator_dev.compute_inverse_mass_matrix(dof_handler,
                                                   affine_constraints);
  std::vector<double> deposition_cos(
      geometry.get_triangulation().n_locally_owned_active_cells(), 1.);
  std::vector<double> deposition_sin(
      geometry.get_triangulation().n_locally_owned_active_cells(), 0.);
  thermal_operator_dev.reinit(dof_handler, affine_constraints, q_collection);
  thermal_operator_dev.set_material_deposition_orientation(deposition_cos,
                                                           deposition_sin);
  thermal_operator_dev.get_state_from_material_properties();
  BOOST_TEST(thermal_operator_dev.m() == thermal_operator_dev.n());

  adamantine::ThermalOperator<2, false, 4, 2, adamantine::SolidLiquidPowder,
                              dealii::MemorySpace::Host>
      thermal_operator_host(communicator, adamantine::BoundaryType::adiabatic,
                            mat_properties_host, heat_sources);
  thermal_operator_host.compute_inverse_mass_matrix(dof_handler,
                                                    affine_constraints);
  thermal_operator_host.reinit(dof_handler, affine_constraints, q_collection);
  thermal_operator_host.set_material_deposition_orientation(deposition_cos,
                                                            deposition_sin);
  thermal_operator_host.get_state_from_material_properties();

  // Compare vmult using matrix free and building the matrix
  dealii::LA::distributed::Vector<double, dealii::MemorySpace::Default> src_dev;
  dealii::LA::distributed::Vector<double, dealii::MemorySpace::Default> dst_dev;

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
      BOOST_TEST(rw_value == dst_host_value);
    }
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
  adamantine::Geometry<3> geometry(communicator, geometry_database);
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
  adamantine::MaterialProperty<3, 3, adamantine::SolidLiquidPowder,
                               dealii::MemorySpace::Default>
      mat_properties(communicator, geometry.get_triangulation(),
                     mat_prop_database);

  // Initialize the ThermalOperatorDevice
  adamantine::ThermalOperatorDevice<3, false, 3, 2,
                                    adamantine::SolidLiquidPowder,
                                    dealii::MemorySpace::Default>
      thermal_operator_dev(communicator, adamantine::BoundaryType::adiabatic,
                           mat_properties);
  double constexpr deposition_angle = M_PI / 6.;
  std::vector<double> deposition_cos(
      geometry.get_triangulation().n_locally_owned_active_cells(),
      std::cos(deposition_angle));
  std::vector<double> deposition_sin(
      geometry.get_triangulation().n_locally_owned_active_cells(),
      std::sin(deposition_angle));
  thermal_operator_dev.reinit(dof_handler, affine_constraints, q_collection);
  thermal_operator_dev.set_material_deposition_orientation(deposition_cos,
                                                           deposition_sin);
  thermal_operator_dev.get_state_from_material_properties();
  BOOST_TEST(thermal_operator_dev.m() == thermal_operator_dev.n());

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
  dealii::LA::distributed::Vector<double, dealii::MemorySpace::Default> src_dev;
  dealii::LA::distributed::Vector<double, dealii::MemorySpace::Default> dst_dev;
  thermal_operator_dev.initialize_dof_vector(src_dev);
  thermal_operator_dev.initialize_dof_vector(dst_dev);

  dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host> src_host(
      src_dev.get_partitioner());
  dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host> dst_host(
      dst_dev.get_partitioner());
  dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host>
      dst_dev_to_host(dst_dev.get_partitioner());

  for (unsigned int i = 0; i < thermal_operator_dev.m(); ++i)
  {
    src_host = 0.;
    src_host[i] = 1;
    src_dev.import(src_host, dealii::VectorOperation::insert);
    thermal_operator_dev.vmult(dst_dev, src_dev);
    sparse_matrix.vmult(dst_host, src_host);
    dst_dev_to_host.import(dst_dev, dealii::VectorOperation::insert);
    for (unsigned int j = 0; j < thermal_operator_dev.m(); ++j)
      BOOST_TEST(dst_dev_to_host[j] == -dst_host[j]);
  }
}
