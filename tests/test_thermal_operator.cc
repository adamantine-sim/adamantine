/* Copyright (c) 2016 - 2021, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#define BOOST_TEST_MODULE ThermalOperator

#include <Geometry.hh>
#include <ThermalOperator.hh>

#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
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

  // Initialize the ThermalOperator
  adamantine::ThermalOperator<2, 2, dealii::MemorySpace::Host> thermal_operator(
      communicator, mat_properties, adamantine::BoundaryType::adiabatic);
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
  std::shared_ptr<adamantine::MaterialProperty<2>> mat_properties(
      new adamantine::MaterialProperty<2>(
          communicator, geometry.get_triangulation(), mat_prop_database));

  // Initialize the ThermalOperator
  adamantine::ThermalOperator<2, 2, dealii::MemorySpace::Host> thermal_operator(
      communicator, mat_properties, adamantine::BoundaryType::adiabatic);
  thermal_operator.reinit(dof_handler, affine_constraints, q_collection);
  thermal_operator.compute_inverse_mass_matrix(dof_handler, affine_constraints,
                                               fe_collection);
  dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host> dummy(
      thermal_operator.m());
  thermal_operator.evaluate_material_properties(dummy);
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
    src[i] = 1;
    thermal_operator.vmult(dst_1, src);
    sparse_matrix.vmult(dst_2, src);
    for (unsigned int j = 0; j < thermal_operator.m(); ++j)
      BOOST_CHECK_CLOSE(dst_1[j], -dst_2[j], tolerance);
  }
}

BOOST_AUTO_TEST_CASE(spmv_rad)
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
  mat_prop_database.put("material_0.solid.emissivity", 1.);
  mat_prop_database.put("material_0.powder.emissivity", 1.);
  mat_prop_database.put("material_0.liquid.emissivity", 1.);
  mat_prop_database.put("material_0.solid.radiation_heat_transfer_coef", 1.);
  mat_prop_database.put("material_0.powder.radiation_heat_transfer_coef", 1.);
  mat_prop_database.put("material_0.liquid.radiation_heat_transfer_coef", 1.);
  mat_prop_database.put("material_0.solid.convection_heat_transfer_coef", 1.);
  mat_prop_database.put("material_0.powder.convection_heat_transfer_coef", 1.);
  mat_prop_database.put("material_0.liquid.convection_heat_transfer_coef", 1.);
  mat_prop_database.put("material_0.radiation_temperature_infty", 0.5);
  mat_prop_database.put("material_0.convection_temperature_infty", 0.5);
  std::shared_ptr<adamantine::MaterialProperty<2>> mat_properties(
      new adamantine::MaterialProperty<2>(
          communicator, geometry.get_triangulation(), mat_prop_database));

  // Initialize the ThermalOperator
  adamantine::ThermalOperator<2, 2, dealii::MemorySpace::Host> thermal_operator(
      communicator, mat_properties, adamantine::BoundaryType::radiative);
  thermal_operator.reinit(dof_handler, affine_constraints, q_collection);
  thermal_operator.compute_inverse_mass_matrix(dof_handler, affine_constraints,
                                               fe_collection);
  dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host>
      temperature(thermal_operator.m());
  for (unsigned int i = 0; i < temperature.local_size(); ++i)
  {
    temperature.local_element(i) = 1.;
  }
  thermal_operator.evaluate_material_properties(temperature);
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
  // Take care of the boundary condition
  unsigned int const fe_degree = 2;
  dealii::FE_Q<2> fe(fe_degree);
  dealii::QGauss<1> face_quadrature_formula(fe_degree + 1);
  unsigned int const n_face_q_points = face_quadrature_formula.size();
  dealii::FEFaceValues<2> fe_face_values(fe, face_quadrature_formula,
                                         dealii::update_values |
                                             dealii::update_quadrature_points |
                                             dealii::update_JxW_values);
  double const heat_transfer_coeff =
      1. * adamantine::Constant::stefan_boltzmann * 1.5 * 1.25;
  std::cout << heat_transfer_coeff << std::endl;
  unsigned int const dofs_per_cell = fe.n_dofs_per_cell();
  dealii::FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  std::vector<dealii::types::global_dof_index> local_dof_indices(dofs_per_cell);
  for (auto const &cell : dof_handler.active_cell_iterators())
  {
    if (cell->at_boundary())
    {
      cell_matrix = 0.;
      for (auto const &face : cell->face_iterators())
      {
        if (face->at_boundary())
        {
          fe_face_values.reinit(cell, face);
          for (unsigned int q = 0; q < n_face_q_points; ++q)
            for (unsigned i = 0; i < dofs_per_cell; ++i)
              for (unsigned j = 0; j < dofs_per_cell; ++j)
              {
                cell_matrix(i, j) +=
                    heat_transfer_coeff * fe_face_values.shape_value(i, q) *
                    fe_face_values.shape_value(j, q) * fe_face_values.JxW(q);
              }
        }
      }
      cell->get_dof_indices(local_dof_indices);
      affine_constraints.distribute_local_to_global(
          cell_matrix, local_dof_indices, sparse_matrix);
    }
  }

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
    src[i] = 1;
    thermal_operator.vmult(dst_1, src);
    sparse_matrix.vmult(dst_2, src);
    for (unsigned int j = 0; j < thermal_operator.m(); ++j)
      BOOST_CHECK_CLOSE(dst_1[j], -dst_2[j], tolerance);
  }
}
