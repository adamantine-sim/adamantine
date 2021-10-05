/* Copyright (c) 2017 - 2021, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#define BOOST_TEST_MODULE ImplicitOperator

#include <Geometry.hh>
#include <GoldakHeatSource.hh>
#include <ImplicitOperator.hh>
#include <ThermalOperator.hh>

#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/hp/fe_collection.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/numerics/matrix_tools.h>

#include <boost/property_tree/ptree.hpp>

#include "main.cc"

BOOST_AUTO_TEST_CASE(implicit_operator)
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
  std::shared_ptr<adamantine::ThermalOperator<2, 2, dealii::MemorySpace::Host>>
      thermal_operator = std::make_shared<
          adamantine::ThermalOperator<2, 2, dealii::MemorySpace::Host>>(
          communicator, adamantine::BoundaryType::adiabatic, mat_properties,
          heat_sources);
  thermal_operator->reinit(dof_handler, affine_constraints, q_collection);
  thermal_operator->compute_inverse_mass_matrix(dof_handler, affine_constraints,
                                                fe_collection);
  dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host> dummy(
      thermal_operator->m());
  thermal_operator->get_state_from_material_properties();

  // Initialize the ImplicitOperator
  adamantine::ImplicitOperator<dealii::MemorySpace::Host> implicit_operator(
      thermal_operator, false);
  BOOST_CHECK(implicit_operator.m() == 99);
  BOOST_CHECK(implicit_operator.n() == 99);
  BOOST_CHECK_THROW(implicit_operator.Tvmult(dummy, dummy),
                    adamantine::NotImplementedExc);
  BOOST_CHECK_THROW(implicit_operator.vmult_add(dummy, dummy),
                    adamantine::NotImplementedExc);
  BOOST_CHECK_THROW(implicit_operator.Tvmult_add(dummy, dummy),
                    adamantine::NotImplementedExc);

  // Initialize the ImplicitOperator with JFNK
  adamantine::ImplicitOperator<dealii::MemorySpace::Host>
      implicit_operator_jfnk(thermal_operator, true);

  // Check that ImplicitOperator with and without JFNK give the same results.
  unsigned int const size = thermal_operator->m();
  dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host> source(
      size);
  std::shared_ptr<
      dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host>>
      inverse_mass_matrix(
          new dealii::LA::distributed::Vector<double,
                                              dealii::MemorySpace::Host>(size));
  dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host> dst(size);
  dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host> dst_jfnk(
      size);
  for (unsigned int i = 0; i < size; ++i)
  {
    source[i] = 1.;
    (*inverse_mass_matrix)[i] = 1.;
  }

  implicit_operator.set_tau(1.);
  implicit_operator_jfnk.set_tau(1.);

  implicit_operator.set_inverse_mass_matrix(inverse_mass_matrix);
  implicit_operator_jfnk.set_inverse_mass_matrix(inverse_mass_matrix);

  implicit_operator.vmult(dst, source);
  implicit_operator_jfnk.vmult(dst_jfnk, source);

  double const tolerance = 1e-7;
  BOOST_CHECK_CLOSE(dst.l2_norm(), dst_jfnk.l2_norm(), tolerance);
}
