/* Copyright (c) 2016, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#define BOOST_TEST_MODULE PostProcessor

#include "main.cc"

#include "Geometry.hh"
#include "MaterialProperty.hh"
#include "PostProcessor.hh"
#include "ThermalOperator.hh"
#include <boost/filesystem.hpp>

BOOST_AUTO_TEST_CASE(post_processor)
{
  boost::mpi::communicator communicator;
  boost::property_tree::ptree mat_prop_database;
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
  std::shared_ptr<adamantine::MaterialProperty> mat_properties(
      new adamantine::MaterialProperty(mat_prop_database));

  // Create the Geometry
  boost::property_tree::ptree geometry_database;
  geometry_database.put("length", 12);
  geometry_database.put("length_divisions", 4);
  geometry_database.put("height", 6);
  geometry_database.put("height_divisions", 5);
  adamantine::Geometry<2> geometry(communicator, geometry_database);
  // Create the DoFHandler
  dealii::FE_Q<2> fe(2);
  dealii::DoFHandler<2> dof_handler(geometry.get_triangulation());
  dof_handler.distribute_dofs(fe);
  dealii::ConstraintMatrix constraint_matrix;
  constraint_matrix.close();
  dealii::QGauss<1> quad(3);
  //
  // Initialize the ThermalOperator
  adamantine::ThermalOperator<2, 2, double> thermal_operator(communicator,
                                                             mat_properties);
  thermal_operator.setup_dofs(dof_handler, constraint_matrix, quad);
  thermal_operator.reinit(dof_handler, constraint_matrix);

  // Create the PostProcessor
  boost::property_tree::ptree post_processor_database;
  post_processor_database.put("file_name", "test");
  adamantine::PostProcessor<2> post_processor(
      communicator, post_processor_database, dof_handler, mat_properties);
  dealii::LA::distributed::Vector<double> src;
  dealii::MatrixFree<2, double> const &matrix_free =
      thermal_operator.get_matrix_free();
  matrix_free.initialize_dof_vector(src);
  for (unsigned int i = 0; i < src.size(); ++i)
    src[i] = 1.;

  post_processor.output_pvtu(1, 0, 0., src);
  post_processor.output_pvtu(1, 1, 0.1, src);
  post_processor.output_pvtu(1, 2, 0.2, src);
  post_processor.output_pvd();

  // Check that the files exist
  BOOST_CHECK(boost::filesystem::exists("test.pvd"));
  BOOST_CHECK(boost::filesystem::exists("test.01.000000.pvtu"));
  BOOST_CHECK(boost::filesystem::exists("test.01.000001.pvtu"));
  BOOST_CHECK(boost::filesystem::exists("test.01.000002.pvtu"));
  BOOST_CHECK(boost::filesystem::exists("test.01.000000.000000.vtu"));
  BOOST_CHECK(boost::filesystem::exists("test.01.000001.000000.vtu"));
  BOOST_CHECK(boost::filesystem::exists("test.01.000002.000000.vtu"));

  // Delete the files
  std::remove("test.pvd");
  std::remove("test.01.000000.pvtu");
  std::remove("test.01.000001.pvtu");
  std::remove("test.01.000002.pvtu");
  std::remove("test.01.000000.000000.vtu");
  std::remove("test.01.000001.000000.vtu");
  std::remove("test.01.000002.000000.vtu");
}
