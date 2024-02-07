/* Copyright (c) 2016 - 2024, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#define BOOST_TEST_MODULE Integration_Data_Assimilation

#include "../application/adamantine.hh"

#include <boost/property_tree/info_parser.hpp>

#include <filesystem>
#include <fstream>

#include "main.cc"

namespace tt = boost::test_tools;

double integration_da(MPI_Comm communicator, bool l2_norm)
{
  std::vector<adamantine::Timer> timers;
  initialize_timers(communicator, timers);

  // Read the input.
  std::string const filename = "bare_plate_L_da.info";
  adamantine::ASSERT_THROW(std::filesystem::exists(filename) == true,
                           "The file " + filename + " does not exist.");
  boost::property_tree::ptree database;
  boost::property_tree::info_parser::read_info(filename, database);

  // Run the simulation
  auto result = run_ensemble<3, dealii::MemorySpace::Host>(communicator,
                                                           database, timers);

  if (l2_norm)
  {
    double norm = -1.;
    for (auto const &r : result)
    {
      norm = std::max(r.l2_norm(), norm);
    }
    return dealii::Utilities::MPI::max(norm, communicator);
  }

  // Three ensemble members expected
  unsigned int local_result_size = result.size();
  unsigned int global_result_size =
      dealii::Utilities::MPI::sum(local_result_size, communicator);
  BOOST_TEST(global_result_size == 3);

  // Get the average minimum value for each ensemble member, which is very close
  // to the initial temperature
  double sum = 0.0;
  for (unsigned int member = 0; member < local_result_size; ++member)
  {
    double min_val = std::numeric_limits<double>::max();
    for (unsigned int i = 0;
         i < result.at(member).block(0).locally_owned_size(); ++i)
    {
      if (result.at(member).block(0).local_element(i) < min_val)
        min_val = result.at(member).block(0).local_element(i);
    }
    sum += min_val;
  }
  double partial_average_minimum_value = sum / global_result_size;
  double average_minimum_value =
      dealii::Utilities::MPI::sum(partial_average_minimum_value, communicator);

  // Based on the experimental data, the expected temperature is ~200.0
  BOOST_TEST(average_minimum_value >= 200.0);
  BOOST_TEST(average_minimum_value < 300.0);
  MPI_Barrier(communicator);

  return average_minimum_value;
}

BOOST_AUTO_TEST_CASE(integration_data_assimilation)
{
  bool l2_norm = dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) > 3
                     ? true
                     : false;
  double result_world = integration_da(MPI_COMM_WORLD, l2_norm);
  double result_self = integration_da(MPI_COMM_SELF, l2_norm);

  BOOST_TEST(result_world == result_self, tt::tolerance(1e-12));
}

double integration_da_point_cloud_add_mat(MPI_Comm communicator, bool l2_norm)
{
  std::vector<adamantine::Timer> timers;
  initialize_timers(communicator, timers);

  // Read the input.
  std::string const filename = "integration_da_add_material.info";
  adamantine::ASSERT_THROW(std::filesystem::exists(filename) == true,
                           "The file " + filename + " does not exist.");
  boost::property_tree::ptree database;
  boost::property_tree::info_parser::read_info(filename, database);

  // Run the simulation
  auto result = run_ensemble<3, dealii::MemorySpace::Host>(communicator,
                                                           database, timers);

  if (l2_norm)
  {
    double norm = -1.;
    for (auto const &r : result)
    {
      norm = std::max(r.l2_norm(), norm);
    }
    return dealii::Utilities::MPI::max(norm, communicator);
  }

  // Three ensemble members expected
  unsigned int local_result_size = result.size();
  unsigned int global_result_size =
      dealii::Utilities::MPI::sum(local_result_size, communicator);
  BOOST_TEST(global_result_size == 3);

  // Get the average minimum value for each ensemble member, which is very close
  // to the initial temperature
  double sum = 0.0;
  for (unsigned int member = 0; member < local_result_size; ++member)
  {
    double min_val = std::numeric_limits<double>::max();
    for (unsigned int i = 0;
         i < result.at(member).block(0).locally_owned_size(); ++i)
    {
      if (result.at(member).block(0).local_element(i) < min_val)
        min_val = result.at(member).block(0).local_element(i);
    }
    sum += min_val;
  }
  double partial_average_minimum_value = sum / global_result_size;
  double average_minimum_value =
      dealii::Utilities::MPI::sum(partial_average_minimum_value, communicator);

  // Based on the experimental data, the expected temperature is ~200.0
  BOOST_CHECK(average_minimum_value >= 200.0);
  BOOST_CHECK(average_minimum_value < 300.0);
  MPI_Barrier(communicator);

  return average_minimum_value;
}

BOOST_AUTO_TEST_CASE(integration_3D_da_point_cloud_add_material)
{
  /*
   * This integration test checks that the data assimilation using point cloud
   * data works while adding material.
   */
  bool l2_norm = dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) > 3
                     ? true
                     : false;
  double result_world =
      integration_da_point_cloud_add_mat(MPI_COMM_WORLD, l2_norm);
  double result_self =
      integration_da_point_cloud_add_mat(MPI_COMM_SELF, l2_norm);

  BOOST_TEST(result_world == result_self, tt::tolerance(1e-12));
}

double integration_da_ray_add_mat(MPI_Comm communicator, bool l2_norm)
{
  std::vector<adamantine::Timer> timers;
  initialize_timers(communicator, timers);

  // Read the input.
  std::string const filename = "integration_da_add_material.info";
  adamantine::ASSERT_THROW(std::filesystem::exists(filename) == true,
                           "The file " + filename + " does not exist.");
  boost::property_tree::ptree database;

  boost::property_tree::info_parser::read_info(filename, database);

  // Change experiment file name
  database.put("experiment.file",
               "integration_da_add_material_expt_ray_#camera_#frame.csv");
  // Change experiment data format to ray
  database.put("experiment.format", "ray");

  // Run the simulation
  auto result = run_ensemble<3, dealii::MemorySpace::Host>(communicator,
                                                           database, timers);

  if (l2_norm)
  {
    double norm = -1.;
    for (auto const &r : result)
    {
      norm = std::max(r.l2_norm(), norm);
    }
    return dealii::Utilities::MPI::max(norm, communicator);
  }

  // Three ensemble members expected
  unsigned int local_result_size = result.size();
  unsigned int global_result_size =
      dealii::Utilities::MPI::sum(local_result_size, communicator);
  BOOST_TEST(global_result_size == 3);

  // Get the average minimum value for each ensemble member, which is very close
  // to the initial temperature
  double sum = 0.0;
  for (unsigned int member = 0; member < result.size(); ++member)
  {
    double min_val = std::numeric_limits<double>::max();
    for (unsigned int i = 0;
         i < result.at(member).block(0).locally_owned_size(); ++i)
    {
      if (result.at(member).block(0).local_element(i) < min_val)
        min_val = result.at(member).block(0).local_element(i);
    }
    sum += min_val;
  }
  double partial_average_minimum_value = sum / global_result_size;
  double average_minimum_value =
      dealii::Utilities::MPI::sum(partial_average_minimum_value, communicator);

  // Based on the experimental data, the expected temperature is ~200.0
  BOOST_CHECK(average_minimum_value >= 200.0);
  BOOST_CHECK(average_minimum_value < 300.0);
  MPI_Barrier(communicator);

  return average_minimum_value;
}

BOOST_AUTO_TEST_CASE(integration_3D_da_ray_add_material)
{
  /*
   * This integration test checks that the data assimilation using point cloud
   * data works while adding material.
   */
  bool l2_norm = dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) > 3
                     ? true
                     : false;
  double result_world = integration_da_ray_add_mat(MPI_COMM_WORLD, l2_norm);
  double result_self = integration_da_ray_add_mat(MPI_COMM_SELF, l2_norm);

  BOOST_TEST(result_world == result_self, tt::tolerance(1e-12));
}
