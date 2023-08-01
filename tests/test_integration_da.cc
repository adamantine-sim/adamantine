/* Copyright (c) 2016 - 2023, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#define BOOST_TEST_MODULE Integration_Data_Assimilation

#include "../application/adamantine.hh"

#include <filesystem>
#include <fstream>

#include "main.cc"

BOOST_AUTO_TEST_CASE(integration_data_assimilation)
{

  MPI_Comm communicator = MPI_COMM_WORLD;

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

  // Three ensemble members expected
  BOOST_TEST(result.size() == 3);

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
  double average_minimum_value = sum / result.size();

  // Based on the experimental data, the expected temperature is ~200.0
  BOOST_TEST(average_minimum_value >= 200.0);
  BOOST_TEST(average_minimum_value < 300.0);
}

BOOST_AUTO_TEST_CASE(integration_3D_da_point_cloud_add_material)
{
  /*
   * This integration test checks that the data assimilation using point cloud
   * data works while adding material.
   */
  MPI_Comm communicator = MPI_COMM_WORLD;

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

  // Three ensemble members expected
  BOOST_TEST(result.size() == 3);

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
  double average_minimum_value = sum / result.size();

  // Based on the experimental data, the expected temperature is ~200.0
  BOOST_CHECK(average_minimum_value >= 200.0);
  BOOST_CHECK(average_minimum_value < 300.0);
}

BOOST_AUTO_TEST_CASE(integration_3D_da_ray_add_material)
{
  /*
   * This integration test checks that the data assimilation using point cloud
   * data works while adding material.
   */
  MPI_Comm communicator = MPI_COMM_WORLD;

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

  // Three ensemble members expected
  BOOST_TEST(result.size() == 3);

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
  double average_minimum_value = sum / result.size();

  // Based on the experimental data, the expected temperature is ~200.0
  BOOST_CHECK(average_minimum_value >= 200.0);
  BOOST_CHECK(average_minimum_value < 300.0);
}
