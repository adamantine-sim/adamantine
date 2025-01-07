/* SPDX-FileCopyrightText: Copyright (c) 2021 - 2025, the adamantine authors.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include <ensemble_management.hh>

#include <deal.II/base/mpi.h>

#define BOOST_TEST_MODULE EnsembleManagement

#include <filesystem>

#include "main.cc"

namespace utf = boost::unit_test;

BOOST_AUTO_TEST_CASE(database_ensemble)
{
  MPI_Comm communicator = MPI_COMM_WORLD;

  unsigned int const global_ensemble_size = 6;
  unsigned int const n_procs =
      dealii::Utilities::MPI::n_mpi_processes(communicator);
  unsigned int const rank =
      dealii::Utilities::MPI::this_mpi_process(communicator);
  unsigned int const local_ensemble_size = global_ensemble_size / n_procs;
  unsigned int const fist_local_member = rank * local_ensemble_size;

  boost::property_tree::ptree database;
  database.put("alpha.beta", 100.);
  database.put("ensemble.alpha.beta_stddev", 10.);
  database.put("gamma.delta", 200.);
  database.put("ensemble.gamma.delta_stddev", 20.);
  database.put("post_processor.filename_prefix", "ensemble_management");

  auto database_ensemble = adamantine::create_database_ensemble(
      database, communicator, fist_local_member, local_ensemble_size,
      global_ensemble_size);

  if (n_procs == 1)
  {
    BOOST_TEST(database_ensemble[0].get<double>("alpha.beta") ==
               101.34529658472329);
    BOOST_TEST(database_ensemble[1].get<double>("alpha.beta") ==
               98.536182188102771);
    BOOST_TEST(database_ensemble[2].get<double>("alpha.beta") ==
               104.60650182383064);
    BOOST_TEST(database_ensemble[3].get<double>("alpha.beta") ==
               81.286156895893981);
    BOOST_TEST(database_ensemble[4].get<double>("alpha.beta") ==
               101.63711684234313);
    BOOST_TEST(database_ensemble[5].get<double>("alpha.beta") ==
               97.857466117910562);

    BOOST_TEST(database_ensemble[0].get<double>("gamma.delta") ==
               205.97190519115932);
    BOOST_TEST(database_ensemble[1].get<double>("gamma.delta") ==
               183.44111635285606);
    BOOST_TEST(database_ensemble[2].get<double>("gamma.delta") ==
               200.20430868691838);
    BOOST_TEST(database_ensemble[3].get<double>("gamma.delta") ==
               221.10932887760762);
    BOOST_TEST(database_ensemble[4].get<double>("gamma.delta") ==
               189.06317994350883);
    BOOST_TEST(database_ensemble[5].get<double>("gamma.delta") ==
               223.49135354909728);

    std::filesystem::remove("ensemble_management_m0_data.txt");
    std::filesystem::remove("ensemble_management_m1_data.txt");
    std::filesystem::remove("ensemble_management_m2_data.txt");
    std::filesystem::remove("ensemble_management_m3_data.txt");
    std::filesystem::remove("ensemble_management_m4_data.txt");
    std::filesystem::remove("ensemble_management_m5_data.txt");
  }
  else
  {
    if (rank == 0)
    {
      BOOST_TEST(database_ensemble[0].get<double>("alpha.beta") ==
                 101.34529658472329);
      BOOST_TEST(database_ensemble[1].get<double>("alpha.beta") ==
                 98.536182188102771);
      BOOST_TEST(database_ensemble[2].get<double>("alpha.beta") ==
                 104.60650182383064);

      BOOST_TEST(database_ensemble[0].get<double>("gamma.delta") ==
                 205.97190519115932);
      BOOST_TEST(database_ensemble[1].get<double>("gamma.delta") ==
                 183.44111635285606);
      BOOST_TEST(database_ensemble[2].get<double>("gamma.delta") ==
                 200.20430868691838);

      std::filesystem::remove("ensemble_management_m0_data.txt");
      std::filesystem::remove("ensemble_management_m1_data.txt");
      std::filesystem::remove("ensemble_management_m2_data.txt");
    }
    else
    {
      BOOST_TEST(database_ensemble[0].get<double>("alpha.beta") ==
                 81.286156895893981);
      BOOST_TEST(database_ensemble[1].get<double>("alpha.beta") ==
                 101.63711684234313);
      BOOST_TEST(database_ensemble[2].get<double>("alpha.beta") ==
                 97.857466117910562);

      BOOST_TEST(database_ensemble[0].get<double>("gamma.delta") ==
                 221.10932887760762);
      BOOST_TEST(database_ensemble[1].get<double>("gamma.delta") ==
                 189.06317994350883);
      BOOST_TEST(database_ensemble[2].get<double>("gamma.delta") ==
                 223.49135354909728);

      std::filesystem::remove("ensemble_management_m3_data.txt");
      std::filesystem::remove("ensemble_management_m4_data.txt");
      std::filesystem::remove("ensemble_management_m5_data.txt");
    }
  }
}
