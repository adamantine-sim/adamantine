/* SPDX-FileCopyrightText: Copyright (c) 2021-2025, the adamantine authors.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include <CubeHeatSource.hh>
#include <ElectronBeamHeatSource.hh>
#include <GoldakHeatSource.hh>
#include <ensemble_management.hh>
#include <utils.hh>

#include <fstream>
#include <random>

namespace adamantine
{
namespace
{
std::vector<double> get_normal_random_vector(unsigned int length,
                                             unsigned int n_rejected_draws,
                                             double mean, double stddev,
                                             bool verbose)
{
  ASSERT(stddev >= 0., "Internal Error");

  std::mt19937 pseudorandom_number_generator;
  std::normal_distribution<> normal_dist_generator(mean, stddev);
  for (unsigned int i = 0; i < n_rejected_draws; ++i)
  {
    normal_dist_generator(pseudorandom_number_generator);
  }

  std::vector<double> output_vector(length);
  for (unsigned int i = 0; i < length; ++i)
  {
    // We reject negative values because physical quantities we care about are
    // all positive and we cannot guarantee that the normal distribution will
    // always be positive.
    do
    {
      output_vector[i] = normal_dist_generator(pseudorandom_number_generator);

      if (verbose && output_vector[i] < 0.)
      {
        std::cout << "Random value rejected because it was negative: "
                  << output_vector[i] << std::endl;
      }

    } while (output_vector[i] < 0.);
  }

  return output_vector;
}
} // namespace

template <int dim>
std::vector<std::shared_ptr<HeatSource<dim>>> get_bounding_heat_sources(
    std::vector<boost::property_tree::ptree> const &database_ensemble,
    MPI_Comm global_communicator)
{
  // To bound the volumes of the Goldak and electron beam sources, we just need
  // to know their diameter and their depths. This is not enough for the cube
  // source but because the source is static and only use for testing, we just
  // assume that the source is identical for all ensemble members.
  // Use a std::vector instead of std::pair so deal.II can serialize the object.
  std::vector<std::vector<std::vector<double>>> diameter_depth_ensemble;
  for (auto const &database : database_ensemble)
  {
    std::vector<std::vector<double>> diameter_depth_beams;
    boost::property_tree::ptree const &source_database =
        database.get_child("sources");
    // PropertyTreeInput sources.n_beams
    unsigned int const n_beams = source_database.get<unsigned int>("n_beams");
    for (unsigned int i = 0; i < n_beams; ++i)
    {
      boost::property_tree::ptree const &beam_database =
          source_database.get_child("beam_" + std::to_string(i));

      // PropertyTreeInput sources.beam_X.type
      std::string type = beam_database.get<std::string>("type");
      if (type != "cube")
      {
        // PropertyTreeInput sources.beam_X.diameter
        double diameter = beam_database.get<double>("diameter");
        // PropertyTreeInput sources.beam_X.depth
        double depth = beam_database.get<double>("diameter");
        diameter_depth_beams.push_back({diameter, depth});
      }
    }
    diameter_depth_ensemble.push_back(diameter_depth_beams);
  }

  // Use all_gather to communicate the diameters and the depths
  auto all_diameter_depth = dealii::Utilities::MPI::all_gather(
      global_communicator, diameter_depth_ensemble);

  // Compute the maximum diameter and depth
  std::vector<double> diameter_max(diameter_depth_ensemble[0].size(), -1.);
  std::vector<double> depth_max(diameter_depth_ensemble[0].size(), -1.);
  for (auto const &diameter_depth_rank : all_diameter_depth)
  {
    for (unsigned int member = 0; member < diameter_depth_rank.size(); ++member)
    {
      for (unsigned int beam = 0; beam < diameter_depth_rank[member].size();
           ++beam)
      {
        if (diameter_depth_rank[member][beam][0] > diameter_max[beam])
        {
          diameter_max[beam] = diameter_depth_rank[member][beam][0];
        }
        if (diameter_depth_rank[member][beam][1] > depth_max[beam])
        {
          depth_max[beam] = diameter_depth_rank[member][beam][1];
        }
      }
    }
  }

  // Get the units database
  boost::optional<boost::property_tree::ptree const &> units_optional_database =
      database_ensemble[0].get_child_optional("units");

  // Create the bounding sources
  boost::property_tree::ptree const &source_database =
      database_ensemble[0].get_child("sources");
  // PropertyTreeInput sources.n_beams
  unsigned int const n_beams = source_database.get<unsigned int>("n_beams");
  std::vector<std::shared_ptr<HeatSource<dim>>> bounding_heat_sources(n_beams);
  unsigned int bounding_source = 0;
  for (unsigned int i = 0; i < n_beams; ++i)
  {
    boost::property_tree::ptree const &beam_database =
        source_database.get_child("beam_" + std::to_string(i));

    // PropertyTreeInput sources.beam_X.type
    std::string type = beam_database.get<std::string>("type");
    if (type == "goldak")
    {
      boost::property_tree::ptree modified_beam_database = beam_database;
      modified_beam_database.put("diameter", diameter_max[bounding_source]);
      modified_beam_database.put("depth", depth_max[bounding_source]);
      ++bounding_source;

      bounding_heat_sources[i] = std::make_shared<GoldakHeatSource<dim>>(
          modified_beam_database, units_optional_database);
    }
    else if (type == "electron_beam")
    {
      boost::property_tree::ptree modified_beam_database = beam_database;
      modified_beam_database.put("diameter", diameter_max[bounding_source]);
      modified_beam_database.put("depth", depth_max[bounding_source]);
      ++bounding_source;

      bounding_heat_sources[i] = std::make_shared<ElectronBeamHeatSource<dim>>(
          modified_beam_database, units_optional_database);
    }
    else if (type == "cube")
    {
      bounding_heat_sources[i] = std::make_shared<CubeHeatSource<dim>>(
          beam_database, units_optional_database);
    }
  }

  return bounding_heat_sources;
}

void traverse(boost::property_tree::ptree const &ensemble_ptree,
              std::vector<boost::property_tree::ptree> &database_ensemble,
              std::vector<std::string> &keys, unsigned int &n_rejected_draws,
              unsigned int local_ensemble_size,
              unsigned int global_ensemble_size, std::string const &path = "")
{
  for (auto const &[key, child] : ensemble_ptree)
  {
    std::string full_path = path.empty() ? key : path + "." + key;

    // If the child is empty, we have a full path
    if (child.empty())
    {
      // Skip ensemble_simulation and ensemble_size
      if ((full_path == "ensemble_simulation") ||
          (full_path == "ensemble_size"))
      {
        continue;
      }

      double stddev = database_ensemble[0].get<double>("ensemble." + full_path);
      // The key in database_ensemble is full_path without the _stddev.
      full_path.erase(full_path.length() - 7);
      double mean = database_ensemble[0].get<double>(full_path);

      std::vector<double> values = get_normal_random_vector(
          local_ensemble_size, n_rejected_draws, mean, stddev,
          database_ensemble[0].get("verbose_output", false));
      n_rejected_draws += global_ensemble_size;

      // Store full_path for when we write the value in a file
      keys.push_back(full_path);

      // Update the database_ensemble
      for (unsigned int member = 0; member < local_ensemble_size; ++member)
      {
        database_ensemble[member].put(full_path, values[member]);
      }
    }
    else
    {
      traverse(child, database_ensemble, keys, n_rejected_draws,
               local_ensemble_size, global_ensemble_size, full_path);
    }
  }
}

std::vector<boost::property_tree::ptree> create_database_ensemble(
    boost::property_tree::ptree const &database, MPI_Comm local_communicator,
    unsigned int first_local_member, unsigned int local_ensemble_size,
    unsigned int global_ensemble_size)
{
  std::vector<boost::property_tree::ptree> database_ensemble(
      local_ensemble_size, database);

  // The structure inside ensemble needs to match the rest of the database
  // Do a try catch to get a nice error message other it's going to be
  // strange once we erase the _stddev
  std::vector<std::string> keys;
  try
  {
    unsigned int n_rejected_draws = first_local_member;
    traverse(database.get_child("ensemble"), database_ensemble, keys,
             n_rejected_draws, local_ensemble_size, global_ensemble_size);
  }
  catch (boost::property_tree::ptree_bad_path &exception)
  {
    std::cerr << std::endl;
    std::cerr << "Aborting in ensemble management." << std::endl;
    std::cerr << "Error: " << exception.what() << std::endl << std::endl;
    std::cerr << "There is a problem with the input file." << std::endl;
    std::cerr << "Make sure that the keys in ensemble match" << std::endl;
    std::cerr << "the keys in the rest of the input file." << std::endl;
    std::cerr << "Note the keys must have been set explicitly" << std::endl;
    std::cerr << "even if they normally have a default value." << std::endl;
    std::cerr << std::endl;
    std::abort();
  }

  // Write the ensemble variables in files to simplify data analysis.
  if (dealii::Utilities::MPI::this_mpi_process(local_communicator) == 0)
  {
    for (unsigned int member = 0; member < local_ensemble_size; ++member)
    {

      std::string member_data_filename =
          database.get<std::string>("post_processor.filename_prefix") + "_m" +
          std::to_string(first_local_member + member) + "_data.txt";
      std::ofstream file(member_data_filename);
      for (auto const &k : keys)
      {
        file << k << ": " << database_ensemble[member].get<double>(k) << "\n";
      }
    }
  }

  return database_ensemble;
}
} // namespace adamantine

//-------------------- Explicit Instantiations --------------------//
namespace adamantine
{
template std::vector<std::shared_ptr<HeatSource<2>>> get_bounding_heat_sources(
    std::vector<boost::property_tree::ptree> const &property_trees,
    MPI_Comm global_communicator);
template std::vector<std::shared_ptr<HeatSource<3>>> get_bounding_heat_sources(
    std::vector<boost::property_tree::ptree> const &property_trees,
    MPI_Comm global_communicator);
} // namespace adamantine
