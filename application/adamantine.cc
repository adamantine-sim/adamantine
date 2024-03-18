/* Copyright (c) 2016 - 2024, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#include "adamantine.hh"

#include "MaterialStates.hh"
#include "utils.hh"
#include <validate_input_database.hh>

#include <boost/program_options.hpp>
#include <boost/property_tree/info_parser.hpp>
#include <boost/property_tree/json_parser.hpp>

#include <Kokkos_Core.hpp>

#include <filesystem>
#include <utility>

#ifdef ADAMANTINE_WITH_ADIAK
#include <adiak.hpp>
#endif

#ifdef ADAMANTINE_WITH_CALIPER
#include <caliper/cali-manager.h>
#endif

std::pair<int, int>
get_p_order_and_n_material_states(boost::property_tree::ptree &database)
{
  // We need to detect the degree of the polynomial. There are two cases. First,
  // we are using a table format. In this case, we return zero. Second, we are
  // using the polynomial format. In this case, we need to loop over all the
  // materials, all the states, and all the properties to determine the
  // polynomial order.

  unsigned int p_order = 0;
  unsigned int n_material_states = 0;
  // PropertyTreeInput materials.property_format
  bool use_table = database.get<std::string>("property_format") == "table";

  // PropertyTreeInput materials.n_materials
  unsigned int const n_materials = database.get<unsigned int>("n_materials");
  // Find all the material_ids being used.
  std::vector<dealii::types::material_id> material_ids;
  for (dealii::types::material_id id = 0;
       id < dealii::numbers::invalid_material_id; ++id)
  {
    if (database.count("material_" + std::to_string(id)) != 0)
      material_ids.push_back(id);
    if (material_ids.size() == n_materials)
      break;
  }

  for (auto const material_id : material_ids)
  {
    // Get the material property tree.
    boost::property_tree::ptree const &material_database =
        database.get_child("material_" + std::to_string(material_id));
    // For each material, loop over the possible states.
    for (unsigned int state = 0;
         state < adamantine::SolidLiquidPowder::n_material_states; ++state)
    {
      // The state may or may not exist for the material.
      boost::optional<boost::property_tree::ptree const &> state_database =
          material_database.get_child_optional(
              adamantine::material_state_names[state]);
      if (state_database)
      {
        // For each state, loop over the possible properties.
        for (unsigned int p = 0; p < adamantine::g_n_state_properties; ++p)
        {
          // The property may or may not exist for that state
          boost::optional<std::string> const property =
              state_database.get().get_optional<std::string>(
                  adamantine::state_property_names[p]);
          // If the property exists, put it in the map. If the property does
          // not exist, we have a nullptr.
          if (property)
          {
            n_material_states = std::max(state, n_material_states);
            if (!use_table)
            {
              // Remove blank spaces
              std::string property_string = property.get();
              property_string.erase(std::remove_if(property_string.begin(),
                                                   property_string.end(),
                                                   [](unsigned char x)
                                                   { return std::isspace(x); }),
                                    property_string.end());
              std::vector<std::string> parsed_property;
              boost::split(parsed_property, property_string,
                           [](char c) { return c == ','; });
              p_order = std::max(
                  static_cast<unsigned int>(parsed_property.size() - 1),
                  p_order);
            }
          }
        }
      }
    }
  }

  // Sanity check
  adamantine::ASSERT_THROW(
      p_order >= 0 && p_order < 5,
      "Error when computing the polynomial order of the material properties");
  adamantine::ASSERT_THROW(
      n_material_states > 0 && n_material_states < 4,
      "Error when computing the number of material states");

  return std::make_pair(p_order, n_material_states);
}

int main(int argc, char *argv[])
{
#ifdef ADAMANTINE_WITH_CALIPER
  CALI_MARK_BEGIN("main");
#endif

  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(
      argc, argv, dealii::numbers::invalid_unsigned_int);
  MPI_Comm communicator = MPI_COMM_WORLD;

#ifdef ADAMANTINE_WITH_ADIAK
  adiak_init(&communicator);
  adiak::user();
  adiak::launchdate();
  adiak::executablepath();
  adiak::libraries();
  adiak::cmdline();
  adiak::clustername();
  adiak::jobsize();
#endif

  std::vector<adamantine::Timer> timers;
  initialize_timers(communicator, timers);
  timers[adamantine::main].start();
  bool profiling = false;
  try
  {
    namespace boost_po = boost::program_options;

    // Get the name of the input file from the command line.
    // First declare the possible options.
    boost_po::options_description description("Options:");
    description.add_options()("help,h", "Produce help message.")(
        "input-file,i", boost_po::value<std::string>(),
        "Name of the input file.");
    // Declare a map that will contains the values read. Parse the command
    // line and finally populate the map.
    boost_po::variables_map map;
    boost_po::store(boost_po::parse_command_line(argc, argv, description), map);
    boost_po::notify(map);
    // Output the help message is asked for
    if (map.count("help") == 1)
    {
      std::cout << description << std::endl;
      return 1;
    }

    // Read the input.
    std::string const filename = map["input-file"].as<std::string>();
    adamantine::wait_for_file(filename, "Waiting for input file: " + filename);
    boost::property_tree::ptree database;
    if (std::filesystem::path(filename).extension().native() == ".json")
    {
      boost::property_tree::json_parser::read_json(filename, database);
    }
    else
    {
      boost::property_tree::info_parser::read_info(filename, database);
    }
    try
    {
      adamantine::validate_input_database(database);
    }
    catch (std::runtime_error const &exception)
    {
      std::cerr << exception.what() << std::endl;
      return 0;
    }

#ifdef ADAMANTINE_WITH_CALIPER
    cali::ConfigManager caliper_manager;
#endif
    boost::optional<boost::property_tree::ptree &> profiling_optional_database =
        database.get_child_optional("profiling");
    if (profiling_optional_database)
    {
      auto profiling_database = profiling_optional_database.get();
      // PropertyTreeInput profiling.timer
      if (profiling_database.get("timer", false))
        profiling = true;
#ifdef ADAMANTINE_WITH_CALIPER
      // PropertyTreeInput profiling.caliper
      auto caliper_optional_string =
          profiling_database.get_optional<std::string>("caliper");
      if (caliper_optional_string)
        caliper_manager.add(caliper_optional_string.get().c_str());
#endif
    }
#ifdef ADAMANTINE_WITH_CALIPER
    caliper_manager.start();
#endif

    boost::optional<boost::property_tree::ptree &> ensemble_optional_database =
        database.get_child_optional("ensemble");
    bool ensemble_calc = false;
    if (ensemble_optional_database)
    {
      auto ensemble_database = ensemble_optional_database.get();
      // PropertyTreeInput ensemble.ensemble_simulation
      ensemble_calc = ensemble_database.get<bool>("ensemble_simulation", false);
    }

    boost::property_tree::ptree geometry_database =
        database.get_child("geometry");
    // PropertyTreeInput geometry.dim
    int const dim = geometry_database.get<int>("dim");

    // Get the polynomial order used in the material properties
    auto const [p_order, n_material_states] =
        get_p_order_and_n_material_states(database.get_child("materials"));
    adamantine::ASSERT_THROW(p_order < 5,
                             "Material properties have too many coefficients.");

    unsigned int rank = dealii::Utilities::MPI::this_mpi_process(communicator);

    // PropertyTreeInput memory_space
    std::string memory_space =
        database.get<std::string>("memory_space", "host");

#ifdef ADAMANTINE_WITH_ADIAK
    if (memory_space == "device")
      adiak::value("MemorySpace", "Device");
    else
      adiak::value("MemorySpace", "Host");
#endif

    if (dim == 2)
    {
      if (ensemble_calc)
      {
        if (rank == 0)
          std::cout << "Starting ensemble simulation" << std::endl;
        if (memory_space == "device")
        {
          // TODO: Add device version of run_ensemble and call it here
          adamantine::ASSERT_THROW(
              false, "Error: Device version of ensemble simulations not "
                     "yet implemented.");
        }
        else
        {
          switch (p_order)
          {
          // p_order case
          case 0:
          {
            switch (n_material_states)
            {
            case 1:
            {
              run_ensemble<2, 0, adamantine::Solid, dealii::MemorySpace::Host>(
                  communicator, database, timers);
              break;
            }
            case 2:
            {
              run_ensemble<2, 0, adamantine::SolidLiquid,
                           dealii::MemorySpace::Host>(communicator, database,
                                                      timers);
              break;
            }
            default:
            {
              run_ensemble<2, 0, adamantine::SolidLiquidPowder,
                           dealii::MemorySpace::Host>(communicator, database,
                                                      timers);
            }
            }

            break;
          }
          // p_order case
          case 1:
          {
            switch (n_material_states)
            {
            case 1:
            {
              run_ensemble<2, 1, adamantine::Solid, dealii::MemorySpace::Host>(
                  communicator, database, timers);
              break;
            }
            case 2:
            {
              run_ensemble<2, 1, adamantine::SolidLiquid,
                           dealii::MemorySpace::Host>(communicator, database,
                                                      timers);
              break;
            }
            default:
            {
              run_ensemble<2, 1, adamantine::SolidLiquidPowder,
                           dealii::MemorySpace::Host>(communicator, database,
                                                      timers);
            }
            }

            break;
          }
          // p_order case
          case 2:
          {
            switch (n_material_states)
            {
            case 1:
            {
              run_ensemble<2, 2, adamantine::Solid, dealii::MemorySpace::Host>(
                  communicator, database, timers);
              break;
            }
            case 2:
            {
              run_ensemble<2, 2, adamantine::SolidLiquid,
                           dealii::MemorySpace::Host>(communicator, database,
                                                      timers);
              break;
            }
            default:
            {
              run_ensemble<2, 2, adamantine::SolidLiquidPowder,
                           dealii::MemorySpace::Host>(communicator, database,
                                                      timers);
            }
            }

            break;
          }
          // p_order case
          case 3:
          {
            switch (n_material_states)
            {
            case 1:
            {
              run_ensemble<2, 3, adamantine::Solid, dealii::MemorySpace::Host>(
                  communicator, database, timers);
              break;
            }
            case 2:
            {
              run_ensemble<2, 3, adamantine::SolidLiquid,
                           dealii::MemorySpace::Host>(communicator, database,
                                                      timers);
              break;
            }
            default:
            {
              run_ensemble<2, 3, adamantine::SolidLiquidPowder,
                           dealii::MemorySpace::Host>(communicator, database,
                                                      timers);
            }
            }

            break;
          }
          // p_order case
          default:
          {
            switch (n_material_states)
            {
            case 1:
            {
              run_ensemble<2, 4, adamantine::Solid, dealii::MemorySpace::Host>(
                  communicator, database, timers);
              break;
            }
            case 2:
            {
              run_ensemble<2, 4, adamantine::SolidLiquid,
                           dealii::MemorySpace::Host>(communicator, database,
                                                      timers);
              break;
            }
            default:
            {
              run_ensemble<2, 4, adamantine::SolidLiquidPowder,
                           dealii::MemorySpace::Host>(communicator, database,
                                                      timers);
            }
            }
          }
          }
        }
      }
      else
      {
        if (rank == 0)
          std::cout << "Starting non-ensemble simulation" << std::endl;

        if (memory_space == "device")
        {
          switch (p_order)
          {
          // p_order case
          case 0:
          {
            switch (n_material_states)
            {
            case 1:
            {
              run<2, 0, adamantine::Solid, dealii::MemorySpace::Default>(
                  communicator, database, timers);
              break;
            }
            case 2:
            {
              run<2, 0, adamantine::SolidLiquid, dealii::MemorySpace::Default>(
                  communicator, database, timers);
              break;
            }
            default:
            {
              run<2, 0, adamantine::SolidLiquidPowder,
                  dealii::MemorySpace::Default>(communicator, database, timers);
            }
            }

            break;
          }
          // p_order case
          case 1:
          {
            switch (n_material_states)
            {
            case 1:
            {
              run<2, 1, adamantine::Solid, dealii::MemorySpace::Default>(
                  communicator, database, timers);
              break;
            }
            case 2:
            {
              run<2, 1, adamantine::SolidLiquid, dealii::MemorySpace::Default>(
                  communicator, database, timers);
              break;
            }
            default:
            {
              run<2, 1, adamantine::SolidLiquidPowder,
                  dealii::MemorySpace::Default>(communicator, database, timers);
            }
            }

            break;
          }
          // p_order case
          case 2:
          {
            switch (n_material_states)
            {
            case 1:
            {
              run<2, 2, adamantine::Solid, dealii::MemorySpace::Default>(
                  communicator, database, timers);
              break;
            }
            case 2:
            {
              run<2, 2, adamantine::SolidLiquid, dealii::MemorySpace::Default>(
                  communicator, database, timers);
              break;
            }
            default:
            {
              run<2, 2, adamantine::SolidLiquidPowder,
                  dealii::MemorySpace::Default>(communicator, database, timers);
            }
            }

            break;
          }
          // p_order case
          case 3:
          {
            switch (n_material_states)
            {
            case 1:
            {
              run<2, 3, adamantine::Solid, dealii::MemorySpace::Default>(
                  communicator, database, timers);
              break;
            }
            case 2:
            {
              run<2, 3, adamantine::SolidLiquid, dealii::MemorySpace::Default>(
                  communicator, database, timers);
              break;
            }
            default:
            {
              run<2, 3, adamantine::SolidLiquidPowder,
                  dealii::MemorySpace::Default>(communicator, database, timers);
            }
            }

            break;
          }
          // p_order case
          default:
          {
            switch (n_material_states)
            {
            case 1:
            {
              run<2, 4, adamantine::Solid, dealii::MemorySpace::Default>(
                  communicator, database, timers);
              break;
            }
            case 2:
            {
              run<2, 4, adamantine::SolidLiquid, dealii::MemorySpace::Default>(
                  communicator, database, timers);
              break;
            }
            default:
            {
              run<2, 4, adamantine::SolidLiquidPowder,
                  dealii::MemorySpace::Default>(communicator, database, timers);
            }
            }
          }
          }
        }
        else
        {
          switch (p_order)
          {
          // p_order case
          case 0:
          {
            switch (n_material_states)
            {
            case 1:
            {
              run<2, 0, adamantine::Solid, dealii::MemorySpace::Host>(
                  communicator, database, timers);
              break;
            }
            case 2:
            {
              run<2, 0, adamantine::SolidLiquid, dealii::MemorySpace::Host>(
                  communicator, database, timers);
              break;
            }
            default:
            {
              run<2, 0, adamantine::SolidLiquidPowder,
                  dealii::MemorySpace::Host>(communicator, database, timers);
            }
            }

            break;
          }
          // p_order case
          case 1:
          {
            switch (n_material_states)
            {
            case 1:
            {
              run<2, 1, adamantine::Solid, dealii::MemorySpace::Host>(
                  communicator, database, timers);
              break;
            }
            case 2:
            {
              run<2, 1, adamantine::SolidLiquid, dealii::MemorySpace::Host>(
                  communicator, database, timers);
              break;
            }
            default:
            {
              run<2, 1, adamantine::SolidLiquidPowder,
                  dealii::MemorySpace::Host>(communicator, database, timers);
            }
            }

            break;
          }
          // p_order case
          case 2:
          {
            switch (n_material_states)
            {
            case 1:
            {
              run<2, 2, adamantine::Solid, dealii::MemorySpace::Host>(
                  communicator, database, timers);
              break;
            }
            case 2:
            {
              run<2, 2, adamantine::SolidLiquid, dealii::MemorySpace::Host>(
                  communicator, database, timers);
              break;
            }
            default:
            {
              run<2, 2, adamantine::SolidLiquidPowder,
                  dealii::MemorySpace::Host>(communicator, database, timers);
            }
            }

            break;
          }
          // p_order case
          case 3:
          {
            switch (n_material_states)
            {
            case 1:
            {
              run<2, 3, adamantine::Solid, dealii::MemorySpace::Host>(
                  communicator, database, timers);
              break;
            }
            case 2:
            {
              run<2, 3, adamantine::SolidLiquid, dealii::MemorySpace::Host>(
                  communicator, database, timers);
              break;
            }
            default:
            {
              run<2, 3, adamantine::SolidLiquidPowder,
                  dealii::MemorySpace::Host>(communicator, database, timers);
            }
            }

            break;
          }
          // p_order case
          default:
          {
            switch (n_material_states)
            {
            case 1:
            {
              run<2, 4, adamantine::Solid, dealii::MemorySpace::Host>(
                  communicator, database, timers);
              break;
            }
            case 2:
            {
              run<2, 4, adamantine::SolidLiquid, dealii::MemorySpace::Host>(
                  communicator, database, timers);
              break;
            }
            default:
            {
              run<2, 4, adamantine::SolidLiquidPowder,
                  dealii::MemorySpace::Host>(communicator, database, timers);
            }
            }
          }
          }
        }
      }
    }
    else
    {
      if (ensemble_calc)
      {
        if (rank == 0)
          std::cout << "Starting ensemble simulation" << std::endl;

        if (memory_space == "device")
        {
          // TODO: Add device version of run_ensemble and call it here
          adamantine::ASSERT_THROW(
              false, "Error: Device version of ensemble simulations not "
                     "yet implemented.");
        }
        else
        {
          switch (p_order)
          {
          // p_order case
          case 0:
          {
            switch (n_material_states)
            {
            case 1:
            {
              run_ensemble<3, 0, adamantine::Solid, dealii::MemorySpace::Host>(
                  communicator, database, timers);
              break;
            }
            case 2:
            {
              run_ensemble<3, 0, adamantine::SolidLiquid,
                           dealii::MemorySpace::Host>(communicator, database,
                                                      timers);
              break;
            }
            default:
            {
              run_ensemble<3, 0, adamantine::SolidLiquidPowder,
                           dealii::MemorySpace::Host>(communicator, database,
                                                      timers);
            }
            }

            break;
          }
          // p_order case
          case 1:
          {
            switch (n_material_states)
            {
            case 1:
            {
              run_ensemble<3, 1, adamantine::Solid, dealii::MemorySpace::Host>(
                  communicator, database, timers);
              break;
            }
            case 2:
            {
              run_ensemble<3, 1, adamantine::SolidLiquid,
                           dealii::MemorySpace::Host>(communicator, database,
                                                      timers);
              break;
            }
            default:
            {
              run_ensemble<3, 1, adamantine::SolidLiquidPowder,
                           dealii::MemorySpace::Host>(communicator, database,
                                                      timers);
            }
            }

            break;
          }
          // p_order case
          case 2:
          {
            switch (n_material_states)
            {
            case 1:
            {
              run_ensemble<3, 2, adamantine::Solid, dealii::MemorySpace::Host>(
                  communicator, database, timers);
              break;
            }
            case 2:
            {
              run_ensemble<3, 2, adamantine::SolidLiquid,
                           dealii::MemorySpace::Host>(communicator, database,
                                                      timers);
              break;
            }
            default:
            {
              run_ensemble<3, 2, adamantine::SolidLiquidPowder,
                           dealii::MemorySpace::Host>(communicator, database,
                                                      timers);
            }
            }

            break;
          }
          // p_order case
          case 3:
          {
            switch (n_material_states)
            {
            case 1:
            {
              run_ensemble<3, 3, adamantine::Solid, dealii::MemorySpace::Host>(
                  communicator, database, timers);
              break;
            }
            case 2:
            {
              run_ensemble<3, 3, adamantine::SolidLiquid,
                           dealii::MemorySpace::Host>(communicator, database,
                                                      timers);
              break;
            }
            default:
            {
              run_ensemble<3, 3, adamantine::SolidLiquidPowder,
                           dealii::MemorySpace::Host>(communicator, database,
                                                      timers);
            }
            }

            break;
          }
          // p_order case
          default:
          {
            switch (n_material_states)
            {
            case 1:
            {
              run_ensemble<3, 4, adamantine::Solid, dealii::MemorySpace::Host>(
                  communicator, database, timers);
              break;
            }
            case 2:
            {
              run_ensemble<3, 4, adamantine::SolidLiquid,
                           dealii::MemorySpace::Host>(communicator, database,
                                                      timers);
              break;
            }
            default:
            {
              run_ensemble<3, 4, adamantine::SolidLiquidPowder,
                           dealii::MemorySpace::Host>(communicator, database,
                                                      timers);
            }
            }
          }
          }
        }
      }
      else
      {
        if (rank == 0)
          std::cout << "Starting non-ensemble simulation" << std::endl;

        if (memory_space == "device")
        {
          switch (p_order)
          {
          // p_order case
          case 0:
          {
            switch (n_material_states)
            {
            case 1:
            {
              run<3, 0, adamantine::Solid, dealii::MemorySpace::Default>(
                  communicator, database, timers);
              break;
            }
            case 2:
            {
              run<3, 0, adamantine::SolidLiquid, dealii::MemorySpace::Default>(
                  communicator, database, timers);
              break;
            }
            default:
            {
              run<3, 0, adamantine::SolidLiquidPowder,
                  dealii::MemorySpace::Default>(communicator, database, timers);
            }
            }

            break;
          }
          // p_order case
          case 1:
          {
            switch (n_material_states)
            {
            case 1:
            {
              run<3, 1, adamantine::Solid, dealii::MemorySpace::Default>(
                  communicator, database, timers);
              break;
            }
            case 2:
            {
              run<3, 1, adamantine::SolidLiquid, dealii::MemorySpace::Default>(
                  communicator, database, timers);
              break;
            }
            }

            break;
          }
          // p_order case
          case 2:
          {
            switch (n_material_states)
            {
            case 1:
            {
              run<3, 2, adamantine::Solid, dealii::MemorySpace::Default>(
                  communicator, database, timers);
              break;
            }
            case 2:
            {
              run<3, 2, adamantine::SolidLiquid, dealii::MemorySpace::Default>(
                  communicator, database, timers);
              break;
            }
            default:
            {
              run<3, 2, adamantine::SolidLiquidPowder,
                  dealii::MemorySpace::Default>(communicator, database, timers);
            }
            }

            break;
          }
          // p_order case
          case 3:
          {
            switch (n_material_states)
            {
            case 1:
            {
              run<3, 3, adamantine::Solid, dealii::MemorySpace::Default>(
                  communicator, database, timers);
              break;
            }
            case 2:
            {
              run<3, 3, adamantine::SolidLiquid, dealii::MemorySpace::Default>(
                  communicator, database, timers);
              break;
            }
            default:
            {
              run<3, 3, adamantine::SolidLiquidPowder,
                  dealii::MemorySpace::Default>(communicator, database, timers);
            }
            }

            break;
          }
          // p_order case
          default:
          {
            switch (n_material_states)
            {
            case 1:
            {
              run<3, 4, adamantine::Solid, dealii::MemorySpace::Default>(
                  communicator, database, timers);
              break;
            }
            case 2:
            {
              run<3, 4, adamantine::SolidLiquid, dealii::MemorySpace::Default>(
                  communicator, database, timers);
              break;
            }
            default:
            {
              run<3, 4, adamantine::SolidLiquidPowder,
                  dealii::MemorySpace::Default>(communicator, database, timers);
            }
            }

            break;
          }
          }
        }
        else
        {
          switch (p_order)
          {
          // p_order case
          case 0:
          {
            switch (n_material_states)
            {
            case 1:
            {
              run<3, 0, adamantine::Solid, dealii::MemorySpace::Host>(
                  communicator, database, timers);
              break;
            }
            case 2:
            {
              run<3, 0, adamantine::SolidLiquid, dealii::MemorySpace::Host>(
                  communicator, database, timers);
              break;
            }
            default:
            {
              run<3, 0, adamantine::SolidLiquidPowder,
                  dealii::MemorySpace::Host>(communicator, database, timers);
            }
            }

            break;
          }
          // p_order case
          case 1:
          {
            switch (n_material_states)
            {
            case 1:
            {
              run<3, 1, adamantine::Solid, dealii::MemorySpace::Host>(
                  communicator, database, timers);
              break;
            }
            case 2:
            {
              run<3, 1, adamantine::SolidLiquid, dealii::MemorySpace::Host>(
                  communicator, database, timers);
              break;
            }
            default:
            {
              run<3, 1, adamantine::SolidLiquid, dealii::MemorySpace::Host>(
                  communicator, database, timers);
            }
            }

            break;
          }
          // p_order case
          case 2:
          {
            switch (n_material_states)
            {
            case 1:
            {
              run<3, 2, adamantine::Solid, dealii::MemorySpace::Host>(
                  communicator, database, timers);
              break;
            }
            case 2:
            {
              run<3, 2, adamantine::SolidLiquid, dealii::MemorySpace::Host>(
                  communicator, database, timers);
              break;
            }
            default:
            {
              run<3, 2, adamantine::SolidLiquidPowder,
                  dealii::MemorySpace::Host>(communicator, database, timers);
            }
            }

            break;
          }
          // p_order case
          case 3:
          {
            switch (n_material_states)
            {
            case 1:
            {
              run<3, 3, adamantine::Solid, dealii::MemorySpace::Host>(
                  communicator, database, timers);
              break;
            }
            case 2:
            {
              run<3, 3, adamantine::SolidLiquid, dealii::MemorySpace::Host>(
                  communicator, database, timers);
              break;
            }
            default:
            {
              run<3, 3, adamantine::SolidLiquidPowder,
                  dealii::MemorySpace::Host>(communicator, database, timers);
            }
            }

            break;
          }
          // p_order case
          default:
          {
            switch (n_material_states)
            {
            case 1:
            {
              run<3, 4, adamantine::Solid, dealii::MemorySpace::Host>(
                  communicator, database, timers);
              break;
            }
            case 2:
            {
              run<3, 4, adamantine::SolidLiquid, dealii::MemorySpace::Host>(
                  communicator, database, timers);
              break;
            }
            default:
            {
              run<3, 4, adamantine::SolidLiquidPowder,
                  dealii::MemorySpace::Host>(communicator, database, timers);
            }
            }
          }
          }
        }
      }
    }

    if (rank == 0)
      std::cout << "Simulation done" << std::endl;

#ifdef ADAMANTINE_WITH_CALIPER
    CALI_MARK_END("main");
    caliper_manager.flush();
#endif
  }
  catch (boost::bad_any_cast &exception)
  {
    std::cerr << std::endl;
    std::cerr << "Aborting." << std::endl;
    std::cerr << "Error: " << exception.what() << std::endl << std::endl;
    std::cerr << "There is a problem with the input file." << std::endl;
    std::cerr << "Make sure that the input file is correct" << std::endl;
    std::cerr << "and that you are using the following command" << std::endl;
    std::cerr << "to run adamantine:" << std::endl;
    std::cerr << "./adamantine --input-file=my_input_file" << std::endl;
    std::cerr << std::endl;
  }
  catch (std::exception &exception)
  {
    std::cerr << std::endl;
    std::cerr << "Aborting." << std::endl;
    std::cerr << "Error: " << exception.what() << std::endl;
    std::cerr << std::endl;

    return 1;
  }
  catch (...)
  {
    std::cerr << std::endl;
    std::cerr << "Aborting." << std::endl;
    std::cerr << "No error message." << std::endl;
    std::cerr << std::endl;

    return 1;
  }

  timers[adamantine::main].stop();
  if (profiling == true)
    for (auto &timer : timers)
      timer.print();

#ifdef ADAMANTINE_WITH_ADIAK
  adiak::fini();
#endif

  return 0;
}
