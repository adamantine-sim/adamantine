/* Copyright (c) 2016 - 2021, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#include "adamantine.hh"

#include <validate_input_database.hh>

#ifdef ADAMANTINE_WITH_ADIAK
#include <adiak.hpp>
#endif

#ifdef ADAMANTINE_WITH_CALIPER
#include <caliper/cali-manager.h>
#endif

#include <Kokkos_Core.hpp>

int main(int argc, char *argv[])
{
#ifdef ADAMANTINE_WITH_CALIPER
  CALI_MARK_BEGIN("main");
#endif

  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(
      argc, argv, dealii::numbers::invalid_unsigned_int);
  MPI_Comm communicator = MPI_COMM_WORLD;

  Kokkos::ScopeGuard guard(argc, argv);

#ifdef ADAMANTINE_WITH_ADIAK
  adiak_init(&communicator);
  adiak::user();
  adiak::launchdate();
  adiak::executablepath();
  adiak::libraries();
  adiak::cmdline();
  adiak::clustername();
  adiak::jobsize();
  adiak::value("MemorySpace", "Host");
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
    // Declare a map that will contains the values read. Parse the command line
    // and finally populate the map.
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
    adamantine::ASSERT_THROW(boost::filesystem::exists(filename) == true,
                             "The file " + filename + " does not exist.");
    boost::property_tree::ptree database;
    boost::property_tree::info_parser::read_info(filename, database);
    try
    {
      adamantine::validate_input_database(database);
    }
    catch (std::runtime_error const &exception)
    {
      std::cerr << exception.what();
      return 1;
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

    unsigned int rank = dealii::Utilities::MPI::this_mpi_process(communicator);
    if (rank == 0)
      std::cout << "Starting simulation" << std::endl;

    if (dim == 2)
    {
      if (ensemble_calc)
      {
        run_ensemble<2, dealii::MemorySpace::Host>(communicator, database,
                                                   timers);
      }
      else
      {
        run<2, dealii::MemorySpace::Host>(communicator, database, timers);
      }
    }
    else
    {
      if (ensemble_calc)
      {
        run_ensemble<3, dealii::MemorySpace::Host>(communicator, database,
                                                   timers);
      }
      else
      {
        run<3, dealii::MemorySpace::Host>(communicator, database, timers);
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
