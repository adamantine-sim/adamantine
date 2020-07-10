/* Copyright (c) 2016 - 2020, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#include "adamantine.hh"

int main(int argc, char *argv[])
{
  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(
      argc, argv, dealii::numbers::invalid_unsigned_int);
  MPI_Comm communicator = MPI_COMM_WORLD;

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

    boost::optional<boost::property_tree::ptree &> profiling_database =
        database.get_child_optional("profiling");
    if ((profiling_database) && (profiling_database.get().get("timer", 0) != 0))
      profiling = true;

    boost::property_tree::ptree geometry_database =
        database.get_child("geometry");
    int const dim = geometry_database.get<int>("dim");
    adamantine::ASSERT_THROW((dim == 2) || (dim == 3), "dim should be 2 or 3");

    unsigned int rank = dealii::Utilities::MPI::this_mpi_process(communicator);
    if (rank == 0)
      std::cout << "Starting simulation" << std::endl;

    if (dim == 2)
    {
      run<2, dealii::MemorySpace::Host>(communicator, database, timers);
    }
    else
    {
      run<3, dealii::MemorySpace::Host>(communicator, database, timers);
    }

    if (rank == 0)
      std::cout << "Simulation done" << std::endl;
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

  return 0;
}
