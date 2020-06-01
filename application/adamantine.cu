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
    description.add_options()("help", "Produce help message.")(
        "input-file", boost_po::value<std::string>(),
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
    boost::property_tree::ptree discretization_database =
        database.get_child("discretization");
    unsigned int const fe_degree =
        discretization_database.get<unsigned int>("fe_degree");
    std::string quadrature_type =
        discretization_database.get("quadrature", "gauss");
    std::transform(quadrature_type.begin(), quadrature_type.end(),
                   quadrature_type.begin(),
                   [](unsigned char c) { return std::tolower(c); });

    unsigned int rank = dealii::Utilities::MPI::this_mpi_process(communicator);
    if (rank == 0)
      std::cout << "Starting simulation" << std::endl;

    if (dim == 2)
    {
      boost::property_tree::ptree time_stepping_database =
          database.get_child("time_stepping");
      boost::property_tree::ptree post_processor_database =
          database.get_child("post_processor");
      boost::property_tree::ptree refinement_database =
          database.get_child("refinement");
      adamantine::Geometry<2> geometry(communicator, geometry_database);
      std::unique_ptr<adamantine::Physics<2, dealii::MemorySpace::CUDA>>
          thermal_physics;
      std::vector<std::unique_ptr<adamantine::ElectronBeam<2>>>
          &electron_beams =
              initialize_thermal_physics<2>(fe_degree, quadrature_type,
                                            communicator, database, geometry,
                                            thermal_physics);
      adamantine::PostProcessor<2> post_processor(
          communicator, post_processor_database,
          thermal_physics->get_dof_handler(),
          thermal_physics->get_material_property());
      double const initial_temperature =
          database.get("materials.initial_temperature", 300.);
      run<2>(communicator, thermal_physics, post_processor, electron_beams,
             time_stepping_database, refinement_database, fe_degree,
             initial_temperature, timers);
    }
    else
    {
      boost::property_tree::ptree time_stepping_database =
          database.get_child("time_stepping");
      boost::property_tree::ptree post_processor_database =
          database.get_child("post_processor");
      boost::property_tree::ptree refinement_database =
          database.get_child("refinement");
      adamantine::Geometry<3> geometry(communicator, geometry_database);
      std::unique_ptr<adamantine::Physics<3, dealii::MemorySpace::CUDA>>
          thermal_physics;
      std::vector<std::unique_ptr<adamantine::ElectronBeam<3>>>
          &electron_beams =
              initialize_thermal_physics<3>(fe_degree, quadrature_type,
                                            communicator, database, geometry,
                                            thermal_physics);
      adamantine::PostProcessor<3> post_processor(
          communicator, post_processor_database,
          thermal_physics->get_dof_handler(),
          thermal_physics->get_material_property());
      double const initial_temperature =
          database.get("materials.initial_temperature", 300.);
      run<3>(communicator, thermal_physics, post_processor, electron_beams,
             time_stepping_database, refinement_database, fe_degree,
             initial_temperature, timers);
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
