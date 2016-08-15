/* Copyright (c) 2016, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#include "Geometry.hh"
#include "PostProcessor.hh"
#include "ThermalPhysics.hh"
#include "utils.hh"
#include <deal.II/base/mpi.h>
#include <boost/filesystem.hpp>
#include <boost/mpi.hpp>
#include <boost/program_options.hpp>
#include <boost/property_tree/info_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <iostream>

template <int dim, int fe_degree, typename QuadratureType>
void initialize(
    boost::mpi::communicator &communicator,
    boost::property_tree::ptree const &database,
    adamantine::Geometry<dim> &geometry,
    std::unique_ptr<adamantine::Physics<dim, double>> &thermal_physics)
{
  thermal_physics.reset(
      new adamantine::ThermalPhysics<dim, fe_degree, double, QuadratureType>(
          communicator, database, geometry));
}

template <int dim, int fe_degree>
void initialize_quadrature(
    std::string const &quadrature_type, boost::mpi::communicator &communicator,
    boost::property_tree::ptree const &database,
    adamantine::Geometry<dim> &geometry,
    std::unique_ptr<adamantine::Physics<dim, double>> &thermal_physics)
{
  if (quadrature_type.compare("gauss") == 0)
    initialize<dim, fe_degree, dealii::QGauss<1>>(communicator, database,
                                                  geometry, thermal_physics);
  else if (quadrature_type.compare("lobatto") == 0)
    initialize<dim, fe_degree, dealii::QGaussLobatto<1>>(
        communicator, database, geometry, thermal_physics);
  else
    adamantine::ASSERT_THROW(false, "quadrature should be Gauss or Lobatto.");
}

template <int dim>
void initialize_thermal_physics(
    unsigned int fe_degree, std::string const &quadrature_type,
    boost::mpi::communicator &communicator,
    boost::property_tree::ptree const &database,
    adamantine::Geometry<dim> &geometry,
    std::unique_ptr<adamantine::Physics<dim, double>> &thermal_physics)
{
  switch (fe_degree)
  {
  case 1:
  {
    initialize_quadrature<dim, 1>(quadrature_type, communicator, database,
                                  geometry, thermal_physics);

    break;
  }
  case 2:
  {
    initialize_quadrature<dim, 2>(quadrature_type, communicator, database,
                                  geometry, thermal_physics);

    break;
  }
  case 3:
  {
    initialize_quadrature<dim, 3>(quadrature_type, communicator, database,
                                  geometry, thermal_physics);

    break;
  }
  case 4:
  {
    initialize_quadrature<dim, 4>(quadrature_type, communicator, database,
                                  geometry, thermal_physics);

    break;
  }
  case 5:
  {
    initialize_quadrature<dim, 5>(quadrature_type, communicator, database,
                                  geometry, thermal_physics);

    break;
  }
  case 6:
  {
    initialize_quadrature<dim, 6>(quadrature_type, communicator, database,
                                  geometry, thermal_physics);

    break;
  }
  case 7:
  {
    initialize_quadrature<dim, 7>(quadrature_type, communicator, database,
                                  geometry, thermal_physics);

    break;
  }
  case 8:
  {
    initialize_quadrature<dim, 8>(quadrature_type, communicator, database,
                                  geometry, thermal_physics);

    break;
  }
  case 9:
  {
    initialize_quadrature<dim, 9>(quadrature_type, communicator, database,
                                  geometry, thermal_physics);

    break;
  }
  case 10:
  {
    initialize_quadrature<dim, 10>(quadrature_type, communicator, database,
                                   geometry, thermal_physics);

    break;
  }
  default:
  {
    adamantine::ASSERT_THROW(false, "fe_degree should be between 1 and 10.");
  }
  }
}

template <int dim>
void run(std::unique_ptr<adamantine::Physics<dim, double>> &thermal_physics,
         adamantine::PostProcessor<dim> &post_processor,
         boost::property_tree::ptree const &time_stepping_database)
{
  thermal_physics->reinit();
  dealii::LA::distributed::Vector<double> solution;
  thermal_physics->initialize_dof_vector(solution);
  unsigned int cycle = 0;
  unsigned int n_time_step = 0;
  double time = 0.;
  // Output the initial solution
  post_processor.output_pvtu(cycle, n_time_step, time, solution);
  ++n_time_step;

  double time_step = time_stepping_database.get<double>("time_step");
  double const duration = time_stepping_database.get<double>("duration");
  while (time < duration)
  {
    if ((time + time_step) > duration)
      time_step = duration - time;
    // time can be different than time + time_step if an embedded scheme is
    // used.
    time = thermal_physics->evolve_one_time_step(time, time_step, solution);

    // Get the new time step
    time_step = thermal_physics->get_delta_t_guess();

    // Output the solution
    post_processor.output_pvtu(cycle, n_time_step, time, solution);
    ++n_time_step;
  }
  post_processor.output_pvd();
}

int main(int argc, char *argv[])
{
  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(
      argc, argv, dealii::numbers::invalid_unsigned_int);
  boost::mpi::communicator communicator;

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
                   quadrature_type.begin(), [](unsigned char c)
                   {
                     return std::tolower(c);
                   });

    if (dim == 2)
    {
      boost::property_tree::ptree time_stepping_database =
          database.get_child("time_stepping");
      boost::property_tree::ptree post_processor_database =
          database.get_child("post_processor");
      adamantine::Geometry<2> geometry(communicator, geometry_database);
      std::unique_ptr<adamantine::Physics<2, double>> thermal_physics;
      initialize_thermal_physics<2>(fe_degree, quadrature_type, communicator,
                                    database, geometry, thermal_physics);
      adamantine::PostProcessor<2> post_processor(
          communicator, post_processor_database,
          thermal_physics->get_dof_handler());
      run(thermal_physics, post_processor, time_stepping_database);
    }
    else
    {
      boost::property_tree::ptree time_stepping_database =
          database.get_child("time_stepping");
      boost::property_tree::ptree post_processor_database =
          database.get_child("post_processor");
      adamantine::Geometry<3> geometry(communicator, geometry_database);
      std::unique_ptr<adamantine::Physics<3, double>> thermal_physics;
      initialize_thermal_physics<3>(fe_degree, quadrature_type, communicator,
                                    database, geometry, thermal_physics);
      adamantine::PostProcessor<3> post_processor(
          communicator, post_processor_database,
          thermal_physics->get_dof_handler());
      run(thermal_physics, post_processor, time_stepping_database);
    }
  }
  catch (boost::bad_any_cast &exception)
  {
    std::cerr << std::endl;
    std::cerr << "Aborting." << std::endl;
    std::cerr << "Error: " << exception.what() << std::endl;
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

  return 0;
}
