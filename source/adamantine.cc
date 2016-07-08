/* Copyright (c) 2016, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#include "Geometry.hh"
#include "utils.hh"
#include <deal.II/base/mpi.h>
#include <boost/filesystem.hpp>
#include <boost/mpi.hpp>
#include <boost/program_options.hpp>
#include <boost/property_tree/info_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <iostream>

int main(int argc, char* argv[])
{
  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv,
      dealii::numbers::invalid_unsigned_int);
  boost::mpi::communicator communicator;

  try
  {                                           
    namespace boost_po = boost::program_options;

    // Get the name of the input file from the command line.
    // First declare the possible options.
    boost_po::options_description description("Options:");
    description.add_options()
      ("help", "Produce help message.")
      ("input-file", boost_po::value<std::string>(), "Name of the input file.")
      ;
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
    adamantine::ASSERT_THROW(map.count("input-file") == 0, "No input file.");
    std::string const filename = map["input-file"].as<std::string>();
    adamantine::ASSERT_THROW(boost::filesystem::exists(filename) == true, 
        "The file " + filename + " does not exist.");
    boost::property_tree::ptree database;
    boost::property_tree::info_parser::read_info(filename, database);

    int const dim = database.get<int>("dim");
    adamantine::ASSERT_THROW((dim == 2) || (dim == 3), 
        "dim should be 2 or 3");
    if (dim == 2)
    {
      adamantine::Geometry<2> geometry(communicator, database);
    }
    else
    {
      adamantine::Geometry<2> geometry(communicator, database);
    }
  }
  catch(std::exception &exception)
  {
    std::cerr<<std::endl;
    std::cerr<<"Aborting."<<std::endl;
    std::cerr<<"Error: "<<exception.what()<<std::endl;

    return 1;
  }
  catch(...)
  {
    std::cerr<<std::endl;
    std::cerr<<"Aborting."<<std::endl;
    std::cerr<<"No error message."<<std::endl;

    return 1;
  }

  return 0;
}
