/* SPDX-FileCopyrightText: Copyright (c) 2026, the adamantine authors.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include <Geometry.hh>
#include <ScanPath.hh>
#include <material_deposition.hh>
#include <utils.hh>

#include <deal.II/base/mpi.h>
#include <deal.II/base/types.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/hp/fe_collection.h>
#include <deal.II/numerics/data_out.h>

#include <boost/program_options.hpp>
#include <boost/property_tree/info_parser.hpp>
#include <boost/property_tree/json_parser.hpp>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

template <int dim>
using DepositionData =
    std::tuple<std::vector<dealii::BoundingBox<dim>>, std::vector<double>,
               std::vector<double>, std::vector<double>>;

template <int dim>
DepositionData<dim>
read_deposition_data(boost::property_tree::ptree const &database,
                     boost::property_tree::ptree const &geometry_database,
                     boost::optional<boost::property_tree::ptree const &> const
                         &units_optional_database)
{
  boost::property_tree::ptree const &sources_database =
      database.get_child("sources");
  unsigned int const n_beams = sources_database.get<unsigned int>("n_beams", 1);

  std::vector<DepositionData<dim>> deposition_paths;
  deposition_paths.reserve(n_beams);
  for (unsigned int beam_id = 0; beam_id < n_beams; ++beam_id)
  {
    boost::property_tree::ptree const &beam_database =
        sources_database.get_child("beam_" + std::to_string(beam_id));
    std::string const scan_path_file =
        beam_database.get<std::string>("scan_path_file");
    std::string const scan_path_file_format =
        beam_database.get<std::string>("scan_path_file_format");

    adamantine::ScanPath scan_path(scan_path_file, scan_path_file_format,
                                   units_optional_database);
    deposition_paths.push_back(adamantine::deposition_along_scan_path<dim>(
        geometry_database, scan_path));
  }

  return adamantine::merge_deposition_paths<dim>(deposition_paths);
}

template <int dim>
void write_activation_output(
    MPI_Comm const &communicator,
    dealii::parallel::distributed::Triangulation<dim> const &triangulation,
    std::filesystem::path const &output_directory,
    std::string const &filename_prefix, unsigned int const time_step,
    double const time, dealii::Vector<double> const &activation_time,
    std::vector<std::pair<double, std::string>> &times_filenames)
{
  dealii::DataOut<dim> data_out;
  data_out.attach_triangulation(triangulation);
  data_out.add_data_vector(activation_time, "activation_time",
                           dealii::DataOut<dim>::type_cell_data);
  data_out.build_patches();

  unsigned int const subdomain_id = triangulation.locally_owned_subdomain();
  std::string const time_step_string = dealii::Utilities::to_string(time_step);
  std::string const subdomain_string =
      dealii::Utilities::to_string(subdomain_id);
  std::string const local_filename = filename_prefix + "." + time_step_string +
                                     "." + subdomain_string + ".vtu";
  std::filesystem::path const local_path = output_directory / local_filename;

  std::ofstream output(local_path);
  if (!output)
    throw std::runtime_error("Unable to open " + local_path.string());
  dealii::DataOutBase::VtkFlags flags(time);
  data_out.set_flags(flags);
  data_out.write_vtu(output);

  MPI_Barrier(communicator);

  unsigned int const rank =
      dealii::Utilities::MPI::this_mpi_process(communicator);
  if (rank == 0)
  {
    unsigned int const n_processes =
        dealii::Utilities::MPI::n_mpi_processes(communicator);
    std::vector<std::string> filenames;
    filenames.reserve(n_processes);
    for (unsigned int process = 0; process < n_processes; ++process)
    {
      filenames.push_back(filename_prefix + "." + time_step_string + "." +
                          dealii::Utilities::to_string(process) + ".vtu");
    }

    std::string const pvtu_filename =
        filename_prefix + "." + time_step_string + ".pvtu";
    std::filesystem::path const pvtu_path = output_directory / pvtu_filename;
    std::ofstream pvtu_output(pvtu_path);
    if (!pvtu_output)
      throw std::runtime_error("Unable to open " + pvtu_path.string());
    data_out.write_pvtu_record(pvtu_output, filenames);
    times_filenames.emplace_back(time, pvtu_filename);
  }

  MPI_Barrier(communicator);
}

template <int dim>
void visualize_activation(MPI_Comm const &communicator,
                          boost::property_tree::ptree const &database,
                          std::filesystem::path const &output_directory)
{
  boost::optional<boost::property_tree::ptree const &> units_optional_database =
      database.get_child_optional("units");
  boost::property_tree::ptree const &geometry_database =
      database.get_child("geometry");

  adamantine::Geometry<dim> geometry(communicator, geometry_database,
                                     units_optional_database);
  dealii::parallel::distributed::Triangulation<dim> &triangulation =
      geometry.get_triangulation();

  // get_elements_to_activate() intentionally searches cells with active FE
  // index 1. FE_Nothing is used here to represent cells that have not yet
  // been activated, just as it is in the thermal solver. Cells below
  // material_height start with active FE index 0 because they already contain
  // material.
  dealii::hp::FECollection<dim> fe_collection;
  fe_collection.push_back(dealii::FE_Q<dim>(1));
  fe_collection.push_back(dealii::FE_Nothing<dim>());
  dealii::DoFHandler<dim> activation_dof_handler(triangulation);
  activation_dof_handler.distribute_dofs(fe_collection);

  double const material_height =
      geometry_database.get<double>("material_height", 1e9);

  dealii::Vector<double> activation_time(triangulation.n_active_cells());
  for (double &time : activation_time)
  {
    time = -1.;
  }

  for (auto const &cell :
       dealii::filter_iterators(activation_dof_handler.active_cell_iterators(),
                                dealii::IteratorFilters::LocallyOwnedCell()))
  {
    bool const initially_active =
        cell->center()[adamantine::axis<dim>::z] < material_height;
    cell->set_active_fe_index(initially_active ? 0 : 1);
    if (initially_active)
    {
      unsigned int const cell_id = cell->active_cell_index();
      activation_time[cell_id] = 0.;
    }
  }

  [[maybe_unused]] auto [material_deposition_boxes, deposition_times,
                         deposition_cos, deposition_sin] =
      read_deposition_data<dim>(database, geometry_database,
                                units_optional_database);

  auto elements_to_activate = adamantine::get_elements_to_activate(
      geometry, activation_dof_handler, material_deposition_boxes);

  if (dealii::Utilities::MPI::this_mpi_process(communicator) == 0)
    std::filesystem::create_directories(output_directory);
  MPI_Barrier(communicator);

  std::string const filename_prefix =
      database.get<std::string>("post_processor.filename_prefix", "activation");
  std::vector<std::pair<double, std::string>> times_filenames;

  unsigned int output_time_step = 0;
  write_activation_output(communicator, triangulation, output_directory,
                          filename_prefix, output_time_step, 0.,
                          activation_time, times_filenames);
  ++output_time_step;

  // The material-deposition API returns one cell list for every deposition
  // box. Apply those lists in chronological order and write one cumulative
  // snapshot whenever the deposition time changes.
  for (unsigned int i = 0; i < elements_to_activate.size(); ++i)
  {
    double const time = deposition_times[i];
    for (auto const &cell : elements_to_activate[i])
    {
      unsigned int const cell_id = cell->active_cell_index();
      if (activation_time[cell_id] < 0.)
      {
        activation_time[cell_id] = time;
      }
    }

    write_activation_output(communicator, triangulation, output_directory,
                            filename_prefix, output_time_step, time,
                            activation_time, times_filenames);
    ++output_time_step;
  }

  if (dealii::Utilities::MPI::this_mpi_process(communicator) == 0)
  {
    std::filesystem::path const pvd_path =
        output_directory / (filename_prefix + ".pvd");
    std::ofstream pvd_output(pvd_path);
    if (!pvd_output)
      throw std::runtime_error("Unable to open " + pvd_path.string());
    dealii::DataOutBase::write_pvd_record(pvd_output, times_filenames);
  }
}

int main(int argc, char *argv[])
{
  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(
      argc, argv, dealii::numbers::invalid_unsigned_int);
  MPI_Comm communicator = MPI_COMM_WORLD;

  unsigned int const rank =
      dealii::Utilities::MPI::this_mpi_process(communicator);
  try
  {
    namespace boost_po = boost::program_options;
    boost_po::options_description description("Options:");
    description.add_options()("help,h", "Produce help message.")(
        "input-file,i", boost_po::value<std::string>(),
        "Adamantine .info or .json file containing the geometry and scan "
        "path.")(
        "output-dir,o", boost_po::value<std::string>(),
        "Directory for the VTU/PVTU/PVD output; overrides the input file.");

    boost_po::variables_map map;
    auto parsed_line = boost_po::command_line_parser(argc, argv)
                           .options(description)
                           .allow_unregistered()
                           .run();
    boost_po::store(parsed_line, map);
    boost_po::notify(map);

    if (map.count("help") != 0)
    {
      if (rank == 0)
        std::cout << description << std::endl;
      return 0;
    }

    if (map.count("input-file") == 0)
      throw std::invalid_argument(
          "An input file is required; use --input-file.");

    std::string const input_filename = map["input-file"].as<std::string>();
    std::filesystem::path const input_path =
        std::filesystem::absolute(input_filename);
    std::filesystem::path output_directory;
    if (map.count("output-dir") != 0)
      output_directory =
          std::filesystem::absolute(map["output-dir"].as<std::string>());

    boost::property_tree::ptree database;
    std::string filename = input_path.string();
    if (std::filesystem::path(filename).extension().native() == ".json")
      boost::property_tree::json_parser::read_json(filename, database);
    else
      boost::property_tree::info_parser::read_info(filename, database);
    std::filesystem::current_path(input_path.parent_path());

    if (output_directory.empty())
    {
      std::string const configured_output_directory =
          database.get<std::string>("post_processor.output_dir", "");
      output_directory =
          configured_output_directory.empty()
              ? std::filesystem::current_path()
              : std::filesystem::absolute(configured_output_directory);
    }

    int const dim = database.get<int>("geometry.dim");
    if (dim == 2)
      visualize_activation<2>(communicator, database, output_directory);
    else if (dim == 3)
      visualize_activation<3>(communicator, database, output_directory);
    else
      throw std::invalid_argument("geometry.dim must be 2 or 3.");

    if (rank == 0)
      std::cout << "Activation visualization done." << std::endl;
  }
  catch (std::exception const &exception)
  {
    if (rank == 0)
      std::cerr << "Activation visualization failed: " << exception.what()
                << std::endl;
    return 1;
  }

  return 0;
}
