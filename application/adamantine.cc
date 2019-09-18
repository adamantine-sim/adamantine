/* Copyright (c) 2016, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#include "Geometry.hh"
#include "PostProcessor.hh"
#include "ThermalPhysics.hh"
#include "Timer.hh"
#include "utils.hh"
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <boost/property_tree/info_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <cmath>
#include <deal.II/base/mpi.h>
#include <deal.II/distributed/solution_transfer.h>
#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/numerics/error_estimator.h>
#include <iostream>

void initialize_timers(MPI_Comm const &communicator,
                       std::vector<adamantine::Timer> &timers)
{
  timers.push_back(adamantine::Timer(communicator, "Main"));
  timers.push_back(adamantine::Timer(communicator, "Refinement"));
  timers.push_back(adamantine::Timer(communicator, "Evolve One Time Step"));
  timers.push_back(adamantine::Timer(
      communicator, "Evolve One Time Step: evaluate_thermal_physics"));
  timers.push_back(adamantine::Timer(
      communicator, "Evolve One Time Step: id_minus_tau_J_inverse"));
  timers.push_back(adamantine::Timer(
      communicator, "Evolve One Time Step: evaluate_material_properties"));
}

template <int dim, int fe_degree, typename QuadratureType>
std::vector<std::unique_ptr<adamantine::ElectronBeam<dim>>> &
initialize(MPI_Comm const &communicator,
           boost::property_tree::ptree const &database,
           adamantine::Geometry<dim> &geometry,
           std::unique_ptr<adamantine::Physics<dim, double>> &thermal_physics)
{
  thermal_physics.reset(
      new adamantine::ThermalPhysics<dim, fe_degree, double, QuadratureType>(
          communicator, database, geometry));
  return static_cast<adamantine::ThermalPhysics<dim, fe_degree, double,
                                                QuadratureType> *>(
             thermal_physics.get())
      ->get_electron_beams();
}

template <int dim, int fe_degree>
std::vector<std::unique_ptr<adamantine::ElectronBeam<dim>>> &
initialize_quadrature(
    std::string const &quadrature_type, MPI_Comm const &communicator,
    boost::property_tree::ptree const &database,
    adamantine::Geometry<dim> &geometry,
    std::unique_ptr<adamantine::Physics<dim, double>> &thermal_physics)
{
  if (quadrature_type.compare("gauss") == 0)
    return initialize<dim, fe_degree, dealii::QGauss<1>>(
        communicator, database, geometry, thermal_physics);
  else
  {
    adamantine::ASSERT_THROW(quadrature_type.compare("lobatto") == 0,
                             "quadrature should be Gauss or Lobatto.");
    return initialize<dim, fe_degree, dealii::QGaussLobatto<1>>(
        communicator, database, geometry, thermal_physics);
  }
}

template <int dim>
std::vector<std::unique_ptr<adamantine::ElectronBeam<dim>>> &
initialize_thermal_physics(
    unsigned int fe_degree, std::string const &quadrature_type,
    MPI_Comm const &communicator, boost::property_tree::ptree const &database,
    adamantine::Geometry<dim> &geometry,
    std::unique_ptr<adamantine::Physics<dim, double>> &thermal_physics)
{
  switch (fe_degree)
  {
  case 1:
  {
    return initialize_quadrature<dim, 1>(quadrature_type, communicator,
                                         database, geometry, thermal_physics);
  }
  case 2:
  {
    return initialize_quadrature<dim, 2>(quadrature_type, communicator,
                                         database, geometry, thermal_physics);
  }
  case 3:
  {
    return initialize_quadrature<dim, 3>(quadrature_type, communicator,
                                         database, geometry, thermal_physics);
  }
  case 4:
  {
    return initialize_quadrature<dim, 4>(quadrature_type, communicator,
                                         database, geometry, thermal_physics);
  }
  case 5:
  {
    return initialize_quadrature<dim, 5>(quadrature_type, communicator,
                                         database, geometry, thermal_physics);
  }
  case 6:
  {
    return initialize_quadrature<dim, 6>(quadrature_type, communicator,
                                         database, geometry, thermal_physics);
  }
  case 7:
  {
    return initialize_quadrature<dim, 7>(quadrature_type, communicator,
                                         database, geometry, thermal_physics);
  }
  case 8:
  {
    return initialize_quadrature<dim, 8>(quadrature_type, communicator,
                                         database, geometry, thermal_physics);
  }
  case 9:
  {
    return initialize_quadrature<dim, 9>(quadrature_type, communicator,
                                         database, geometry, thermal_physics);
  }
  default:
  {
    adamantine::ASSERT_THROW(fe_degree == 10,
                             "fe_degree should be between 1 and 10.");
    return initialize_quadrature<dim, 10>(quadrature_type, communicator,
                                          database, geometry, thermal_physics);
  }
  }
}

template <int dim>
void refine_and_transfer(
    std::unique_ptr<adamantine::Physics<dim, double>> &thermal_physics,
    dealii::DoFHandler<dim> &dof_handler,
    dealii::LA::distributed::Vector<double> &solution)
{
  dealii::parallel::distributed::Triangulation<dim> &triangulation =
      dynamic_cast<dealii::parallel::distributed::Triangulation<dim> &>(
          const_cast<dealii::Triangulation<dim> &>(
              dof_handler.get_triangulation()));

  std::shared_ptr<adamantine::MaterialProperty<dim>> material_property =
      thermal_physics->get_material_property();

  dealii::parallel::distributed::SolutionTransfer<
      dim, dealii::LA::distributed::Vector<double>>
      solution_transfer(dof_handler);
  std::vector<dealii::parallel::distributed::SolutionTransfer<
      dim, dealii::LA::distributed::Vector<double>>>
  material_state(static_cast<unsigned int>(adamantine::MaterialState::SIZE),
                 dealii::parallel::distributed::SolutionTransfer<
                     dim, dealii::LA::distributed::Vector<double>>(
                     material_property->get_dof_handler()));

  // We need to update the ghost values before we can do the interpolation on
  // the new mesh.
  solution.update_ghost_values();

  // Prepare the Triangulation and the diffent SolutionTransfers for refinement
  triangulation.prepare_coarsening_and_refinement();
  solution_transfer.prepare_for_coarsening_and_refinement(solution);
  for (unsigned int i = 0;
       i < static_cast<unsigned int>(adamantine::MaterialState::SIZE); ++i)
    material_state[i].prepare_for_coarsening_and_refinement(
        material_property->get_state()[i]);

  // Execute the refinement
  triangulation.execute_coarsening_and_refinement();

  // Update the AffineConstraints and resize the solution
  thermal_physics->setup_dofs();
  thermal_physics->initialize_dof_vector(solution);

  // Update MaterialProperty DoFHandler and resize the state vectors
  material_property->reinit_dofs();

  // Interpolate the solution and the state onto the new mesh
  solution_transfer.interpolate(solution);
  for (unsigned int i = 0;
       i < static_cast<unsigned int>(adamantine::MaterialState::SIZE); ++i)
    material_state[i].interpolate(material_property->get_state()[i]);

#if ADAMANTINE_DEBUG
  // Check that we are not losing material
  std::array<dealii::LA::distributed::Vector<double>,
             static_cast<unsigned int>(adamantine::MaterialState::SIZE)>
      state = material_property->get_state();
  unsigned int const local_size = state[0].local_size();
  unsigned int constexpr n_material_states =
      static_cast<unsigned int>(adamantine::MaterialState::SIZE);
  for (unsigned int i = 0; i < local_size; ++i)
  {
    double material_ratio = 0.;
    for (unsigned int j = 0; j < n_material_states; ++j)
      material_ratio += state[j].local_element(i);
    adamantine::ASSERT(std::abs(material_ratio - 1.) < 1e-14,
                       "Material is lost.");
  }
#endif
}

template <int dim>
std::vector<typename dealii::parallel::distributed::Triangulation<
    dim>::active_cell_iterator>
compute_cells_to_refine(
    dealii::parallel::distributed::Triangulation<dim> &triangulation,
    double const time, double const next_refinement_time,
    unsigned int const n_time_steps,
    std::vector<std::unique_ptr<adamantine::ElectronBeam<dim>>> &electron_beams)
{
  // The save_time()/rewind_time() functions are used to save the current time
  // and row in the list of beam positions. It does nothing when the position is
  // given by an analytic formula.
  for (auto &beam : electron_beams)
    beam->save_time();

  // Compute the position of the beams between time and next_refinement_time and
  // refine the mesh where the source is greater than 1e-15. This cut-off is due
  // to the fact that the source is gaussian and thus never strictly zero. If
  // the beams intersect, some cells will appear twice in the vector. This is
  // not a problem.
  std::vector<typename dealii::parallel::distributed::Triangulation<
      dim>::active_cell_iterator>
      cells_to_refine;
  for (unsigned int i = 0; i < n_time_steps; ++i)
  {
    double const current_time = time + static_cast<double>(i) /
                                           static_cast<double>(n_time_steps) *
                                           (next_refinement_time - time);
    for (auto &beam : electron_beams)
    {
      beam->set_time(current_time);
      for (auto cell : dealii::filter_iterators(
               triangulation.active_cell_iterators(),
               dealii::IteratorFilters::LocallyOwnedCell()))
        if (beam->value(cell->center()) > 1e-15)
          cells_to_refine.push_back(cell);
    }
  }

  for (auto &beam : electron_beams)
    beam->rewind_time();

  return cells_to_refine;
}

template <int dim, int fe_degree>
void refine_mesh(
    std::unique_ptr<adamantine::Physics<dim, double>> &thermal_physics,
    dealii::LA::distributed::Vector<double> &solution,
    std::vector<std::unique_ptr<adamantine::ElectronBeam<dim>>> &electron_beams,
    double const time, double const next_refinement_time,
    unsigned int const time_steps_refinement,
    boost::property_tree::ptree const &refinement_database)
{
  dealii::DoFHandler<dim> &dof_handler = thermal_physics->get_dof_handler();
  // Use the Kelly error estimator to refine the mesh. This is done so that the
  // part of the domain that were heated stay refined.
  unsigned int const n_kelly_refinements =
      refinement_database.get("n_heat_refinements", 2);
  double coarsening_fraction = 0.3;
  double refining_fraction = 0.6;
  double cells_fraction = refinement_database.get("heat_cell_ratio", 1.);
  dealii::parallel::distributed::Triangulation<dim> &triangulation =
      dynamic_cast<dealii::parallel::distributed::Triangulation<dim> &>(
          const_cast<dealii::Triangulation<dim> &>(
              dof_handler.get_triangulation()));
  // Number of times the mesh on the beam paths will be refined and maximum
  // number of time a cell can be refined.
  unsigned int const n_beam_refinements =
      refinement_database.get("n_beam_refinements", 2);
  int max_level = refinement_database.get<int>("max_level");
  for (unsigned int i = 0; i < n_kelly_refinements; ++i)
  {
    // Estimate the error. For simplicity, always use dealii::QGauss
    dealii::Vector<float> estimated_error_per_cell(
        triangulation.n_active_cells());
    dealii::KellyErrorEstimator<dim>::estimate(
        dof_handler, dealii::QGauss<dim - 1>(fe_degree + 1),
        typename dealii::FunctionMap<dim>::type(), solution,
        estimated_error_per_cell, dealii::ComponentMask(), nullptr, 0,
        triangulation.locally_owned_subdomain());

    // Flag the cells for refinement.
    unsigned int new_n_cells = static_cast<unsigned int>(
        cells_fraction *
        static_cast<double>(triangulation.n_global_active_cells()));
    dealii::GridRefinement::refine_and_coarsen_fixed_fraction(
        triangulation, estimated_error_per_cell, refining_fraction,
        coarsening_fraction, new_n_cells);

    // Don't refine cells that are already as much refined as it is allowed.
    for (auto cell :
         dealii::filter_iterators(triangulation.active_cell_iterators(),
                                  dealii::IteratorFilters::LocallyOwnedCell()))
      if (cell->level() >= max_level)
        cell->clear_refine_flag();

    // Execute the refinement and transfer the solution onto the new mesh.
    refine_and_transfer(thermal_physics, dof_handler, solution);
  }

  // Refine the mesh along the trajectory of the sources.
  for (unsigned int i = 0; i < n_beam_refinements; ++i)
  {
    // Compute the cells to be refined.
    std::vector<typename dealii::parallel::distributed::Triangulation<
        dim>::active_cell_iterator>
        cells_to_refine =
            compute_cells_to_refine(triangulation, time, next_refinement_time,
                                    time_steps_refinement, electron_beams);

    // Flag the cells for refinement.
    for (auto &cell : cells_to_refine)
      if (cell->level() < max_level)
        cell->set_refine_flag();

    // Execute the refinement and transfer the solution onto the new mesh.
    refine_and_transfer(thermal_physics, dof_handler, solution);
  }

  // Recompute the inverse of the mass matrix and update the material
  // properties.
  thermal_physics->reinit();
}

template <int dim>
void refine_mesh(
    std::unique_ptr<adamantine::Physics<dim, double>> &thermal_physics,
    dealii::LA::distributed::Vector<double> &solution,
    std::vector<std::unique_ptr<adamantine::ElectronBeam<dim>>> &electron_beams,
    double const time, double const next_refinement_time,
    unsigned int const time_steps_refinement,
    boost::property_tree::ptree const &refinement_database,
    unsigned int const fe_degree)
{
  switch (fe_degree)
  {
  case 1:
  {
    refine_mesh<dim, 1>(thermal_physics, solution, electron_beams, time,
                        next_refinement_time, time_steps_refinement,
                        refinement_database);
    break;
  }
  case 2:
  {
    refine_mesh<dim, 2>(thermal_physics, solution, electron_beams, time,
                        next_refinement_time, time_steps_refinement,
                        refinement_database);
    break;
  }
  case 3:
  {
    refine_mesh<dim, 3>(thermal_physics, solution, electron_beams, time,
                        next_refinement_time, time_steps_refinement,
                        refinement_database);
    break;
  }
  case 4:
  {
    refine_mesh<dim, 4>(thermal_physics, solution, electron_beams, time,
                        next_refinement_time, time_steps_refinement,
                        refinement_database);
    break;
  }
  case 5:
  {
    refine_mesh<dim, 5>(thermal_physics, solution, electron_beams, time,
                        next_refinement_time, time_steps_refinement,
                        refinement_database);
    break;
  }
  case 6:
  {
    refine_mesh<dim, 6>(thermal_physics, solution, electron_beams, time,
                        next_refinement_time, time_steps_refinement,
                        refinement_database);
    break;
  }
  case 7:
  {
    refine_mesh<dim, 7>(thermal_physics, solution, electron_beams, time,
                        next_refinement_time, time_steps_refinement,
                        refinement_database);
    break;
  }
  case 8:
  {
    refine_mesh<dim, 8>(thermal_physics, solution, electron_beams, time,
                        next_refinement_time, time_steps_refinement,
                        refinement_database);
    break;
  }
  case 9:
  {
    refine_mesh<dim, 9>(thermal_physics, solution, electron_beams, time,
                        next_refinement_time, time_steps_refinement,
                        refinement_database);
    break;
  }
  case 10:
  {
    refine_mesh<dim, 10>(thermal_physics, solution, electron_beams, time,
                         next_refinement_time, time_steps_refinement,
                         refinement_database);
    break;
  }
  default:
  {
    adamantine::ASSERT_THROW(false, "fe_degree should be between 1 and 10.");
  }
  }
}

template <int dim>
void run(
    MPI_Comm const &communicator,
    std::unique_ptr<adamantine::Physics<dim, double>> &thermal_physics,
    adamantine::PostProcessor<dim> &post_processor,
    std::vector<std::unique_ptr<adamantine::ElectronBeam<dim>>> &electron_beams,
    boost::property_tree::ptree const &time_stepping_database,
    boost::property_tree::ptree const &refinement_database,
    unsigned int const fe_degree, double const initial_temperature,
    std::vector<adamantine::Timer> &timers)
{
  thermal_physics->setup_dofs();
  thermal_physics->reinit();
  dealii::LA::distributed::Vector<double> solution;
  thermal_physics->initialize_dof_vector(initial_temperature, solution);
  unsigned int progress = 0;
  unsigned int cycle = 0;
  unsigned int n_time_step = 0;
  double time = 0.;
  // Output the initial solution
  dealii::AffineConstraints<double> &affine_constraints =
      thermal_physics->get_affine_constraints();
  affine_constraints.distribute(solution);
  post_processor.output_pvtu(cycle, n_time_step, time, solution);
  ++n_time_step;

  bool const verbose_refinement = refinement_database.get("verbose", false);
  unsigned int const time_steps_refinement =
      refinement_database.get("time_steps_between_refinement", 10);
  double next_refinement_time = time;
  double time_step = time_stepping_database.get<double>("time_step");
  double const duration = time_stepping_database.get<double>("duration");
  while (time < duration)
  {
    if ((time + time_step) > duration)
      time_step = duration - time;
    unsigned int rank = dealii::Utilities::MPI::this_mpi_process(communicator);

    // Refine the mesh after time_steps_refinement time steps or when time is
    // greater or equal than the next predicted time for refinement. This is
    // necessary when using an embedded method.
    if (((n_time_step % time_steps_refinement) == 0) ||
        (time >= next_refinement_time))
    {
      next_refinement_time = time + time_steps_refinement * time_step;
      timers[adamantine::refine].start();
      refine_mesh(thermal_physics, solution, electron_beams, time,
                  next_refinement_time, time_steps_refinement,
                  refinement_database, fe_degree);
      timers[adamantine::refine].stop();
      if ((rank == 0) && (verbose_refinement == true))
        std::cout << "n_dofs: " << thermal_physics->get_dof_handler().n_dofs()
                  << std::endl;
    }

    // time can be different than time + time_step if an embedded scheme is
    // used.
    timers[adamantine::evol_time].start();
    time = thermal_physics->evolve_one_time_step(time, time_step, solution,
                                                 timers);
    timers[adamantine::evol_time].stop();

    // Get the new time step
    time_step = thermal_physics->get_delta_t_guess();

    // Output progress on screen
    if (rank == 0)
    {
      double adim_time = time / (duration / 10.);
      double int_part = 0;
      std::modf(adim_time, &int_part);
      if (int_part > progress)
      {
        std::cout << int_part * 10 << '%' << " completed" << std::endl;
        ++progress;
      }
    }

    // Output the solution
    affine_constraints.distribute(solution);
    post_processor.output_pvtu(cycle, n_time_step, time, solution);
    ++n_time_step;
  }
  post_processor.output_pvd();
}

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
      std::unique_ptr<adamantine::Physics<2, double>> thermal_physics;
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
      std::unique_ptr<adamantine::Physics<3, double>> thermal_physics;
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
