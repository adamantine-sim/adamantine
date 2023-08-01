/* Copyright (c) 2016 - 2023, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#ifndef ADAMANTINE_HH
#define ADAMANTINE_HH

#include <DataAssimilator.hh>
#include <ExperimentalData.hh>
#include <Geometry.hh>
#include <MaterialProperty.hh>
#include <MechanicalPhysics.hh>
#include <MemoryBlock.hh>
#include <PointCloud.hh>
#include <PostProcessor.hh>
#include <RayTracing.hh>
#include <ThermalPhysics.hh>
#include <ThermalPhysicsInterface.hh>
#include <Timer.hh>
#include <ensemble_management.hh>
#include <experimental_data_utils.hh>
#include <material_deposition.hh>
#include <utils.hh>

#include <deal.II/base/index_set.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/types.h>
#include <deal.II/distributed/cell_data_transfer.templates.h>
#include <deal.II/distributed/solution_transfer.h>
#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/vector_operation.h>
#include <deal.II/numerics/error_estimator.h>

#include <boost/program_options.hpp>
#include <boost/property_tree/info_parser.hpp>
#include <boost/property_tree/ptree.hpp>

#include <memory>
#include <type_traits>
#include <unordered_map>

#ifdef ADAMANTINE_WITH_CALIPER
#include <caliper/cali.h>
#endif

#include <cmath>
#include <iostream>

template <int dim, typename MemorySpaceType,
          std::enable_if_t<
              std::is_same<MemorySpaceType, dealii::MemorySpace::Host>::value,
              int> = 0>
void output_pvtu(
    adamantine::PostProcessor<dim> &post_processor, unsigned int cycle,
    unsigned int n_time_step, double time,
    std::unique_ptr<
        adamantine::ThermalPhysicsInterface<dim, MemorySpaceType>> const
        &thermal_physics,
    dealii::LinearAlgebra::distributed::Vector<double, MemorySpaceType>
        &temperature,
    std::unique_ptr<adamantine::MechanicalPhysics<dim, MemorySpaceType>> const
        &mechanical_physics,
    [[maybe_unused]] dealii::LA::distributed::Vector<
        double, dealii::MemorySpace::Host> &displacement,
    adamantine::MaterialProperty<dim, MemorySpaceType> const
        &material_properties,
    std::vector<adamantine::Timer> &timers)
{
#ifdef ADAMANTINE_WITH_CALIPER
  CALI_CXX_MARK_FUNCTION;
#endif
  timers[adamantine::output].start();
  if (thermal_physics)
  {
    thermal_physics->get_affine_constraints().distribute(temperature);
    if (mechanical_physics)
    {
      mechanical_physics->get_affine_constraints().distribute(displacement);
      post_processor.write_output(cycle, n_time_step, time, temperature,
                                  displacement, material_properties.get_state(),
                                  material_properties.get_dofs_map(),
                                  material_properties.get_dof_handler());
    }
    else
    {
      post_processor.write_thermal_output(
          cycle, n_time_step, time, temperature,
          material_properties.get_state(), material_properties.get_dofs_map(),
          material_properties.get_dof_handler());
    }
  }
  else
  {
    mechanical_physics->get_affine_constraints().distribute(displacement);
    post_processor.write_mechanical_output(
        cycle, n_time_step, time, displacement, material_properties.get_state(),
        material_properties.get_dofs_map(),
        material_properties.get_dof_handler());
  }
  timers[adamantine::output].stop();
}

#ifdef ADAMANTINE_HAVE_CUDA
template <int dim, typename MemorySpaceType,
          std::enable_if_t<
              std::is_same<MemorySpaceType, dealii::MemorySpace::CUDA>::value,
              int> = 0>
void output_pvtu(
    adamantine::PostProcessor<dim> &post_processor, unsigned int cycle,
    unsigned int n_time_step, double time,
    std::unique_ptr<
        adamantine::ThermalPhysicsInterface<dim, MemorySpaceType>> const
        &thermal_physics,
    dealii::LinearAlgebra::distributed::Vector<double, MemorySpaceType>
        &temperature,
    std::unique_ptr<adamantine::MechanicalPhysics<dim, MemorySpaceType>> const
        &mechanical_physics,
    dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host>
        &displacement,
    adamantine::MaterialProperty<dim, MemorySpaceType> const
        &material_properties,
    std::vector<adamantine::Timer> &timers)
{
#ifdef ADAMANTINE_WITH_CALIPER
  CALI_CXX_MARK_FUNCTION;
#endif
  timers[adamantine::output].start();
  auto state = material_properties.get_state();
  adamantine::MemoryBlock<double, dealii::MemorySpace::Host> state_host(
      state.extent(0), state.extent(1));
  adamantine::MemoryBlockView<double, dealii::MemorySpace::Host>
      state_host_view(state_host);
  adamantine::deep_copy(state_host_view, state);
  if (thermal_physics)
  {
    dealii::LinearAlgebra::distributed::Vector<double,
                                               dealii::MemorySpace::Host>
        temperature_host(temperature.get_partitioner());
    temperature_host.import(temperature, dealii::VectorOperation::insert);
    thermal_physics->get_affine_constraints().distribute(temperature_host);
    if (mechanical_physics)
    {
      mechanical_physics->get_affine_constraints().distribute(displacement);
      post_processor.write_output(cycle, n_time_step, time, temperature_host,
                                  displacement, state_host_view,
                                  material_properties.get_dofs_map(),
                                  material_properties.get_dof_handler());
    }
    else
    {
      post_processor.write_thermal_output(
          cycle, n_time_step, time, temperature_host, state_host_view,
          material_properties.get_dofs_map(),
          material_properties.get_dof_handler());
    }
  }
  else
  {
    mechanical_physics->get_affine_constraints().distribute(displacement);
    post_processor.write_mechanical_output(
        cycle, n_time_step, time, displacement, state_host_view,
        material_properties.get_dofs_map(),
        material_properties.get_dof_handler());
  }
  timers[adamantine::output].stop();
}
#endif

template <int dim, typename MemorySpaceType,
          std::enable_if_t<
              std::is_same<MemorySpaceType, dealii::MemorySpace::Host>::value,
              int> = 0>
dealii::Vector<float> estimate_error(
    dealii::parallel::distributed::Triangulation<dim> const &triangulation,
    dealii::DoFHandler<dim> const &dof_handler, int fe_degree,
    dealii::LA::distributed::Vector<double, MemorySpaceType> const &solution)
{
  dealii::Vector<float> estimated_error_per_cell(
      triangulation.n_active_cells());
  dealii::KellyErrorEstimator<dim>::estimate(
      dof_handler, dealii::QGauss<dim - 1>(fe_degree + 1),
      std::map<dealii::types::boundary_id,
               const dealii::Function<dim, double> *>(),
      solution, estimated_error_per_cell, dealii::ComponentMask(), nullptr, 0,
      triangulation.locally_owned_subdomain());

  return estimated_error_per_cell;
}

#ifdef ADAMANTINE_HAVE_CUDA
template <int dim, typename MemorySpaceType,
          std::enable_if_t<
              std::is_same<MemorySpaceType, dealii::MemorySpace::CUDA>::value,
              int> = 0>
dealii::Vector<float> estimate_error(
    dealii::parallel::distributed::Triangulation<dim> const &triangulation,
    dealii::DoFHandler<dim> const &dof_handler, int fe_degree,
    dealii::LA::distributed::Vector<double, MemorySpaceType> const &solution)
{
  dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host>
      solution_host(solution.get_partitioner());
  solution_host.import(solution, dealii::VectorOperation::insert);
  dealii::Vector<float> estimated_error_per_cell(
      triangulation.n_active_cells());
  dealii::KellyErrorEstimator<dim>::estimate(
      dof_handler, dealii::QGauss<dim - 1>(fe_degree + 1),
      std::map<dealii::types::boundary_id,
               const dealii::Function<dim, double> *>(),
      solution_host, estimated_error_per_cell, dealii::ComponentMask(), nullptr,
      0, triangulation.locally_owned_subdomain());

  return estimated_error_per_cell;
}
#endif

// inlining this function so we can have in the header
inline void initialize_timers(MPI_Comm const &communicator,
                              std::vector<adamantine::Timer> &timers)
{
  timers.push_back(adamantine::Timer(communicator, "Main"));
  timers.push_back(adamantine::Timer(communicator, "Refinement"));
  timers.push_back(adamantine::Timer(communicator, "Add Material, Search"));
  timers.push_back(adamantine::Timer(communicator, "Add Material, Activate"));
  timers.push_back(
      adamantine::Timer(communicator, "Data Assimilation, Exp. Data"));
  timers.push_back(
      adamantine::Timer(communicator, "Data Assimilation, DOF Mapping"));
  timers.push_back(
      adamantine::Timer(communicator, "Data Assimilation, Cov. Sparsity"));
  timers.push_back(
      adamantine::Timer(communicator, "Data Assimilation, Exp. Cov."));
  timers.push_back(
      adamantine::Timer(communicator, "Data Assimilation, Update Ensemble"));
  timers.push_back(adamantine::Timer(communicator, "Evolve One Time Step"));
  timers.push_back(adamantine::Timer(
      communicator, "Evolve One Time Step: evaluate_thermal_physics"));
  timers.push_back(adamantine::Timer(
      communicator, "Evolve One Time Step: id_minus_tau_J_inverse"));
  timers.push_back(adamantine::Timer(
      communicator, "Evolve One Time Step: evaluate_material_properties"));
  timers.push_back(adamantine::Timer(communicator, "Output"));
}

template <int dim, int fe_degree, typename MemorySpaceType,
          typename QuadratureType>
std::unique_ptr<adamantine::ThermalPhysicsInterface<dim, MemorySpaceType>>
initialize(
    MPI_Comm const &communicator, boost::property_tree::ptree const &database,
    adamantine::Geometry<dim> &geometry,
    adamantine::MaterialProperty<dim, MemorySpaceType> &material_properties)
{
  return std::make_unique<adamantine::ThermalPhysics<
      dim, fe_degree, MemorySpaceType, QuadratureType>>(
      communicator, database, geometry, material_properties);
}

template <int dim, int fe_degree, typename MemorySpaceType>
std::unique_ptr<adamantine::ThermalPhysicsInterface<dim, MemorySpaceType>>
initialize_quadrature(
    std::string const &quadrature_type, MPI_Comm const &communicator,
    boost::property_tree::ptree const &database,
    adamantine::Geometry<dim> &geometry,
    adamantine::MaterialProperty<dim, MemorySpaceType> &material_properties)
{
  if (quadrature_type.compare("gauss") == 0)
    return initialize<dim, fe_degree, MemorySpaceType, dealii::QGauss<1>>(
        communicator, database, geometry, material_properties);
  else
  {
    adamantine::ASSERT_THROW(quadrature_type.compare("lobatto") == 0,
                             "quadrature should be Gauss or Lobatto.");
    return initialize<dim, fe_degree, MemorySpaceType,
                      dealii::QGaussLobatto<1>>(communicator, database,
                                                geometry, material_properties);
  }
}

template <int dim, typename MemorySpaceType>
std::unique_ptr<adamantine::ThermalPhysicsInterface<dim, MemorySpaceType>>
initialize_thermal_physics(
    unsigned int fe_degree, std::string const &quadrature_type,
    MPI_Comm const &communicator, boost::property_tree::ptree const &database,
    adamantine::Geometry<dim> &geometry,
    adamantine::MaterialProperty<dim, MemorySpaceType> &material_properties)
{
  switch (fe_degree)
  {
  case 1:
  {
    return initialize_quadrature<dim, 1, MemorySpaceType>(
        quadrature_type, communicator, database, geometry, material_properties);
  }
  case 2:
  {
    return initialize_quadrature<dim, 2, MemorySpaceType>(
        quadrature_type, communicator, database, geometry, material_properties);
  }
  case 3:
  {
    return initialize_quadrature<dim, 3, MemorySpaceType>(
        quadrature_type, communicator, database, geometry, material_properties);
  }
  case 4:
  {
    return initialize_quadrature<dim, 4, MemorySpaceType>(
        quadrature_type, communicator, database, geometry, material_properties);
  }
  case 5:
  {
    return initialize_quadrature<dim, 5, MemorySpaceType>(
        quadrature_type, communicator, database, geometry, material_properties);
  }
  case 6:
  {
    return initialize_quadrature<dim, 6, MemorySpaceType>(
        quadrature_type, communicator, database, geometry, material_properties);
  }
  case 7:
  {
    return initialize_quadrature<dim, 7, MemorySpaceType>(
        quadrature_type, communicator, database, geometry, material_properties);
  }
  case 8:
  {
    return initialize_quadrature<dim, 8, MemorySpaceType>(
        quadrature_type, communicator, database, geometry, material_properties);
  }
  case 9:
  {
    return initialize_quadrature<dim, 9, MemorySpaceType>(
        quadrature_type, communicator, database, geometry, material_properties);
  }
  default:
  {
    adamantine::ASSERT_THROW(fe_degree == 10,
                             "fe_degree should be between 1 and 10.");
    return initialize_quadrature<dim, 10, MemorySpaceType>(
        quadrature_type, communicator, database, geometry, material_properties);
  }
  }
}

template <int dim, typename MemorySpaceType>
void refine_and_transfer(
    std::unique_ptr<adamantine::ThermalPhysicsInterface<dim, MemorySpaceType>>
        &thermal_physics,
    adamantine::MaterialProperty<dim, MemorySpaceType> &material_properties,
    dealii::DoFHandler<dim> &dof_handler,
    dealii::LA::distributed::Vector<double, MemorySpaceType> &solution)
{
#ifdef ADAMANTINE_WITH_CALIPER
  CALI_CXX_MARK_FUNCTION;
#endif

  dealii::parallel::distributed::Triangulation<dim> &triangulation =
      dynamic_cast<dealii::parallel::distributed::Triangulation<dim> &>(
          const_cast<dealii::Triangulation<dim> &>(
              dof_handler.get_triangulation()));

  // Update the material state from the ThermalOperator to MaterialProperty
  // because, for now, we need to use state from MaterialProperty to perform the
  // transfer to the refined mesh.
  thermal_physics->set_state_to_material_properties();

  // Transfer of the solution
  dealii::parallel::distributed::SolutionTransfer<
      dim, dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host>>
      solution_transfer(dof_handler);

  // Transfer material state
  unsigned int const direction_data_size = 2;
  unsigned int const phase_history_data_size = 1;
  unsigned int constexpr n_material_states =
      static_cast<unsigned int>(adamantine::MaterialState::SIZE);
  std::vector<std::vector<double>> data_to_transfer;
  std::vector<double> dummy_cell_data(n_material_states + direction_data_size +
                                          phase_history_data_size,
                                      std::numeric_limits<double>::infinity());
  adamantine::MemoryBlockView<double, MemorySpaceType> material_state_view =
      material_properties.get_state();
  adamantine::MemoryBlock<double, dealii::MemorySpace::Host>
      material_state_host(material_state_view.extent(0),
                          material_state_view.extent(1));
  typename decltype(material_state_view)::memory_space memspace;
  adamantine::deep_copy(material_state_host.data(), dealii::MemorySpace::Host{},
                        material_state_view.data(), memspace,
                        material_state_view.size());

  adamantine::MemoryBlockView<double, dealii::MemorySpace::Host>
      state_host_view(material_state_host);
  unsigned int cell_id = 0;
  unsigned int activated_cell_id = 0;
  for (auto const &cell : dof_handler.active_cell_iterators())
  {
    if (cell->is_locally_owned())
    {
      std::vector<double> cell_data(n_material_states + direction_data_size +
                                    phase_history_data_size);
      for (unsigned int i = 0; i < n_material_states; ++i)
        cell_data[i] = state_host_view(i, cell_id);
      if (cell->active_fe_index() == 0)
      {
        cell_data[n_material_states] =
            thermal_physics->get_deposition_cos(activated_cell_id);
        cell_data[n_material_states + 1] =
            thermal_physics->get_deposition_sin(activated_cell_id);

        if (thermal_physics->get_has_melted(activated_cell_id))
          cell_data[n_material_states + direction_data_size] = 1.0;
        else
          cell_data[n_material_states + direction_data_size] = 0.0;

        ++activated_cell_id;
      }
      else
      {
        cell_data[n_material_states] = std::numeric_limits<double>::infinity();
        cell_data[n_material_states + 1] =
            std::numeric_limits<double>::infinity();
        cell_data[n_material_states + direction_data_size] =
            std::numeric_limits<double>::infinity();
      }
      data_to_transfer.push_back(cell_data);
      ++cell_id;
    }
    else
    {
      data_to_transfer.push_back(dummy_cell_data);
    }
  }

  // Prepare the Triangulation and the diffent data transfer objects for
  // refinement
  triangulation.prepare_coarsening_and_refinement();
  // Prepare for refinement of the solution
  dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host>
      solution_host;
  if constexpr (std::is_same_v<MemorySpaceType, dealii::MemorySpace::Host>)
  {
    // We need to apply the constraints before the mesh transfer
    thermal_physics->get_affine_constraints().distribute(solution);
    // We need to update the ghost values before we can do the interpolation on
    // the new mesh.
    solution.update_ghost_values();
    solution_transfer.prepare_for_coarsening_and_refinement(solution);
  }
  else
  {
    solution_host.reinit(solution.get_partitioner());
    solution_host.import(solution, dealii::VectorOperation::insert);
    // We need to apply the constraints before the mesh transfer
    thermal_physics->get_affine_constraints().distribute(solution_host);
    // We need to update the ghost values before we can do the interpolation on
    // the new mesh.
    solution_host.update_ghost_values();
    solution_transfer.prepare_for_coarsening_and_refinement(solution_host);
  }

  dealii::parallel::distributed::CellDataTransfer<
      dim, dim, std::vector<std::vector<double>>>
      cell_data_trans(triangulation);
  cell_data_trans.prepare_for_coarsening_and_refinement(data_to_transfer);

#ifdef ADAMANTINE_WITH_CALIPER
  CALI_MARK_BEGIN("refine triangulation");
#endif
  // Execute the refinement
  triangulation.execute_coarsening_and_refinement();
#ifdef ADAMANTINE_WITH_CALIPER
  CALI_MARK_END("refine triangulation");
#endif

  // Update the AffineConstraints and resize the solution
  thermal_physics->setup_dofs();
  thermal_physics->initialize_dof_vector(solution);

  // Update MaterialProperty DoFHandler and resize the state vectors
  material_properties.reinit_dofs();

  // Interpolate the solution
  if constexpr (std::is_same_v<MemorySpaceType, dealii::MemorySpace::Host>)
  {
    solution_transfer.interpolate(solution);
  }
  else
  {
    solution_host.reinit(solution.get_partitioner());
    solution_transfer.interpolate(solution_host);
    solution.import(solution_host, dealii::VectorOperation::insert);
  }

  // Unpack the material state and repopulate the material state
  std::vector<std::vector<double>> transferred_data(
      triangulation.n_active_cells(),
      std::vector<double>(n_material_states + direction_data_size +
                          phase_history_data_size));
  cell_data_trans.unpack(transferred_data);
  material_state_view = material_properties.get_state();
  material_state_host.reinit(material_state_view.extent(0),
                             material_state_view.extent(1));
  state_host_view.reinit(material_state_host);
  unsigned int total_cell_id = 0;
  cell_id = 0;
  std::vector<double> transferred_cos;
  std::vector<double> transferred_sin;
  std::vector<bool> has_melted;
  for (auto const &cell : dof_handler.active_cell_iterators())
  {
    if (cell->is_locally_owned())
    {
      for (unsigned int i = 0; i < n_material_states; ++i)
      {
        state_host_view(i, cell_id) = transferred_data[total_cell_id][i];
      }
      if (cell->active_fe_index() == 0)
      {
        transferred_cos.push_back(
            transferred_data[total_cell_id][n_material_states]);
        transferred_sin.push_back(
            transferred_data[total_cell_id][n_material_states + 1]);

        // Convert from double back to bool
        if (transferred_data[total_cell_id]
                            [n_material_states + direction_data_size] > 0.5)
          has_melted.push_back(true);
        else
          has_melted.push_back(false);
      }
      ++cell_id;
    }
    ++total_cell_id;
  }

  // Update the deposition cos and sin
  thermal_physics->set_material_deposition_orientation(transferred_cos,
                                                       transferred_sin);

  // Update the melted indicator
  thermal_physics->set_has_melted_vector(has_melted);

  // Copy the data back to material_property
  adamantine::deep_copy(material_state_view.data(), memspace,
                        material_state_host.data(), dealii::MemorySpace::Host{},
                        material_state_view.size());

  // Update the material states in the ThermalOperator
  thermal_physics->get_state_from_material_properties();

#if ADAMANTINE_DEBUG
  // Check that we are not losing material
  cell_id = 0;
  for (auto const &cell : dof_handler.active_cell_iterators())
  {
    if (cell->is_locally_owned())
    {
      double material_ratio = 0.;
      for (unsigned int i = 0; i < n_material_states; ++i)
      {
        material_ratio += state_host_view(i, cell_id);
      }
      ASSERT(std::abs(material_ratio - 1.) < 1e-14, "Material is lost.");
      ++cell_id;
    }
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
    std::vector<std::shared_ptr<adamantine::HeatSource<dim>>> &heat_sources,
    double const current_source_height, double const refinement_beam_cutoff)
{

  // Compute the position of the beams between time and next_refinement_time and
  // refine the mesh where the source is greater than refinement_beam_cutoff.
  // This cut-off is due to the fact that the source is gaussian and thus never
  // strictly zero. If the beams intersect, some cells will appear twice in the
  // vector. This is not a problem.
  std::vector<typename dealii::parallel::distributed::Triangulation<
      dim>::active_cell_iterator>
      cells_to_refine;
  for (unsigned int i = 0; i < n_time_steps; ++i)
  {
    double const current_time = time + static_cast<double>(i) /
                                           static_cast<double>(n_time_steps) *
                                           (next_refinement_time - time);
    for (auto &beam : heat_sources)
    {
      beam->update_time(current_time);
      for (auto cell : dealii::filter_iterators(
               triangulation.active_cell_iterators(),
               dealii::IteratorFilters::LocallyOwnedCell()))
      {
        // Check the value at the center of the cell faces. For most cases this
        // should be sufficient, but if the beam is small compared to the
        // coarsest mesh we may need to add other points to check (e.g.
        // quadrature points, vertices).
        for (unsigned int f = 0; f < cell->reference_cell().n_faces(); ++f)
        {
          if (beam->value(cell->face(f)->center(), current_source_height) >
              refinement_beam_cutoff)
          {
            cells_to_refine.push_back(cell);
            break;
          }
        }
      }
    }
  }

  return cells_to_refine;
}

template <int dim, int fe_degree, typename MemorySpaceType>
void refine_mesh(
    std::unique_ptr<adamantine::ThermalPhysicsInterface<dim, MemorySpaceType>>
        &thermal_physics,
    adamantine::MaterialProperty<dim, MemorySpaceType> &material_properties,
    dealii::LA::distributed::Vector<double, MemorySpaceType> &solution,
    std::vector<std::shared_ptr<adamantine::HeatSource<dim>>> &heat_sources,
    double const time, double const next_refinement_time,
    unsigned int const time_steps_refinement,
    boost::property_tree::ptree const &refinement_database)
{
#ifdef ADAMANTINE_WITH_CALIPER
  CALI_CXX_MARK_FUNCTION;
#endif
  dealii::DoFHandler<dim> &dof_handler = thermal_physics->get_dof_handler();
  // Use the Kelly error estimator to refine the mesh. This is done so that the
  // part of the domain that were heated stay refined.
  // PropertyTreeInput refinement.n_heat_refinements
  unsigned int const n_kelly_refinements =
      refinement_database.get("n_heat_refinements", 2);
  double coarsening_fraction = 0.3;
  double refining_fraction = 0.6;
  // PropertyTreeInput refinement.heat_cell_ratio
  double cells_fraction = refinement_database.get("heat_cell_ratio", 1.);
  dealii::parallel::distributed::Triangulation<dim> &triangulation =
      dynamic_cast<dealii::parallel::distributed::Triangulation<dim> &>(
          const_cast<dealii::Triangulation<dim> &>(
              dof_handler.get_triangulation()));
  // Number of times the mesh on the beam paths will be refined and maximum
  // number of time a cell can be refined.
  // PropertyTreeInput refinement.n_beam_refinements
  unsigned int const n_beam_refinements =
      refinement_database.get("n_beam_refinements", 2);
  // PropertyTreeInput refinement.max_level
  int max_level = refinement_database.get<int>("max_level");

  // PropertyTreeInput refinement.beam_cutoff
  const double refinement_beam_cutoff =
      refinement_database.get<double>("beam_cutoff", 1.0e-15);

  for (unsigned int i = 0; i < n_kelly_refinements; ++i)
  {
    // Estimate the error. For simplicity, always use dealii::QGauss
    dealii::Vector<float> estimated_error_per_cell =
        estimate_error(triangulation, dof_handler, fe_degree, solution);

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
    refine_and_transfer(thermal_physics, material_properties, dof_handler,
                        solution);
  }

  // Refine the mesh along the trajectory of the sources.
  double current_source_height =
      dynamic_cast<adamantine::ThermalPhysics<dim, fe_degree, MemorySpaceType,
                                              dealii::QGauss<1>> *>(
          thermal_physics.get())
          ? dynamic_cast<adamantine::ThermalPhysics<
                dim, fe_degree, MemorySpaceType, dealii::QGauss<1>> *>(
                thermal_physics.get())
                ->get_current_source_height()
          : dynamic_cast<adamantine::ThermalPhysics<
                dim, fe_degree, MemorySpaceType, dealii::QGaussLobatto<1>> *>(
                thermal_physics.get())
                ->get_current_source_height();

  for (unsigned int i = 0; i < n_beam_refinements; ++i)
  {
    // Compute the cells to be refined.
    std::vector<typename dealii::parallel::distributed::Triangulation<
        dim>::active_cell_iterator>
        cells_to_refine = compute_cells_to_refine(
            triangulation, time, next_refinement_time, time_steps_refinement,
            heat_sources, current_source_height, refinement_beam_cutoff);

    // PropertyTreeInput refinement.coarsen_after_beam
    const bool coarsen_after_beam =
        refinement_database.get<bool>("coarsen_after_beam", false);

    // If coarsening is allowed, set the coarsening flag everywhere
    if (coarsen_after_beam)
    {
      for (auto cell : dealii::filter_iterators(
               triangulation.active_cell_iterators(),
               dealii::IteratorFilters::LocallyOwnedCell()))
      {
        if (cell->level() > 0)
          cell->set_coarsen_flag();
      }
    }

    // Flag the cells for refinement.
    for (auto &cell : cells_to_refine)
    {
      if (coarsen_after_beam)
        cell->clear_coarsen_flag();
      if (cell->level() < max_level)
        cell->set_refine_flag();
    }

    // Execute the refinement and transfer the solution onto the new mesh.
    refine_and_transfer(thermal_physics, material_properties, dof_handler,
                        solution);
  }

  // Recompute the inverse of the mass matrix
  thermal_physics->compute_inverse_mass_matrix();
}

template <int dim, typename MemorySpaceType>
void refine_mesh(
    std::unique_ptr<adamantine::ThermalPhysicsInterface<dim, MemorySpaceType>>
        &thermal_physics,
    adamantine::MaterialProperty<dim, MemorySpaceType> &material_properties,
    dealii::LA::distributed::Vector<double, MemorySpaceType> &solution,
    std::vector<std::shared_ptr<adamantine::HeatSource<dim>>> &heat_sources,
    double const time, double const next_refinement_time,
    unsigned int const time_steps_refinement,
    boost::property_tree::ptree const &refinement_database)
{
  if (!thermal_physics)
    return;

  switch (thermal_physics->get_fe_degree())
  {
  case 1:
  {
    refine_mesh<dim, 1>(thermal_physics, material_properties, solution,
                        heat_sources, time, next_refinement_time,
                        time_steps_refinement, refinement_database);
    break;
  }
  case 2:
  {
    refine_mesh<dim, 2>(thermal_physics, material_properties, solution,
                        heat_sources, time, next_refinement_time,
                        time_steps_refinement, refinement_database);
    break;
  }
  case 3:
  {
    refine_mesh<dim, 3>(thermal_physics, material_properties, solution,
                        heat_sources, time, next_refinement_time,
                        time_steps_refinement, refinement_database);
    break;
  }
  case 4:
  {
    refine_mesh<dim, 4>(thermal_physics, material_properties, solution,
                        heat_sources, time, next_refinement_time,
                        time_steps_refinement, refinement_database);
    break;
  }
  case 5:
  {
    refine_mesh<dim, 5>(thermal_physics, material_properties, solution,
                        heat_sources, time, next_refinement_time,
                        time_steps_refinement, refinement_database);
    break;
  }
  case 6:
  {
    refine_mesh<dim, 6>(thermal_physics, material_properties, solution,
                        heat_sources, time, next_refinement_time,
                        time_steps_refinement, refinement_database);
    break;
  }
  case 7:
  {
    refine_mesh<dim, 7>(thermal_physics, material_properties, solution,
                        heat_sources, time, next_refinement_time,
                        time_steps_refinement, refinement_database);
    break;
  }
  case 8:
  {
    refine_mesh<dim, 8>(thermal_physics, material_properties, solution,
                        heat_sources, time, next_refinement_time,
                        time_steps_refinement, refinement_database);
    break;
  }
  case 9:
  {
    refine_mesh<dim, 9>(thermal_physics, material_properties, solution,
                        heat_sources, time, next_refinement_time,
                        time_steps_refinement, refinement_database);
    break;
  }
  case 10:
  {
    refine_mesh<dim, 10>(thermal_physics, material_properties, solution,
                         heat_sources, time, next_refinement_time,
                         time_steps_refinement, refinement_database);
    break;
  }
  default:
  {
    adamantine::ASSERT_THROW(false, "fe_degree should be between 1 and 10.");
  }
  }
}

template <int dim, typename MemorySpaceType>
std::pair<dealii::LinearAlgebra::distributed::Vector<double,
                                                     dealii::MemorySpace::Host>,
          dealii::LinearAlgebra::distributed::Vector<double,
                                                     dealii::MemorySpace::Host>>
run(MPI_Comm const &communicator, boost::property_tree::ptree const &database,
    std::vector<adamantine::Timer> &timers)
{
#ifdef ADAMANTINE_WITH_CALIPER
  CALI_CXX_MARK_FUNCTION;
#endif

  // Create the Geometry
  boost::property_tree::ptree geometry_database =
      database.get_child("geometry");
  adamantine::Geometry<dim> geometry(communicator, geometry_database);

  // Create the MaterialProperty
  boost::property_tree::ptree material_database =
      database.get_child("materials");
  adamantine::MaterialProperty<dim, MemorySpaceType> material_properties(
      communicator, geometry.get_triangulation(), material_database);

  // Extract the physics property tree
  boost::property_tree::ptree physics_database = database.get_child("physics");
  bool const use_thermal_physics = physics_database.get<bool>("thermal");
  [[maybe_unused]] bool const use_mechanical_physics =
      physics_database.get<bool>("mechanical");

  // Extract the discretization property tree
  boost::property_tree::ptree discretization_database =
      database.get_child("discretization");

  // Extract the post-processor property tree
  boost::property_tree::ptree post_processor_database =
      database.get_child("post_processor");

  // Extract the verbosity
  // PropertyTreeInput verbose_output
  bool const verbose_output = database.get("verbose_output", false);

  // Create ThermalPhysics if necessary
  std::unique_ptr<adamantine::ThermalPhysicsInterface<dim, MemorySpaceType>>
      thermal_physics;
  std::vector<std::shared_ptr<adamantine::HeatSource<dim>>> heat_sources;
  if (use_thermal_physics)
  {
    // PropertyTreeInput discretization.thermal.fe_degree
    unsigned int const fe_degree =
        discretization_database.get<unsigned int>("thermal.fe_degree");
    // PropertyTreeInput discretization.thermal.quadrature
    std::string quadrature_type =
        discretization_database.get("thermal.quadrature", "gauss");
    thermal_physics = initialize_thermal_physics<dim>(
        fe_degree, quadrature_type, communicator, database, geometry,
        material_properties);
    heat_sources = thermal_physics->get_heat_sources();
    post_processor_database.put("thermal_output", true);
  }

  // PropertyTreeInput materials.initial_temperature
  double const initial_temperature =
      material_database.get("initial_temperature", 300.);

  // Get the reference temperature(s) for the substrate and deposited
  // material(s)
  std::vector<double> material_reference_temps;
  // PropertyTreeInput materials.n_materials
  unsigned int const n_materials =
      material_database.get<unsigned int>("n_materials");
  for (unsigned int i = 0; i < n_materials; ++i)
  {
    // Use the solidus as the reference temperature, in the future we may want
    // to use something else
    // PropertyTreeInput materials.material_n.solidus
    double const reference_temperature = material_database.get<double>(
        "material_" + std::to_string(i) + ".solidus");
    material_reference_temps.push_back(reference_temperature);
  }
  material_reference_temps.push_back(initial_temperature);

  // Create MechanicalPhysics
  std::unique_ptr<adamantine::MechanicalPhysics<dim, MemorySpaceType>>
      mechanical_physics;
  if (use_mechanical_physics)
  {
    // PropertyTreeInput discretization.mechanical.fe_degree
    unsigned int const fe_degree =
        discretization_database.get<unsigned int>("mechanical.fe_degree");
    mechanical_physics =
        std::make_unique<adamantine::MechanicalPhysics<dim, MemorySpaceType>>(
            communicator, fe_degree, geometry, material_properties,
            material_reference_temps);
    post_processor_database.put("mechanical_output", true);
  }

  // Create PostProcessor
  std::unique_ptr<adamantine::PostProcessor<dim>> post_processor;
  bool thermal_output = post_processor_database.get("thermal_output", false);
  bool mechanical_output =
      post_processor_database.get("mechanical_output", false);
  if ((thermal_output) && (mechanical_output))
  {
    post_processor = std::make_unique<adamantine::PostProcessor<dim>>(
        communicator, post_processor_database,
        thermal_physics->get_dof_handler(),
        mechanical_physics->get_dof_handler());
  }
  else
  {
    post_processor = std::make_unique<adamantine::PostProcessor<dim>>(
        communicator, post_processor_database,
        thermal_output ? thermal_physics->get_dof_handler()
                       : mechanical_physics->get_dof_handler());
  }

  dealii::LA::distributed::Vector<double, MemorySpaceType> temperature;
  dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host>
      displacement;
  if (use_thermal_physics)
  {
    thermal_physics->setup_dofs();
    thermal_physics->update_material_deposition_orientation();
    thermal_physics->compute_inverse_mass_matrix();
    thermal_physics->initialize_dof_vector(initial_temperature, temperature);
    thermal_physics->get_state_from_material_properties();
  }

  if (use_mechanical_physics)
  {
    if (use_thermal_physics)
    {
      // Thermo-mechanical simulation
      dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host>
          temperature_host(temperature.get_partitioner());
      temperature_host.import(temperature, dealii::VectorOperation::insert);
      mechanical_physics->setup_dofs(thermal_physics->get_dof_handler(),
                                     temperature_host,
                                     thermal_physics->get_has_melted_vector());
    }
    else
    {
      // Mechanical only simulation
      mechanical_physics->setup_dofs();
    }
    displacement = mechanical_physics->solve();
  }

  unsigned int progress = 0;
  unsigned int cycle = 0;
  unsigned int n_time_step = 0;
  double time = 0.;

  // Output the initial solution
  output_pvtu(*post_processor, cycle, n_time_step, time, thermal_physics,
              temperature, mechanical_physics, displacement,
              material_properties, timers);
  ++n_time_step;

  // Create the bounding boxes used for material deposition
  auto [material_deposition_boxes, deposition_times, deposition_cos,
        deposition_sin] =
      adamantine::create_material_deposition_boxes<dim>(geometry_database,
                                                        heat_sources);

  // Unless we use an embedded method, we know in advance the time step.
  // Thus we can get for each time step, the list of elements that we will
  // need to activate. This list will be invalidated every time we refine the
  // mesh.
  timers[adamantine::add_material_search].start();
  auto elements_to_activate = adamantine::get_elements_to_activate(
      thermal_physics->get_dof_handler(), material_deposition_boxes);
  timers[adamantine::add_material_search].stop();

  // Extract the time-stepping database
  boost::property_tree::ptree time_stepping_database =
      database.get_child("time_stepping");
  // PropertyTreeInput time_stepping.time_step
  double time_step = time_stepping_database.get<double>("time_step");
  // PropertyTreeInput time_stepping.duration
  double const duration = time_stepping_database.get<double>("duration");

  // Extract the refinement database
  boost::property_tree::ptree refinement_database =
      database.get_child("refinement");
  // PropertyTreeInput refinement.time_steps_between_refinement
  unsigned int const time_steps_refinement =
      refinement_database.get("time_steps_between_refinement", 10);
  // PropertyTreeInput post_processor.time_steps_between_output
  unsigned int const time_steps_output =
      post_processor_database.get("time_steps_between_output", 1);

  double next_refinement_time = time;
  // PropertyTreeInput materials.new_material_temperature
  double const new_material_temperature =
      database.get("materials.new_material_temperature", 300.);

#ifdef ADAMANTINE_WITH_CALIPER
  CALI_CXX_MARK_LOOP_BEGIN(main_loop_id, "main_loop");
#endif
  while (time < duration)
  {
#ifdef ADAMANTINE_WITH_CALIPER
    CALI_CXX_MARK_LOOP_ITERATION(main_loop_id, n_time_step - 1);
#endif
    if ((time + time_step) > duration)
      time_step = duration - time;
    unsigned int rank = dealii::Utilities::MPI::this_mpi_process(communicator);

    // Refine the mesh after time_steps_refinement time steps or when time
    // is greater or equal than the next predicted time for refinement. This
    // is necessary when using an embedded method.
    if ((((n_time_step % time_steps_refinement) == 0) ||
         (time >= next_refinement_time)) &&
        use_thermal_physics)
    {
      next_refinement_time = time + time_steps_refinement * time_step;
      timers[adamantine::refine].start();
      refine_mesh(thermal_physics, material_properties, temperature,
                  heat_sources, time, next_refinement_time,
                  time_steps_refinement, refinement_database);
      timers[adamantine::refine].stop();
      if ((rank == 0) && (verbose_output == true))
        std::cout << "n_dofs: " << thermal_physics->get_dof_handler().n_dofs()
                  << std::endl;

      timers[adamantine::add_material_search].start();
      elements_to_activate = adamantine::get_elements_to_activate(
          thermal_physics->get_dof_handler(), material_deposition_boxes);

      timers[adamantine::add_material_search].stop();
    }

    // Add material if necessary.
    // We use an epsilon to get the "expected" behavior when the deposition
    // time and the time match should match exactly but don't because of
    // floating point accuracy.
    timers[adamantine::add_material_activate].start();

    double const eps = time_step / 1e12;
    auto activation_start =
        std::lower_bound(deposition_times.begin(), deposition_times.end(),
                         time - eps) -
        deposition_times.begin();
    auto activation_end =
        std::lower_bound(deposition_times.begin(), deposition_times.end(),
                         time + time_step - eps) -
        deposition_times.begin();
    if (activation_start < activation_end)
    {
      if (use_thermal_physics)
      {
        // For now assume that all deposited material has never been melted
        // (may or may not be reasonable)
        std::vector<bool> has_melted(deposition_cos.size(), false);

        thermal_physics->add_material(elements_to_activate, deposition_cos,
                                      deposition_sin, has_melted,
                                      activation_start, activation_end,
                                      new_material_temperature, temperature);
      }
    }

    if ((rank == 0) && (verbose_output == true) &&
        (activation_end - activation_start > 0) && use_thermal_physics)
    {
      std::cout << "n_dofs: " << thermal_physics->get_dof_handler().n_dofs()
                << std::endl;
    }
    timers[adamantine::add_material_activate].stop();

    // If thermomechanics are being solved, mark cells that are above the
    // solidus as cells that should have their reference temperature reset.
    // This cannot be in the thermomechanics solve because some cells may go
    // above the solidus and then back below the solidus in the time between
    // thermomechanical solves.
    if (use_thermal_physics && use_mechanical_physics)
    {
      thermal_physics->mark_has_melted(material_reference_temps[0],
                                       temperature);
    }

    // Time can be different than time + time_step if an embedded scheme is
    // used. Note that this is a problem when adding material because it
    // means that the amount of material that needs to be added is not
    // known.
#if ADAMANTINE_DEBUG
    double const old_time = time;
    bool const adding_material =
        (activation_start == activation_end) ? false : true;
#endif
    timers[adamantine::evol_time].start();

    // Solve the thermal problem
    if (use_thermal_physics)
    {
      time = thermal_physics->evolve_one_time_step(time, time_step, temperature,
                                                   timers);
    }

    // Solve the (thermo-)mechanical problem
    if (use_mechanical_physics)
    {
      // Since there is no history dependence in the model, only calculate
      // mechanics when outputting
      if (n_time_step % time_steps_output == 0)
      {
        if (use_thermal_physics)
        {
          // Update the material state
          thermal_physics->set_state_to_material_properties();
          dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host>
              temperature_host(temperature.get_partitioner());
          temperature_host.import(temperature, dealii::VectorOperation::insert);
          mechanical_physics->setup_dofs(
              thermal_physics->get_dof_handler(), temperature_host,
              thermal_physics->get_has_melted_vector());
        }
        else
        {
          mechanical_physics->setup_dofs();
        }
        displacement = mechanical_physics->solve();
      }
    }

#if ADAMANTINE_DEBUG
    ASSERT(!adding_material || ((time - old_time) < time_step / 1e-9),
           "Unexpected time step while adding material.");
#endif
    timers[adamantine::evol_time].stop();

    // Get the new time step
    if (use_thermal_physics)
    {
      time_step = thermal_physics->get_delta_t_guess();
    }
    else
    {
      time_step += time_step;
    }

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
    if (n_time_step % time_steps_output == 0)
    {
      if (use_thermal_physics)
      {
        thermal_physics->set_state_to_material_properties();
      }
      output_pvtu(*post_processor, cycle, n_time_step, time, thermal_physics,
                  temperature, mechanical_physics, displacement,
                  material_properties, timers);
    }
    ++n_time_step;
  }
#ifdef ADAMANTINE_WITH_CALIPER
  CALI_CXX_MARK_LOOP_END(main_loop_id);
#endif

  post_processor->write_pvd();

  // This is only used for integration test
  if constexpr (std::is_same_v<MemorySpaceType, dealii::MemorySpace::Host>)
  {
    if (use_thermal_physics)
    {
      thermal_physics->get_affine_constraints().distribute(temperature);
    }
    if (use_mechanical_physics)
    {
      mechanical_physics->get_affine_constraints().distribute(displacement);
    }
    return std::make_pair(temperature, displacement);
  }
  else
  {
    dealii::LinearAlgebra::distributed::Vector<double,
                                               dealii::MemorySpace::Host>
        temperature_host(temperature.get_partitioner());
    temperature_host.import(temperature, dealii::VectorOperation::insert);
    if (use_thermal_physics)
    {
      thermal_physics->get_affine_constraints().distribute(temperature_host);
    }
    if (use_mechanical_physics)
    {
      mechanical_physics->get_affine_constraints().distribute(displacement);
    }
    return std::make_pair(temperature_host, displacement);
  }
}

template <int dim, typename MemorySpaceType>
std::vector<dealii::LA::distributed::BlockVector<double>>
run_ensemble(MPI_Comm const &communicator,
             boost::property_tree::ptree const &database,
             std::vector<adamantine::Timer> &timers)
{
#ifdef ADAMANTINE_WITH_CALIPER
  CALI_CXX_MARK_FUNCTION;
#endif

  unsigned int rank = dealii::Utilities::MPI::this_mpi_process(communicator);

  // ------ Extract child property trees -----
  // Mandatory subtrees
  boost::property_tree::ptree geometry_database =
      database.get_child("geometry");
  boost::property_tree::ptree discretization_database =
      database.get_child("discretization");
  boost::property_tree::ptree time_stepping_database =
      database.get_child("time_stepping");
  boost::property_tree::ptree post_processor_database =
      database.get_child("post_processor");
  boost::property_tree::ptree refinement_database =
      database.get_child("refinement");
  boost::property_tree::ptree ensemble_database =
      database.get_child("ensemble");
  boost::property_tree::ptree material_database =
      database.get_child("materials");

  // Optional subtrees
  boost::optional<const boost::property_tree::ptree &>
      experiment_optional_database = database.get_child_optional("experiment");
  boost::optional<const boost::property_tree::ptree &>
      data_assimilation_optional_database =
          database.get_child_optional("data_assimilation");

  // Verbosity
  // PropertyTreeInput verbose_output
  bool const verbose_output = database.get("verbose_output", false);

  // ------ Get finite element implementation parameters -----
  // PropertyTreeInput discretization.fe_degree
  unsigned int const fe_degree =
      discretization_database.get<unsigned int>("thermal.fe_degree");
  // PropertyTreeInput discretization.quadrature
  std::string quadrature_type =
      discretization_database.get("thermal.quadrature", "gauss");
  std::transform(quadrature_type.begin(), quadrature_type.end(),
                 quadrature_type.begin(),
                 [](unsigned char c) { return std::tolower(c); });

  // ------ Get means of ensemble parameters -----
  // PropertyTreeInput materials.initial_temperature
  double const initial_temperature_mean =
      database.get("materials.initial_temperature", 300.);
  // Get the nominal (mean) values of the ensemble parameters
  // PropertyTreeInput materials.new_material_temperature
  double const new_material_temperature_mean =
      database.get("materials.new_material_temperature", 300.);

  // PropertyTreeInput sources.n_beams
  unsigned int const n_beams = database.get<unsigned int>("sources.n_beams");
  double beam_0_max_power_mean = 0.0;
  double beam_0_absorption_mean = 0.0;
  if (n_beams > 0)
  {
    // PropertyTreeInput sources.beam_0.max_power
    beam_0_max_power_mean = database.get<double>("sources.beam_0.max_power");

    // PropertyTreeInput sources.beam_0.absorption_efficiency
    beam_0_absorption_mean =
        database.get<double>("sources.beam_0.absorption_efficiency");
  }

  // ------ Set up the ensemble members -----
  // There might be a more efficient way to share some of these objects
  // between ensemble members. For now, we'll do the simpler approach of
  // duplicating everything. PropertyTreeInput ensemble.ensemble_size
  const unsigned int ensemble_size = ensemble_database.get("ensemble_size", 5);

  // PropertyTreeInput ensemble.initial_temperature_stddev
  const double initial_temperature_stddev =
      ensemble_database.get("initial_temperature_stddev", 0.0);

  std::vector<double> initial_temperature =
      adamantine::fill_and_sync_random_vector(
          ensemble_size, initial_temperature_mean, initial_temperature_stddev);

  // PropertyTreeInput ensemble.new_material_temperature_stddev
  const double new_material_temperature_stddev =
      ensemble_database.get("new_material_temperature_stddev", 0.0);

  std::vector<double> new_material_temperature =
      adamantine::fill_and_sync_random_vector(ensemble_size,
                                              new_material_temperature_mean,
                                              new_material_temperature_stddev);

  // PropertyTreeInput ensemble.beam_0_max_power_stddev
  const double beam_0_max_power_stddev =
      ensemble_database.get("beam_0_max_power_stddev", 0.0);

  std::vector<double> beam_0_max_power =
      adamantine::fill_and_sync_random_vector(
          ensemble_size, beam_0_max_power_mean, beam_0_max_power_stddev);

  // PropertyTreeInput ensemble.beam_0_absorption_stddev
  const double beam_0_absorption_stddev =
      ensemble_database.get("beam_0_absorption_stddev", 0.0);

  std::vector<double> beam_0_absorption =
      adamantine::fill_and_sync_random_vector(
          ensemble_size, beam_0_absorption_mean, beam_0_absorption_stddev);

  // Create a new property tree database for each ensemble member
  std::vector<boost::property_tree::ptree> database_ensemble(ensemble_size,
                                                             database);

  std::vector<std::unique_ptr<
      adamantine::ThermalPhysicsInterface<dim, MemorySpaceType>>>
      thermal_physics_ensemble(ensemble_size);

  std::vector<std::vector<std::shared_ptr<adamantine::HeatSource<dim>>>>
      heat_sources_ensemble(ensemble_size);

  std::vector<std::unique_ptr<adamantine::Geometry<dim>>> geometry_ensemble;

  std::vector<
      std::unique_ptr<adamantine::MaterialProperty<dim, MemorySpaceType>>>
      material_properties_ensemble;

  std::vector<std::unique_ptr<adamantine::PostProcessor<dim>>>
      post_processor_ensemble;

  // Create the vector of augmented state vectors
  std::vector<dealii::LA::distributed::BlockVector<double>>
      solution_augmented_ensemble(ensemble_size);

  // Give names to the blocks in the augmented state vector
  int constexpr base_state = 0;
  int constexpr augmented_state = 1;

  // ----- Initialize the data assimilation object -----
  boost::property_tree::ptree data_assimilation_database;
  bool assimilate_data = false;
  std::vector<adamantine::AugmentedStateParameters> augmented_state_parameters;

  if (data_assimilation_optional_database)
  {
    // PropertyTreeInput data_assimilation
    data_assimilation_database = data_assimilation_optional_database.get();

    // PropertyTreeInput data_assimilation.assimilate_data
    if (data_assimilation_database.get("assimilate_data", false))
    {
      assimilate_data = true;

      // PropertyTreeInput data_assimilation.augment_with_beam_0_absorption
      if (data_assimilation_database.get("augment_with_beam_0_absorption",
                                         false))
      {
        augmented_state_parameters.push_back(
            adamantine::AugmentedStateParameters::beam_0_absorption);
      }
      // PropertyTreeInput data_assimilation.augment_with_beam_0_max_power
      if (data_assimilation_database.get("augment_with_beam_0_max_power",
                                         false))
      {
        augmented_state_parameters.push_back(
            adamantine::AugmentedStateParameters::beam_0_max_power);
      }
    }
  }
  adamantine::DataAssimilator data_assimilator(data_assimilation_database);

  for (unsigned int member = 0; member < ensemble_size; ++member)
  {
    // Resize the augmented ensemble block vector to have two blocks
    solution_augmented_ensemble[member].reinit(2);

    // Edit the database for the ensemble
    if (n_beams > 0)
    {
      // PropertyTreeInput sources.beam_0.max_power
      database_ensemble[member].put("sources.beam_0.max_power",
                                    beam_0_max_power[member]);

      // PropertyTreeInput sources.beam_0.absorption_efficiency
      database_ensemble[member].put("sources.beam_0.absorption_efficiency",
                                    beam_0_absorption[member]);

      // Populate the parameter augmentation block of the augmented state
      // ensemble
      if (assimilate_data)
      {
        solution_augmented_ensemble[member]
            .block(augmented_state)
            .reinit(augmented_state_parameters.size());

        for (unsigned int index = 0; index < augmented_state_parameters.size();
             ++index)
        {
          // FIXME: Need to consider how we want to generalize this. It could
          // get unwieldy if we want to specify every parameter of an
          // arbitrary number of beams.
          if (augmented_state_parameters.at(index) ==
              adamantine::AugmentedStateParameters::beam_0_absorption)
          {
            solution_augmented_ensemble[member].block(augmented_state)[index] =
                beam_0_absorption[member];
          }
          else if (augmented_state_parameters.at(index) ==
                   adamantine::AugmentedStateParameters::beam_0_max_power)
          {
            solution_augmented_ensemble[member].block(augmented_state)[index] =
                beam_0_max_power[member];
          }
        }
      }
    }

    solution_augmented_ensemble[member].collect_sizes();

    geometry_ensemble.push_back(std::make_unique<adamantine::Geometry<dim>>(
        communicator, geometry_database));

    material_properties_ensemble.push_back(
        std::make_unique<adamantine::MaterialProperty<dim, MemorySpaceType>>(
            communicator, geometry_ensemble.back()->get_triangulation(),
            material_database));

    thermal_physics_ensemble[member] = initialize_thermal_physics<dim>(
        fe_degree, quadrature_type, communicator, database_ensemble[member],
        *geometry_ensemble[member], *material_properties_ensemble[member]);
    heat_sources_ensemble[member] =
        thermal_physics_ensemble[member]->get_heat_sources();

    thermal_physics_ensemble[member]->setup_dofs();
    thermal_physics_ensemble[member]->update_material_deposition_orientation();
    thermal_physics_ensemble[member]->compute_inverse_mass_matrix();

    thermal_physics_ensemble[member]->initialize_dof_vector(
        initial_temperature[member],
        solution_augmented_ensemble[member].block(base_state));

    solution_augmented_ensemble[member].collect_sizes();

    thermal_physics_ensemble[member]->get_state_from_material_properties();

    // For now we only output temperature
    post_processor_database.put("thermal_output", true);
    post_processor_ensemble.push_back(
        std::make_unique<adamantine::PostProcessor<dim>>(
            communicator, post_processor_database,
            thermal_physics_ensemble[member]->get_dof_handler(), member));
  }

  // PostProcessor for outputting the experimental data
  boost::property_tree::ptree post_processor_expt_database;
  // PropertyTreeInput post_processor.file_name
  std::string expt_file_prefix =
      post_processor_database.get<std::string>("filename_prefix") + ".expt";
  post_processor_expt_database.put("filename_prefix", expt_file_prefix);
  post_processor_expt_database.put("thermal_output", true);
  adamantine::PostProcessor<dim> post_processor_expt(
      communicator, post_processor_expt_database,
      thermal_physics_ensemble[0]->get_dof_handler());

  // ----- Read the experimental data -----
  std::vector<std::vector<double>> frame_time_stamps;
  std::unique_ptr<adamantine::ExperimentalData<dim>> experimental_data;
  unsigned int experimental_frame_index = -1;

  if (experiment_optional_database)
  {
    auto experiment_database = experiment_optional_database.get();

    // PropertyTreeInput experiment.read_in_experimental_data
    bool const read_in_experimental_data =
        experiment_database.get("read_in_experimental_data", false);

    if (read_in_experimental_data)
    {
      if (rank == 0)
        std::cout << "Reading the experimental log file..." << std::endl;

      frame_time_stamps =
          adamantine::read_frame_timestamps(experiment_database);

      adamantine::ASSERT_THROW(
          frame_time_stamps.size() > 0,
          "Error: Experimental data parsing is activated, but "
          "the log shows zero cameras.");
      adamantine::ASSERT_THROW(
          frame_time_stamps[0].size() > 0,
          "Error: Experimental data parsing is activated, but "
          "the log shows zero data frames.");

      if (rank == 0)
        std::cout << "Done. Log entries found for " << frame_time_stamps.size()
                  << " camera(s), with " << frame_time_stamps[0].size()
                  << " frame(s)." << std::endl;

      // PropertyTreeInput experiment.format
      std::string experiment_format =
          experiment_database.get<std::string>("format");

      if (boost::iequals(experiment_format, "point_cloud"))
      {
        experimental_data = std::make_unique<adamantine::PointCloud<dim>>(
            adamantine::PointCloud<dim>(experiment_database));
      }
      else
      {
        if constexpr (dim == 3)
        {
          experimental_data =
              std::make_unique<adamantine::RayTracing>(adamantine::RayTracing(
                  experiment_database,
                  thermal_physics_ensemble[0]->get_dof_handler()));
        }
      }
    }
  }

  // ----- Initialize time and time stepping counters -----
  unsigned int progress = 0;
  unsigned int cycle = 0;
  unsigned int n_time_step = 0;
  double time = 0.;

  // ----- Output the initial solution -----
  std::unique_ptr<adamantine::MechanicalPhysics<dim, MemorySpaceType>>
      mechanical_physics;
  dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host>
      displacement;
  for (unsigned int member = 0; member < ensemble_size; ++member)
  {
    output_pvtu(*post_processor_ensemble[member], cycle, n_time_step, time,
                thermal_physics_ensemble[member],
                solution_augmented_ensemble[member].block(base_state),
                mechanical_physics, displacement,
                *material_properties_ensemble[member], timers);
  }

  // ----- Increment the time step -----
  ++n_time_step;

  // ----- Get refinement and time stepping parameters -----
  // PropertyTreeInput refinement.time_steps_between_refinement
  unsigned int const time_steps_refinement =
      refinement_database.get("time_steps_between_refinement", 10);
  double next_refinement_time = time;
  // PropertyTreeInput time_stepping.time_step
  double time_step = time_stepping_database.get<double>("time_step");
  // PropertyTreeInput time_stepping.duration
  double const duration = time_stepping_database.get<double>("duration");
  // PropertyTreeInput post_processor.time_steps_between_output
  unsigned int const time_steps_output =
      post_processor_database.get("time_steps_between_output", 1);

  // ----- Deposit material -----
  // For now assume that all ensemble members share the same geometry (they
  // have independent adamantine::Geometry objects, but all are constructed
  // from identical parameters), base new additions on the 0th ensemble member
  auto [material_deposition_boxes, deposition_times, deposition_cos,
        deposition_sin] =
      adamantine::create_material_deposition_boxes<dim>(
          geometry_database, heat_sources_ensemble[0]);

  // Unless we use an embedded method, we know in advance the time step.

  // Thus we can get for each time step, the list of elements that we will
  // need to activate. This list will be invalidated every time we refine the
  // mesh.
  timers[adamantine::add_material_search].start();
  std::vector<std::vector<
      std::vector<typename dealii::DoFHandler<dim>::active_cell_iterator>>>
      elements_to_activate_ensemble(ensemble_size);
  for (unsigned int member = 0; member < ensemble_size; ++member)
  {
    elements_to_activate_ensemble[member] =
        adamantine::get_elements_to_activate(
            thermal_physics_ensemble[member]->get_dof_handler(),
            material_deposition_boxes);
  }
  timers[adamantine::add_material_search].stop();

  // ----- Main time stepping loop -----
  if (rank == 0)
    std::cout << "Starting the main time stepping loop..." << std::endl;

#ifdef ADAMANTINE_WITH_CALIPER
  CALI_CXX_MARK_LOOP_BEGIN(main_loop_id, "main_loop");
#endif
  while (time < duration)
  {
#ifdef ADAMANTINE_WITH_CALIPER
    CALI_CXX_MARK_LOOP_ITERATION(main_loop_id, n_time_step - 1);
#endif
    if ((time + time_step) > duration)
      time_step = duration - time;

    // ----- Refine the mesh if necessary -----
    // Refine the mesh after time_steps_refinement time steps or when time
    // is greater or equal than the next predicted time for refinement. This
    // is necessary when using an embedded method.
    if (((n_time_step % time_steps_refinement) == 0) ||
        (time >= next_refinement_time))
    {
      next_refinement_time = time + time_steps_refinement * time_step;
      timers[adamantine::refine].start();

      for (unsigned int member = 0; member < ensemble_size; ++member)
      {
        refine_mesh(thermal_physics_ensemble[member],
                    *material_properties_ensemble[member],
                    solution_augmented_ensemble[member].block(base_state),
                    heat_sources_ensemble[member], time, next_refinement_time,
                    time_steps_refinement, refinement_database);
      }

      timers[adamantine::refine].stop();
      if ((rank == 0) && (verbose_output == true))
        std::cout << "n_dofs: "
                  << thermal_physics_ensemble[0]->get_dof_handler().n_dofs()
                  << std::endl;

      // ----- Add material if necessary -----
      timers[adamantine::add_material_search].start();
      for (unsigned int member = 0; member < ensemble_size; ++member)
      {
        elements_to_activate_ensemble[member] =
            adamantine::get_elements_to_activate(
                thermal_physics_ensemble[member]->get_dof_handler(),
                material_deposition_boxes);
        solution_augmented_ensemble[member].collect_sizes();
      }
      timers[adamantine::add_material_search].stop();
    }

    // We use an epsilon to get the "expected" behavior when the deposition
    // time and the time match should match exactly but don't because of
    // floating point accuracy.
    timers[adamantine::add_material_activate].start();

    double const eps = time_step / 1e12;
    auto activation_start =
        std::lower_bound(deposition_times.begin(), deposition_times.end(),
                         time - eps) -
        deposition_times.begin();
    auto activation_end =
        std::lower_bound(deposition_times.begin(), deposition_times.end(),
                         time + time_step - eps) -
        deposition_times.begin();
    if (activation_start < activation_end)
      for (unsigned int member = 0; member < ensemble_size; ++member)
      {
        // For now assume that all deposited material has never been melted
        // (may or may not be reasonable)
        std::vector<bool> has_melted(deposition_cos.size(), false);

        thermal_physics_ensemble[member]->add_material(
            elements_to_activate_ensemble[member], deposition_cos,
            deposition_sin, has_melted, activation_start, activation_end,
            new_material_temperature[member],
            solution_augmented_ensemble[member].block(base_state));

        solution_augmented_ensemble[member].collect_sizes();
      }

    if ((rank == 0) && (verbose_output == true) &&
        (activation_end - activation_start > 0))
      std::cout << "n_dofs: "
                << thermal_physics_ensemble[0]->get_dof_handler().n_dofs()
                << std::endl;

    timers[adamantine::add_material_activate].stop();

    // ----- Evolve the solution by one time step -----
    // Time can be different than time + time_step if an embedded scheme is
    // used. Note that this is a problem when adding material because it
    // means that the amount of material that needs to be added is not
    // known.
    double const old_time = time;
#if ADAMANTINE_DEBUG
    bool const adding_material =
        (activation_start == activation_end) ? false : true;
#endif
    timers[adamantine::evol_time].start();

    for (unsigned int member = 0; member < ensemble_size; ++member)
    {
      time = thermal_physics_ensemble[member]->evolve_one_time_step(
          old_time, time_step,
          solution_augmented_ensemble[member].block(base_state), timers);
    }

#if ADAMANTINE_DEBUG
    ASSERT(!adding_material || ((time - old_time) < time_step / 1e-9),
           "Unexpected time step while adding material.");
#endif
    timers[adamantine::evol_time].stop();

    // ----- Get the new time step size -----
    // Needs to be the same for all ensemble members, obtained from the 0th
    // member
    time_step = thermal_physics_ensemble[0]->get_delta_t_guess();

    // ----- Perform data assimilation -----
    if (assimilate_data)
    {
      for (unsigned int member = 0; member < ensemble_size; ++member)
      {
        thermal_physics_ensemble[member]->get_affine_constraints().distribute(
            solution_augmented_ensemble[member].block(base_state));
      }

      // Currently assume that all frames are synced so that the 0th camera
      // frame time is the relevant time
      double frame_time;
      if (((experimental_frame_index + 1) >= frame_time_stamps[0].size()) ||
          (frame_time_stamps[0][experimental_frame_index + 1] > time))
      {
        frame_time = std::numeric_limits<double>::max();
      }
      else
      {
        experimental_frame_index = experimental_data->read_next_frame();
        frame_time = frame_time_stamps[0][experimental_frame_index];
      }

      if (frame_time <= time)
      {
        adamantine::ASSERT_THROW(
            frame_time > old_time || n_time_step == 1,
            "Unexpectedly missed a data assimilation frame.");
        if (rank == 0)
          std::cout << "Performing data assimilation at time " << time << "..."
                    << std::endl;

        // Print out the augmented parameters
        if (rank == 0)
        {
          for (unsigned int member = 0; member < ensemble_size; ++member)
          {
            std::cout << "Old parameters for member " << member << ": ";
            for (auto param : solution_augmented_ensemble[member].block(1))
              std::cout << param << " ";

            std::cout << std::endl;
          }
        }

        timers[adamantine::da_experimental_data].start();
#ifdef ADAMANTINE_WITH_CALIPER
        CALI_MARK_BEGIN("da_experimental_data");
#endif
        auto points_values = experimental_data->get_points_values();
        auto const &thermal_dof_handler =
            thermal_physics_ensemble[0]->get_dof_handler();
        auto expt_to_dof_mapping = adamantine::get_expt_to_dof_mapping(
            points_values, thermal_dof_handler);
        if (rank == 0)
        {
          std::cout << "Number expt sites mapped to DOFs: "
                    << expt_to_dof_mapping.first.size() << std::endl;
        }
#ifdef ADAMANTINE_WITH_CALIPER
        CALI_MARK_END("da_experimental_data");
#endif
        timers[adamantine::da_experimental_data].stop();

        // Optionally output the experimental data projected onto the mesh
        // PropertyTreeInput experiment.output_experiment_on_mesh
        bool const output_experiment_on_mesh =
            experiment_optional_database.get().get("output_experiment_on_mesh",
                                                   true);
        if (output_experiment_on_mesh)
        {
          dealii::LA::distributed::Vector<double, MemorySpaceType>
              temperature_expt;
          temperature_expt.reinit(
              solution_augmented_ensemble[0].block(base_state));
          temperature_expt.add(1.0e10);
          adamantine::set_with_experimental_data(communicator,
              points_values, expt_to_dof_mapping, temperature_expt,
              verbose_output);

          thermal_physics_ensemble[0]->get_affine_constraints().distribute(
              temperature_expt);
          post_processor_expt.write_thermal_output(
              cycle, n_time_step, time, temperature_expt,
              material_properties_ensemble[0]->get_state(),
              material_properties_ensemble[0]->get_dofs_map(),
              material_properties_ensemble[0]->get_dof_handler());
        }

        // NOTE: As is, this updates the dof mapping and covariance sparsity
        // pattern for every data assimilation operation. Strictly, this is
        // only necessary if the mesh changes (both updates) or the locations
        // of observations changes (the dof mapping). In practice, changes to
        // the mesh due to deposition likely cause the updates to be required
        // for each operation. If this is a bottleneck, it can be fixed in the
        // future.
        timers[adamantine::da_dof_mapping].start();
#ifdef ADAMANTINE_WITH_CALIPER
        CALI_MARK_BEGIN("da_dof_mapping");
#endif
        data_assimilator.update_dof_mapping<dim>(expt_to_dof_mapping);
#ifdef ADAMANTINE_WITH_CALIPER
        CALI_MARK_END("da_dof_mapping");
#endif
        timers[adamantine::da_dof_mapping].stop();

        timers[adamantine::da_covariance_sparsity].start();
#ifdef ADAMANTINE_WITH_CALIPER
        CALI_MARK_BEGIN("da_covariance_sparsity");
#endif
        data_assimilator.update_covariance_sparsity_pattern<dim>(
            thermal_dof_handler,
            solution_augmented_ensemble[0].block(augmented_state).size());
#ifdef ADAMANTINE_WITH_CALIPER
        CALI_MARK_END("da_covariance_sparsity");
#endif
        timers[adamantine::da_covariance_sparsity].start();

        unsigned int experimental_data_size = points_values.values.size();

        // Create the R matrix (the observation covariance matrix)
        // PropertyTreeInput experiment.estimated_uncertainty
        timers[adamantine::da_obs_covariance].start();
#ifdef ADAMANTINE_WITH_CALIPER
        CALI_MARK_BEGIN("da_obs_covariance");
#endif
        double variance_entries = experiment_optional_database.get().get(
            "estimated_uncertainty", 0.0);
        variance_entries = variance_entries * variance_entries;

        dealii::SparsityPattern pattern(experimental_data_size,
                                        experimental_data_size, 1);
        for (unsigned int i = 0; i < experimental_data_size; ++i)
        {
          pattern.add(i, i);
        }
        pattern.compress();

        dealii::SparseMatrix<double> R(pattern);
        for (unsigned int i = 0; i < experimental_data_size; ++i)
        {
          R.add(i, i, variance_entries);
        }
#ifdef ADAMANTINE_WITH_CALIPER
        CALI_MARK_END("da_obs_covariance");
#endif
        timers[adamantine::da_obs_covariance].stop();

        // Perform data assimilation to update the augmented state ensemble
        timers[adamantine::da_update_ensemble].start();
#ifdef ADAMANTINE_WITH_CALIPER
        CALI_MARK_BEGIN("da_update_ensemble");
#endif
        data_assimilator.update_ensemble(
            communicator, solution_augmented_ensemble, points_values.values, R);
#ifdef ADAMANTINE_WITH_CALIPER
        CALI_MARK_END("da_update_ensemble");
#endif
        timers[adamantine::da_update_ensemble].stop();

        // Extract the parameters from the augmented state
        for (unsigned int member = 0; member < ensemble_size; ++member)
        {
          for (unsigned int index = 0;
               index < augmented_state_parameters.size(); ++index)
          {
            // FIXME: Need to consider how we want to generalize this. It
            // could get unwieldy if we want to specify every parameter of an
            // arbitrary number of beams.
            if (augmented_state_parameters.at(index) ==
                adamantine::AugmentedStateParameters::beam_0_absorption)
            {
              database_ensemble[member].put(
                  "sources.beam_0.absorption_efficiency",
                  solution_augmented_ensemble[member].block(
                      augmented_state)[index]);
            }
            else if (augmented_state_parameters.at(index) ==
                     adamantine::AugmentedStateParameters::beam_0_max_power)
            {
              database_ensemble[member].put(
                  "sources.beam_0.max_power",
                  solution_augmented_ensemble[member].block(
                      augmented_state)[index]);
            }
          }
        }

        if (rank == 0)
          std::cout << "Done." << std::endl;

        // Print out the augmented parameters
        if (rank == 0)
        {
          for (unsigned int member = 0; member < ensemble_size; ++member)
          {
            std::cout << "New parameters for member " << member << ": ";
            for (auto param : solution_augmented_ensemble[member].block(1))
              std::cout << param << " ";

            std::cout << std::endl;
          }
        }
      }

      // Update the heat source in the ThermalPhysics objects
      for (unsigned int member = 0; member < ensemble_size; ++member)
      {
        thermal_physics_ensemble[member]->update_physics_parameters(
            database_ensemble[member].get_child("sources"));
      }
    }

    // ----- Output progress on screen -----
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

    // ----- Output the solution -----
    if (n_time_step % time_steps_output == 0)
    {
      for (unsigned int member = 0; member < ensemble_size; ++member)
      {
        thermal_physics_ensemble[member]->set_state_to_material_properties();
        output_pvtu(*post_processor_ensemble[member], cycle, n_time_step, time,
                    thermal_physics_ensemble[member],
                    solution_augmented_ensemble[member].block(base_state),
                    mechanical_physics, displacement,
                    *material_properties_ensemble[member], timers);
      }
    }

    ++n_time_step;
  }

#ifdef ADAMANTINE_WITH_CALIPER
  CALI_CXX_MARK_LOOP_END(main_loop_id);
#endif

  for (unsigned int member = 0; member < ensemble_size; ++member)
  {
    post_processor_ensemble[member]->write_pvd();
  }

  // This is only used for integration test
  if constexpr (std::is_same_v<MemorySpaceType, dealii::MemorySpace::Host>)
  {
    for (unsigned int member = 0; member < ensemble_size; ++member)
    {
      thermal_physics_ensemble[member]->get_affine_constraints().distribute(
          solution_augmented_ensemble[member].block(base_state));
    }
    return solution_augmented_ensemble;
  }
  else
  {
    // NOTE: Currently unused. Added for the future case where run_ensemble is
    // functional on the device.
    std::vector<dealii::LA::distributed::BlockVector<double>>
        solution_augmented_ensemble_host(ensemble_size);

    for (unsigned int member = 0; member < ensemble_size; ++member)
    {
      solution_augmented_ensemble[member].reinit(2);

      solution_augmented_ensemble_host[member].block(0).reinit(
          solution_augmented_ensemble[member].block(0).get_partitioner());

      solution_augmented_ensemble_host[member].block(0).import(
          solution_augmented_ensemble[member].block(0),
          dealii::VectorOperation::insert);
      thermal_physics_ensemble[member]->get_affine_constraints().distribute(
          solution_augmented_ensemble_host[member].block(0));

      solution_augmented_ensemble_host[member].block(1).reinit(
          solution_augmented_ensemble[member].block(0).get_partitioner());

      solution_augmented_ensemble_host[member].block(1).import(
          solution_augmented_ensemble[member].block(1),
          dealii::VectorOperation::insert);
    }

    return solution_augmented_ensemble_host;
  }
}
#endif
