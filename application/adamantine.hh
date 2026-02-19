/* SPDX-FileCopyrightText: Copyright (c) 2016 - 2026, the adamantine authors.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#ifndef ADAMANTINE_HH
#define ADAMANTINE_HH

#include <Boundary.hh>
#include <DataAssimilator.hh>
#include <ExperimentalData.hh>
#include <Geometry.hh>
#include <MaterialProperty.hh>
#include <MechanicalPhysics.hh>
#include <Microstructure.hh>
#include <PointCloud.hh>
#include <PostProcessor.hh>
#include <RayTracing.hh>
#include <ThermalPhysics.hh>
#include <ThermalPhysicsInterface.hh>
#include <Timer.hh>
#include <ensemble_management.hh>
#include <experimental_data_utils.hh>
#include <material_deposition.hh>
#include <types.hh>
#include <utils.hh>

#include <deal.II/arborx/bvh.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/types.h>
#include <deal.II/distributed/cell_data_transfer.templates.h>
#include <deal.II/distributed/solution_transfer.h>
#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/grid/grid_refinement.h>
#if DEAL_II_VERSION_GTE(9, 7, 0) && defined(DEAL_II_TRILINOS_WITH_TPETRA)
#include <deal.II/lac/trilinos_tpetra_sparse_matrix.h>
#else
#include <deal.II/lac/trilinos_sparse_matrix.h>
#endif
#include <deal.II/lac/vector_operation.h>
#include <deal.II/numerics/error_estimator.h>

#include <boost/algorithm/string.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/property_tree/ptree.hpp>

#include <fstream>
#include <limits>
#include <memory>
#include <string>
#include <type_traits>
#include <unordered_map>

#ifdef ADAMANTINE_WITH_CALIPER
#include <caliper/cali.h>
#endif

#include <cmath>
#include <iostream>

template <int dim, int n_materials, int p_order, typename MaterialStates,
          typename MemorySpaceType,
          std::enable_if_t<
              std::is_same<MemorySpaceType, dealii::MemorySpace::Host>::value,
              int> = 0>
void output_pvtu(
    adamantine::PostProcessor<dim> &post_processor, unsigned int n_time_step,
    double time,
    std::unique_ptr<
        adamantine::ThermalPhysicsInterface<dim, MemorySpaceType>> const
        &thermal_physics,
    dealii::LinearAlgebra::distributed::Vector<double, MemorySpaceType>
        &temperature,
    std::unique_ptr<adamantine::MechanicalPhysics<
        dim, n_materials, p_order, MaterialStates, MemorySpaceType>> const
        &mechanical_physics,
    dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host>
        &displacement,
    adamantine::MaterialProperty<dim, n_materials, p_order, MaterialStates,
                                 MemorySpaceType> const &material_properties,
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
      post_processor.template write_output<typename Kokkos::View<
          double **, typename MemorySpaceType::kokkos_space>::array_layout>(
          n_time_step, time, temperature, displacement,
          mechanical_physics->get_stress_tensor(),
          material_properties.get_state(), material_properties.get_dofs_map(),
          material_properties.get_dof_handler());
    }
    else
    {
      post_processor.template write_thermal_output<typename Kokkos::View<
          double **, typename MemorySpaceType::kokkos_space>::array_layout>(
          n_time_step, time, temperature, material_properties.get_state(),
          material_properties.get_dofs_map(),
          material_properties.get_dof_handler());
    }
  }
  else
  {
    mechanical_physics->get_affine_constraints().distribute(displacement);
    post_processor.template write_mechanical_output<typename Kokkos::View<
        double **, typename MemorySpaceType::kokkos_space>::array_layout>(
        n_time_step, time, displacement,
        mechanical_physics->get_stress_tensor(),
        material_properties.get_state(), material_properties.get_dofs_map(),
        material_properties.get_dof_handler());
  }
  timers[adamantine::output].stop();
}

template <int dim, int n_materials, int p_order, typename MaterialStates,
          typename MemorySpaceType,
          std::enable_if_t<std::is_same<MemorySpaceType,
                                        dealii::MemorySpace::Default>::value,
                           int> = 0>
void output_pvtu(
    adamantine::PostProcessor<dim> &post_processor, unsigned int n_time_step,
    double time,
    std::unique_ptr<
        adamantine::ThermalPhysicsInterface<dim, MemorySpaceType>> const
        &thermal_physics,
    dealii::LinearAlgebra::distributed::Vector<double, MemorySpaceType>
        &temperature,
    std::unique_ptr<adamantine::MechanicalPhysics<
        dim, n_materials, p_order, MaterialStates, MemorySpaceType>> const
        &mechanical_physics,
    dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host>
        &displacement,
    adamantine::MaterialProperty<dim, n_materials, p_order, MaterialStates,
                                 MemorySpaceType> const &material_properties,
    std::vector<adamantine::Timer> &timers)
{
#ifdef ADAMANTINE_WITH_CALIPER
  CALI_CXX_MARK_FUNCTION;
#endif
  timers[adamantine::output].start();
  auto state = material_properties.get_state();
  auto state_host = Kokkos::create_mirror_view_and_copy(
      dealii::MemorySpace::Host::kokkos_space{}, state);
  if (thermal_physics)
  {
    dealii::LinearAlgebra::distributed::Vector<double,
                                               dealii::MemorySpace::Host>
        temperature_host(temperature.get_partitioner());
    temperature_host.import_elements(temperature,
                                     dealii::VectorOperation::insert);
    thermal_physics->get_affine_constraints().distribute(temperature_host);
    if (mechanical_physics)
    {
      mechanical_physics->get_affine_constraints().distribute(displacement);
      post_processor.template write_output<typename Kokkos::View<
          double **, typename MemorySpaceType::kokkos_space>::array_layout>(
          n_time_step, time, temperature_host, displacement,
          mechanical_physics->get_stress_tensor(), state_host,
          material_properties.get_dofs_map(),
          material_properties.get_dof_handler());
    }
    else
    {
      post_processor.template write_thermal_output<typename Kokkos::View<
          double **, typename MemorySpaceType::kokkos_space>::array_layout>(
          n_time_step, time, temperature_host, state_host,
          material_properties.get_dofs_map(),
          material_properties.get_dof_handler());
    }
  }
  else
  {
    mechanical_physics->get_affine_constraints().distribute(displacement);
    post_processor.template write_mechanical_output<typename Kokkos::View<
        double **, typename MemorySpaceType::kokkos_space>::array_layout>(
        n_time_step, time, displacement,
        mechanical_physics->get_stress_tensor(), state_host,
        material_properties.get_dofs_map(),
        material_properties.get_dof_handler());
  }
  timers[adamantine::output].stop();
}

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
      communicator, "Evolve One Time Step: evaluate_material_properties"));
  timers.push_back(adamantine::Timer(communicator, "Output"));
}

template <int dim, int n_materials, int p_order, int fe_degree,
          typename MaterialStates, typename MemorySpaceType,
          typename QuadratureType>
std::unique_ptr<adamantine::ThermalPhysicsInterface<dim, MemorySpaceType>>
initialize(
    MPI_Comm const &communicator, boost::property_tree::ptree const &database,
    adamantine::Geometry<dim> &geometry, adamantine::Boundary const &boundary,
    adamantine::MaterialProperty<dim, n_materials, p_order, MaterialStates,
                                 MemorySpaceType> &material_properties)
{
  return std::make_unique<adamantine::ThermalPhysics<
      dim, n_materials, p_order, fe_degree, MaterialStates, MemorySpaceType,
      QuadratureType>>(communicator, database, geometry, boundary,
                       material_properties);
}

template <int dim, int n_materials, int p_order, int fe_degree,
          typename MaterialStates, typename MemorySpaceType>
std::unique_ptr<adamantine::ThermalPhysicsInterface<dim, MemorySpaceType>>
initialize_quadrature(
    std::string const &quadrature_type, MPI_Comm const &communicator,
    boost::property_tree::ptree const &database,
    adamantine::Geometry<dim> &geometry, adamantine::Boundary const &boundary,
    adamantine::MaterialProperty<dim, n_materials, p_order, MaterialStates,
                                 MemorySpaceType> &material_properties)
{
  if (quadrature_type.compare("gauss") == 0)
    return initialize<dim, n_materials, p_order, fe_degree, MaterialStates,
                      MemorySpaceType, dealii::QGauss<1>>(
        communicator, database, geometry, boundary, material_properties);
  else
  {
    adamantine::ASSERT_THROW(quadrature_type.compare("lobatto") == 0,
                             "quadrature should be Gauss or Lobatto.");
    return initialize<dim, n_materials, p_order, fe_degree, MaterialStates,
                      MemorySpaceType, dealii::QGaussLobatto<1>>(
        communicator, database, geometry, boundary, material_properties);
  }
}

template <int dim, int n_materials, int p_order, typename MaterialStates,
          typename MemorySpaceType>
std::unique_ptr<adamantine::ThermalPhysicsInterface<dim, MemorySpaceType>>
initialize_thermal_physics(
    unsigned int fe_degree, std::string const &quadrature_type,
    MPI_Comm const &communicator, boost::property_tree::ptree const &database,
    adamantine::Geometry<dim> &geometry, adamantine::Boundary const &boundary,
    adamantine::MaterialProperty<dim, n_materials, p_order, MaterialStates,
                                 MemorySpaceType> &material_properties)
{
  switch (fe_degree)
  {
  case 1:
  {
    return initialize_quadrature<dim, n_materials, p_order, 1, MaterialStates,
                                 MemorySpaceType>(quadrature_type, communicator,
                                                  database, geometry, boundary,
                                                  material_properties);
  }
  case 2:
  {
    return initialize_quadrature<dim, n_materials, p_order, 2, MaterialStates,
                                 MemorySpaceType>(quadrature_type, communicator,
                                                  database, geometry, boundary,
                                                  material_properties);
  }
  case 3:
  {
    return initialize_quadrature<dim, n_materials, p_order, 3, MaterialStates,
                                 MemorySpaceType>(quadrature_type, communicator,
                                                  database, geometry, boundary,
                                                  material_properties);
  }
  case 4:
  {
    return initialize_quadrature<dim, n_materials, p_order, 4, MaterialStates,
                                 MemorySpaceType>(quadrature_type, communicator,
                                                  database, geometry, boundary,
                                                  material_properties);
  }
  default:
  {
    adamantine::ASSERT_THROW(fe_degree == 5,
                             "fe_degree should be between 1 and 5.");
    return initialize_quadrature<dim, n_materials, p_order, 5, MaterialStates,
                                 MemorySpaceType>(quadrature_type, communicator,
                                                  database, geometry, boundary,
                                                  material_properties);
  }
  }
}

template <int dim, int n_materials, int p_order, typename MaterialStates,
          typename MemorySpaceType>
void refine_and_transfer(
    std::unique_ptr<adamantine::ThermalPhysicsInterface<dim, MemorySpaceType>>
        &thermal_physics,
    std::unique_ptr<adamantine::MechanicalPhysics<
        dim, n_materials, p_order, MaterialStates, MemorySpaceType>>
        &mechanical_physics,
    adamantine::MaterialProperty<dim, n_materials, p_order, MaterialStates,
                                 MemorySpaceType> &material_properties,
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
#if DEAL_II_VERSION_GTE(9, 7, 0)
  dealii::SolutionTransfer<
#else
  dealii::parallel::distributed::SolutionTransfer<
#endif
      dim, dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host>>
      solution_transfer(dof_handler);

  // Transfer material state
  unsigned int const direction_data_size = 2;
  unsigned int const phase_history_data_size = 1;
  unsigned int constexpr n_material_states = MaterialStates::n_material_states;
  std::vector<std::vector<double>> data_to_transfer;
  std::vector<double> dummy_cell_data(n_material_states + direction_data_size +
                                          phase_history_data_size,
                                      std::numeric_limits<double>::infinity());
  auto state_host = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, material_properties.get_state());
  unsigned int cell_id = 0;
  unsigned int activated_cell_id = 0;
  for (auto const &cell : dof_handler.active_cell_iterators())
  {
    if (cell->is_locally_owned())
    {
      std::vector<double> cell_data(n_material_states + direction_data_size +
                                    phase_history_data_size);
      for (unsigned int i = 0; i < n_material_states; ++i)
        cell_data[i] = state_host(i, cell_id);
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
    solution_host.import_elements(solution, dealii::VectorOperation::insert);
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

  if (mechanical_physics)
  {
    mechanical_physics->prepare_transfer_mpi();
  }

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
  thermal_physics->initialize_dof_vector(0., solution);

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
    solution.import_elements(solution_host, dealii::VectorOperation::insert);
  }

  // Unpack the material state and repopulate the material state
  std::vector<std::vector<double>> transferred_data(
      triangulation.n_active_cells(),
      std::vector<double>(n_material_states + direction_data_size +
                          phase_history_data_size));
  cell_data_trans.unpack(transferred_data);
  auto state = material_properties.get_state();
  state_host = Kokkos::create_mirror_view(state);
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
        state_host(i, cell_id) = transferred_data[total_cell_id][i];
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
  Kokkos::deep_copy(state, state_host);

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
        material_ratio += state_host(i, cell_id);
      }
      ASSERT(std::abs(material_ratio - 1.) < 1e-14, "Material is lost.");
      ++cell_id;
    }
  }
#endif

  if (mechanical_physics)
  {
    mechanical_physics->complete_transfer_mpi();
  }
}

template <int dim>
std::vector<typename dealii::parallel::distributed::Triangulation<
    dim>::active_cell_iterator>
compute_cells_to_refine(
    dealii::parallel::distributed::Triangulation<dim> &triangulation,
    double const time, double const next_refinement_time,
    unsigned int const n_time_steps,
    std::vector<std::shared_ptr<adamantine::HeatSource<dim>>> const
        &heat_sources)
{
  // Compute the position of the beams between time and next_refinement_time and
  // create a list of cells that will intersect the beam.

  // Build the bounding boxes associated with the locally owned cells
  std::vector<dealii::BoundingBox<dim>> cell_bounding_boxes;
  cell_bounding_boxes.reserve(triangulation.n_locally_owned_active_cells());
  for (auto const &cell : triangulation.active_cell_iterators() |
                              dealii::IteratorFilters::LocallyOwnedCell())
  {
    cell_bounding_boxes.push_back(cell->bounding_box());
  }
  dealii::ArborXWrappers::BVH bvh(cell_bounding_boxes);

  double const bounding_box_scaling = 2.0;
  std::vector<dealii::BoundingBox<dim>> heat_source_bounding_boxes;
  heat_source_bounding_boxes.reserve(n_time_steps * heat_sources.size());
  for (unsigned int i = 0; i < n_time_steps; ++i)
  {
    double const current_time = time + static_cast<double>(i) /
                                           static_cast<double>(n_time_steps) *
                                           (next_refinement_time - time);

    // Build the bounding boxes associated with the heat sources
    for (auto &beam : heat_sources)
    {
      heat_source_bounding_boxes.push_back(
          beam->get_bounding_box(current_time, bounding_box_scaling));
    }
  }

  // Perform the search with ArborX. Since we are only interested in locally
  // owned cells, we use BVH.
  dealii::ArborXWrappers::BoundingBoxIntersectPredicate bb_intersect(
      heat_source_bounding_boxes);
  auto [indices, offset] = bvh.query(bb_intersect);

  // Put the indices into a set to get rid of the duplicates and to make it
  // easier to check if the indices are found.
  std::unordered_set<int> indices_to_refine;
  indices_to_refine.insert(indices.begin(), indices.end());

  std::vector<typename dealii::parallel::distributed::Triangulation<
      dim>::active_cell_iterator>
      cells_to_refine;
  int cell_index = 0;
  for (auto const &cell : triangulation.active_cell_iterators() |
                              dealii::IteratorFilters::LocallyOwnedCell())
  {
    if (indices_to_refine.count(cell_index))
    {
      cells_to_refine.push_back(cell);
    }
    ++cell_index;
  }

  return cells_to_refine;
}

template <int dim, int n_materials, int p_order, int fe_degree,
          typename MaterialStates, typename MemorySpaceType>
void refine_mesh(
    std::unique_ptr<adamantine::ThermalPhysicsInterface<dim, MemorySpaceType>>
        &thermal_physics,
    std::unique_ptr<adamantine::MechanicalPhysics<
        dim, n_materials, p_order, MaterialStates, MemorySpaceType>>
        &mechanical_physics,
    adamantine::MaterialProperty<dim, n_materials, p_order, MaterialStates,
                                 MemorySpaceType> &material_properties,
    dealii::LA::distributed::Vector<double, MemorySpaceType> &solution,
    std::vector<std::shared_ptr<adamantine::HeatSource<dim>>> const
        &heat_sources,
    double const time, double const next_refinement_time,
    unsigned int const time_steps_refinement,
    boost::property_tree::ptree const &refinement_database)
{
#ifdef ADAMANTINE_WITH_CALIPER
  CALI_CXX_MARK_FUNCTION;
#endif
  dealii::DoFHandler<dim> &dof_handler = thermal_physics->get_dof_handler();
  dealii::parallel::distributed::Triangulation<dim> &triangulation =
      dynamic_cast<dealii::parallel::distributed::Triangulation<dim> &>(
          const_cast<dealii::Triangulation<dim> &>(
              dof_handler.get_triangulation()));

  // PropertyTreeInput refinement.n_refinements
  unsigned int const n_refinements =
      refinement_database.get("n_refinements", 2);

  for (unsigned int i = 0; i < n_refinements; ++i)
  {
    // Compute the cells to be refined.
    auto cells_to_refine =
        compute_cells_to_refine(triangulation, time, next_refinement_time,
                                time_steps_refinement, heat_sources);

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

      if (cell->level() < static_cast<int>(n_refinements))
        cell->set_refine_flag();
    }

    // Execute the refinement and transfer the solution onto the new mesh.
    refine_and_transfer(thermal_physics, mechanical_physics,
                        material_properties, dof_handler, solution);
  }

  // Recompute the inverse of the mass matrix
  thermal_physics->compute_inverse_mass_matrix();
}

template <int dim, int n_materials, int p_order, typename MaterialStates,
          typename MemorySpaceType>
void refine_mesh(
    std::unique_ptr<adamantine::ThermalPhysicsInterface<dim, MemorySpaceType>>
        &thermal_physics,
    std::unique_ptr<adamantine::MechanicalPhysics<
        dim, n_materials, p_order, MaterialStates, MemorySpaceType>>
        &mechanical_physics,
    adamantine::MaterialProperty<dim, n_materials, p_order, MaterialStates,
                                 MemorySpaceType> &material_properties,
    dealii::LA::distributed::Vector<double, MemorySpaceType> &solution,
    std::vector<std::shared_ptr<adamantine::HeatSource<dim>>> const
        &heat_sources,
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
    refine_mesh<dim, n_materials, p_order, 1, MaterialStates>(
        thermal_physics, mechanical_physics, material_properties, solution,
        heat_sources, time, next_refinement_time, time_steps_refinement,
        refinement_database);
    break;
  }
  case 2:
  {
    refine_mesh<dim, n_materials, p_order, 2, MaterialStates>(
        thermal_physics, mechanical_physics, material_properties, solution,
        heat_sources, time, next_refinement_time, time_steps_refinement,
        refinement_database);
    break;
  }
  case 3:
  {
    refine_mesh<dim, n_materials, p_order, 3, MaterialStates>(
        thermal_physics, mechanical_physics, material_properties, solution,
        heat_sources, time, next_refinement_time, time_steps_refinement,
        refinement_database);
    break;
  }
  case 4:
  {
    refine_mesh<dim, n_materials, p_order, 4, MaterialStates>(
        thermal_physics, mechanical_physics, material_properties, solution,
        heat_sources, time, next_refinement_time, time_steps_refinement,
        refinement_database);
    break;
  }
  case 5:
  {
    refine_mesh<dim, n_materials, p_order, 5, MaterialStates>(
        thermal_physics, mechanical_physics, material_properties, solution,
        heat_sources, time, next_refinement_time, time_steps_refinement,
        refinement_database);
    break;
  }
  default:
  {
    adamantine::ASSERT_THROW(false, "fe_degree should be between 1 and 5.");
  }
  }
}

template <int dim, int n_materials, int p_order, typename MaterialStates,
          typename MemorySpaceType>
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
  // Get optional units property tree
  boost::optional<boost::property_tree::ptree const &> units_optional_database =
      database.get_child_optional("units");

  // Create the Geometry
  boost::property_tree::ptree geometry_database =
      database.get_child("geometry");
  adamantine::Geometry<dim> geometry(communicator, geometry_database,
                                     units_optional_database);

  // Create the MaterialProperty
  boost::property_tree::ptree material_database =
      database.get_child("materials");
  adamantine::MaterialProperty<dim, n_materials, p_order, MaterialStates,
                               MemorySpaceType>
      material_properties(communicator, geometry.get_triangulation(),
                          material_database);

  // Extract the physics property tree
  boost::property_tree::ptree physics_database = database.get_child("physics");
  bool const use_thermal_physics = physics_database.get<bool>("thermal");
  bool const use_mechanical_physics = physics_database.get<bool>("mechanical");

  // Extract the discretization property tree
  boost::property_tree::ptree discretization_database =
      database.get_child("discretization");

  // Extract the post-processor property tree
  boost::property_tree::ptree post_processor_database =
      database.get_child("post_processor");

  // Extract the verbosity
  // PropertyTreeInput verbose_output
  bool const verbose_output = database.get("verbose_output", false);

  // Create the Boundary
  boost::property_tree::ptree boundary_database =
      database.get_child("boundary");
  adamantine::Boundary boundary(
      boundary_database, geometry.get_triangulation().get_boundary_ids());

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
        fe_degree, quadrature_type, communicator, database, geometry, boundary,
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
  unsigned int const n_materials_runtime =
      material_database.get<unsigned int>("n_materials");
  for (unsigned int i = 0; i < n_materials_runtime; ++i)
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
  std::unique_ptr<adamantine::MechanicalPhysics<
      dim, n_materials, p_order, MaterialStates, MemorySpaceType>>
      mechanical_physics;
  if (use_mechanical_physics)
  {
    // PropertyTreeInput discretization.mechanical.fe_degree
    unsigned int const fe_degree =
        discretization_database.get<unsigned int>("mechanical.fe_degree");
    mechanical_physics = std::make_unique<adamantine::MechanicalPhysics<
        dim, n_materials, p_order, MaterialStates, MemorySpaceType>>(
        communicator, fe_degree, geometry, boundary, material_properties,
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

  // Get the checkpoint and restart subtrees
  boost::optional<boost::property_tree::ptree const &>
      checkpoint_optional_database = database.get_child_optional("checkpoint");
  boost::optional<boost::property_tree::ptree const &>
      restart_optional_database = database.get_child_optional("restart");

  unsigned int time_steps_checkpoint = std::numeric_limits<unsigned int>::max();
  std::string checkpoint_filename;
  bool checkpoint_overwrite = true;
  if (checkpoint_optional_database)
  {
    auto checkpoint_database = checkpoint_optional_database.get();
    // PropertyTreeInput checkpoint.time_steps_between_checkpoint
    time_steps_checkpoint =
        checkpoint_database.get<unsigned int>("time_steps_between_checkpoint");
    // PropertyTreeInput checkpoint.filename_prefix
    checkpoint_filename =
        checkpoint_database.get<std::string>("filename_prefix");
    // PropertyTreeInput checkpoint.overwrite_files
    checkpoint_overwrite = checkpoint_database.get<bool>("overwrite_files");
  }

  bool restart = false;
  std::string restart_filename;
  if (restart_optional_database)
  {
    restart = true;
    auto restart_database = restart_optional_database.get();
    // PropertyTreeInput restart.filename_prefix
    restart_filename = restart_database.get<std::string>("filename_prefix");
  }

  dealii::LA::distributed::Vector<double, MemorySpaceType> temperature;
  dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host>
      displacement;
  if (use_thermal_physics)
  {
    if (restart == false)
    {
      thermal_physics->setup();
      thermal_physics->initialize_dof_vector(initial_temperature, temperature);
    }
    else
    {
#ifdef ADAMANTINE_WITH_CALIPER
      CALI_MARK_BEGIN("restart from file");
#endif
      thermal_physics->load_checkpoint(restart_filename, temperature);
#ifdef ADAMANTINE_WITH_CALIPER
      CALI_MARK_END("restart from file");
#endif
    }
  }

  if (use_mechanical_physics)
  {
    // We currently do not support restarting the mechanical simulation.
    adamantine::ASSERT_THROW(
        restart == false,
        "Mechanical simulation cannot be restarted from a file");
    if (use_thermal_physics)
    {
      // Thermo-mechanical simulation
      dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host>
          temperature_host(temperature.get_partitioner());
      temperature_host.import_elements(temperature,
                                       dealii::VectorOperation::insert);
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
  unsigned int n_time_step = 0;
  double time = 0.;
  double activation_time_end = -1.;
  unsigned int const rank =
      dealii::Utilities::MPI::this_mpi_process(communicator);
  if (restart == true)
  {
    if (rank == 0)
    {
      std::cout << "Restarting from file" << std::endl;
    }
    std::ifstream file{restart_filename + "_time.txt"};
    boost::archive::text_iarchive ia{file};
    ia >> time;
    ia >> n_time_step;
  }
  // PropertyTreeInput geometry.deposition_time
  double const activation_time =
      geometry_database.get<double>("deposition_time", 0.);

  // Output the initial solution
  output_pvtu(*post_processor, n_time_step, time, thermal_physics, temperature,
              mechanical_physics, displacement, material_properties, timers);
  ++n_time_step;

  // Create the bounding boxes used for material deposition
  auto [material_deposition_boxes, deposition_times, deposition_cos,
        deposition_sin] =
      adamantine::create_material_deposition_boxes<dim>(geometry_database,
                                                        heat_sources);
  // Extract the time-stepping database
  boost::property_tree::ptree time_stepping_database =
      database.get_child("time_stepping");
  // PropertyTreeInput time_stepping.time_step
  double time_step = time_stepping_database.get<double>("time_step");
  // PropertyTreeInput time_stepping.mechanical_time_step_factor
  unsigned int const mechanical_time_step_factor =
      time_stepping_database.get<unsigned int>("mechanical_time_step_factor",
                                               1);
  // PropertyTreeInput time_stepping.scan_path_for_duration
  bool const scan_path_for_duration =
      time_stepping_database.get("scan_path_for_duration", false);
  // PropertyTreeInput time_stepping.duration
  double const duration = scan_path_for_duration
                              ? std::numeric_limits<double>::max()
                              : time_stepping_database.get<double>("duration");

  // Extract the refinement database
  boost::property_tree::ptree refinement_database =
      database.get_child("refinement");
  // PropertyTreeInput refinement.time_steps_between_refinement
  unsigned int const time_steps_refinement =
      refinement_database.get("time_steps_between_refinement", 10);
  // PropertyTreeInput post_processor.time_steps_between_output
  unsigned int const time_steps_output =
      post_processor_database.get("time_steps_between_output", 1);
  // PropertyTreeInput materials.new_material_temperature
  double const new_material_temperature =
      database.get("materials.new_material_temperature", 300.);

  // Extract the microstructure database if present
  boost::optional<boost::property_tree::ptree const &>
      microstructure_optional_database =
          database.get_child_optional("microstructure");
  bool const compute_microstructure =
      microstructure_optional_database ? true : false;
  std::unique_ptr<adamantine::Microstructure<dim>> microstructure;
  if (compute_microstructure)
  {
    auto microstructure_database = microstructure_optional_database.get();
    // PropertyTreeInput microstructure.filename_prefix
    std::string microstructure_filename =
        microstructure_database.get<std::string>("filename_prefix");
    microstructure = std::make_unique<adamantine::Microstructure<dim>>(
        communicator, microstructure_filename);
  }

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

    // Refine the mesh the first time we get in the loop and after
    // time_steps_refinement time steps.
    if (((n_time_step == 1) || ((n_time_step % time_steps_refinement) == 0)) &&
        use_thermal_physics)
    {
      timers[adamantine::refine].start();
      double next_refinement_time = time + time_steps_refinement * time_step;
      refine_mesh(thermal_physics, mechanical_physics, material_properties,
                  temperature, heat_sources, time, next_refinement_time,
                  time_steps_refinement, refinement_database);
      timers[adamantine::refine].stop();
      if ((rank == 0) && (verbose_output == true))
      {
        std::cout << "n_time_step: " << n_time_step << " time: " << time
                  << " n_dofs after mesh refinement: "
                  << thermal_physics->get_dof_handler().n_dofs() << std::endl;
      }
    }

    // Add material if necessary.
    // We use an epsilon to get the "expected" behavior when the deposition
    // time and the time match should match exactly but don't because of
    // floating point accuracy.
    timers[adamantine::add_material_activate].start();
    if (time > activation_time_end)
    {
      // If we use scan_path_for_duration, we may need to read the scan path
      // file once again.
      if (scan_path_for_duration)
      {
        // Check if we have reached the end of current scan path.
        bool need_updated_scan_path = false;
        for (auto &source : heat_sources)
        {
          if (time > source->get_scan_path().get_segment_list().back().end_time)
          {
            need_updated_scan_path = true;
            break;
          }
        }

        if (need_updated_scan_path)
        {
          // Check if we have reached the end of the file. If not, read the
          // updated scan path file
          bool scan_path_end = true;
          for (auto &source : heat_sources)
          {
            if (!source->get_scan_path().is_finished())
            {
              scan_path_end = false;
              // This functions waits for the scan path file to be updated
              // before reading the file.
              source->get_scan_path().read_file();
            }
          }

          // If we have reached the end of scan path file for all the heat
          // sources, we just exit.
          if (scan_path_end)
          {
            break;
          }

          std::tie(material_deposition_boxes, deposition_times, deposition_cos,
                   deposition_sin) =
              adamantine::create_material_deposition_boxes<dim>(
                  geometry_database, heat_sources);
        }
      }

      double const eps = time_step / 1e10;

      auto activation_start =
          std::lower_bound(deposition_times.begin(), deposition_times.end(),
                           time - eps) -
          deposition_times.begin();
      activation_time_end =
          std::min(time + std::max(activation_time, time_step), duration) - eps;
      auto activation_end =
          std::lower_bound(deposition_times.begin(), deposition_times.end(),
                           activation_time_end) -
          deposition_times.begin();
      if (activation_start < activation_end)
      {
        if (use_thermal_physics)
        {
#ifdef ADAMANTINE_WITH_CALIPER
          CALI_MARK_BEGIN("add material");
#endif
          // Compute the elements to activate.
          // TODO Right now, we compute the list of cells that get activated for
          // the entire material deposition. We should restrict the list to the
          // cells that are activated between activation_start and
          // activation_end.
          timers[adamantine::add_material_search].start();
          auto elements_to_activate = adamantine::get_elements_to_activate(
              geometry, thermal_physics->get_dof_handler(),
              material_deposition_boxes);
          timers[adamantine::add_material_search].stop();

          // For now assume that all deposited material has never been melted
          // (may or may not be reasonable)
          std::vector<bool> has_melted(deposition_cos.size(), false);

          thermal_physics->add_material_start(
              elements_to_activate, deposition_cos, deposition_sin, has_melted,
              activation_start, activation_end, temperature);

          if (use_mechanical_physics)
          {
            mechanical_physics->prepare_transfer_mpi();
          }

#ifdef ADAMANTINE_WITH_CALIPER
          CALI_MARK_BEGIN("refine triangulation");
#endif
          dealii::parallel::distributed::Triangulation<dim> &triangulation =
              dynamic_cast<dealii::parallel::distributed::Triangulation<dim> &>(
                  const_cast<dealii::Triangulation<dim> &>(
                      thermal_physics->get_dof_handler().get_triangulation()));
          triangulation.execute_coarsening_and_refinement();
#ifdef ADAMANTINE_WITH_CALIPER
          CALI_MARK_END("refine triangulation");
#endif

          thermal_physics->add_material_end(new_material_temperature,
                                            temperature);

          if (use_mechanical_physics)
          {
            mechanical_physics->complete_transfer_mpi();
          }

#ifdef ADAMANTINE_WITH_CALIPER
          CALI_MARK_END("add material");
#endif
        }
      }

      if ((rank == 0) && (verbose_output == true) &&
          (activation_end - activation_start > 0) && use_thermal_physics)
      {
        std::cout << "n_time_step: " << n_time_step << " time: " << time
                  << " n_dofs after cell activation: "
                  << thermal_physics->get_dof_handler().n_dofs() << std::endl;
      }
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

    timers[adamantine::evol_time].start();

    // Solve the thermal problem
    if (use_thermal_physics)
    {
      if (compute_microstructure)
      {
#ifdef ADAMANTINE_WITH_CALIPER
        CALI_MARK_BEGIN("compute microstructure");
#endif
        if constexpr (std::is_same_v<MemorySpaceType,
                                     dealii::MemorySpace::Host>)
        {
          microstructure->set_old_temperature(temperature);
        }
        else
        {
          dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host>
              temperature_host(temperature.get_partitioner());
          temperature_host.import_elements(temperature,
                                           dealii::VectorOperation::insert);
          microstructure->set_old_temperature(temperature_host);
        }
#ifdef ADAMANTINE_WITH_CALIPER
        CALI_MARK_END("compute microstructure");
#endif
      }

      time = thermal_physics->evolve_one_time_step(time, time_step, temperature,
                                                   timers);

      if (compute_microstructure)
      {
#ifdef ADAMANTINE_WITH_CALIPER
        CALI_MARK_BEGIN("compute microstructure");
#endif
        if constexpr (std::is_same_v<MemorySpaceType,
                                     dealii::MemorySpace::Host>)
        {
          microstructure->compute_G_and_R(material_properties,
                                          thermal_physics->get_dof_handler(),
                                          temperature, time_step);
        }
        else
        {
          dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host>
              temperature_host(temperature.get_partitioner());
          temperature_host.import_elements(temperature,
                                           dealii::VectorOperation::insert);
          microstructure->compute_G_and_R(material_properties,
                                          thermal_physics->get_dof_handler(),
                                          temperature_host, time_step);
        }
#ifdef ADAMANTINE_WITH_CALIPER
        CALI_MARK_END("compute microstructure");
#endif
      }
    }

    // Solve the (thermo-)mechanical problem
    if (use_mechanical_physics)
    {
      // Solve if the time step is a multipe of mechanical_time_step_factor or
      // if we will output the results.
      if ((n_time_step % mechanical_time_step_factor == 0) ||
          (n_time_step % time_steps_output == 0))
      {
        if (use_thermal_physics)
        {
          // Update the material state
          thermal_physics->set_state_to_material_properties();
          dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host>
              temperature_host(temperature.get_partitioner());
          temperature_host.import_elements(temperature,
                                           dealii::VectorOperation::insert);
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

    timers[adamantine::evol_time].stop();

    if (n_time_step % time_steps_checkpoint == 0)
    {
#ifdef ADAMANTINE_WITH_CALIPER
      CALI_MARK_BEGIN("save checkpoint");
#endif
      if (rank == 0)
      {
        std::cout << "Checkpoint reached" << std::endl;
      }

      std::string output_dir =
          post_processor_database.get<std::string>("output_dir", "");
      std::string filename_prefix =
          checkpoint_overwrite
              ? checkpoint_filename
              : checkpoint_filename + '_' + std::to_string(n_time_step);
      thermal_physics->save_checkpoint(filename_prefix, temperature);
      std::ofstream file{output_dir + filename_prefix + "_time.txt"};
      boost::archive::text_oarchive oa{file};
      oa << time;
      oa << n_time_step;
#ifdef ADAMANTINE_WITH_CALIPER
      CALI_MARK_END("save checkpoint");
#endif
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
        progress = static_cast<unsigned int>(int_part);
      }
    }

    // Output the solution
    if (n_time_step % time_steps_output == 0)
    {
      if (use_thermal_physics)
      {
        thermal_physics->set_state_to_material_properties();
      }
      output_pvtu(*post_processor, n_time_step, time, thermal_physics,
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
    temperature_host.import_elements(temperature,
                                     dealii::VectorOperation::insert);
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

void split_global_communicator(MPI_Comm global_communicator,
                               unsigned int global_ensemble_size,
                               MPI_Comm &local_communicator,
                               unsigned int &local_ensemble_size,
                               unsigned int &first_local_member, int &my_color)
{
  unsigned int const global_rank =
      dealii::Utilities::MPI::this_mpi_process(global_communicator);
  unsigned int const global_n_procs =
      dealii::Utilities::MPI::n_mpi_processes(global_communicator);

  double const avg_n_procs_per_member =
      static_cast<double>(global_n_procs) /
      static_cast<double>(global_ensemble_size);
  if (avg_n_procs_per_member > 1)
  {
    local_ensemble_size = 1;
    // We need all the members to use the same partitioning otherwise the dofs
    // may be number differently which is problematic for the data assimulation.
    adamantine::ASSERT_THROW(
        avg_n_procs_per_member == std::floor(avg_n_procs_per_member),
        "Number of MPI ranks should be less than the number of ensemble "
        "members or a multiple of this number.");
    // Assign color
    my_color =
        global_rank / static_cast<int>(std::floor(avg_n_procs_per_member));

    first_local_member = my_color;
  }
  else
  {
    my_color = global_rank;
    // Compute the local ensemble size
    unsigned int const members_per_proc = global_ensemble_size / global_n_procs;
    unsigned int const leftover_members =
        global_ensemble_size - global_n_procs * members_per_proc;
    // We assign all the leftover members to processor 0. We could do a
    // better job distributing the leftover
    local_ensemble_size = members_per_proc;
    if (global_rank == 0)
    {
      local_ensemble_size += leftover_members;
      first_local_member = 0;
    }
    else
    {
      first_local_member = global_rank * local_ensemble_size + leftover_members;
    }
  }

  MPI_Comm_split(global_communicator, my_color, 0, &local_communicator);
}

template <int dim, int n_materials, int p_order, typename MaterialStates,
          typename MemorySpaceType>
std::vector<dealii::LA::distributed::BlockVector<double>>
run_ensemble(MPI_Comm const &global_communicator,
             boost::property_tree::ptree const &database,
             std::vector<adamantine::Timer> &timers)
{
#ifdef ADAMANTINE_WITH_CALIPER
  CALI_CXX_MARK_FUNCTION;
#endif

  // ------ Extract child property trees -----
  // Mandatory subtrees
  boost::property_tree::ptree geometry_database =
      database.get_child("geometry");
  boost::property_tree::ptree boundary_database =
      database.get_child("boundary");
  boost::property_tree::ptree discretization_database =
      database.get_child("discretization");
  boost::property_tree::ptree time_stepping_database =
      database.get_child("time_stepping");
  boost::property_tree::ptree post_processor_database =
      database.get_child("post_processor");
  boost::property_tree::ptree refinement_database =
      database.get_child("refinement");
  boost::property_tree::ptree material_database =
      database.get_child("materials");

  // Optional subtrees
  boost::optional<boost::property_tree::ptree const &> units_optional_database =
      database.get_child_optional("units");
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

  // ------ Split MPI communicator -----
  // PropertyTreeInput ensemble.ensemble_size
  unsigned int const global_ensemble_size =
      database.get("ensemble.ensemble_size", 5);
  // Distribute the processors among the ensemble members
  MPI_Comm local_communicator;
  unsigned int local_ensemble_size = std::numeric_limits<unsigned int>::max();
  unsigned int first_local_member = std::numeric_limits<unsigned int>::max();
  int my_color = -1;
  split_global_communicator(global_communicator, global_ensemble_size,
                            local_communicator, local_ensemble_size,
                            first_local_member, my_color);

  // ------ Set up the ensemble members -----
  // There might be a more efficient way to share some of these objects
  // between ensemble members. For now, we'll do the simpler approach of
  // duplicating everything.

  // Create a new property tree database for each ensemble member
  std::vector<boost::property_tree::ptree> database_ensemble =
      adamantine::create_database_ensemble(
          database, local_communicator, first_local_member, local_ensemble_size,
          global_ensemble_size);

  std::vector<std::unique_ptr<
      adamantine::ThermalPhysicsInterface<dim, MemorySpaceType>>>
      thermal_physics_ensemble(local_ensemble_size);

  std::vector<std::vector<std::shared_ptr<adamantine::HeatSource<dim>>>>
      heat_sources_ensemble(local_ensemble_size);

  std::vector<std::unique_ptr<adamantine::Geometry<dim>>> geometry_ensemble;

  std::vector<std::unique_ptr<adamantine::Boundary>> boundary_ensemble;

  std::vector<std::unique_ptr<adamantine::MaterialProperty<
      dim, n_materials, p_order, MaterialStates, MemorySpaceType>>>
      material_properties_ensemble;

  std::vector<std::unique_ptr<adamantine::PostProcessor<dim>>>
      post_processor_ensemble;

  // Create the vector of augmented state vectors
  std::vector<dealii::LA::distributed::BlockVector<double>>
      solution_augmented_ensemble(local_ensemble_size);

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
  adamantine::DataAssimilator data_assimilator(global_communicator,
                                               local_communicator, my_color,
                                               data_assimilation_database);

  // Get the checkpoint and restart subtrees
  boost::optional<boost::property_tree::ptree const &>
      checkpoint_optional_database = database.get_child_optional("checkpoint");
  boost::optional<boost::property_tree::ptree const &>
      restart_optional_database = database.get_child_optional("restart");

  unsigned int const global_rank =
      dealii::Utilities::MPI::this_mpi_process(global_communicator);
  unsigned int time_steps_checkpoint = std::numeric_limits<unsigned int>::max();
  std::string checkpoint_filename;
  bool checkpoint_overwrite = true;
  if (checkpoint_optional_database)
  {
    auto checkpoint_database = checkpoint_optional_database.get();
    // PropertyTreeInput checkpoint.time_steps_between_checkpoint
    time_steps_checkpoint =
        checkpoint_database.get<unsigned int>("time_steps_between_checkpoint");
    // PropertyTreeInput checkpoint.filename_prefix
    checkpoint_filename =
        checkpoint_database.get<std::string>("filename_prefix");
    // PropertyTreeInput checkpoint.overwrite_files
    checkpoint_overwrite = checkpoint_database.get<bool>("overwrite_files");
  }

  bool restart = false;
  std::string restart_filename;
  if (restart_optional_database)
  {
    restart = true;
    auto restart_database = restart_optional_database.get();
    // PropertyTreeInput restart.filename_prefix
    restart_filename = restart_database.get<std::string>("filename_prefix");
  }
  for (unsigned int member = 0; member < local_ensemble_size; ++member)
  {
    // Resize the augmented ensemble block vector to have two blocks
    solution_augmented_ensemble[member].reinit(2);

    // Edit the database for the ensemble
    if (database.get<unsigned int>("sources.n_beams") > 0)
    {
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
                database_ensemble[member].get<double>(
                    "sources.beam_0.absorption_efficiency");
          }
          else if (augmented_state_parameters.at(index) ==
                   adamantine::AugmentedStateParameters::beam_0_max_power)
          {
            solution_augmented_ensemble[member].block(augmented_state)[index] =
                database_ensemble[member].get<double>(
                    "sources.beam_0.max_power");
          }
        }
      }
    }

    solution_augmented_ensemble[member].collect_sizes();

    geometry_ensemble.push_back(std::make_unique<adamantine::Geometry<dim>>(
        local_communicator, geometry_database, units_optional_database));

    boundary_ensemble.push_back(std::make_unique<adamantine::Boundary>(
        boundary_database,
        geometry_ensemble.back()->get_triangulation().get_boundary_ids()));

    material_properties_ensemble.push_back(
        std::make_unique<adamantine::MaterialProperty<
            dim, n_materials, p_order, MaterialStates, MemorySpaceType>>(
            local_communicator, geometry_ensemble.back()->get_triangulation(),
            material_database));

    thermal_physics_ensemble[member] = initialize_thermal_physics<dim>(
        fe_degree, quadrature_type, local_communicator,
        database_ensemble[member], *geometry_ensemble[member],
        *boundary_ensemble[member], *material_properties_ensemble[member]);
    heat_sources_ensemble[member] =
        thermal_physics_ensemble[member]->get_heat_sources();

    if (restart == false)
    {
      thermal_physics_ensemble[member]->setup();
      thermal_physics_ensemble[member]->initialize_dof_vector(
          database_ensemble[member].get("materials.initial_temperature", 300.),
          solution_augmented_ensemble[member].block(base_state));
    }
    else
    {
#ifdef ADAMANTINE_WITH_CALIPER
      CALI_MARK_BEGIN("restart from file");
#endif
      thermal_physics_ensemble[member]->load_checkpoint(
          restart_filename + '_' + std::to_string(member),
          solution_augmented_ensemble[member].block(base_state));
#ifdef ADAMANTINE_WITH_CALIPER
      CALI_MARK_END("restart from file");
#endif
    }
    solution_augmented_ensemble[member].collect_sizes();

    // For now we only output temperature
    post_processor_database.put("thermal_output", true);
    post_processor_ensemble.push_back(
        std::make_unique<adamantine::PostProcessor<dim>>(
            local_communicator, post_processor_database,
            thermal_physics_ensemble[member]->get_dof_handler(),
            first_local_member + member));
  }

  // PostProcessor for outputting the experimental data
  boost::property_tree::ptree post_processor_expt_database;
  // PropertyTreeInput post_processor.file_name
  std::string expt_file_prefix =
      post_processor_database.get<std::string>("filename_prefix") + ".expt";
  post_processor_expt_database.put("filename_prefix", expt_file_prefix);
  post_processor_expt_database.put("thermal_output", true);
  std::unique_ptr<adamantine::PostProcessor<dim>> post_processor_expt;
  // PropertyTreeInput experiment.output_experiment_on_mesh
  bool const output_experiment_on_mesh =
      experiment_optional_database
          ? experiment_optional_database->get("output_experiment_on_mesh", true)
          : false;
  // Optionally output the experimental data projected onto the mesh. Since the
  // experimental data is unique we may not need all the processors to write the
  // output files.
  if (output_experiment_on_mesh && (my_color == 0))
  {
    post_processor_expt = std::make_unique<adamantine::PostProcessor<dim>>(
        local_communicator, post_processor_expt_database,
        thermal_physics_ensemble[0]->get_dof_handler());
  }

  // ----- Initialize time and time stepping counters -----
  unsigned int progress = 0;
  unsigned int n_time_step = 0;
  double time = 0.;
  double activation_time_end = -1.;
  double da_time = -1.;
  if (restart == true)
  {
    if (global_rank == 0)
    {
      std::cout << "Restarting from file" << std::endl;
    }
    std::ifstream file{restart_filename + "_time.txt"};
    boost::archive::text_iarchive ia{file};
    ia >> time;
    ia >> n_time_step;
  }

  // PropertyTreeInput geometry.deposition_time
  double const activation_time =
      geometry_database.get<double>("deposition_time", 0.);

  // ----- Read the experimental data -----
  std::vector<std::vector<double>> frame_time_stamps;
  std::unique_ptr<adamantine::ExperimentalData<dim>> experimental_data;
  unsigned int experimental_frame_index =
      std::numeric_limits<unsigned int>::max();

  if (experiment_optional_database)
  {
    auto experiment_database = experiment_optional_database.get();

    // PropertyTreeInput experiment.read_in_experimental_data
    bool const read_in_experimental_data =
        experiment_database.get("read_in_experimental_data", false);

    if (read_in_experimental_data)
    {
      if (global_rank == 0)
        std::cout << "Reading the experimental log file..." << std::endl;

      frame_time_stamps =
          adamantine::read_frame_timestamps(experiment_database);

      adamantine::ASSERT_THROW(frame_time_stamps.size() > 0,
                               "Experimental data parsing is activated, but "
                               "the log shows zero cameras.");
      adamantine::ASSERT_THROW(frame_time_stamps[0].size() > 0,
                               "Experimental data parsing is activated, but "
                               "the log shows zero data frames.");

      if (global_rank == 0)
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

      // Advance the experimental frame counter upon restart, if necessary
      if (restart)
      {
        bool found_frame = false;
        while (!found_frame)
        {
          if (frame_time_stamps[0][experimental_frame_index + 1] < time)
          {
            experimental_frame_index++;
          }
          else
          {
            found_frame = true;
          }
        }
      }
    }
  }

  // ----- Output the initial solution -----
  std::unique_ptr<adamantine::MechanicalPhysics<
      dim, n_materials, p_order, MaterialStates, MemorySpaceType>>
      mechanical_physics;
  dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host>
      displacement;
  for (unsigned int member = 0; member < local_ensemble_size; ++member)
  {
    output_pvtu(*post_processor_ensemble[member], n_time_step, time,
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
  // PropertyTreeInput time_stepping.time_step
  double time_step = time_stepping_database.get<double>("time_step");
  double const da_time_half_window = 1.01 * time_step;
  // PropertyTreeInput time_stepping.scan_path_for_duration
  bool const scan_path_for_duration =
      time_stepping_database.get("scan_path_for_duration", false);
  // PropertyTreeInput time_stepping.duration
  double const duration = scan_path_for_duration
                              ? std::numeric_limits<double>::max()
                              : time_stepping_database.get<double>("duration");
  // PropertyTreeInput post_processor.time_steps_between_output
  unsigned int const time_steps_output =
      post_processor_database.get("time_steps_between_output", 1);
  // PropertyTreeInput post_processor.output_on_data_assimilation
  bool const output_on_da =
      post_processor_database.get("output_on_data_assimilation", true);

  // ----- Deposit material -----
  // For now assume that all ensemble members share the same geometry (they
  // have independent adamantine::Geometry objects, but all are constructed
  // from identical parameters), base new additions on the 0th ensemble member
  // since all the sources use the same scan path.
  auto [material_deposition_boxes, deposition_times, deposition_cos,
        deposition_sin] =
      adamantine::create_material_deposition_boxes<dim>(
          geometry_database, heat_sources_ensemble[0]);

  // ----- Compute bounding heat sources -----
  // When using AMR, we refine the cells that the heat sources intersect. Since
  // each ensemble members can have slightly different sources, we create a new
  // set of heat sources that encompasses the member sources. We use these new
  // sources when refining the mesh.
  auto bounding_heat_sources = adamantine::get_bounding_heat_sources<dim>(
      database_ensemble, global_communicator);

  // ----- Main time stepping loop -----
  if (global_rank == 0)
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
    // Refine the mesh the first time we get in the loop and after
    // time_steps_refinement time steps.
    if ((n_time_step == 1) || ((n_time_step % time_steps_refinement) == 0))
    {
      timers[adamantine::refine].start();
      double const next_refinement_time =
          time + time_steps_refinement * time_step;

      for (unsigned int member = 0; member < local_ensemble_size; ++member)
      {
        // FIXME
        std::unique_ptr<adamantine::MechanicalPhysics<
            dim, n_materials, p_order, MaterialStates, MemorySpaceType>>
            dummy;
        refine_mesh(thermal_physics_ensemble[member], dummy,
                    *material_properties_ensemble[member],
                    solution_augmented_ensemble[member].block(base_state),
                    bounding_heat_sources, time, next_refinement_time,
                    time_steps_refinement, refinement_database);
        solution_augmented_ensemble[member].collect_sizes();
      }

      timers[adamantine::refine].stop();
      if ((global_rank == 0) && (verbose_output == true))
      {
        std::cout << "n_time_step: " << n_time_step << " time: " << time
                  << " n_dofs: "
                  << thermal_physics_ensemble[0]->get_dof_handler().n_dofs()
                  << std::endl;
      }
    }

    // We use an epsilon to get the "expected" behavior when the deposition
    // time and the time match should match exactly but don't because of
    // floating point accuracy.
    timers[adamantine::add_material_activate].start();
    if (time > activation_time_end)
    {
      // If we use scan_path_for_duration, we may need to read the scan path
      // file once again.
      if (scan_path_for_duration)
      {
        // Check if we have reached the end of current scan path.
        bool need_updated_scan_path = false;
        {
          for (auto &source : heat_sources_ensemble[0])
          {
            if (time >
                source->get_scan_path().get_segment_list().back().end_time)
            {
              need_updated_scan_path = true;
              break;
            }
          }
        }

        if (need_updated_scan_path)
        {
          // Check if we have reached the end of the file. If not, read the
          // updated scan path file. We assume that the scan paths are identical
          // for all the heat sources and thus, we can use
          // heat_sources_ensemble[0] to get the material deposition boxes and
          // times. We still need every ensemble member to read the scan path in
          // order to compute the correct heat sources.
          bool scan_path_end = true;
          for (unsigned int member = 0; member < local_ensemble_size; ++member)
          {
            for (auto &source : heat_sources_ensemble[member])
            {
              if (!source->get_scan_path().is_finished())
              {
                scan_path_end = false;
                // This functions waits for the scan path file to be updated
                // before reading the file.
                source->get_scan_path().read_file();
              }
            }
          }

          // If we have reached the end of scan path file for all the heat
          // sources, we just exit.
          if (scan_path_end)
          {
            break;
          }

          std::tie(material_deposition_boxes, deposition_times, deposition_cos,
                   deposition_sin) =
              adamantine::create_material_deposition_boxes<dim>(
                  geometry_database, heat_sources_ensemble[0]);
        }
      }

      double const eps = time_step / 1e10;
      auto activation_start =
          std::lower_bound(deposition_times.begin(), deposition_times.end(),
                           time - eps) -
          deposition_times.begin();
      activation_time_end =
          std::min(time + std::max(activation_time, time_step), duration) - eps;
      auto activation_end =
          std::lower_bound(deposition_times.begin(), deposition_times.end(),
                           activation_time_end) -
          deposition_times.begin();
      if (activation_start < activation_end)
      {
#ifdef ADAMANTINE_WITH_CALIPER
        CALI_MARK_BEGIN("add material");
#endif
        for (unsigned int member = 0; member < local_ensemble_size; ++member)
        {
          // Compute the elements to activate.
          // TODO Right now, we compute the list of cells that get activated
          // for the entire material deposition. We should restrict the list
          // to the cells that are activated between activation_start and
          // activation_end.
          timers[adamantine::add_material_search].start();
          auto elements_to_activate = adamantine::get_elements_to_activate(
              *geometry_ensemble[member],
              thermal_physics_ensemble[member]->get_dof_handler(),
              material_deposition_boxes);
          timers[adamantine::add_material_search].stop();
          // For now assume that all deposited material has never been
          // melted (may or may not be reasonable)
          std::vector<bool> has_melted(deposition_cos.size(), false);

          thermal_physics_ensemble[member]->add_material_start(
              elements_to_activate, deposition_cos, deposition_sin, has_melted,
              activation_start, activation_end,
              solution_augmented_ensemble[member].block(base_state));

#ifdef ADAMANTINE_WITH_CALIPER
          CALI_MARK_BEGIN("refine triangulation");
#endif
          dealii::DoFHandler<dim> &dof_handler =
              thermal_physics_ensemble[member]->get_dof_handler();
          dealii::parallel::distributed::Triangulation<dim> &triangulation =
              dynamic_cast<dealii::parallel::distributed::Triangulation<dim> &>(
                  const_cast<dealii::Triangulation<dim> &>(
                      dof_handler.get_triangulation()));

          triangulation.execute_coarsening_and_refinement();
#ifdef ADAMANTINE_WITH_CALIPER
          CALI_MARK_END("refine triangulation");
#endif

          thermal_physics_ensemble[member]->add_material_end(
              database_ensemble[member].get(
                  "materials.new_material_temperature", 300.),
              solution_augmented_ensemble[member].block(base_state));

          solution_augmented_ensemble[member].collect_sizes();
        }
#ifdef ADAMANTINE_WITH_CALIPER
        CALI_MARK_END("add material");
#endif
      }

      if ((global_rank == 0) && (verbose_output == true) &&
          (activation_end - activation_start > 0))
      {
        std::cout << "n_time_step: " << n_time_step << " time: " << time
                  << " n_dofs after cell activation: "
                  << thermal_physics_ensemble[0]->get_dof_handler().n_dofs()
                  << std::endl;
      }
    }
    timers[adamantine::add_material_activate].stop();

    // ----- Evolve the solution by one time step -----
    double const old_time = time;
    timers[adamantine::evol_time].start();

    for (unsigned int member = 0; member < local_ensemble_size; ++member)
    {
      time = thermal_physics_ensemble[member]->evolve_one_time_step(
          old_time, time_step,
          solution_augmented_ensemble[member].block(base_state), timers);
    }
    timers[adamantine::evol_time].stop();

    // ----- Perform data assimilation -----
    if (assimilate_data)
    {
      for (unsigned int member = 0; member < local_ensemble_size; ++member)
      {
        thermal_physics_ensemble[member]->get_affine_constraints().distribute(
            solution_augmented_ensemble[member].block(base_state));
      }

      // Currently assume that all frames are synced so that the 0th camera
      // frame time is the relevant time
      double frame_time = std::numeric_limits<double>::max();
      if ((experimental_frame_index + 1) < frame_time_stamps[0].size())
      {
        if (time > da_time + da_time_half_window)
        {
          da_time = frame_time_stamps[0][experimental_frame_index + 1];
        }
        if (frame_time_stamps[0][experimental_frame_index + 1] <= time)
        {
          experimental_frame_index = experimental_data->read_next_frame();
          frame_time = frame_time_stamps[0][experimental_frame_index];
        }
      }

      if (frame_time <= time)
      {
        adamantine::ASSERT_THROW(
            frame_time > old_time || n_time_step == 1,
            "Unexpectedly missed a data assimilation frame.");
        if (global_rank == 0)
          std::cout << "Performing data assimilation at time " << time << "..."
                    << std::endl;

        // Print out the augmented parameters
        for (unsigned int member = 0; member < local_ensemble_size; ++member)
        {
          std::cout << "Rank: " << global_rank
                    << " | Old parameters for member "
                    << first_local_member + member << ": ";
          for (auto param : solution_augmented_ensemble[member].block(1))
            std::cout << param << " ";

          std::cout << std::endl;
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
        std::cout << "Rank: " << global_rank
                  << " | Number expt sites mapped to DOFs: "
                  << expt_to_dof_mapping.first.size() << std::endl;
#ifdef ADAMANTINE_WITH_CALIPER
        CALI_MARK_END("da_experimental_data");
#endif
        timers[adamantine::da_experimental_data].stop();

        if (post_processor_expt)
        {
          dealii::LA::distributed::Vector<double, MemorySpaceType>
              temperature_expt;
          temperature_expt.reinit(
              solution_augmented_ensemble[0].block(base_state));
          temperature_expt.add(1.0e10);
          adamantine::set_with_experimental_data(
              global_communicator, points_values, expt_to_dof_mapping,
              temperature_expt, verbose_output);

          thermal_physics_ensemble[0]->get_affine_constraints().distribute(
              temperature_expt);
          post_processor_expt
              ->template write_thermal_output<typename Kokkos::View<
                  double **,
                  typename MemorySpaceType::kokkos_space>::array_layout>(
                  n_time_step, time, temperature_expt,
                  material_properties_ensemble[0]->get_state(),
                  material_properties_ensemble[0]->get_dofs_map(),
                  material_properties_ensemble[0]->get_dof_handler());
        }

        // Only continue data assimilation if some of the observations are
        // mapped to DOFs
        if (expt_to_dof_mapping.first.size() > 0)
        {
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
          data_assimilator.update_ensemble(solution_augmented_ensemble,
                                           points_values.values, R);
#ifdef ADAMANTINE_WITH_CALIPER
          CALI_MARK_END("da_update_ensemble");
#endif
          timers[adamantine::da_update_ensemble].stop();

          // Extract the parameters from the augmented state
          for (unsigned int member = 0; member < local_ensemble_size; ++member)
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

          if (global_rank == 0)
            std::cout << "Done." << std::endl;

          // Print out the augmented parameters
          if (solution_augmented_ensemble[0].block(1).size() > 0)
          {
            for (unsigned int member = 0; member < local_ensemble_size;
                 ++member)
            {
              std::cout << "Rank: " << global_rank
                        << " | New parameters for member "
                        << first_local_member + member << ": ";
              for (auto param : solution_augmented_ensemble[member].block(1))
                std::cout << param << " ";

              std::cout << std::endl;
            }
          }
        }
        else
        {
          if (global_rank == 0)
            std::cout
                << "WARNING: NO EXPERIMENTAL DATA POINTS MAPPED ONTO THE "
                   "SIMULATION MESH. SKIPPING DATA ASSIMILATION OPERATION."
                << std::endl;
        }
      }

      // Update the heat source in the ThermalPhysics objects
      for (unsigned int member = 0; member < local_ensemble_size; ++member)
      {
        thermal_physics_ensemble[member]->update_physics_parameters(
            database_ensemble[member].get_child("sources"));
      }
    }

    // ----- Checkpoint the ensemble members -----
    if (n_time_step % time_steps_checkpoint == 0)
    {
#ifdef ADAMANTINE_WITH_CALIPER
      CALI_MARK_BEGIN("save checkpoint");
#endif
      if (global_rank == 0)
      {
        std::cout << "Checkpoint reached" << std::endl;
      }

      std::string output_dir =
          post_processor_database.get<std::string>("output_dir", "");
      std::string filename_prefix =
          checkpoint_overwrite
              ? checkpoint_filename
              : checkpoint_filename + '_' + std::to_string(n_time_step);
      for (unsigned int member = 0; member < local_ensemble_size; ++member)
      {
        thermal_physics_ensemble[member]->save_checkpoint(
            filename_prefix + '_' + std::to_string(first_local_member + member),
            solution_augmented_ensemble[member].block(base_state));
      }
      std::ofstream file{output_dir + filename_prefix + "_time.txt"};
      boost::archive::text_oarchive oa{file};
      oa << time;
      oa << n_time_step;
#ifdef ADAMANTINE_WITH_CALIPER
      CALI_MARK_END("save checkpoint");
#endif
    }

    // ----- Output progress on screen -----
    if (global_rank == 0)
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
    if ((n_time_step % time_steps_output == 0) ||
        (output_on_da && time > (da_time - da_time_half_window) &&
         time < (da_time + da_time_half_window)))
    {
      for (unsigned int member = 0; member < local_ensemble_size; ++member)
      {
        thermal_physics_ensemble[member]->set_state_to_material_properties();
        output_pvtu(*post_processor_ensemble[member], n_time_step, time,
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

  for (unsigned int member = 0; member < local_ensemble_size; ++member)
  {
    post_processor_ensemble[member]->write_pvd();
  }

  // This is only used for integration test
  if constexpr (std::is_same_v<MemorySpaceType, dealii::MemorySpace::Host>)
  {
    for (unsigned int member = 0; member < local_ensemble_size; ++member)
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
        solution_augmented_ensemble_host(local_ensemble_size);

    for (unsigned int member = 0; member < local_ensemble_size; ++member)
    {
      solution_augmented_ensemble[member].reinit(2);

      solution_augmented_ensemble_host[member].block(0).reinit(
          solution_augmented_ensemble[member].block(0).get_partitioner());

      solution_augmented_ensemble_host[member].block(0).import_elements(
          solution_augmented_ensemble[member].block(0),
          dealii::VectorOperation::insert);
      thermal_physics_ensemble[member]->get_affine_constraints().distribute(
          solution_augmented_ensemble_host[member].block(0));

      solution_augmented_ensemble_host[member].block(1).reinit(
          solution_augmented_ensemble[member].block(0).get_partitioner());

      solution_augmented_ensemble_host[member].block(1).import_elements(
          solution_augmented_ensemble[member].block(1),
          dealii::VectorOperation::insert);
    }

    return solution_augmented_ensemble_host;
  }
}
#endif
