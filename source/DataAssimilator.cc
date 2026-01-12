/* SPDX-FileCopyrightText: Copyright (c) 2021 - 2025, the adamantine authors.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include <DataAssimilator.hh>
#include <utils.hh>

#include <deal.II/arborx/distributed_tree.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/mpi.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/linear_operator_tools.h>
#include <deal.II/lac/read_write_vector.h>
#if DEAL_II_VERSION_GTE(9, 7, 0) && defined(DEAL_II_TRILINOS_WITH_TPETRA)
#include <deal.II/lac/trilinos_tpetra_sparse_matrix.h>
#else
#include <deal.II/lac/trilinos_sparse_matrix.h>
#endif
#include <deal.II/lac/vector_operation.h>

#include <boost/algorithm/string/predicate.hpp>

#include <ArborX.hpp>

#ifdef ADAMANTINE_WITH_CALIPER
#include <caliper/cali.h>
#endif

namespace adamantine
{

DataAssimilator::DataAssimilator(MPI_Comm const &global_communicator,
                                 MPI_Comm const &local_communicator, int color,
                                 boost::property_tree::ptree const &database)
    : _global_communicator(global_communicator),
      _local_communicator(local_communicator), _color(color)
{
  _global_rank = dealii::Utilities::MPI::this_mpi_process(_global_communicator);
  _global_comm_size =
      dealii::Utilities::MPI::n_mpi_processes(_global_communicator);

  // We need all the processors to know the cutoff distance for ArborX
  // DistributedTree to work correctly.
  // PropertyTreeInput data_assimilation.localization_cutoff_distance
  _localization_cutoff_distance = database.get(
      "localization_cutoff_distance", std::numeric_limits<double>::max());

  if (_global_rank == 0)
  {
    // Set the solver parameters from the input database
    // PropertyTreeInput data_assimilation.solver.max_number_of_temp_vectors
    if (boost::optional<unsigned int> max_num_temp_vectors =
            database.get_optional<unsigned int>(
                "solver.max_number_of_temp_vectors"))
      _additional_data.max_basis_size = *max_num_temp_vectors;

    // PropertyTreeInput data_assimilation.solver.max_iterations
    if (boost::optional<unsigned int> max_iterations =
            database.get_optional<unsigned int>("solver.max_iterations"))
      _solver_control.set_max_steps(*max_iterations);

    // PropertyTreeInput data_assimilation.solver.convergence_tolerance
    if (boost::optional<double> tolerance =
            database.get_optional<double>("solver.convergence_tolerance"))
      _solver_control.set_tolerance(*tolerance);

    // PropertyTreeInput data_assimilation.localization_cutoff_function
    std::string localization_cutoff_function_str =
        database.get("localization_cutoff_function", "none");

    if (boost::iequals(localization_cutoff_function_str, "gaspari_cohn"))
    {
      _localization_cutoff_function = LocalizationCutoff::gaspari_cohn;
    }
    else if (boost::iequals(localization_cutoff_function_str, "step_function"))
    {
      _localization_cutoff_function = LocalizationCutoff::step_function;
    }
    else if (boost::iequals(localization_cutoff_function_str, "none"))
    {
      _localization_cutoff_function = LocalizationCutoff::none;
    }
    else
    {
      ASSERT_THROW(false, "Unknown localization cutoff function. Valid options "
                          "are 'gaspari_cohn', 'step_function', and 'none'.");
    }
  }
}

void DataAssimilator::update_ensemble(
    std::vector<dealii::LA::distributed::BlockVector<double>>
        &augmented_state_ensemble,
    std::vector<double> const &expt_data, dealii::SparseMatrix<double> const &R)
{

  std::vector<dealii::LA::distributed::BlockVector<double>>
      global_augmented_state_ensemble;
  std::vector<unsigned int> local_n_ensemble_members(_global_comm_size);
  std::vector<block_size_type> block_sizes(2, 0);

  gather_ensemble_members(augmented_state_ensemble,
                          global_augmented_state_ensemble,
                          local_n_ensemble_members, block_sizes);

  if (_global_rank == 0)
  {
    // Set some constants
    _num_ensemble_members = global_augmented_state_ensemble.size();
    _sim_size = block_sizes[0];
    _parameter_size = block_sizes[1];

    adamantine::ASSERT_THROW(_expt_size == expt_data.size(),
                             "Unexpected experiment vector size.");

    // Check if R is diagonal, needed for filling the noise vector
    auto bandwidth = R.get_sparsity_pattern().bandwidth();
    bool const R_is_diagonal = bandwidth == 0 ? true : false;

    // Get the perturbed innovation, ( y+u - Hx )
    // This is determined using the unaugmented state because the parameters
    // are not observable
    std::cout << "Getting the perturbed innovation..." << std::endl;

#ifdef ADAMANTINE_WITH_CALIPER
    CALI_MARK_BEGIN("da_get_pert_inno");
#endif

    int constexpr base_state = 0;
    std::vector<dealii::Vector<double>> perturbed_innovation(
        _num_ensemble_members);
    for (unsigned int member = 0; member < _num_ensemble_members; ++member)
    {
      perturbed_innovation[member].reinit(_expt_size);
      fill_noise_vector(perturbed_innovation[member], R, R_is_diagonal);
      dealii::Vector<double> temporary =
          calc_Hx(global_augmented_state_ensemble[member].block(base_state));

      for (unsigned int i = 0; i < _expt_size; ++i)
      {
        perturbed_innovation[member][i] += expt_data[i] - temporary[i];
      }
    }

#ifdef ADAMANTINE_WITH_CALIPER
    CALI_MARK_END("da_get_pert_inno");
#endif

    // Apply the Kalman gain to update the augmented state ensemble
    std::cout << "Applying the Kalman gain..." << std::endl;

#ifdef ADAMANTINE_WITH_CALIPER
    CALI_MARK_BEGIN("da_apply_K");
#endif

    // Apply the Kalman filter to the perturbed innovation, K ( y+u - Hx )
    std::vector<dealii::LA::distributed::BlockVector<double>> forecast_shift =
        apply_kalman_gain(global_augmented_state_ensemble, R,
                          perturbed_innovation);

#ifdef ADAMANTINE_WITH_CALIPER
    CALI_MARK_END("da_apply_K");
#endif

    // Update the ensemble, x = x + K ( y+u - Hx )
    std::cout << "Updating the ensemble members..." << std::endl;

#ifdef ADAMANTINE_WITH_CALIPER
    CALI_MARK_BEGIN("da_update_members");
#endif

    for (unsigned int member = 0; member < _num_ensemble_members; ++member)
    {
      global_augmented_state_ensemble[member] += forecast_shift[member];
    }

#ifdef ADAMANTINE_WITH_CALIPER
    CALI_MARK_END("da_update_members");
#endif
  }

  scatter_ensemble_members(augmented_state_ensemble,
                           global_augmented_state_ensemble,
                           local_n_ensemble_members, block_sizes);
}

void DataAssimilator::gather_ensemble_members(
    std::vector<dealii::LA::distributed::BlockVector<double>>
        &augmented_state_ensemble,
    std::vector<dealii::LA::distributed::BlockVector<double>>
        &global_augmented_state_ensemble,
    std::vector<unsigned int> &local_n_ensemble_members,
    std::vector<block_size_type> &block_sizes)
{
  // We need to gather the augmented_state_ensemble from the other processors.
  // BlockVector is a complex structure with its own communicator and so we
  // cannot simple use dealii's gather to perform the communication. Instead, we
  // extract the locally owned data and gather it to processor zero. We do this
  // in a two step process. First, we move the data to local rank zero. Second,
  // we move the data to global rank zero. The first step allows to simply move
  // complete vectors to global rank zero. Otherwise, we have the vector data
  // divided in multiple chunks when gathered on global rank zero and we have to
  // reconstruct the vector. Finally we can build new BlockVector using the
  // local communicator.

  // Extract relevant data
  unsigned int const n_local_ensemble_members = augmented_state_ensemble.size();
  std::vector<std::vector<std::vector<double>>> block_data(
      n_local_ensemble_members, std::vector<std::vector<double>>(2));
  for (unsigned int i = 0; i < n_local_ensemble_members; ++i)
  {
    for (unsigned int j = 0; j < 2; ++j)
    {
      auto data_ptr = augmented_state_ensemble[i].block(j).get_values();
      block_data[i][j].insert(
          block_data[i][j].end(), data_ptr,
          data_ptr + augmented_state_ensemble[i].block(j).locally_owned_size());
    }
  }

  // Perform the communications on the local communicator
  auto local_block_data =
      dealii::Utilities::MPI::gather(_local_communicator, block_data);
  auto local_indexsets_block_0 = dealii::Utilities::MPI::gather(
      _local_communicator,
      augmented_state_ensemble[0].block(0).locally_owned_elements());
  auto local_indexsets_block_1 = dealii::Utilities::MPI::gather(
      _local_communicator,
      augmented_state_ensemble[0].block(1).locally_owned_elements());
  // The local processor zero has all the data. Reorder the data before
  // sending it to the global processor zero
  std::vector<std::vector<std::vector<double>>> reordered_local_block_data;
  if (dealii::Utilities::MPI::this_mpi_process(_local_communicator) == 0)
  {
    reordered_local_block_data.resize(n_local_ensemble_members,
                                      std::vector<std::vector<double>>(2));

    for (unsigned int i = 0; i < local_indexsets_block_0.size(); ++i)
    {
      block_sizes[0] += local_indexsets_block_0[i].n_elements();
    }
    // The augmented parameters are not distributed.
    block_sizes[1] = local_indexsets_block_1[0].n_elements();

    // Loop over the ensemble members
    for (unsigned int i = 0; i < n_local_ensemble_members; ++i)
    {
      // Loop over the processors
      for (unsigned int j = 0; j < local_block_data.size(); ++j)
      {
        // Loop over the blocks
        for (unsigned int k = 0; k < 2; ++k)
        {
          reordered_local_block_data[i][k].resize(block_sizes[k]);
          // Loop over the dofs
          for (std::size_t m = 0; m < local_block_data[j][i][k].size(); ++m)
          {
            auto pos = k == 0 ? local_indexsets_block_0[j].nth_index_in_set(m)
                              : local_indexsets_block_1[j].nth_index_in_set(m);
            reordered_local_block_data[i][k][pos] =
                local_block_data[j][i][k][m];
          }
        }
      }
    }
  }

  // Perform the global communication.
  auto all_local_block_data = dealii::Utilities::MPI::gather(
      _global_communicator, reordered_local_block_data);

  if (_global_rank == 0)
  {
    // Build the new BlockVector
    for (unsigned int i = 0; i < all_local_block_data.size(); ++i)
    {
      auto const &data = all_local_block_data[i];
      local_n_ensemble_members[i] = data.size();
      if (data.size())
      {
        // Loop over the ensemble members
        for (unsigned int j = 0; j < data.size(); ++j)
        {
          dealii::LA::distributed::BlockVector<double> ensemble_member(
              block_sizes);
          // Copy the values in data to ensemble_member
          for (unsigned int k = 0; k < data[j].size(); ++k)
          {
            for (unsigned int m = 0; m < data[j][k].size(); ++m)
            {
              ensemble_member.block(k)[m] = data[j][k][m];
            }
          }
          global_augmented_state_ensemble.push_back(ensemble_member);
        }
      }
    }
  }
}

void DataAssimilator::scatter_ensemble_members(
    std::vector<dealii::LA::distributed::BlockVector<double>>
        &augmented_state_ensemble,
    std::vector<dealii::LA::distributed::BlockVector<double>> const
        &global_augmented_state_ensemble,
    std::vector<unsigned int> const &local_n_ensemble_members,
    std::vector<block_size_type> const &block_sizes)
{
  // Scatter global_augmented_state_ensemble to augmented_state_ensemble.
  // First we split the data to the root of the local communicators and then,
  // the data is moved inside each local communicator.
  // deal.II has isend and irecv functions but we cannot them. Using these
  // functions will result in a deadlock because the future returned by isend
  // is blocking until we call the get function of the future returned by
  // irecv.

  std::vector<char> packed_recv_buffer;
  if (_global_rank == 0)
  {
    std::vector<std::vector<char>> packed_send_buffers(_global_comm_size);
    std::vector<MPI_Request> mpi_requests(_global_comm_size);
    unsigned int global_member_id = 0;
    for (int i = 0; i < _global_comm_size; ++i)
    {
      unsigned int const local_size = local_n_ensemble_members[i];
      std::vector<std::vector<double>> send_buffer(local_size);
      for (unsigned int j = 0; j < local_size; ++j)
      {
        auto const &global_state =
            global_augmented_state_ensemble[global_member_id];
        send_buffer[j].reserve(global_state.size());
        for (auto block_vector_it = global_state.begin();
             block_vector_it != global_state.end(); ++block_vector_it)
        {
          send_buffer[j].push_back(*block_vector_it);
        }
        ++global_member_id;
      }
      // Pack and send the data to the local root rank
      packed_send_buffers[i] = dealii::Utilities::pack(send_buffer);
      MPI_Isend(packed_send_buffers[i].data(), packed_send_buffers[i].size(),
                MPI_CHAR, i, 0, _global_communicator, &mpi_requests[i]);
    }

    // Receive the data. First, call MPI_Probe to get the size of the message.
    MPI_Status status;
    MPI_Probe(0, 0, _global_communicator, &status);
    int packed_recv_buffer_size = -1;
    MPI_Get_count(&status, MPI_CHAR, &packed_recv_buffer_size);
    packed_recv_buffer.resize(packed_recv_buffer_size);
    MPI_Recv(packed_recv_buffer.data(), packed_recv_buffer_size, MPI_CHAR, 0, 0,
             _global_communicator, MPI_STATUS_IGNORE);

    // Wait for all the sends to be over before freeing the buffers
    for (auto &request : mpi_requests)
    {
      MPI_Wait(&request, MPI_STATUS_IGNORE);
    }
  }
  else
  {
    MPI_Status status;
    MPI_Probe(0, 0, _global_communicator, &status);
    int packed_recv_buffer_size = -1;
    MPI_Get_count(&status, MPI_CHAR, &packed_recv_buffer_size);
    packed_recv_buffer.resize(packed_recv_buffer_size);
    MPI_Recv(packed_recv_buffer.data(), packed_recv_buffer_size, MPI_CHAR, 0, 0,
             _global_communicator, MPI_STATUS_IGNORE);
  }

  // Unpack the data
  auto recv_buffer =
      dealii::Utilities::unpack<std::vector<std::vector<double>>>(
          packed_recv_buffer);

  // The local root ranks have all the data, now we need to update
  // augmented_state_ensemble. This communication is easier to do than the
  // other communications because we can use deal.II's built-in functions.
  for (unsigned int m = 0; m < augmented_state_ensemble.size(); ++m)
  {
    for (unsigned int b = 0; b < 2; ++b)
    {
      dealii::LA::ReadWriteVector<double> rw_vector(block_sizes[b]);
      unsigned int offset = b == 0 ? 0 : block_sizes[0];
      for (std::size_t i = 0; i < block_sizes[b]; ++i)
      {
        rw_vector[i] = recv_buffer[m][offset + i];
      }
      // We cannot insert the elements because deal.II checks that the local
      // elements and the remote ones match. Instead, we set everything to zero
      // and then, we add the imported elements.
      augmented_state_ensemble[m].block(b) = 0.;
      augmented_state_ensemble[m].block(b).import_elements(
          rw_vector, dealii::VectorOperation::add);
    }
  }
}

std::vector<dealii::LA::distributed::BlockVector<double>>
DataAssimilator::apply_kalman_gain(
    std::vector<dealii::LA::distributed::BlockVector<double>>
        &augmented_state_ensemble,
    dealii::SparseMatrix<double> const &R,
    std::vector<dealii::Vector<double>> const &perturbed_innovation)
{
  unsigned int augmented_state_size = _sim_size + _parameter_size;

  /*
   * Currently this function uses GMRES to apply the inverse of HPH^T+R in the
   * Kalman gain calculation for each ensemble member individually. Depending
   * on the size of the datasets, the number of ensembles, and other factors
   * doing a direct solve of (HPH^T+R)^-1 once and then applying to the
   * perturbed innovation from each ensemble member might be more efficient.
   */
  dealii::SparsityPattern pattern_H(_expt_size, augmented_state_size,
                                    _expt_size);
  auto H = calc_H(pattern_H);
  auto P = calc_sample_covariance_sparse(augmented_state_ensemble);

  ASSERT(H.n() == P.m(), "Matrices dimensions not compatible");
  ASSERT(H.m() == R.m(), "Matrices dimensions not compatible");

  const auto op_H = dealii::linear_operator(H);
  const auto op_P = dealii::linear_operator(P);
  const auto op_R = dealii::linear_operator(R);

  const auto op_HPH_plus_R =
      op_H * op_P * dealii::transpose_operator(op_H) + op_R;

  const std::vector<dealii::types::global_dof_index> block_sizes = {
      _sim_size, _parameter_size};
  std::vector<dealii::LA::distributed::BlockVector<double>> output(
      _num_ensemble_members,
      dealii::LA::distributed::BlockVector<double>(block_sizes));

  // Create non-member versions of these for use in the lambda function
  auto solver_control = _solver_control;
  auto additional_data = _additional_data;

  // Apply the Kalman gain to the perturbed innovation for the ensemble
  // members in parallel
  std::transform(
      perturbed_innovation.begin(), perturbed_innovation.end(), output.begin(),
      [&](dealii::Vector<double> const &entry)
      {
        dealii::SolverGMRES<dealii::Vector<double>> HPH_plus_R_inv_solver(
            solver_control, additional_data);

        auto op_HPH_plus_R_inv =
            dealii::inverse_operator(op_HPH_plus_R, HPH_plus_R_inv_solver);

        const auto op_K =
            op_P * dealii::transpose_operator(op_H) * op_HPH_plus_R_inv;

        // Apply the Kalman gain to each innovation vector
        dealii::Vector<double> temporary = op_K * entry;

        // Copy into a distributed block vector, this is the only place where
        // the mismatch matters, using dealii::Vector for the experimental
        // data and dealii::LA::distributed::BlockVector for the simulation
        // data.
        dealii::LA::distributed::BlockVector<double> output_member(block_sizes);
        for (unsigned int i = 0; i < augmented_state_size; ++i)
        {
          output_member(i) = temporary(i);
        }

        return output_member;
      });

  return output;
}

dealii::SparseMatrix<double>
DataAssimilator::calc_H(dealii::SparsityPattern &pattern) const
{
  unsigned int num_expt_dof_map_entries = _expt_to_dof_mapping.first.size();

  for (unsigned int i = 0; i < num_expt_dof_map_entries; ++i)
  {
    auto sim_index = _expt_to_dof_mapping.second[i];
    auto expt_index = _expt_to_dof_mapping.first[i];
    pattern.add(expt_index, sim_index);
  }

  pattern.compress();

  dealii::SparseMatrix<double> H(pattern);

  for (unsigned int i = 0; i < num_expt_dof_map_entries; ++i)
  {
    auto sim_index = _expt_to_dof_mapping.second[i];
    auto expt_index = _expt_to_dof_mapping.first[i];
    H.add(expt_index, sim_index, 1.0);
  }

  return H;
}

template <int dim>
void DataAssimilator::update_dof_mapping(
    std::pair<std::vector<int>, std::vector<int>> const &expt_to_dof_mapping)
{
  _expt_size = expt_to_dof_mapping.first.size();
  _expt_to_dof_mapping = expt_to_dof_mapping;
}

template <int dim>
void DataAssimilator::update_covariance_sparsity_pattern(
    dealii::DoFHandler<dim> const &dof_handler,
    const unsigned int parameter_size)
{
  if (_color == 0)
  {
    _sim_size = dof_handler.n_dofs();
    _parameter_size = parameter_size;
    unsigned int augmented_state_size = _sim_size + _parameter_size;

    auto [dof_indices, support_points] =
        get_dof_to_support_mapping(dof_handler);

    // Perform the spatial search using ArborX
    dealii::ArborXWrappers::DistributedTree distributed_tree(
        _local_communicator, support_points);

    std::vector<std::pair<dealii::Point<dim, double>, double>> spheres;
    if (dim == 2)
      for (auto const pt : support_points)
        spheres.push_back({{pt[0], pt[1]}, _localization_cutoff_distance});
    else
      for (auto const pt : support_points)
        spheres.push_back(
            {{pt[0], pt[1], pt[2]}, _localization_cutoff_distance});
    dealii::ArborXWrappers::SphereIntersectPredicate sph_intersect(spheres);
    auto [indices_ranks, offsets] = distributed_tree.query(sph_intersect);
    ASSERT(offsets.size() == spheres.size() + 1,
           "There was a problem in ArborX.");

    auto locally_owned_dofs_per_rank = dealii::Utilities::MPI::gather(
        _local_communicator, dof_handler.locally_owned_dofs());
    auto support_points_per_rank =
        dealii::Utilities::MPI::gather(_local_communicator, support_points);
    auto indices_ranks_per_rank =
        dealii::Utilities::MPI::gather(_local_communicator, indices_ranks);
    auto offsets_per_rank =
        dealii::Utilities::MPI::gather(_local_communicator, offsets);

    if (_global_rank == 0)
    {
      // We need IndexSet to build the sparsity pattern. Since the data
      // assimilation is done is serial, the IndexSet just contains
      // everything.
      dealii::IndexSet parallel_partitioning =
          dealii::complete_index_set(augmented_state_size);
      parallel_partitioning.compress();

      dealii::DynamicSparsityPattern dsp(parallel_partitioning);
      // Fill in the SparsityPattern
      unsigned int const local_comm_size =
          dealii::Utilities::MPI::n_mpi_processes(_local_communicator);
      for (unsigned int rank = 0; rank < local_comm_size; ++rank)
      {
        if (offsets_per_rank[rank].size() != 0)
        {
          for (unsigned int i = 0; i < offsets_per_rank[rank].size() - 1; ++i)
          {
            for (int j = offsets_per_rank[rank][i];
                 j < offsets_per_rank[rank][i + 1]; ++j)
            {
              unsigned int row =
                  locally_owned_dofs_per_rank[rank].nth_index_in_set(i);
              unsigned int other_rank = indices_ranks_per_rank[rank][j].second;
              unsigned int other_i = indices_ranks_per_rank[rank][j].first;
              unsigned int column =
                  locally_owned_dofs_per_rank[other_rank].nth_index_in_set(
                      other_i);
              _covariance_distance_map[std::make_pair(row, column)] =
                  support_points_per_rank[rank][i].distance(
                      support_points_per_rank[other_rank][other_i]);
              dsp.add(row, column);
            }
          }
        }
      }

      // Add entries for the parameter augmentation
      for (unsigned int i1 = _sim_size; i1 < augmented_state_size; ++i1)
      {
        for (unsigned int j1 = 0; j1 < augmented_state_size; ++j1)
        {
          dsp.add(i1, j1);
        }
      }

      for (unsigned int i1 = 0; i1 < _sim_size; ++i1)
      {
        for (unsigned int j1 = _sim_size; j1 < augmented_state_size; ++j1)
        {
          dsp.add(i1, j1);
        }
      }

      _covariance_sparsity_pattern.reinit(parallel_partitioning, dsp,
                                          MPI_COMM_SELF);
      _covariance_sparsity_pattern.compress();
    }
  }
}

dealii::Vector<double> DataAssimilator::calc_Hx(
    dealii::LA::distributed::Vector<double> const &sim_ensemble_member) const
{
  dealii::Vector<double> out_vec(_expt_size);

  // Loop through the observation map to get the observation indices
  for (unsigned int i = 0; i < _expt_size; ++i)
  {
    auto sim_index = _expt_to_dof_mapping.second[i];
    auto expt_index = _expt_to_dof_mapping.first[i];
    out_vec(expt_index) = sim_ensemble_member(sim_index);
  }

  return out_vec;
}

void DataAssimilator::fill_noise_vector(dealii::Vector<double> &vec,
                                        dealii::SparseMatrix<double> const &R,
                                        bool const R_is_diagonal)
{
  auto vector_size = vec.size();

  // If R is diagonal, then the entries in the noise vector are independent
  // and each are simply a scaled output of the pseudo-random number
  // generator. For a more general R, one needs to multiply by the Cholesky
  // decomposition of R to achieve the appropriate correlation between the
  // entries. Deal.II only has a specific Cholesky function for full matrices,
  // which is used in the implementation below. The Cholesky decomposition is
  // a special case of LU decomposition, so we can use a sparse LU solver to
  // obtain the "L" below if needed in the future.

  if (R_is_diagonal)
  {
    for (unsigned int i = 0; i < vector_size; ++i)
    {
      vec(i) = _normal_dist_generator(_prng) * std::sqrt(R(i, i));
    }
  }
  else
  {
    // Do Cholesky decomposition
    dealii::FullMatrix<double> L(vector_size);
    dealii::FullMatrix<double> R_full(vector_size);
    R_full.copy_from(R);
    L.cholesky(R_full);

    // Get a vector of normally distributed values
    dealii::Vector<double> uncorrelated_noise_vector(vector_size);

    for (unsigned int i = 0; i < vector_size; ++i)
    {
      uncorrelated_noise_vector(i) = _normal_dist_generator(_prng);
    }

    L.vmult(vec, uncorrelated_noise_vector);
  }
}

double DataAssimilator::gaspari_cohn_function(double const r) const
{
  if (r < 1.0)
  {
    return 1. - 5. / 3. * std::pow(r, 2) + 5. / 8. * std::pow(r, 3) +
           0.5 * std::pow(r, 4) - 0.25 * std::pow(r, 5);
  }
  else if (r < 2)
  {
    return 4. - 5. * r + 5. / 3. * std::pow(r, 2) + 5. / 8. * std::pow(r, 3) -
           0.5 * std::pow(r, 4) + 1. / 12. * std::pow(r, 5) - 2. / (3. * r);
  }
  else
  {
    return 0.;
  }
}

typename DataAssimilator::TrilinosMatrixType
DataAssimilator::calc_sample_covariance_sparse(
    std::vector<dealii::LA::distributed::BlockVector<double>> const
        &vec_ensemble) const
{
  unsigned int const local_size = vec_ensemble[0].locally_owned_size();

  // Calculate the mean
  dealii::Vector<double> mean(local_size);
  auto sum = vec_ensemble[0];
  for (unsigned int sample = 1; sample < _num_ensemble_members; ++sample)
  {
    sum += vec_ensemble[sample];
  }
  unsigned int ii = 0;
  for (auto index : sum.locally_owned_elements())
  {
    mean[ii] = sum[index] / _num_ensemble_members;
    ++ii;
  }

  TrilinosMatrixType cov(_covariance_sparsity_pattern);

  unsigned int pos = 0;
  for (auto conv_iter = cov.begin(); conv_iter != cov.end(); ++conv_iter, ++pos)
  {
    unsigned int i = conv_iter->row();
    unsigned int j = conv_iter->column();

    // Do the element-wise matrix multiply by hand
    double element_value = 0;
    for (unsigned int k = 0; k < _num_ensemble_members; ++k)
    {
      element_value +=
          (vec_ensemble[k][i] - mean[i]) * (vec_ensemble[k][j] - mean[j]);
    }

    element_value /= (_num_ensemble_members - 1.0);

    // Apply localization
    double localization_scaling = 1.0;
    if (i < _sim_size && j < _sim_size)
    {
      double dist = _covariance_distance_map.find(std::make_pair(i, j))->second;
      if (_localization_cutoff_function == LocalizationCutoff::gaspari_cohn)
      {
        localization_scaling =
            gaspari_cohn_function(2.0 * dist / _localization_cutoff_distance);
      }
      else if ((_localization_cutoff_function ==
                LocalizationCutoff::step_function) &&
               (dist > _localization_cutoff_distance))
      {
        localization_scaling = 0.0;
      }
    }

    conv_iter->value() = element_value * localization_scaling;
  }

  cov.compress(dealii::VectorOperation::insert);
  return cov;
}

// Explicit instantiation
template void DataAssimilator::update_dof_mapping<2>(
    std::pair<std::vector<int>, std::vector<int>> const &expt_to_dof_mapping);
template void DataAssimilator::update_dof_mapping<3>(
    std::pair<std::vector<int>, std::vector<int>> const &expt_to_dof_mapping);
template void DataAssimilator::update_covariance_sparsity_pattern<2>(
    dealii::DoFHandler<2> const &dof_handler,
    const unsigned int parameter_size);
template void DataAssimilator::update_covariance_sparsity_pattern<3>(
    dealii::DoFHandler<3> const &dof_handler,
    const unsigned int parameter_size);

} // namespace adamantine
