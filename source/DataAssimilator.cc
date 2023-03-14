/* Copyright (c) 2021-2022, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#include <DataAssimilator.hh>
#include <utils.hh>

#include <deal.II/arborx/distributed_tree.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/linear_operator_tools.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

#include <boost/algorithm/string/predicate.hpp>

#include <ArborX.hpp>

#ifdef ADAMANTINE_WITH_CALIPER
#include <caliper/cali.h>
#endif

namespace adamantine
{

DataAssimilator::DataAssimilator(boost::property_tree::ptree const &database)
{
  // Set the solver parameters from the input database
  // PropertyTreeInput data_assimilation.solver.max_number_of_temp_vectors
  if (boost::optional<unsigned int> max_num_temp_vectors =
          database.get_optional<unsigned int>(
              "solver.max_number_of_temp_vectors"))
    _additional_data.max_n_tmp_vectors = *max_num_temp_vectors;

  // PropertyTreeInput data_assimilation.solver.max_iterations
  if (boost::optional<unsigned int> max_iterations =
          database.get_optional<unsigned int>("solver.max_iterations"))
    _solver_control.set_max_steps(*max_iterations);

  // PropertyTreeInput data_assimilation.solver.convergence_tolerance
  if (boost::optional<double> tolerance =
          database.get_optional<double>("solver.convergence_tolerance"))
    _solver_control.set_tolerance(*tolerance);

  // PropertyTreeInput data_assimilation.localization_cutoff_distance
  _localization_cutoff_distance = database.get(
      "localization_cutoff_distance", std::numeric_limits<double>::max());

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
    ASSERT_THROW(false,
                 "Error: Unknown localization cutoff function. Valid options "
                 "are 'gaspari_cohn', 'step_function', and 'none'.");
  }
}

void DataAssimilator::update_ensemble(
    MPI_Comm const &communicator,
    std::vector<dealii::LA::distributed::BlockVector<double>>
        &augmented_state_ensemble,
    std::vector<double> const &expt_data, dealii::SparseMatrix<double> const &R)
{
  unsigned int rank = dealii::Utilities::MPI::this_mpi_process(communicator);

  // Give names to the blocks in the augmented state vector
  int constexpr base_state = 0;
  int constexpr augmented_state = 1;

  // Set some constants
  _num_ensemble_members = augmented_state_ensemble.size();
  if (_num_ensemble_members > 0)
  {
    _sim_size = augmented_state_ensemble[0].block(base_state).size();
    _parameter_size = augmented_state_ensemble[0].block(augmented_state).size();
  }
  else
  {
    _sim_size = 0;
    _parameter_size = 0;
  }

  adamantine::ASSERT_THROW(_expt_size == expt_data.size(),
                           "Error: Unexpected experiment vector size.");

  // Get the perturbed innovation, ( y+u - Hx )
  // This is determined using the unaugmented state because the parameters are
  // not observable
  if (rank == 0)
    std::cout << "Getting the perturbed innovation..." << std::endl;

#ifdef ADAMANTINE_WITH_CALIPER
  CALI_MARK_BEGIN("da_get_pert_inno");
#endif

  std::vector<dealii::Vector<double>> perturbed_innovation(
      _num_ensemble_members);
  for (unsigned int member = 0; member < _num_ensemble_members; ++member)
  {
    perturbed_innovation[member] = get_noise_vector(R);
    dealii::Vector<double> temporary =
        calc_Hx(augmented_state_ensemble[member].block(base_state));

    for (unsigned int i = 0; i < _expt_size; ++i)
    {
      perturbed_innovation[member][i] += expt_data[i] - temporary[i];
    }
  }

#ifdef ADAMANTINE_WITH_CALIPER
  CALI_MARK_END("da_get_pert_inno");
#endif

  // Apply the Kalman gain to update the augmented state ensemble
  if (rank == 0)
    std::cout << "Applying the Kalman gain..." << std::endl;

#ifdef ADAMANTINE_WITH_CALIPER
  CALI_MARK_BEGIN("da_apply_K");
#endif

  // Apply the Kalman filter to the perturbed innovation, K ( y+u - Hx )
  std::vector<dealii::LA::distributed::BlockVector<double>> forecast_shift =
      apply_kalman_gain(augmented_state_ensemble, R, perturbed_innovation);

#ifdef ADAMANTINE_WITH_CALIPER
  CALI_MARK_END("da_apply_K");
#endif

  // Update the ensemble, x = x + K ( y+u - Hx )
  if (rank == 0)
    std::cout << "Updating the ensemble members..." << std::endl;

#ifdef ADAMANTINE_WITH_CALIPER
  CALI_MARK_BEGIN("da_update_members");
#endif

  for (unsigned int member = 0; member < _num_ensemble_members; ++member)
  {
    augmented_state_ensemble[member] += forecast_shift[member];
  }

#ifdef ADAMANTINE_WITH_CALIPER
  CALI_MARK_END("da_update_members");
#endif
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
      [&](dealii::Vector<double> entry)
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
        // the mismatch matters, using dealii::Vector for the experimental data
        // and dealii::LA::distributed::BlockVector for the simulation data.
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
  int num_expt_dof_map_entries = _expt_to_dof_mapping.first.size();

  for (auto i = 0; i < num_expt_dof_map_entries; ++i)
  {
    auto sim_index = _expt_to_dof_mapping.second[i];
    auto expt_index = _expt_to_dof_mapping.first[i];
    pattern.add(expt_index, sim_index);
  }

  pattern.compress();

  dealii::SparseMatrix<double> H(pattern);

  for (auto i = 0; i < num_expt_dof_map_entries; ++i)
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
  _sim_size = dof_handler.n_dofs();
  _parameter_size = parameter_size;
  unsigned int augmented_state_size = _sim_size + _parameter_size;

  auto [dof_indices, support_points] = get_dof_to_support_mapping(dof_handler);

  // Perform the spatial search using ArborX
  auto communicator = dof_handler.get_communicator();
  dealii::ArborXWrappers::DistributedTree distributed_tree(communicator,
                                                           support_points);

  std::vector<std::pair<dealii::Point<dim, double>, double>> spheres;
  if (dim == 2)
    for (auto const pt : support_points)
      spheres.push_back({{pt[0], pt[1]}, _localization_cutoff_distance});
  else
    for (auto const pt : support_points)
      spheres.push_back({{pt[0], pt[1], pt[2]}, _localization_cutoff_distance});
  dealii::ArborXWrappers::SphereIntersectPredicate sph_intersect(spheres);
  auto [indices_ranks, offsets] = distributed_tree.query(sph_intersect);
  ASSERT(offsets.size() == spheres.size() + 1,
         "There was a problem in ArborX.");

  // We need IndexSet to build the sparsity pattern. The IndexSet is the same as
  // the one from the DoFHandler but augmented by parameter size;
  dealii::IndexSet parallel_partitioning(augmented_state_size);
  auto locally_owned_dofs = dof_handler.locally_owned_dofs();
  parallel_partitioning.add_indices(locally_owned_dofs);
  int const my_rank = dealii::Utilities::MPI::this_mpi_process(communicator);
  if (my_rank == 0)
    parallel_partitioning.add_range(_sim_size, augmented_state_size);
  parallel_partitioning.compress();
  _covariance_sparsity_pattern.reinit(parallel_partitioning, communicator);

  // Fill in the SparsityPattern
  if (offsets.size() != 0)
  {
    for (unsigned int i = 0; i < offsets.size() - 1; ++i)
    {
      for (int j = offsets[i]; j < offsets[i + 1]; ++j)
      {
        if (indices_ranks[j].second == my_rank)
        {
          int k = indices_ranks[j].first;
          _covariance_distance_map[std::make_pair(i, k)] =
              support_points[i].distance(support_points[k]);
          // We would like to use the add_entries functions but we cannot
          // because deal.II only accepts unsigned int but ArborX returns int.
          // Note that the entries are unique but not sorted.
          _covariance_sparsity_pattern.add(i, k);
        }
      }
    }
  }

  if (my_rank == 0)
  {
    // Add entries for the parameter augmentation
    for (unsigned int i1 = _sim_size; i1 < augmented_state_size; ++i1)
    {
      for (unsigned int j1 = 0; j1 < augmented_state_size; ++j1)
      {
        _covariance_sparsity_pattern.add(i1, j1);
      }
    }

    for (unsigned int i1 = 0; i1 < _sim_size; ++i1)
    {
      for (unsigned int j1 = _sim_size; j1 < augmented_state_size; ++j1)
      {
        _covariance_sparsity_pattern.add(i1, j1);
      }
    }
  }

  _covariance_sparsity_pattern.compress();
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

dealii::Vector<double>
DataAssimilator::get_noise_vector(dealii::SparseMatrix<double> const &R)
{
  // If R is diagonal, then the entries in the noise vector are independent
  // and each are simply a scaled output of the pseudo-random number
  // generator. For a more general R, one needs to multiply by the Cholesky
  // decomposition of R to achieve the appropriate correlation between the
  // entries.
  dealii::Vector<double> noise(_expt_size);
  for (unsigned int i = 0; i < _expt_size; ++i)
  {
    noise(i) = _normal_dist_generator(_prng) * std::sqrt(R(i, i));
  }

  return noise;
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

dealii::TrilinosWrappers::SparseMatrix
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

  dealii::TrilinosWrappers::SparseMatrix cov(_covariance_sparsity_pattern);

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
    double localization_scaling;
    if (i < _sim_size && j < _sim_size)
    {
      double dist = _covariance_distance_map.find(std::make_pair(i, j))->second;
      if (_localization_cutoff_function == LocalizationCutoff::gaspari_cohn)
      {
        localization_scaling =
            gaspari_cohn_function(2.0 * dist / _localization_cutoff_distance);
      }
      else if (_localization_cutoff_function ==
               LocalizationCutoff::step_function)
      {
        if (dist <= _localization_cutoff_distance)
          localization_scaling = 1.0;
        else
          localization_scaling = 0.0;
      }
      else
      {
        localization_scaling = 1.0;
      }
    }
    else
    {
      localization_scaling = 1.0;
    }

    conv_iter->value() = element_value * localization_scaling;
  }

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
