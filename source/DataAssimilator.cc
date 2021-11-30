/* Copyright (c) 2021, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#include <DataAssimilator.hh>
#include <utils.hh>

#include <deal.II/dofs/dof_tools.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/linear_operator_tools.h>

#include <boost/algorithm/string/predicate.hpp>

// libc++ does not support parallel std library
#ifdef __GLIBCXX__
#include <execution>
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
    std::vector<dealii::LA::distributed::Vector<double>> &sim_data,
    std::vector<double> const &expt_data, dealii::SparseMatrix<double> &R)
{
  unsigned int rank = dealii::Utilities::MPI::this_mpi_process(communicator);

  // Set some constants
  _num_ensemble_members = sim_data.size();
  if (sim_data.size() > 0)
  {
    _sim_size = sim_data[0].size();
  }
  else
  {
    _sim_size = 0;
  }
  adamantine::ASSERT_THROW(_expt_size == expt_data.size(),
                           "Error: Unexpected experiment vector size.");

  // Check if R is diagonal, needed for filling the noise vector
  auto bandwidth = R.get_sparsity_pattern().bandwidth();
  bool const R_is_diagonal = bandwidth == 0 ? true : false;

  // Get the perturbed innovation, ( y+u - Hx )
  if (rank == 0)
    std::cout << "Getting the perturbed innovation..." << std::endl;

  std::vector<dealii::Vector<double>> perturbed_innovation(
      _num_ensemble_members);
  for (unsigned int member = 0; member < _num_ensemble_members; ++member)
  {
    perturbed_innovation[member].reinit(_expt_size);
    fill_noise_vector(perturbed_innovation[member], R, R_is_diagonal);
    dealii::Vector<double> temporary = calc_Hx(sim_data[member]);
    for (unsigned int i = 0; i < _expt_size; ++i)
    {
      perturbed_innovation[member][i] += expt_data[i] - temporary[i];
    }
  }

  if (rank == 0)
    std::cout << "Applying the Kalman gain..." << std::endl;

  // Apply the Kalman filter to the perturbed innovation, K ( y+u - Hx )
  std::vector<dealii::LA::distributed::Vector<double>> forecast_shift =
      apply_kalman_gain(sim_data, R, perturbed_innovation);

  // Update the ensemble, x = x + K ( y+u - Hx )
  if (rank == 0)
    std::cout << "Updating the ensemble members..." << std::endl;

  for (unsigned int member = 0; member < _num_ensemble_members; ++member)
  {
    sim_data[member] += forecast_shift[member];
  }
}

std::vector<dealii::LA::distributed::Vector<double>>
DataAssimilator::apply_kalman_gain(
    std::vector<dealii::LA::distributed::Vector<double>> &vec_ensemble,
    dealii::SparseMatrix<double> &R,
    std::vector<dealii::Vector<double>> &perturbed_innovation)
{
  /*
   * Currently this function uses GMRES to apply the inverse of HPH^T+R in the
   * Kalman gain calculation for each ensemble member individually. Depending on
   * the size of the datasets, the number of ensembles, and other factors doing
   * a direct solve of (HPH^T+R)^-1 once and then applying to the perturbed
   * innovation from each ensemble member might be more efficient.
   */
  dealii::SparsityPattern pattern_H(_expt_size, _sim_size, _expt_size);

  dealii::SparseMatrix<double> H = calc_H(pattern_H);

  dealii::SparseMatrix<double> P(_covariance_sparsity_pattern);
  P = calc_sample_covariance_sparse(vec_ensemble);

  const auto op_H = dealii::linear_operator(H);
  const auto op_P = dealii::linear_operator(P);
  const auto op_R = dealii::linear_operator(R);

  const auto op_HPH_plus_R =
      op_H * op_P * dealii::transpose_operator(op_H) + op_R;

  std::vector<dealii::LA::distributed::Vector<double>> output(
      _num_ensemble_members,
      dealii::LA::distributed::Vector<double>(_sim_size));

  // Create non-member versions of these for use in the lambda function
  auto solver_control = _solver_control;
  auto additional_data = _additional_data;
  auto sim_size = _sim_size;

  // Apply the Kalman gain to the perturbed innovation for the ensemble members
  // in parallel
  std::transform(
#ifdef __GLIBCXX__
      std::execution::par,
#endif
      perturbed_innovation.begin(), perturbed_innovation.end(), output.begin(),
      [&](dealii::Vector<double> entry) {
        dealii::SolverGMRES<dealii::Vector<double>> HPH_plus_R_inv_solver(
            solver_control, additional_data);

        auto op_HPH_plus_R_inv =
            dealii::inverse_operator(op_HPH_plus_R, HPH_plus_R_inv_solver);

        const auto op_K =
            op_P * dealii::transpose_operator(op_H) * op_HPH_plus_R_inv;

        // Apply the Kalman gain to each innovation vector
        dealii::Vector<double> temporary = op_K * entry;

        // Copy into a distributed vector, this is the only place where the
        // mismatch matters, using dealii::Vector for the experimental data
        // and dealii::LA::distributed::Vector for the simulation data.
        dealii::LA::distributed::Vector<double> output_member(sim_size);
        for (unsigned int i = 0; i < sim_size; ++i)
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
    dealii::DoFHandler<dim> const &dof_handler,
    std::pair<std::vector<int>, std::vector<int>> const &indices_and_offsets)
{
  _expt_size = indices_and_offsets.first.size();

  std::map<dealii::types::global_dof_index, dealii::Point<dim>> indices_points;
  dealii::DoFTools::map_dofs_to_support_points(
      dealii::StaticMappingQ1<dim>::mapping, dof_handler, indices_points);
  // Change the format to the one used by ArborX
  std::vector<dealii::types::global_dof_index> dof_indices(
      indices_points.size());
  unsigned int pos = 0;
  for (auto map_it = indices_points.begin(); map_it != indices_points.end();
       ++map_it, ++pos)
  {
    dof_indices[pos] = map_it->first;
  }

  _expt_to_dof_mapping.first.resize(indices_and_offsets.first.size());
  _expt_to_dof_mapping.second.resize(indices_and_offsets.first.size());

  for (unsigned int i = 0; i < _expt_size; ++i)
  {
    for (int j = indices_and_offsets.second[i];
         j < indices_and_offsets.second[i + 1]; ++j)
    {
      _expt_to_dof_mapping.first[j] = i;
      _expt_to_dof_mapping.second[j] =
          dof_indices[indices_and_offsets.first[j]];
    }
  }
}

template <int dim>
void DataAssimilator::update_covariance_sparsity_pattern(
    dealii::DoFHandler<dim> const &dof_handler)
{
  _sim_size = dof_handler.n_dofs();

  // Use a DynamicSparsityPattern temporarily because the number of entries per
  // row is difficult to guess.
  dealii::DynamicSparsityPattern dsp(dof_handler.n_dofs());

  // Loop through the dofs to see which pairs are within the specified distance
  std::map<dealii::types::global_dof_index, dealii::Point<dim>> indices_points;

  dealii::DoFTools::map_dofs_to_support_points(
      dealii::StaticMappingQ1<dim>::mapping, dof_handler, indices_points);

  unsigned int i = 0;
  _covariance_distance_map.clear();
  for (auto map_it_i = indices_points.begin(); map_it_i != indices_points.end();
       ++map_it_i, ++i)
  {
    unsigned int j = 0;
    for (auto map_it_j = indices_points.begin();
         map_it_j != indices_points.end(); ++map_it_j, ++j)
    {
      double dist = map_it_i->second.distance(map_it_j->second);
      if (dist <= _localization_cutoff_distance)
      {
        dsp.add(i, j);
        _covariance_distance_map[std::make_pair(i, j)] = dist;
      }
    }
  }

  // Copy the DynamicSparsityPattern into a regular SparsityPattern for use
  _covariance_sparsity_pattern.copy_from(dsp);
}

dealii::Vector<double> DataAssimilator::calc_Hx(
    const dealii::LA::distributed::Vector<double> &sim_ensemble_member) const
{
  int num_expt_dof_map_entries = _expt_to_dof_mapping.first.size();

  dealii::Vector<double> out_vec(_expt_size);

  // Loop through the observation map to get the observation indices
  for (auto i = 0; i < num_expt_dof_map_entries; ++i)
  {
    auto sim_index = _expt_to_dof_mapping.second[i];
    auto expt_index = _expt_to_dof_mapping.first[i];
    out_vec(expt_index) = sim_ensemble_member(sim_index);
  }

  return out_vec;
}

void DataAssimilator::fill_noise_vector(dealii::Vector<double> &vec,
                                        dealii::SparseMatrix<double> &R,
                                        bool R_is_diagonal)
{
  auto vector_size = vec.size();

  // If R is diagonal, then the entries in the noise vector are independent and
  // each are simply a scaled output of the pseudo-random number generator. For
  // a more general R, one needs to multiply by the Cholesky decomposition of R
  // to achieve the appropriate correlation between the entries. Deal.II only
  // has a specific Cholesky function for full matrices, which is used in the
  // implementation below. The Cholesky decomposition is a special case of LU
  // decomposition, so we can use a sparse LU solver to obtain the "L" below if
  // needed in the future.

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

double DataAssimilator::gaspari_cohn_function(double r) const
{
  if (r < 1.0)
  {
    return 1. - 5. / 3. * r * r + 5. / 8. * r * r * r + 0.5 * r * r * r * r -
           0.25 * r * r * r * r * r;
  }
  else if (r < 2)
  {
    return 4. - 5. * r + 5. / 3. * r * r + 5. / 8. * r * r * r -
           0.5 * r * r * r * r + 1. / 12. * r * r * r * r * r - 2. / (3. * r);
  }
  else
  {
    return 0.;
  }
}

template <typename VectorType>
dealii::SparseMatrix<double> DataAssimilator::calc_sample_covariance_sparse(
    std::vector<VectorType> vec_ensemble) const
{
  unsigned int num_ensemble_members = vec_ensemble.size();
  unsigned int vec_size = 0;
  if (vec_ensemble.size() > 0)
  {
    vec_size = vec_ensemble[0].size();
  }

  // Calculate the mean
  dealii::Vector<double> mean(vec_size);
  for (unsigned int i = 0; i < vec_size; ++i)
  {
    double sum = 0.0;
    for (unsigned int sample = 0; sample < num_ensemble_members; ++sample)
    {
      sum += vec_ensemble[sample][i];
    }
    mean[i] = sum / num_ensemble_members;
  }

  // This could be calculated on the fly, but for now we're pre-calculating it.
  dealii::FullMatrix<double> anomaly(vec_size, num_ensemble_members);
  for (unsigned int member = 0; member < num_ensemble_members; ++member)
  {
    for (unsigned int i = 0; i < vec_size; ++i)
    {
      anomaly(i, member) = (vec_ensemble[member][i] - mean[i]) /
                           std::sqrt(num_ensemble_members - 1.0);
    }
  }

  // Do the element-wise matrix multiply by hand
  dealii::SparseMatrix<double> cov(_covariance_sparsity_pattern);

  unsigned int pos = 0;
  for (auto conv_iter = cov.begin(); conv_iter != cov.end(); ++conv_iter, ++pos)
  {
    unsigned int i = conv_iter->row();
    unsigned int j = conv_iter->column();

    double element_value = 0;
    for (unsigned int k = 0; k < num_ensemble_members; ++k)
    {
      element_value += anomaly(i, k) * anomaly(j, k);
    }

    double localization_scaling;
    double dist = _covariance_distance_map.find(std::make_pair(i, j))->second;
    if (_localization_cutoff_function == LocalizationCutoff::gaspari_cohn)
    {
      localization_scaling =
          gaspari_cohn_function(2.0 * dist / _localization_cutoff_distance);
    }
    else if (_localization_cutoff_function == LocalizationCutoff::step_function)
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

    conv_iter->value() = element_value * localization_scaling;
  }

  return cov;
}

// Explicit instantiation
template void DataAssimilator::update_dof_mapping<2>(
    dealii::DoFHandler<2> const &dof_handler,
    std::pair<std::vector<int>, std::vector<int>> const &indices_and_offsets);
template void DataAssimilator::update_dof_mapping<3>(
    dealii::DoFHandler<3> const &dof_handler,
    std::pair<std::vector<int>, std::vector<int>> const &indices_and_offsets);
template void DataAssimilator::update_covariance_sparsity_pattern<2>(
    dealii::DoFHandler<2> const &dof_handler);
template void DataAssimilator::update_covariance_sparsity_pattern<3>(
    dealii::DoFHandler<3> const &dof_handler);
template dealii::SparseMatrix<double>
DataAssimilator::calc_sample_covariance_sparse<dealii::Vector<double>>(
    std::vector<dealii::Vector<double>> vec_ensemble) const;
template dealii::SparseMatrix<double>
DataAssimilator::calc_sample_covariance_sparse<
    dealii::LA::distributed::Vector<double>>(
    std::vector<dealii::LA::distributed::Vector<double>> vec_ensemble) const;

} // namespace adamantine
