/* Copyright (c) 2021, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#include <DataAssimilator.hh>

#include <deal.II/dofs/dof_tools.h>
#include <deal.II/lac/linear_operator_tools.h>

namespace adamantine
{

template <typename SimVectorType>
DataAssimilator<SimVectorType>::DataAssimilator(
    boost::property_tree::ptree const &database)
    : _normal_dist_generator(_prng, boost::normal_distribution<>())
{

  // Set the solver parameters from the input database
  if (boost::optional<unsigned int> max_num_temp_vectors =
          database.get_optional<unsigned int>("maximum number of temp vectors"))
    _additional_data.max_n_tmp_vectors = *max_num_temp_vectors;

  if (boost::optional<unsigned int> max_iterations =
          database.get_optional<unsigned int>("maximum iterations"))
    _solver_control.set_max_steps(*max_iterations);

  if (boost::optional<double> tolerance =
          database.get_optional<double>("convergence tolerance"))
    _solver_control.set_tolerance(*tolerance);
};

template <typename SimVectorType>
void DataAssimilator<SimVectorType>::update_ensemble(
    std::vector<SimVectorType> &sim_data, std::vector<double> const &expt_data,
    dealii::SparseMatrix<double> &R)
{
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
  _expt_size = expt_data.size();

  // Get the perturbed innovation, ( y+u - Hx )
  std::vector<dealii::Vector<double>> perturbed_innovation(
      _num_ensemble_members);
  for (unsigned int member = 0; member < _num_ensemble_members; ++member)
  {
    perturbed_innovation[member].reinit(_expt_size);
    fillNoiseVector(perturbed_innovation[member], R);
    dealii::Vector<double> temp = calcHx(sim_data[member]);
    for (unsigned int i = 0; i < _expt_size; ++i)
    {
      perturbed_innovation[member][i] += expt_data[i] - temp[i];
    }
  }

  // Apply the Kalman filter to the perturbed innovation, K ( y+u - Hxf )
  std::vector<dealii::Vector<double>> forcast_shift =
      applyKalmanGain(sim_data, R, perturbed_innovation);

  // Update the ensemble, xa = xf + K ( y+u - Hxf )
  for (unsigned int member = 0; member < _num_ensemble_members; ++member)
  {
    sim_data[member] += forcast_shift[member];
  }
}

template <typename SimVectorType>
std::vector<dealii::Vector<double>>
DataAssimilator<SimVectorType>::applyKalmanGain(
    std::vector<SimVectorType> &vec_ensemble, dealii::SparseMatrix<double> &R,
    std::vector<dealii::Vector<double>> &perturbed_innovation)
{
  dealii::SparsityPattern patternH(_expt_size, _sim_size, _expt_size);
  dealii::SparseMatrix<double> H = calcH(patternH);

  dealii::FullMatrix<double> P = calcSampleCovarianceDense(vec_ensemble);

  const auto op_H = dealii::linear_operator(H);
  const auto op_P = dealii::linear_operator(P);
  const auto op_R = dealii::linear_operator(R);

  const auto op_HPH_plus_R =
      op_H * op_P * dealii::transpose_operator(op_H) + op_R;

  dealii::SolverGMRES<dealii::Vector<double>> R_inv_solver(_solver_control,
                                                           _additional_data);

  auto op_HPH_plus_R_inv =
      dealii::inverse_operator(op_HPH_plus_R, R_inv_solver);

  const auto op_K = op_P * dealii::transpose_operator(op_H) * op_HPH_plus_R_inv;

  // Apply the Kalman gain to each innovation vector
  // (Unclear if this is really inefficient because op_K is calculated fresh
  // for each application)
  std::vector<dealii::Vector<double>> output;
  for (unsigned int member = 0; member < _num_ensemble_members; ++member)
  {
    dealii::Vector<double> temp = op_K * perturbed_innovation[member];
    output.push_back(temp);
  }

  return output;
}

template <typename SimVectorType>
dealii::SparseMatrix<double>
DataAssimilator<SimVectorType>::calcH(dealii::SparsityPattern &pattern) const
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

template <typename SimVectorType>
template <int dim>
void DataAssimilator<SimVectorType>::updateDofMapping(
    dealii::DoFHandler<dim> const &dof_handler,
    std::pair<std::vector<int>, std::vector<int>> const &indices_and_offsets)
{
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

template <typename SimVectorType>
dealii::Vector<double> DataAssimilator<SimVectorType>::calcHx(
    const SimVectorType &sim_ensemble_member) const
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

template <typename SimVectorType>
void DataAssimilator<SimVectorType>::fillNoiseVector(
    dealii::Vector<double> &vec, dealii::SparseMatrix<double> &R)
{
  auto vector_size = vec.size();

  // Do Cholesky decomposition
  dealii::FullMatrix<double> L(vector_size);
  dealii::FullMatrix<double> Rfull(vector_size);
  Rfull.copy_from(R);
  L.cholesky(Rfull);

  // Get a vector of normally distributed values
  dealii::Vector<double> uncorrelated_noise_vector(vector_size);

  for (unsigned int i = 0; i < vector_size; ++i)
  {
    uncorrelated_noise_vector(i) = _normal_dist_generator();
  }

  L.vmult(vec, uncorrelated_noise_vector);
}

template <typename SimVectorType>
dealii::FullMatrix<double>
DataAssimilator<SimVectorType>::calcSampleCovarianceDense(
    std::vector<dealii::Vector<double>> vec_ensemble) const
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

  // Calculate the anomaly
  dealii::FullMatrix<double> anomaly(vec_size, num_ensemble_members);
  for (unsigned int member = 0; member < num_ensemble_members; ++member)
  {
    for (unsigned int i = 0; i < vec_size; ++i)
    {
      anomaly(i, member) = (vec_ensemble[member][i] - mean[i]) /
                           std::sqrt(num_ensemble_members - 1.0);
    }
  }

  dealii::FullMatrix<double> cov(vec_size);
  anomaly.mTmult(cov, anomaly);

  return cov;
}

// Explicit instantiation
template class DataAssimilator<dealii::Vector<double>>;
template void DataAssimilator<dealii::Vector<double>>::updateDofMapping<2>(
    dealii::DoFHandler<2> const &dof_handler,
    std::pair<std::vector<int>, std::vector<int>> const &indices_and_offsets);
template void DataAssimilator<dealii::Vector<double>>::updateDofMapping<3>(
    dealii::DoFHandler<3> const &dof_handler,
    std::pair<std::vector<int>, std::vector<int>> const &indices_and_offsets);

} // namespace adamantine
