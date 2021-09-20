/* Copyright (c) 2020 - 2021, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#include <DataAssimilator.hh>

#include <deal.II/dofs/dof_tools.h>
#include <deal.II/lac/linear_operator_tools.h>
#include <deal.II/lac/sparse_direct.h>

namespace adamantine
{

template <int dim, typename SimVectorType>
DataAssimilator<dim, SimVectorType>::DataAssimilator()
    : _gen(_igen, boost::normal_distribution<>()){};

template <int dim, typename SimVectorType>
void DataAssimilator<dim, SimVectorType>::updateEnsemble(
    std::vector<SimVectorType> &sim_data, PointsValues<dim> const &expt_data,
    const std::pair<std::vector<int>, std::vector<int>> &indices_and_offsets,
    dealii::SparseMatrix<double> &R) const
{
  // Get the perturbed y

  // Get H

  // Get K

  // Update the ensemble
}

template <int dim, typename SimVectorType>
std::vector<dealii::Vector<double>>
DataAssimilator<dim, SimVectorType>::applyKalmanGain(
    std::vector<SimVectorType> &vec_ensemble, int expt_size,
    dealii::SparseMatrix<double> &R,
    dealii::Vector<double> &perturbed_innovation) const
{
  int num_samples = vec_ensemble.size();
  int sim_size = vec_ensemble[0].size();

  dealii::SparsityPattern patternH(expt_size, sim_size, expt_size);
  dealii::SparseMatrix<double> H = calcH(patternH);

  dealii::FullMatrix<double> P(sim_size);
  calcSampleCovarianceDense(vec_ensemble, P);

  const auto op_H = dealii::linear_operator(H);
  const auto op_P = dealii::linear_operator(P);
  const auto op_R = dealii::linear_operator(R);

  const auto op_HPH_plus_R =
      op_H * op_P * dealii::transpose_operator(op_H) + op_R;

  dealii::SparseDirectUMFPACK HPH_plus_R_inv;
  const auto op_HPH_plus_R_inv =
      dealii::linear_operator(op_HPH_plus_R, HPH_plus_R_inv);

  const auto op_K = op_P * dealii::transpose_operator(op_H) * op_HPH_plus_R_inv;

  std::vector<dealii::Vector<double>> output;
  for (int sample = 0; sample < vec_ensemble.size(); ++sample)
  {
    dealii::Vector<double> temp = op_K * perturbed_innovation;
    output.push_back(temp);
  }

  return output;
}

template <int dim, typename SimVectorType>
dealii::SparseMatrix<double> DataAssimilator<dim, SimVectorType>::calcH(
    dealii::SparsityPattern &pattern) const
{
  int num_expt_dof_map_entries = _expt_to_dof_mapping.first.size();

  for (auto i = 0; i < num_expt_dof_map_entries; ++i)
  {
    auto sim_index = _expt_to_dof_mapping.second[i];
    auto expt_index = _expt_to_dof_mapping.first[i];
    std::cout << "adding: " << sim_index << " and " << expt_index << std::endl;
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

template <int dim, typename SimVectorType>
void DataAssimilator<dim, SimVectorType>::updateDofMapping(
    dealii::DoFHandler<dim> const &dof_handler, int num_expt_points,
    std::pair<std::vector<int>, std::vector<int>> const &indices_and_offsets)
{
  std::map<dealii::types::global_dof_index, dealii::Point<dim>> indices_points;
  dealii::DoFTools::map_dofs_to_support_points(
      dealii::StaticMappingQ1<dim>::mapping, dof_handler, indices_points);
  // Change the format to used by ArborX
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

  for (unsigned int i = 0; i < num_expt_points; ++i)
  {
    std::cout << "here i: " << i << std::endl;
    for (int j = indices_and_offsets.second[i];
         j < indices_and_offsets.second[i + 1]; ++j)
    {
      std::cout << "here j:  " << j << std::endl;
      _expt_to_dof_mapping.first[j] = i;
      _expt_to_dof_mapping.second[j] =
          dof_indices[indices_and_offsets.first[j]];

      std::cout << "updateDofMapping: " << _expt_to_dof_mapping.first[j] << " "
                << _expt_to_dof_mapping.second[j] << std::endl;
    }
  }
}

template <int dim, typename SimVectorType>
dealii::Vector<double> DataAssimilator<dim, SimVectorType>::calcHx(
    const int expt_size, const SimVectorType &sim_ensemble_member) const
{
  int num_expt_dof_map_entries = _expt_to_dof_mapping.first.size();

  dealii::Vector<double> out_vec(expt_size);

  // Loop through the observation map to get the observation indices
  for (auto i = 0; i < num_expt_dof_map_entries; ++i)
  {
    auto sim_index = _expt_to_dof_mapping.second[i];
    auto expt_index = _expt_to_dof_mapping.first[i];
    out_vec(expt_index) = sim_ensemble_member(sim_index);
  }

  return out_vec;
}

template <int dim, typename SimVectorType>
void DataAssimilator<dim, SimVectorType>::fillNoiseVector(
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

  for (int i = 0; i < vector_size; ++i)
  {
    uncorrelated_noise_vector(i) = _gen();
  }

  L.vmult(vec, uncorrelated_noise_vector);
}

template <int dim, typename SimVectorType>
void DataAssimilator<dim, SimVectorType>::calcSampleCovarianceDense(
    std::vector<dealii::Vector<double>> vec_ensemble,
    dealii::FullMatrix<double> &cov) const
{
  auto size_per_sample = vec_ensemble[0].size();
  // Calculate the mean
  dealii::Vector<double> mean(size_per_sample);
  for (unsigned int i = 0; i < size_per_sample; ++i)
  {
    double sum = 0.0;
    for (unsigned int sample = 0; sample < vec_ensemble.size(); ++sample)
    {
      sum += vec_ensemble[sample][i];
    }
    mean[i] = sum / vec_ensemble.size();
  }

  // Calculate the anomaly
  dealii::FullMatrix<double> anomaly(size_per_sample, vec_ensemble.size());
  for (unsigned int sample = 0; sample < vec_ensemble.size(); ++sample)
  {
    for (unsigned int i = 0; i < size_per_sample; ++i)
    {
      anomaly(i, sample) = (vec_ensemble[sample][i] - mean[i]) /
                           std::sqrt(vec_ensemble.size() - 1.0);
    }
  }
  anomaly.mTmult(cov, anomaly);
}

// Explicit instantiation
template class DataAssimilator<2, dealii::Vector<double>>;
template class DataAssimilator<3, dealii::Vector<double>>;

} // namespace adamantine
