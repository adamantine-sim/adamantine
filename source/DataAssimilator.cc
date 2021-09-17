/* Copyright (c) 2020 - 2021, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#include <DataAssimilator.hh>
namespace adamantine
{

template <int dim, typename SimVectorType>
DataAssimilator<dim, SimVectorType>::DataAssimilator()
    : _gen(_igen, boost::normal_distribution<>()){};

template <int dim, typename SimVectorType>
void DataAssimilator<dim, SimVectorType>::updateEnsemble(
    std::vector<SimVectorType> &sim_data, PointsValues<dim> const &expt_data,
    const std::pair<std::vector<int>, std::vector<int>> &expt_to_sim_mapping,
    dealii::SparseMatrix<double> &R)
{
  // Get the perturbed y

  // Get H

  // Get K

  // Update the ensemble
}

template <int dim, typename SimVectorType>
dealii::SparseMatrix<double>
DataAssimilator<dim, SimVectorType>::calcKalmanGain(
    std::vector<SimVectorType> &sim_data,
    std::map<dealii::types::global_dof_index, double> &expt_data)
{
}

template <int dim, typename SimVectorType>
dealii::Vector<double> DataAssimilator<dim, SimVectorType>::calcHx(
    const SimVectorType &sim_ensemble_member,
    const std::pair<std::vector<int>, std::vector<int>> &expt_to_sim_mapping)
    const
{
  auto num_observations = expt_to_sim_mapping.first.size();

  dealii::Vector<double> out_vec(num_observations);

  /*
  // Loop through the observation map to get the observation indices
  for (auto i = 0; i < num_observations; ++i)
  {
    out_vec()
  }
  for (auto const &x : expt_to_sim_mapping)
  {
    out_vec(x.first) = sim_ensemble_member(x.second);
  }
  */
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
    dealii::FullMatrix<double> &cov)
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
