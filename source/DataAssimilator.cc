/* Copyright (c) 2020 - 2021, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#include <DataAssimilator.hh>
namespace adamantine
{
template <typename SimVectorType>
void DataAssimilator<SimVectorType>::updateEnsemble(
    std::vector<SimVectorType> &sim_data,
    std::map<dealii::types::global_dof_index, double> &expt_data)
{
}

template <typename SimVectorType>
dealii::SparseMatrix<double> DataAssimilator<SimVectorType>::calcKalmanGain(
    std::vector<SimVectorType> &sim_data,
    std::map<dealii::types::global_dof_index, double> &expt_data)
{
}

template <typename SimVectorType>
dealii::LA::Vector<double> DataAssimilator<SimVectorType>::calcHx(
    const SimVectorType &sim_ensemble_member,
    const std::map<dealii::types::global_dof_index,
                   dealii::types::global_dof_index> &expt_to_sim_mapping) const
{
  dealii::LA::Vector<double> out_vec(expt_to_sim_mapping.size());

  // Loop through the observation map to get the observation indices
  for (auto const &x : expt_to_sim_mapping)
  {
    out_vec(x.first) = sim_ensemble_member(x.second);
  }

  return out_vec;
}

// Explicit instantiation
template class DataAssimilator<dealii::LA::Vector<double>>;

} // namespace adamantine
