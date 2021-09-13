/* Copyright (c) 2020 - 2021, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#ifndef DATA_ASSIMILATOR_HH
#define DATA_ASSIMILATOR_HH

#include <types.hh>

#include <deal.II/lac/la_vector.h>
#include <deal.II/lac/sparse_matrix.h>

#include <map>

namespace adamantine
{
/**
 * Forward declaration of the tester friend class to DataAssimilator.
 */
class DataAssimilatorTester;

template <typename SimVectorType>
class DataAssimilator
{
  friend class DataAssimilatorTester;

public:
  void
  updateEnsemble(std::vector<SimVectorType> &sim_data,
                 std::map<dealii::types::global_dof_index, double> &expt_data);

  dealii::LA::Vector<double>
  calcHx(const SimVectorType &sim_ensemble_member,
         const std::map<dealii::types::global_dof_index,
                        dealii::types::global_dof_index> &expt_to_sim_mapping)
      const;

private:
  dealii::SparseMatrix<double>
  calcKalmanGain(std::vector<SimVectorType> &sim_data,
                 std::map<dealii::types::global_dof_index, double> &expt_data);
};
} // namespace adamantine

#endif
