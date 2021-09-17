/* Copyright (c) 2020 - 2021, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#ifndef DATA_ASSIMILATOR_HH
#define DATA_ASSIMILATOR_HH

#include <experimental_data.hh>
#include <types.hh>

#include <deal.II/fe/mapping_q1_eulerian.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/la_vector.h>
#include <deal.II/lac/sparse_matrix.h>

#include <boost/random.hpp>
#include <boost/random/mersenne_twister.hpp>

#include <map>

namespace adamantine
{
/**
 * Forward declaration of the tester friend class to DataAssimilator.
 */
class DataAssimilatorTester;

template <int dim, typename SimVectorType>
class DataAssimilator
{
  friend class DataAssimilatorTester;

public:
  DataAssimilator();

  void updateEnsemble(
      std::vector<SimVectorType> &sim_data, PointsValues<dim> const &expt_data,
      const std::pair<std::vector<int>, std::vector<int>> &expt_to_sim_mapping,
      dealii::SparseMatrix<double> &R);

private:
  dealii::SparseMatrix<double>
  calcKalmanGain(std::vector<SimVectorType> &sim_data,
                 std::map<dealii::types::global_dof_index, double> &expt_data);

  dealii::Vector<double>
  calcHx(const SimVectorType &sim_ensemble_member,
         const std::pair<std::vector<int>, std::vector<int>>
             &expt_to_sim_mapping) const;

  void fillNoiseVector(dealii::Vector<double> &vec,
                       dealii::SparseMatrix<double> &R);

  void
  calcSampleCovarianceDense(std::vector<dealii::Vector<double>> vec_ensemble,
                            dealii::FullMatrix<double> &cov);

  boost::mt19937 _igen;
  boost::variate_generator<boost::mt19937, boost::normal_distribution<>> _gen;
};
} // namespace adamantine

#endif
