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

  void updateEnsemble(std::vector<SimVectorType> &sim_data,
                      std::vector<double> const &expt_data,
                      dealii::SparseMatrix<double> &R);

private:
  std::vector<dealii::Vector<double>> applyKalmanGain(
      std::vector<SimVectorType> &vec_ensemble, dealii::SparseMatrix<double> &R,
      std::vector<dealii::Vector<double>> &perturbed_innovation) const;

  void updateDofMapping(
      dealii::DoFHandler<dim> const &dof_handler,
      std::pair<std::vector<int>, std::vector<int>> const &indices_and_offsets);

  dealii::SparseMatrix<double> calcH(dealii::SparsityPattern &pattern) const;

  dealii::Vector<double> calcHx(const SimVectorType &sim_ensemble_member) const;

  void fillNoiseVector(dealii::Vector<double> &vec,
                       dealii::SparseMatrix<double> &R);

  dealii::FullMatrix<double> calcSampleCovarianceDense(
      std::vector<dealii::Vector<double>> vec_ensemble) const;

  unsigned int _num_ensemble_members;
  unsigned int _sim_size;
  unsigned int _expt_size;

  boost::mt19937 _igen;
  boost::variate_generator<boost::mt19937, boost::normal_distribution<>> _gen;

  std::pair<std::vector<int>, std::vector<int>> _expt_to_dof_mapping;
};
} // namespace adamantine

#endif
