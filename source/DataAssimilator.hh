/* Copyright (c) 2021, the adamantine authors.
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
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparse_matrix.h>

#include <map>
#include <random>

namespace adamantine
{
/**
 * Forward declaration of the tester friend class to DataAssimilator.
 */
class DataAssimilatorTester;

/**
 * This class implements an ensemble Kalman filter (EnKF) for data assimilation.
 * It updates an ensemble of solution vectors to be consistent with experimental
 * observations.
 *
 * The EnKF implementation here is largely based on Chapter 6 of Data
 * Assimilation: Method and Applications by Asch, Bocquet, and Nodet.
 */
class DataAssimilator
{
  friend class DataAssimilatorTester;

public:
  /**
   * Constructor.
   */
  DataAssimilator(boost::property_tree::ptree const &database);

  /**
   * This is the main public interface for the class and is called to perform
   * one assimilation process. It takes in an ensemble of simulation data
   * (sim_data), experimental observation data (expt_data), and the covariance
   * matrix for the experimental observations (R). This function updates
   * sim_data using the EnKF method.
   *
   * Overall it performs x = x + K (y + u - H x) for each ensemble member,
   * where:
   * x is the simulation solution
   * K is the Kalman gain
   * y is the observed solution
   * u is a perturbation to the observed solution
   * H is the observation matrix
   */
  void update_ensemble(
      MPI_Comm const &communicator,
      std::vector<dealii::LA::distributed::Vector<double>> &sim_data,
      std::vector<double> const &expt_data, dealii::SparseMatrix<double> &R);

  /**
   * This updates the internal mapping between the indices of the entries in
   * expt_data and the indices of the entries in the sim_data ensemble members
   * in updateEnsemble. This must be called before updateEnsemble whenever there
   * are changes to the simulation mesh or the observation locations. The
   * indices_and_offsets variable is the output from
   * adamantine::set_with_experimental_data.
   */
  template <int dim>
  void update_dof_mapping(
      dealii::DoFHandler<dim> const &dof_handler,
      std::pair<std::vector<int>, std::vector<int>> const &indices_and_offsets);

private:
  /**
   * This calculates the Kalman gain and applies it to the perturbed innovation.
   */
  std::vector<dealii::LA::distributed::Vector<double>> apply_kalman_gain(
      std::vector<dealii::LA::distributed::Vector<double>> &vec_ensemble,
      dealii::SparseMatrix<double> &R,
      std::vector<dealii::Vector<double>> &perturbed_innovation);

  /**
   * This calculates the observation matrix.
   */
  dealii::SparseMatrix<double> calc_H(dealii::SparsityPattern &pattern) const;

  /**
   * This calculates the application of the observation matrix on an ensemble
   * member from the simulation (sim_ensemble_member), avoiding explicit
   * construction of a sparse matrix for H.
   */
  dealii::Vector<double> calc_Hx(
      const dealii::LA::distributed::Vector<double> &sim_ensemble_member) const;

  /**
   * This fills a vector (vec) with noise from a multivariate normal
   * distribution defined by a covariance matrix (R).
   */
  void fill_noise_vector(dealii::Vector<double> &vec,
                         dealii::SparseMatrix<double> &R);

  /**
   * This calculates the sample covariance for an input ensemble of vectors
   * (vec_ensemble). Currently this is for a full (i.e. not sparse) matrix. For
   * improved computational performance and reduced spurious correlations a
   * sparse matrix version of this with localization should also be implemented.
   * This is templated so that it can be called on both simulated and
   * experimental data.
   */
  template <typename VectorType>
  dealii::FullMatrix<double>
  calc_sample_covariance_dense(std::vector<VectorType> vec_ensemble) const;

  /**
   * The number of ensemble members in the simulation.
   */
  unsigned int _num_ensemble_members;

  /**
   * The length of the data vector for each simulation ensemble member.
   */
  unsigned int _sim_size;

  /**
   * The length of the data vector the experimental observations.
   */
  unsigned int _expt_size;

  /**
   * The pseudo-random number generator, used for the perturbations to the
   * innovation vectors.
   */
  std::mt19937 _prng;

  /**
   * Random variate generator for a normal distribution, used for the
   * perturbations to the innovation vectors.
   */
  std::normal_distribution<> _normal_dist_generator;

  /**
   * The mapping between the index in the experimental observation data vector
   * to the DoF in the simulation data vectors. This is simpler to use than the
   * indices and offsets that are passed into updateEnsemble, and means that the
   * simulation DoFHandler only needs to be used by updateDofMapping where this
   * variable is set.
   */
  std::pair<std::vector<int>, std::vector<int>> _expt_to_dof_mapping;

  /**
   * Standardized settings for the GMRES solver needed for the matrix inversion
   * in the Kalman gain calculation.
   */
  dealii::SolverControl _solver_control;

  /**
   * Additional settings for the GMRES solver needed for the matrix inversion in
   * the Kalman gain calculation.
   */
  dealii::SolverGMRES<dealii::Vector<double>>::AdditionalData _additional_data;
};
} // namespace adamantine

#endif
