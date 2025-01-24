/* Copyright (c) 2021-2023, the adamantine authors.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#ifndef DATA_ASSIMILATOR_HH
#define DATA_ASSIMILATOR_HH

#include <experimental_data_utils.hh>
#include <types.hh>

#include <deal.II/fe/mapping_q1_eulerian.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>

#include <map>
#include <random>

namespace adamantine
{

/**
 * Enum for different options for the functions that determine how the
 * covariance is decreased with distance for localization. The 'gaspari_cohn'
 * option corrseponds to a function defined in Gaspari and Cohn, Quarterly
 * Journal of the Royal Meteorological Society, 125, 1999.
 */
enum class LocalizationCutoff
{
  gaspari_cohn,
  step_function,
  none
};

enum class AugmentedStateParameters
{
  beam_0_absorption,
  beam_0_max_power
};

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
  DataAssimilator(MPI_Comm const &global_communicator,
                  MPI_Comm const &local_communicator, int my_color,
                  boost::property_tree::ptree const &database);

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
  void update_ensemble(std::vector<dealii::LA::distributed::BlockVector<double>>
                           &augmented_state_ensemble,
                       std::vector<double> const &expt_data,
                       dealii::SparseMatrix<double> const &R);

  /**
   * This updates the internal mapping between the indices of the entries in
   * expt_data and the indices of the entries in the sim_data ensemble members
   * in updateEnsemble. This must be called before updateEnsemble whenever there
   * are changes to the simulation mesh or the observation locations.
   */
  template <int dim>
  void update_dof_mapping(
      std::pair<std::vector<int>, std::vector<int>> const &expt_to_dof_mapping);

  /**
   * This updates the sparsity pattern for the sample covariance matrix for the
   * simulation ensemble. This must be called before updateEnsemble whenever
   * there are changes to the simulation mesh.
   */
  template <int dim>
  void
  update_covariance_sparsity_pattern(dealii::DoFHandler<dim> const &dof_handler,
                                     const unsigned int parameter_size);

private:
  using block_size_type =
      dealii::LA::distributed::BlockVector<double>::size_type;

  /**
   * Gather the augmented ensemble members on the global root node in order to
   * apply the Kalman gain.
   */
  void gather_ensemble_members(
      std::vector<dealii::LA::distributed::BlockVector<double>>
          &augmented_state_ensemble,
      std::vector<dealii::LA::distributed::BlockVector<double>>
          &global_augmented_state_ensemble,
      std::vector<unsigned int> &local_n_ensemble_members,
      std::vector<block_size_type> &block_sizes);

  /**
   * Scatter the @p global_augmented_state_ensemble back into @p
   * augmented_state_ensemble.
   */
  void scatter_ensemble_members(
      std::vector<dealii::LA::distributed::BlockVector<double>>
          &augmented_state_ensemble,
      std::vector<dealii::LA::distributed::BlockVector<double>> const
          &global_augmented_state_ensemble,
      std::vector<unsigned int> const &local_n_ensemble_members,
      std::vector<block_size_type> const &block_sizes);

  /**
   * This calculates the Kalman gain and applies it to the perturbed innovation.
   */
  std::vector<dealii::LA::distributed::BlockVector<double>> apply_kalman_gain(
      std::vector<dealii::LA::distributed::BlockVector<double>>
          &augmented_state_ensemble,
      dealii::SparseMatrix<double> const &R,
      std::vector<dealii::Vector<double>> const &perturbed_innovation);

  /**
   * This calculates the observation matrix.
   */
  dealii::SparseMatrix<double> calc_H(dealii::SparsityPattern &pattern) const;

  /**
   * This calculates the application of the observation matrix on an unaugmented
   * ensemble member from the simulation, avoiding explicit construction of a
   * sparse matrix for H.
   */
  dealii::Vector<double> calc_Hx(
      dealii::LA::distributed::Vector<double> const &sim_ensemble_member) const;

  /**
   * This fills a vector (vec) with noise from a multivariate normal
   * distribution defined by a covariance matrix (R). Note: For non-diagonal R
   * this method currently uses full matrices, which substantially limits the
   * allowable problem size.
   */
  void fill_noise_vector(dealii::Vector<double> &vec,
                         dealii::SparseMatrix<double> const &R,
                         bool const R_is_diagonal);

  /**
   * A standard localization function, resembles a Gaussian, but with finite
   * support. From Gaspari and Cohn, Quarterly Journal of the Royal
   * Meteorological Society, 125, 1999.
   */
  double gaspari_cohn_function(double const r) const;

  /**
   * This calculates the sample covariance for an input ensemble of vectors
   * (vec_ensemble). Currently this is tied to the simulation ensemble, through
   * the use of member variables inside. If needed, the interface could be
   * redone to make it more generally applicable.
   */
  dealii::TrilinosWrappers::SparseMatrix calc_sample_covariance_sparse(
      std::vector<dealii::LA::distributed::BlockVector<double>> const
          &vec_ensemble) const;

  /**
   * Global MPI communicator.
   */
  MPI_Comm _global_communicator;

  /**
   * Local MPI communicator split off the global MPI communicator.
   */
  MPI_Comm _local_communicator;

  /**
   * Rank of the process when using the global MPI communicator.
   */
  int _global_rank = -1;

  /**
   * Number of processes in the gglbal MPI communicator.
   */
  int _global_comm_size = -1;

  /**
   * Color associated with the local
   */
  int _color = -1;

  /**
   * The number of ensemble members in the simulation.
   */
  unsigned int _num_ensemble_members = 0;

  /**
   * The length of the data vector for each simulation ensemble member.
   */
  unsigned int _sim_size = 0;

  /**
   * The length of the parameter vector for each simulation ensemble member.
   */
  unsigned int _parameter_size = 0;

  /**
   * The length of the data vector the experimental observations.
   */
  unsigned int _expt_size = 0;

  /**
   * The sparsity pattern for the localized covariance matrix.
   */
  dealii::TrilinosWrappers::SparsityPattern _covariance_sparsity_pattern;

  /**
   * Map between the indices in the covariance matrix and the distance between
   * the support points for the associated dofs on the mesh. This is necessary
   * for localization.
   */
  std::map<const std::pair<unsigned int, unsigned int>, double>
      _covariance_distance_map;

  /**
   * The distance at which the sample covariance is truncated.
   */
  double _localization_cutoff_distance;

  /**
   * The function used to reduce the sample covariance entries based on the
   * distance between the relevant points.
   */
  LocalizationCutoff _localization_cutoff_function;

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
