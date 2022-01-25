/* Copyright (c) 2021, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#define BOOST_TEST_MODULE DataAssimilator

#include <DataAssimilator.hh>
#include <Geometry.hh>

#include <deal.II/fe/fe_q.h>
#include <deal.II/lac/la_parallel_vector.h>

#include "main.cc"

namespace adamantine
{
class DataAssimilatorTester
{
public:
  void test_constructor()
  {
    boost::property_tree::ptree database;

    // First checking the dealii default values
    DataAssimilator da0(database);

    double tol = 1.0e-12;
    BOOST_CHECK_SMALL(da0._solver_control.tolerance() - 1.0e-10, tol);
    BOOST_CHECK(da0._solver_control.max_steps() == 100);
    BOOST_CHECK(da0._additional_data.max_n_tmp_vectors == 30);

    // Now explicitly setting them
    database.put("solver.convergence_tolerance", 1.0e-6);
    database.put("solver.max_iterations", 25);
    database.put("solver.max_number_of_temp_vectors", 4);
    DataAssimilator da1(database);
    BOOST_CHECK_SMALL(da1._solver_control.tolerance() - 1.0e-6, tol);
    BOOST_CHECK(da1._solver_control.max_steps() == 25);
    BOOST_CHECK(da1._additional_data.max_n_tmp_vectors == 4);
  };

  void test_calc_kalman_gain()
  {
    // Create the DoF mapping
    MPI_Comm communicator = MPI_COMM_WORLD;

    boost::property_tree::ptree database;
    database.put("import_mesh", false);
    database.put("length", 1);
    database.put("length_divisions", 1);
    database.put("height", 1);
    database.put("height_divisions", 1);
    adamantine::Geometry<2> geometry(communicator, database);
    dealii::parallel::distributed::Triangulation<2> const &tria =
        geometry.get_triangulation();

    dealii::FE_Q<2> fe(1);
    dealii::DoFHandler<2> dof_handler(tria);
    dof_handler.distribute_dofs(fe);

    unsigned int sim_size = 4;
    unsigned int expt_size = 2;

    dealii::Vector<double> expt_vec(2);
    expt_vec(0) = 2.5;
    expt_vec(1) = 9.5;

    std::pair<std::vector<int>, std::vector<int>> indices_and_offsets;
    indices_and_offsets.first.resize(2);
    indices_and_offsets.second.resize(3); // Offset vector is one longer
    indices_and_offsets.first[0] = 1;
    indices_and_offsets.first[1] = 3;
    indices_and_offsets.second[0] = 0;
    indices_and_offsets.second[1] = 1;
    indices_and_offsets.second[2] = 2;

    boost::property_tree::ptree solver_settings_database;
    DataAssimilator da(solver_settings_database);
    da._sim_size = sim_size;
    da._expt_size = expt_size;
    da._parameter_size = 0;
    da._num_ensemble_members = 3;
    da.update_dof_mapping<2>(dof_handler, indices_and_offsets);
    da.update_covariance_sparsity_pattern<2>(dof_handler);

    // Create the simulation data
    std::vector<dealii::LA::distributed::BlockVector<double>>
        augmented_state_ensemble(3);

    augmented_state_ensemble[0].reinit(2);
    augmented_state_ensemble[0].block(0).reinit(4);
    augmented_state_ensemble[0].block(0)(0) = 1.0;
    augmented_state_ensemble[0].block(0)(1) = 3.0;
    augmented_state_ensemble[0].block(0)(2) = 6.0;
    augmented_state_ensemble[0].block(0)(3) = 9.0;
    augmented_state_ensemble[0].collect_sizes();
    augmented_state_ensemble[1].reinit(2);
    augmented_state_ensemble[1].block(0).reinit(4);
    augmented_state_ensemble[1].block(0)(0) = 1.5;
    augmented_state_ensemble[1].block(0)(1) = 3.2;
    augmented_state_ensemble[1].block(0)(2) = 6.3;
    augmented_state_ensemble[1].block(0)(3) = 9.7;
    augmented_state_ensemble[1].collect_sizes();
    augmented_state_ensemble[2].reinit(2);
    augmented_state_ensemble[2].block(0).reinit(4);
    augmented_state_ensemble[2].block(0)(0) = 1.1;
    augmented_state_ensemble[2].block(0)(1) = 3.1;
    augmented_state_ensemble[2].block(0)(2) = 6.1;
    augmented_state_ensemble[2].block(0)(3) = 9.1;
    augmented_state_ensemble[2].collect_sizes();

    // Build the sparse experimental covariance matrix
    dealii::SparsityPattern pattern(expt_size, expt_size, 1);
    pattern.add(0, 0);
    pattern.add(1, 1);
    pattern.compress();

    dealii::SparseMatrix<double> R(pattern);
    R.add(0, 0, 0.002);
    R.add(1, 1, 0.001);

    // Create the (perturbed) innovation
    std::vector<dealii::Vector<double>> perturbed_innovation(3);
    for (unsigned int sample = 0; sample < perturbed_innovation.size();
         ++sample)
    {
      perturbed_innovation[sample].reinit(expt_size);
      dealii::Vector<double> temp =
          da.calc_Hx(augmented_state_ensemble[sample].block(0));
      for (unsigned int i = 0; i < expt_size; ++i)
      {
        perturbed_innovation[sample][i] = expt_vec[i] - temp[i];
      }
    }

    perturbed_innovation[0][0] = perturbed_innovation[0][0] + 0.0008;
    perturbed_innovation[0][1] = perturbed_innovation[0][1] - 0.0005;
    perturbed_innovation[1][0] = perturbed_innovation[1][0] - 0.001;
    perturbed_innovation[1][1] = perturbed_innovation[1][1] + 0.0002;
    perturbed_innovation[2][0] = perturbed_innovation[2][0] + 0.0002;
    perturbed_innovation[2][1] = perturbed_innovation[2][1] - 0.0009;

    // Apply the Kalman gain
    std::vector<dealii::LA::distributed::BlockVector<double>> forecast_shift =
        da.apply_kalman_gain(augmented_state_ensemble, R, perturbed_innovation);

    double tol = 1.0e-4;

    // Reference solution calculated using Python
    BOOST_CHECK_CLOSE(forecast_shift[0][0], 0.21352564, tol);
    BOOST_CHECK_CLOSE(forecast_shift[0][1], -0.14600986, tol);
    BOOST_CHECK_CLOSE(forecast_shift[0][2], -0.02616469, tol);
    BOOST_CHECK_CLOSE(forecast_shift[0][3], 0.45321598, tol);
    BOOST_CHECK_CLOSE(forecast_shift[1][0], -0.27786325, tol);
    BOOST_CHECK_CLOSE(forecast_shift[1][1], -0.32946285, tol);
    BOOST_CHECK_CLOSE(forecast_shift[1][2], -0.31226298, tol);
    BOOST_CHECK_CLOSE(forecast_shift[1][3], -0.24346351, tol);
    BOOST_CHECK_CLOSE(forecast_shift[2][0], 0.12767094, tol);
    BOOST_CHECK_CLOSE(forecast_shift[2][1], -0.20319395, tol);
    BOOST_CHECK_CLOSE(forecast_shift[2][2], -0.09290565, tol);
    BOOST_CHECK_CLOSE(forecast_shift[2][3], 0.34824753, tol);
  };

  void test_update_dof_mapping()
  {
    MPI_Comm communicator = MPI_COMM_WORLD;

    boost::property_tree::ptree database;
    database.put("import_mesh", false);
    database.put("length", 1);
    database.put("length_divisions", 2);
    database.put("height", 1);
    database.put("height_divisions", 2);
    adamantine::Geometry<2> geometry(communicator, database);
    dealii::parallel::distributed::Triangulation<2> const &tria =
        geometry.get_triangulation();

    dealii::FE_Q<2> fe(1);
    dealii::DoFHandler<2> dof_handler(tria);
    dof_handler.distribute_dofs(fe);

    unsigned int sim_size = 4;
    unsigned int expt_size = 3;

    std::pair<std::vector<int>, std::vector<int>> indices_and_offsets;
    indices_and_offsets.first.resize(3);
    indices_and_offsets.second.resize(4); // offset vector is one longer
    indices_and_offsets.first[0] = 0;
    indices_and_offsets.first[1] = 1;
    indices_and_offsets.first[2] = 3;
    indices_and_offsets.second[0] = 0;
    indices_and_offsets.second[1] = 1;
    indices_and_offsets.second[2] = 2;
    indices_and_offsets.second[3] = 3;

    boost::property_tree::ptree solver_settings_database;
    DataAssimilator da(solver_settings_database);
    da._sim_size = sim_size;
    da._expt_size = expt_size;
    da.update_dof_mapping<2>(dof_handler, indices_and_offsets);

    BOOST_CHECK(da._expt_to_dof_mapping.first[0] == 0);
    BOOST_CHECK(da._expt_to_dof_mapping.first[1] == 1);
    BOOST_CHECK(da._expt_to_dof_mapping.first[2] == 2);
    BOOST_CHECK(da._expt_to_dof_mapping.second[0] == 0);
    BOOST_CHECK(da._expt_to_dof_mapping.second[1] == 1);
    BOOST_CHECK(da._expt_to_dof_mapping.second[2] == 3);
  };

  void test_calc_H()
  {
    MPI_Comm communicator = MPI_COMM_WORLD;

    boost::property_tree::ptree database;
    database.put("import_mesh", false);
    database.put("length", 1);
    database.put("length_divisions", 2);
    database.put("height", 1);
    database.put("height_divisions", 2);
    adamantine::Geometry<2> geometry(communicator, database);
    dealii::parallel::distributed::Triangulation<2> const &tria =
        geometry.get_triangulation();

    dealii::FE_Q<2> fe(1);
    dealii::DoFHandler<2> dof_handler(tria);
    dof_handler.distribute_dofs(fe);

    unsigned int sim_size = 4;
    unsigned int expt_size = 3;

    std::pair<std::vector<int>, std::vector<int>> indices_and_offsets;
    indices_and_offsets.first.resize(3);
    indices_and_offsets.second.resize(4); // offset vector is one longer
    indices_and_offsets.first[0] = 0;
    indices_and_offsets.first[1] = 1;
    indices_and_offsets.first[2] = 3;
    indices_and_offsets.second[0] = 0;
    indices_and_offsets.second[1] = 1;
    indices_and_offsets.second[2] = 2;
    indices_and_offsets.second[3] = 3;

    boost::property_tree::ptree solver_settings_database;
    DataAssimilator da(solver_settings_database);
    da._sim_size = sim_size;
    da._expt_size = expt_size;
    da.update_dof_mapping<2>(dof_handler, indices_and_offsets);

    dealii::SparsityPattern pattern(expt_size, sim_size, expt_size);

    dealii::SparseMatrix<double> H = da.calc_H(pattern);

    double tol = 1e-12;
    for (unsigned int i = 0; i < expt_size; ++i)
    {
      for (unsigned int j = 0; j < sim_size; ++j)
      {
        if (i == 0 && j == 0)
          BOOST_CHECK_CLOSE(H(i, j), 1.0, tol);
        else if (i == 1 && j == 1)
          BOOST_CHECK_CLOSE(H(i, j), 1.0, tol);
        else if (i == 2 && j == 3)
          BOOST_CHECK_CLOSE(H(i, j), 1.0, tol);
        else
          BOOST_CHECK_CLOSE(H.el(i, j), 0.0, tol);
      }
    }
  };
  void test_calc_Hx()
  {
    MPI_Comm communicator = MPI_COMM_WORLD;

    boost::property_tree::ptree database;
    database.put("import_mesh", false);
    database.put("length", 1);
    database.put("length_divisions", 2);
    database.put("height", 1);
    database.put("height_divisions", 2);
    adamantine::Geometry<2> geometry(communicator, database);
    dealii::parallel::distributed::Triangulation<2> const &tria =
        geometry.get_triangulation();

    dealii::FE_Q<2> fe(1);
    dealii::DoFHandler<2> dof_handler(tria);
    dof_handler.distribute_dofs(fe);

    int sim_size = 4;
    int expt_size = 3;

    dealii::LA::distributed::Vector<double> sim_vec(dof_handler.n_dofs());
    sim_vec(0) = 2.0;
    sim_vec(1) = 4.0;
    sim_vec(2) = 5.0;
    sim_vec(3) = 7.0;

    dealii::Vector<double> expt_vec(3);
    expt_vec(0) = 2.5;
    expt_vec(1) = 4.5;
    expt_vec(2) = 8.5;

    std::pair<std::vector<int>, std::vector<int>> indices_and_offsets;
    indices_and_offsets.first.resize(3);
    indices_and_offsets.second.resize(4); // Offset vector is one longer
    indices_and_offsets.first[0] = 0;
    indices_and_offsets.first[1] = 1;
    indices_and_offsets.first[2] = 3;
    indices_and_offsets.second[0] = 0;
    indices_and_offsets.second[1] = 1;
    indices_and_offsets.second[2] = 2;
    indices_and_offsets.second[3] = 3;

    boost::property_tree::ptree solver_settings_database;
    DataAssimilator da(solver_settings_database);
    da._sim_size = sim_size;
    da._expt_size = expt_size;
    da.update_dof_mapping<2>(dof_handler, indices_and_offsets);
    dealii::Vector<double> Hx = da.calc_Hx(sim_vec);

    double tol = 1e-10;
    BOOST_CHECK_CLOSE(Hx(0), 2.0, tol);
    BOOST_CHECK_CLOSE(Hx(1), 4.0, tol);
    BOOST_CHECK_CLOSE(Hx(2), 7.0, tol);
  };

  void test_update_covariance_sparsity_pattern()
  {
    MPI_Comm communicator = MPI_COMM_WORLD;

    boost::property_tree::ptree database;
    database.put("import_mesh", false);
    database.put("length", 1);
    database.put("length_divisions", 2);
    database.put("height", 1);
    database.put("height_divisions", 2);
    adamantine::Geometry<2> geometry(communicator, database);
    dealii::parallel::distributed::Triangulation<2> const &tria =
        geometry.get_triangulation();

    dealii::FE_Q<2> fe(1);
    dealii::DoFHandler<2> dof_handler(tria);
    dof_handler.distribute_dofs(fe);

    // Effectively a dense matrix
    boost::property_tree::ptree solver_settings_database;
    DataAssimilator da(solver_settings_database);
    da._localization_cutoff_distance = 100.0;
    da._localization_cutoff_function = LocalizationCutoff::step_function;
    da.update_covariance_sparsity_pattern<2>(dof_handler);
    BOOST_CHECK(da._covariance_sparsity_pattern.n_nonzero_elements() == 81);

    // Sparse diagonal matrix
    da._localization_cutoff_distance = 1.0e-6;
    da._localization_cutoff_function = LocalizationCutoff::step_function;
    da.update_covariance_sparsity_pattern<2>(dof_handler);
    BOOST_CHECK(da._covariance_sparsity_pattern.n_nonzero_elements() == 9);

    // More general sparse matrix, cuts off interactions between the corners
    // of the domain
    da._localization_cutoff_distance = 1.2;
    da._localization_cutoff_function = LocalizationCutoff::step_function;
    da.update_covariance_sparsity_pattern<2>(dof_handler);
    BOOST_CHECK(da._covariance_sparsity_pattern.n_nonzero_elements() == 77);
  }

  void test_calc_sample_covariance_sparse()
  {
    MPI_Comm communicator = MPI_COMM_WORLD;

    boost::property_tree::ptree database;
    database.put("import_mesh", false);
    database.put("length", 1);
    database.put("length_divisions", 1);
    database.put("height", 1);
    database.put("height_divisions", 1);
    adamantine::Geometry<2> geometry(communicator, database);
    dealii::parallel::distributed::Triangulation<2> const &tria =
        geometry.get_triangulation();

    dealii::FE_Q<2> fe(1);
    dealii::DoFHandler<2> dof_handler(tria);
    dof_handler.distribute_dofs(fe);

    // Trivial case of identical vectors, effectively dense, covariance should
    // be the zero matrix
    dealii::LA::distributed::Vector<double> sim_vec(dof_handler.n_dofs());
    sim_vec(0) = 2.0;
    sim_vec(1) = 4.0;
    sim_vec(2) = 5.0;
    sim_vec(3) = 7.0;

    std::vector<dealii::LA::distributed::Vector<double>> vec_ensemble;
    vec_ensemble.push_back(sim_vec);
    vec_ensemble.push_back(sim_vec);

    boost::property_tree::ptree solver_settings_database;
    solver_settings_database.put("localization_cutoff_distance", 100.0);
    solver_settings_database.put("localization_cutoff_function",
                                 "step_function");
    DataAssimilator da(solver_settings_database);
    da._sim_size = sim_vec.size();
    da._num_ensemble_members = vec_ensemble.size();

    da.update_covariance_sparsity_pattern<2>(dof_handler);

    dealii::SparseMatrix<double> cov =
        da.calc_sample_covariance_sparse(vec_ensemble);

    // Check results
    double tol = 1e-10;
    for (unsigned int i = 0; i < 4; ++i)
    {
      for (unsigned int j = 0; j < 4; ++j)
      {
        BOOST_CHECK_SMALL(std::abs(cov(i, j)), tol);
      }
    }

    // Non-trivial case, still effectively dense, using NumPy solution as the
    // reference
    dealii::LA::distributed::Vector<double> sim_vec1(dof_handler.n_dofs());
    sim_vec1(0) = 2.1;
    sim_vec1(1) = 4.3;
    sim_vec1(2) = 5.2;
    sim_vec1(3) = 7.4;

    std::vector<dealii::LA::distributed::Vector<double>> vec_ensemble1;
    vec_ensemble1.push_back(sim_vec);
    vec_ensemble1.push_back(sim_vec1);

    dealii::SparseMatrix<double> cov1 =
        da.calc_sample_covariance_sparse(vec_ensemble1);

    BOOST_CHECK_CLOSE(cov1(0, 0), 0.005, tol);
    BOOST_CHECK_CLOSE(cov1(0, 1), 0.015, tol);
    BOOST_CHECK_CLOSE(cov1(0, 2), 0.01, tol);
    BOOST_CHECK_CLOSE(cov1(0, 3), 0.02, tol);
    BOOST_CHECK_CLOSE(cov1(1, 0), 0.015, tol);
    BOOST_CHECK_CLOSE(cov1(1, 1), 0.045, tol);
    BOOST_CHECK_CLOSE(cov1(1, 2), 0.03, tol);
    BOOST_CHECK_CLOSE(cov1(1, 3), 0.06, tol);
    BOOST_CHECK_CLOSE(cov1(2, 0), 0.01, tol);
    BOOST_CHECK_CLOSE(cov1(2, 1), 0.03, tol);
    BOOST_CHECK_CLOSE(cov1(2, 2), 0.02, tol);
    BOOST_CHECK_CLOSE(cov1(2, 3), 0.04, tol);
    BOOST_CHECK_CLOSE(cov1(3, 0), 0.02, tol);
    BOOST_CHECK_CLOSE(cov1(3, 1), 0.06, tol);
    BOOST_CHECK_CLOSE(cov1(3, 2), 0.04, tol);
    BOOST_CHECK_CLOSE(cov1(3, 3), 0.08, tol);

    // Non-trivial case with step-function sparsity
    da._localization_cutoff_distance = 1.0e-6;
    da.update_covariance_sparsity_pattern<2>(dof_handler);
    BOOST_CHECK(da._covariance_sparsity_pattern.n_nonzero_elements() == 4);
    dealii::SparseMatrix<double> cov2 =
        da.calc_sample_covariance_sparse(vec_ensemble1);

    BOOST_CHECK_CLOSE(cov2.el(0, 0), 0.005, tol);
    BOOST_CHECK_CLOSE(cov2.el(0, 1), 0.0, tol);
    BOOST_CHECK_CLOSE(cov2.el(0, 2), 0.0, tol);
    BOOST_CHECK_CLOSE(cov2.el(0, 3), 0.0, tol);
    BOOST_CHECK_CLOSE(cov2.el(1, 0), 0.0, tol);
    BOOST_CHECK_CLOSE(cov2.el(1, 1), 0.045, tol);
    BOOST_CHECK_CLOSE(cov2.el(1, 2), 0.0, tol);
    BOOST_CHECK_CLOSE(cov2.el(1, 3), 0.0, tol);
    BOOST_CHECK_CLOSE(cov2.el(2, 0), 0.0, tol);
    BOOST_CHECK_CLOSE(cov2.el(2, 1), 0.0, tol);
    BOOST_CHECK_CLOSE(cov2.el(2, 2), 0.02, tol);
    BOOST_CHECK_CLOSE(cov2.el(2, 3), 0.0, tol);
    BOOST_CHECK_CLOSE(cov2.el(3, 0), 0.0, tol);
    BOOST_CHECK_CLOSE(cov2.el(3, 1), 0.0, tol);
    BOOST_CHECK_CLOSE(cov2.el(3, 2), 0.0, tol);
    BOOST_CHECK_CLOSE(cov2.el(3, 3), 0.08, tol);

    // Non-trivial case with Gaspari-Cohn sparsity
    da._localization_cutoff_distance = 3.0;
    da._localization_cutoff_function = LocalizationCutoff::gaspari_cohn;
    da.update_covariance_sparsity_pattern<2>(dof_handler);
    BOOST_CHECK(da._covariance_sparsity_pattern.n_nonzero_elements() == 16);
    dealii::SparseMatrix<double> cov3 =
        da.calc_sample_covariance_sparse(vec_ensemble1);

    BOOST_CHECK_CLOSE(cov3.el(0, 0), 0.005, tol);
    BOOST_CHECK(cov3.el(0, 1) > 0.0 && cov3.el(0, 1) < 0.015);
    BOOST_CHECK(cov3.el(0, 2) > 0.0 && cov3.el(0, 2) < 0.01);
    BOOST_CHECK(cov3.el(0, 3) > 0.0 && cov3.el(0, 3) < 0.02);
    BOOST_CHECK(cov3.el(1, 0) > 0.0 && cov3.el(1, 0) < 0.015);
    BOOST_CHECK_CLOSE(cov3.el(1, 1), 0.045, tol);
    BOOST_CHECK(cov3.el(1, 2) > 0.0 && cov3.el(1, 2) < 0.03);
    BOOST_CHECK(cov3.el(1, 3) > 0.0 && cov3.el(1, 3) < 0.06);
    BOOST_CHECK(cov3.el(2, 0) > 0.0 && cov3.el(2, 0) < 0.01);
    BOOST_CHECK(cov3.el(2, 1) > 0.0 && cov3.el(2, 1) < 0.03);
    BOOST_CHECK_CLOSE(cov3.el(2, 2), 0.02, tol);
    BOOST_CHECK(cov3.el(2, 3) > 0.0 && cov3.el(2, 3) < 0.04);
    BOOST_CHECK(cov3.el(3, 0) > 0.0 && cov3.el(3, 0) < 0.02);
    BOOST_CHECK(cov3.el(3, 1) > 0.0 && cov3.el(3, 1) < 0.06);
    BOOST_CHECK(cov3.el(3, 2) > 0.0 && cov3.el(3, 2) < 0.04);
    BOOST_CHECK_CLOSE(cov3.el(3, 3), 0.08, tol);
  };

  void test_fill_noise_vector(bool R_is_diagonal)
  {
    if (R_is_diagonal)
    {
      boost::property_tree::ptree solver_settings_database;
      DataAssimilator da(solver_settings_database);

      dealii::SparsityPattern pattern(3, 3, 1);
      pattern.add(0, 0);
      pattern.add(1, 1);
      pattern.add(2, 2);
      pattern.compress();

      dealii::SparseMatrix<double> R(pattern);

      R.add(0, 0, 0.1);
      R.add(1, 1, 1.0);
      R.add(2, 2, 0.2);

      std::vector<dealii::Vector<double>> data;
      dealii::Vector<double> ensemble_member(3);
      for (unsigned int i = 0; i < 1000; ++i)
      {
        da.fill_noise_vector(ensemble_member, R, R_is_diagonal);
        data.push_back(ensemble_member);
      }

      dealii::FullMatrix<double> Rtest = calc_sample_covariance_dense(data);

      double tol =
          20.; // Loose 20% tolerance because this is a statistical check
      BOOST_CHECK_CLOSE(R(0, 0), Rtest(0, 0), tol);
      BOOST_CHECK_CLOSE(R(1, 1), Rtest(1, 1), tol);
      BOOST_CHECK_CLOSE(R(2, 2), Rtest(2, 2), tol);
    }
    else
    {
      boost::property_tree::ptree solver_settings_database;
      DataAssimilator da(solver_settings_database);

      dealii::SparsityPattern pattern(3, 3, 3);
      pattern.add(0, 0);
      pattern.add(1, 0);
      pattern.add(1, 1);
      pattern.add(0, 1);
      pattern.add(2, 2);
      pattern.compress();

      dealii::SparseMatrix<double> R(pattern);

      R.add(0, 0, 0.1);
      R.add(1, 0, 0.3);
      R.add(1, 1, 1.0);
      R.add(0, 1, 0.3);
      R.add(2, 2, 0.2);

      std::vector<dealii::Vector<double>> data;
      dealii::Vector<double> ensemble_member(3);
      for (unsigned int i = 0; i < 1000; ++i)
      {
        da.fill_noise_vector(ensemble_member, R, R_is_diagonal);
        data.push_back(ensemble_member);
      }

      dealii::FullMatrix<double> Rtest = calc_sample_covariance_dense(data);

      double tol =
          20.; // Loose 20% tolerance because this is a statistical check
      BOOST_CHECK_CLOSE(R(0, 0), Rtest(0, 0), tol);
      BOOST_CHECK_CLOSE(R(1, 0), Rtest(1, 0), tol);
      BOOST_CHECK_CLOSE(R(1, 1), Rtest(1, 1), tol);
      BOOST_CHECK_CLOSE(R(0, 1), Rtest(0, 1), tol);
      BOOST_CHECK_CLOSE(R(2, 2), Rtest(2, 2), tol);
    }
  }; // namespace adamantine

  void test_update_ensemble()
  {
    // Create the DoF mapping
    MPI_Comm communicator = MPI_COMM_WORLD;

    boost::property_tree::ptree database;
    database.put("import_mesh", false);
    database.put("length", 1);
    database.put("length_divisions", 1);
    database.put("height", 1);
    database.put("height_divisions", 1);
    adamantine::Geometry<2> geometry(communicator, database);
    dealii::parallel::distributed::Triangulation<2> const &tria =
        geometry.get_triangulation();

    dealii::FE_Q<2> fe(1);
    dealii::DoFHandler<2> dof_handler(tria);
    dof_handler.distribute_dofs(fe);

    int sim_size = 4;
    int expt_size = 2;

    std::vector<double> expt_vec(2);
    expt_vec[0] = 2.5;
    expt_vec[1] = 9.5;

    std::pair<std::vector<int>, std::vector<int>> indices_and_offsets;
    indices_and_offsets.first.resize(2);
    indices_and_offsets.second.resize(3); // Offset vector is one longer
    indices_and_offsets.first[0] = 1;
    indices_and_offsets.first[1] = 3;
    indices_and_offsets.second[0] = 0;
    indices_and_offsets.second[1] = 1;
    indices_and_offsets.second[2] = 2;

    boost::property_tree::ptree solver_settings_database;
    DataAssimilator da(solver_settings_database);
    da._sim_size = sim_size;
    da._parameter_size = 0;
    da._expt_size = expt_size;
    da._num_ensemble_members = 3;

    da.update_covariance_sparsity_pattern<2>(dof_handler);
    da.update_dof_mapping<2>(dof_handler, indices_and_offsets);

    // Create the simulation data
    std::vector<dealii::LA::distributed::BlockVector<double>>
        augmented_state_ensemble(3);

    augmented_state_ensemble[0].reinit(2);
    augmented_state_ensemble[0].block(0).reinit(4);
    augmented_state_ensemble[0].block(0)(0) = 1.0;
    augmented_state_ensemble[0].block(0)(1) = 3.0;
    augmented_state_ensemble[0].block(0)(2) = 6.0;
    augmented_state_ensemble[0].block(0)(3) = 9.0;
    augmented_state_ensemble[0].collect_sizes();
    augmented_state_ensemble[1].reinit(2);
    augmented_state_ensemble[1].block(0).reinit(4);
    augmented_state_ensemble[1].block(0)(0) = 1.5;
    augmented_state_ensemble[1].block(0)(1) = 3.2;
    augmented_state_ensemble[1].block(0)(2) = 6.3;
    augmented_state_ensemble[1].block(0)(3) = 9.7;
    augmented_state_ensemble[1].collect_sizes();
    augmented_state_ensemble[2].reinit(2);
    augmented_state_ensemble[2].block(0).reinit(4);
    augmented_state_ensemble[2].block(0)(0) = 1.1;
    augmented_state_ensemble[2].block(0)(1) = 3.1;
    augmented_state_ensemble[2].block(0)(2) = 6.1;
    augmented_state_ensemble[2].block(0)(3) = 9.1;
    augmented_state_ensemble[2].collect_sizes();

    // Build the sparse experimental covariance matrix
    dealii::SparsityPattern pattern(expt_size, expt_size, 1);
    pattern.add(0, 0);
    pattern.add(1, 1);
    pattern.compress();

    dealii::SparseMatrix<double> R(pattern);
    R.add(0, 0, 0.002);
    R.add(1, 1, 0.001);

    // Save the data at the observation points before assimilation
    std::vector<double> sim_at_expt_pt_1_before(3);
    sim_at_expt_pt_1_before.push_back(augmented_state_ensemble[0].block(0)[1]);
    sim_at_expt_pt_1_before.push_back(augmented_state_ensemble[1].block(0)[1]);
    sim_at_expt_pt_1_before.push_back(augmented_state_ensemble[2].block(0)[1]);

    std::vector<double> sim_at_expt_pt_2_before(3);
    sim_at_expt_pt_2_before.push_back(augmented_state_ensemble[0].block(0)[3]);
    sim_at_expt_pt_2_before.push_back(augmented_state_ensemble[1].block(0)[3]);
    sim_at_expt_pt_2_before.push_back(augmented_state_ensemble[2].block(0)[3]);

    // Update the simulation data
    da.update_ensemble(communicator, augmented_state_ensemble, expt_vec, R);

    // Save the data at the observation points after assimilation
    std::vector<double> sim_at_expt_pt_1_after(3);
    sim_at_expt_pt_1_after.push_back(augmented_state_ensemble[0].block(0)[1]);
    sim_at_expt_pt_1_after.push_back(augmented_state_ensemble[1].block(0)[1]);
    sim_at_expt_pt_1_after.push_back(augmented_state_ensemble[2].block(0)[1]);

    std::vector<double> sim_at_expt_pt_2_after(3);
    sim_at_expt_pt_2_after.push_back(augmented_state_ensemble[0].block(0)[3]);
    sim_at_expt_pt_2_after.push_back(augmented_state_ensemble[1].block(0)[3]);
    sim_at_expt_pt_2_after.push_back(augmented_state_ensemble[2].block(0)[3]);

    // Check the solution
    // The observed points should get closer to the experimental values
    // Large entries in R could make these fail spuriously
    for (int member = 0; member < 3; ++member)
    {
      BOOST_CHECK(std::abs(expt_vec[0] - sim_at_expt_pt_1_after[member]) <=
                  std::abs(expt_vec[0] - sim_at_expt_pt_1_before[member]));
      BOOST_CHECK(std::abs(expt_vec[1] - sim_at_expt_pt_2_after[member]) <=
                  std::abs(expt_vec[1] - sim_at_expt_pt_2_before[member]));
    }
  };

private:
  template <typename VectorType>
  dealii::FullMatrix<double>
  calc_sample_covariance_dense(std::vector<VectorType> vec_ensemble) const
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

    // FIXME This can be a problem for even moderately sized meshes
    dealii::FullMatrix<double> cov(vec_size);

    anomaly.mTmult(cov, anomaly);

    return cov;
  }
};

BOOST_AUTO_TEST_CASE(data_assimilator)
{
  DataAssimilatorTester dat;

  dat.test_constructor();
  dat.test_update_dof_mapping();
  dat.test_update_covariance_sparsity_pattern();
  dat.test_calc_sample_covariance_sparse();
  dat.test_fill_noise_vector(true);
  dat.test_fill_noise_vector(false);
  dat.test_calc_H();
  dat.test_calc_Hx();
  dat.test_calc_kalman_gain();
  dat.test_update_ensemble();
}
} // namespace adamantine
