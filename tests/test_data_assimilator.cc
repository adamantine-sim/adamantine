/* SPDX-FileCopyrightText: Copyright (c) 2021 - 2024, the adamantine authors.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#define BOOST_TEST_MODULE DataAssimilator

#include <DataAssimilator.hh>
#include <Geometry.hh>

#include <deal.II/fe/fe_q.h>
#include <deal.II/lac/la_parallel_vector.h>

#include "main.cc"

namespace tt = boost::test_tools;

namespace adamantine
{
class DataAssimilatorTester
{
public:
  void test_constructor()
  {
    boost::property_tree::ptree database;

    // First checking the dealii default values
    DataAssimilator da0(MPI_COMM_WORLD, MPI_COMM_WORLD, 0, database);

    double tol = 1.0e-12;
    BOOST_TEST(da0._solver_control.tolerance() - 1.0e-10 == 0.,
               tt::tolerance(tol));
    BOOST_TEST(da0._solver_control.max_steps() == 100u);
    BOOST_TEST(da0._additional_data.max_n_tmp_vectors == 30u);

    // Now explicitly setting them
    database.put("solver.convergence_tolerance", 1.0e-6);
    database.put("solver.max_iterations", 25);
    database.put("solver.max_number_of_temp_vectors", 4);
    DataAssimilator da1(MPI_COMM_WORLD, MPI_COMM_WORLD, 0, database);
    BOOST_TEST(da1._solver_control.tolerance() - 1.0e-6 == 0.,
               tt::tolerance(tol));
    BOOST_TEST(da1._solver_control.max_steps() == 25u);
    BOOST_TEST(da1._additional_data.max_n_tmp_vectors == 4u);
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
    boost::optional<boost::property_tree::ptree const &>
        units_optional_database;
    adamantine::Geometry<2> geometry(communicator, database,
                                     units_optional_database);
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

    std::pair<std::vector<int>, std::vector<int>> expt_to_dof_mapping;
    expt_to_dof_mapping.first.resize(2);
    expt_to_dof_mapping.second.resize(2);
    expt_to_dof_mapping.first[0] = 0;
    expt_to_dof_mapping.first[1] = 1;
    expt_to_dof_mapping.second[0] = 1;
    expt_to_dof_mapping.second[1] = 3;

    boost::property_tree::ptree solver_settings_database;
    DataAssimilator da(communicator, communicator, 0, solver_settings_database);
    da._sim_size = sim_size;
    da._expt_size = expt_size;
    da._parameter_size = 0;
    da._num_ensemble_members = 3;
    da.update_dof_mapping<2>(expt_to_dof_mapping);
    da.update_covariance_sparsity_pattern<2>(dof_handler, 0);

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
    BOOST_TEST(forecast_shift[0][0] == 0.21352564, tt::tolerance(tol));
    BOOST_TEST(forecast_shift[0][1] == -0.14600986, tt::tolerance(tol));
    BOOST_TEST(forecast_shift[0][2] == -0.02616469, tt::tolerance(tol));
    BOOST_TEST(forecast_shift[0][3] == 0.45321598, tt::tolerance(tol));
    BOOST_TEST(forecast_shift[1][0] == -0.27786325, tt::tolerance(tol));
    BOOST_TEST(forecast_shift[1][1] == -0.32946285, tt::tolerance(tol));
    BOOST_TEST(forecast_shift[1][2] == -0.31226298, tt::tolerance(tol));
    BOOST_TEST(forecast_shift[1][3] == -0.24346351, tt::tolerance(tol));
    BOOST_TEST(forecast_shift[2][0] == 0.12767094, tt::tolerance(tol));
    BOOST_TEST(forecast_shift[2][1] == -0.20319395, tt::tolerance(tol));
    BOOST_TEST(forecast_shift[2][2] == -0.09290565, tt::tolerance(tol));
    BOOST_TEST(forecast_shift[2][3] == 0.34824753, tt::tolerance(tol));
  };

  void test_update_dof_mapping()
  {
    unsigned int sim_size = 4;
    unsigned int expt_size = 3;

    std::pair<std::vector<int>, std::vector<int>> expt_to_dof_mapping;
    expt_to_dof_mapping.first.resize(3);
    expt_to_dof_mapping.second.resize(3);
    expt_to_dof_mapping.first[0] = 0;
    expt_to_dof_mapping.first[1] = 1;
    expt_to_dof_mapping.first[2] = 2;
    expt_to_dof_mapping.second[0] = 0;
    expt_to_dof_mapping.second[1] = 1;
    expt_to_dof_mapping.second[2] = 3;

    boost::property_tree::ptree solver_settings_database;
    DataAssimilator da(MPI_COMM_WORLD, MPI_COMM_WORLD, 0,
                       solver_settings_database);
    da._sim_size = sim_size;
    da._expt_size = expt_size;
    da.update_dof_mapping<2>(expt_to_dof_mapping);

    BOOST_TEST(da._expt_to_dof_mapping.first[0] == 0);
    BOOST_TEST(da._expt_to_dof_mapping.first[1] == 1);
    BOOST_TEST(da._expt_to_dof_mapping.first[2] == 2);
    BOOST_TEST(da._expt_to_dof_mapping.second[0] == 0);
    BOOST_TEST(da._expt_to_dof_mapping.second[1] == 1);
    BOOST_TEST(da._expt_to_dof_mapping.second[2] == 3);
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
    boost::optional<boost::property_tree::ptree const &>
        units_optional_database;
    adamantine::Geometry<2> geometry(communicator, database,
                                     units_optional_database);
    dealii::parallel::distributed::Triangulation<2> const &tria =
        geometry.get_triangulation();

    dealii::FE_Q<2> fe(1);
    dealii::DoFHandler<2> dof_handler(tria);
    dof_handler.distribute_dofs(fe);

    unsigned int sim_size = 4;
    unsigned int expt_size = 3;

    std::pair<std::vector<int>, std::vector<int>> expt_to_dof_mapping;
    expt_to_dof_mapping.first.resize(3);
    expt_to_dof_mapping.second.resize(3);
    expt_to_dof_mapping.first[0] = 0;
    expt_to_dof_mapping.first[1] = 1;
    expt_to_dof_mapping.first[2] = 2;
    expt_to_dof_mapping.second[0] = 0;
    expt_to_dof_mapping.second[1] = 1;
    expt_to_dof_mapping.second[2] = 3;

    boost::property_tree::ptree solver_settings_database;
    DataAssimilator da(communicator, communicator, 0, solver_settings_database);
    da._sim_size = sim_size;
    da._expt_size = expt_size;
    da.update_dof_mapping<2>(expt_to_dof_mapping);

    dealii::SparsityPattern pattern(expt_size, sim_size, expt_size);

    dealii::SparseMatrix<double> H = da.calc_H(pattern);

    double tol = 1e-12;
    for (unsigned int i = 0; i < expt_size; ++i)
    {
      for (unsigned int j = 0; j < sim_size; ++j)
      {
        if (i == 0 && j == 0)
          BOOST_TEST(H(i, j) == 1.0, tt::tolerance(tol));
        else if (i == 1 && j == 1)
          BOOST_TEST(H(i, j) == 1.0, tt::tolerance(tol));
        else if (i == 2 && j == 3)
          BOOST_TEST(H(i, j) == 1.0, tt::tolerance(tol));
        else
          BOOST_TEST(H.el(i, j) == 0.0, tt::tolerance(tol));
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
    boost::optional<boost::property_tree::ptree const &>
        units_optional_database;
    adamantine::Geometry<2> geometry(communicator, database,
                                     units_optional_database);
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

    std::pair<std::vector<int>, std::vector<int>> expt_to_dof_mapping;
    expt_to_dof_mapping.first.resize(3);
    expt_to_dof_mapping.second.resize(3);
    expt_to_dof_mapping.first[0] = 0;
    expt_to_dof_mapping.first[1] = 1;
    expt_to_dof_mapping.first[2] = 2;
    expt_to_dof_mapping.second[0] = 0;
    expt_to_dof_mapping.second[1] = 1;
    expt_to_dof_mapping.second[2] = 3;

    boost::property_tree::ptree solver_settings_database;
    DataAssimilator da(communicator, communicator, 0, solver_settings_database);
    da._sim_size = sim_size;
    da._expt_size = expt_size;
    da.update_dof_mapping<2>(expt_to_dof_mapping);
    dealii::Vector<double> Hx = da.calc_Hx(sim_vec);

    double tol = 1e-10;
    BOOST_TEST(Hx(0) == 2.0, tt::tolerance(tol));
    BOOST_TEST(Hx(1) == 4.0, tt::tolerance(tol));
    BOOST_TEST(Hx(2) == 7.0, tt::tolerance(tol));
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
    boost::optional<boost::property_tree::ptree const &>
        units_optional_database;
    adamantine::Geometry<2> geometry(communicator, database,
                                     units_optional_database);
    dealii::parallel::distributed::Triangulation<2> const &tria =
        geometry.get_triangulation();

    dealii::FE_Q<2> fe(1);
    dealii::DoFHandler<2> dof_handler(tria);
    dof_handler.distribute_dofs(fe);

    // Effectively a dense matrix
    boost::property_tree::ptree solver_settings_database;
    DataAssimilator da(communicator, communicator, 0, solver_settings_database);
    da._localization_cutoff_distance = 100.0;
    da._localization_cutoff_function = LocalizationCutoff::step_function;
    da.update_covariance_sparsity_pattern<2>(dof_handler, 0);
    BOOST_TEST(da._covariance_sparsity_pattern.n_nonzero_elements() == 81u);

    // Sparse diagonal matrix
    da._localization_cutoff_distance = 1.0e-6;
    da._localization_cutoff_function = LocalizationCutoff::step_function;
    da.update_covariance_sparsity_pattern<2>(dof_handler, 0);
    BOOST_TEST(da._covariance_sparsity_pattern.n_nonzero_elements() == 9u);

    // More general sparse matrix, cuts off interactions between the corners
    // of the domain
    da._localization_cutoff_distance = 1.2;
    da._localization_cutoff_function = LocalizationCutoff::step_function;
    da.update_covariance_sparsity_pattern<2>(dof_handler, 0);
    BOOST_TEST(da._covariance_sparsity_pattern.n_nonzero_elements() == 77u);

    // Sparse diagonal matrix with two augmentation parameters
    da._localization_cutoff_distance = 1.0e-6;
    da._localization_cutoff_function = LocalizationCutoff::step_function;
    da.update_covariance_sparsity_pattern<2>(dof_handler, 2);
    BOOST_TEST(da._covariance_sparsity_pattern.n_nonzero_elements() == 49u);
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
    boost::optional<boost::property_tree::ptree const &>
        units_optional_database;
    adamantine::Geometry<2> geometry(communicator, database,
                                     units_optional_database);
    dealii::parallel::distributed::Triangulation<2> const &tria =
        geometry.get_triangulation();

    dealii::FE_Q<2> fe(1);
    dealii::DoFHandler<2> dof_handler(tria);
    dof_handler.distribute_dofs(fe);

    // Trivial case of identical vectors, effectively dense, covariance should
    // be the zero matrix
    dealii::LA::distributed::BlockVector<double> sim_vec(1,
                                                         dof_handler.n_dofs());
    sim_vec[0] = 2.0;
    sim_vec[1] = 4.0;
    sim_vec[2] = 5.0;
    sim_vec[3] = 7.0;

    std::vector<dealii::LA::distributed::BlockVector<double>> vec_ensemble;
    vec_ensemble.push_back(sim_vec);
    vec_ensemble.push_back(sim_vec);

    boost::property_tree::ptree solver_settings_database;
    solver_settings_database.put("localization_cutoff_distance", 100.0);
    solver_settings_database.put("localization_cutoff_function",
                                 "step_function");
    DataAssimilator da(communicator, communicator, 0, solver_settings_database);
    da._sim_size = sim_vec.size();
    da._num_ensemble_members = vec_ensemble.size();

    da.update_covariance_sparsity_pattern<2>(dof_handler, 0);

    auto cov = da.calc_sample_covariance_sparse(vec_ensemble);

    // Check results
    double tol = 1e-10;
    for (unsigned int i = 0; i < 4; ++i)
    {
      for (unsigned int j = 0; j < 4; ++j)
      {
        BOOST_TEST(cov.el(i, j) == 0., tt::tolerance(tol));
      }
    }

    // Non-trivial case, still effectively dense, using NumPy solution as the
    // reference
    dealii::LA::distributed::BlockVector<double> sim_vec1(1,
                                                          dof_handler.n_dofs());
    sim_vec1(0) = 2.1;
    sim_vec1(1) = 4.3;
    sim_vec1(2) = 5.2;
    sim_vec1(3) = 7.4;

    std::vector<dealii::LA::distributed::BlockVector<double>> vec_ensemble1;
    vec_ensemble1.push_back(sim_vec);
    vec_ensemble1.push_back(sim_vec1);

    auto cov1 = da.calc_sample_covariance_sparse(vec_ensemble1);

    BOOST_TEST(cov1.el(0, 0) == 0.005, tt::tolerance(tol));
    BOOST_TEST(cov1.el(0, 1) == 0.015, tt::tolerance(tol));
    BOOST_TEST(cov1.el(0, 2) == 0.01, tt::tolerance(tol));
    BOOST_TEST(cov1.el(0, 3) == 0.02, tt::tolerance(tol));
    BOOST_TEST(cov1.el(1, 0) == 0.015, tt::tolerance(tol));
    BOOST_TEST(cov1.el(1, 1) == 0.045, tt::tolerance(tol));
    BOOST_TEST(cov1.el(1, 2) == 0.03, tt::tolerance(tol));
    BOOST_TEST(cov1.el(1, 3) == 0.06, tt::tolerance(tol));
    BOOST_TEST(cov1.el(2, 0) == 0.01, tt::tolerance(tol));
    BOOST_TEST(cov1.el(2, 1) == 0.03, tt::tolerance(tol));
    BOOST_TEST(cov1.el(2, 2) == 0.02, tt::tolerance(tol));
    BOOST_TEST(cov1.el(2, 3) == 0.04, tt::tolerance(tol));
    BOOST_TEST(cov1.el(3, 0) == 0.02, tt::tolerance(tol));
    BOOST_TEST(cov1.el(3, 1) == 0.06, tt::tolerance(tol));
    BOOST_TEST(cov1.el(3, 2) == 0.04, tt::tolerance(tol));
    BOOST_TEST(cov1.el(3, 3) == 0.08, tt::tolerance(tol));

    // Non-trivial case with step-function sparsity
    da._localization_cutoff_distance = 1.0e-6;
    da.update_covariance_sparsity_pattern<2>(dof_handler, 0);
    BOOST_TEST(da._covariance_sparsity_pattern.n_nonzero_elements() == 4);
    auto cov2 = da.calc_sample_covariance_sparse(vec_ensemble1);

    BOOST_TEST(cov2.el(0, 0) == 0.005, tt::tolerance(tol));
    BOOST_TEST(cov2.el(0, 1) == 0.0, tt::tolerance(tol));
    BOOST_TEST(cov2.el(0, 2) == 0.0, tt::tolerance(tol));
    BOOST_TEST(cov2.el(0, 3) == 0.0, tt::tolerance(tol));
    BOOST_TEST(cov2.el(1, 0) == 0.0, tt::tolerance(tol));
    BOOST_TEST(cov2.el(1, 1) == 0.045, tt::tolerance(tol));
    BOOST_TEST(cov2.el(1, 2) == 0.0, tt::tolerance(tol));
    BOOST_TEST(cov2.el(1, 3) == 0.0, tt::tolerance(tol));
    BOOST_TEST(cov2.el(2, 0) == 0.0, tt::tolerance(tol));
    BOOST_TEST(cov2.el(2, 1) == 0.0, tt::tolerance(tol));
    BOOST_TEST(cov2.el(2, 2) == 0.02, tt::tolerance(tol));
    BOOST_TEST(cov2.el(2, 3) == 0.0, tt::tolerance(tol));
    BOOST_TEST(cov2.el(3, 0) == 0.0, tt::tolerance(tol));
    BOOST_TEST(cov2.el(3, 1) == 0.0, tt::tolerance(tol));
    BOOST_TEST(cov2.el(3, 2) == 0.0, tt::tolerance(tol));
    BOOST_TEST(cov2.el(3, 3) == 0.08, tt::tolerance(tol));

    // Non-trivial case with Gaspari-Cohn sparsity
    da._localization_cutoff_distance = 3.0;
    da._localization_cutoff_function = LocalizationCutoff::gaspari_cohn;
    da.update_covariance_sparsity_pattern<2>(dof_handler, 0);
    BOOST_TEST(da._covariance_sparsity_pattern.n_nonzero_elements() == 16);
    auto cov3 = da.calc_sample_covariance_sparse(vec_ensemble1);

    BOOST_TEST(cov3.el(0, 0) == 0.005, tt::tolerance(tol));
    BOOST_TEST(cov3.el(0, 1) > 0.0);
    BOOST_TEST(cov3.el(0, 1) < 0.015);
    BOOST_TEST(cov3.el(0, 2) > 0.0);
    BOOST_TEST(cov3.el(0, 2) < 0.01);
    BOOST_TEST(cov3.el(0, 3) > 0.0);
    BOOST_TEST(cov3.el(0, 3) < 0.02);
    BOOST_TEST(cov3.el(1, 0) > 0.0);
    BOOST_TEST(cov3.el(1, 0) < 0.015);
    BOOST_TEST(cov3.el(1, 1) == 0.045, tt::tolerance(tol));
    BOOST_TEST(cov3.el(1, 2) > 0.0);
    BOOST_TEST(cov3.el(1, 2) < 0.03);
    BOOST_TEST(cov3.el(1, 3) > 0.0);
    BOOST_TEST(cov3.el(1, 3) < 0.06);
    BOOST_TEST(cov3.el(2, 0) > 0.0);
    BOOST_TEST(cov3.el(2, 0) < 0.01);
    BOOST_TEST(cov3.el(2, 1) > 0.0);
    BOOST_TEST(cov3.el(2, 1) < 0.03);
    BOOST_TEST(cov3.el(2, 2) == 0.02, tt::tolerance(tol));
    BOOST_TEST(cov3.el(2, 3) > 0.0);
    BOOST_TEST(cov3.el(2, 3) < 0.04);
    BOOST_TEST(cov3.el(3, 0) > 0.0);
    BOOST_TEST(cov3.el(3, 0) < 0.02);
    BOOST_TEST(cov3.el(3, 1) > 0.0);
    BOOST_TEST(cov3.el(3, 1) < 0.06);
    BOOST_TEST(cov3.el(3, 2) > 0.0);
    BOOST_TEST(cov3.el(3, 2) < 0.04);
    BOOST_TEST(cov3.el(3, 3) == 0.08, tt::tolerance(tol));

    // Non-trivial case with step-function sparsity and two augmented parameters
    dealii::LA::distributed::BlockVector<double> sim_vec2(
        1, dof_handler.n_dofs() + 2);
    sim_vec2(0) = 2.0;
    sim_vec2(1) = 4.0;
    sim_vec2(2) = 5.0;
    sim_vec2(3) = 7.0;
    sim_vec2(4) = 1.0;
    sim_vec2(5) = 1.5;

    dealii::LA::distributed::BlockVector<double> sim_vec3(
        1, dof_handler.n_dofs() + 2);
    sim_vec3(0) = 2.1;
    sim_vec3(1) = 4.3;
    sim_vec3(2) = 5.2;
    sim_vec3(3) = 7.4;
    sim_vec3(4) = 1.1;
    sim_vec3(5) = 1.4;

    std::vector<dealii::LA::distributed::BlockVector<double>> vec_ensemble2;
    vec_ensemble2.push_back(sim_vec2);
    vec_ensemble2.push_back(sim_vec3);

    da._localization_cutoff_distance = 1.0e-6;
    da.update_covariance_sparsity_pattern<2>(dof_handler, 2);
    BOOST_TEST(da._covariance_sparsity_pattern.n_nonzero_elements() == 24u);
    auto cov4 = da.calc_sample_covariance_sparse(vec_ensemble2);

    BOOST_TEST(cov4.el(0, 0) == 0.005, tt::tolerance(tol));
    BOOST_TEST(cov4.el(0, 1) == 0.0, tt::tolerance(tol));
    BOOST_TEST(cov4.el(0, 2) == 0.0, tt::tolerance(tol));
    BOOST_TEST(cov4.el(0, 3) == 0.0, tt::tolerance(tol));
    BOOST_TEST(cov4.el(1, 0) == 0.0, tt::tolerance(tol));
    BOOST_TEST(cov4.el(1, 1) == 0.045, tt::tolerance(tol));
    BOOST_TEST(cov4.el(1, 2) == 0.0, tt::tolerance(tol));
    BOOST_TEST(cov4.el(1, 3) == 0.0, tt::tolerance(tol));
    BOOST_TEST(cov4.el(2, 0) == 0.0, tt::tolerance(tol));
    BOOST_TEST(cov4.el(2, 1) == 0.0, tt::tolerance(tol));
    BOOST_TEST(cov4.el(2, 2) == 0.02, tt::tolerance(tol));
    BOOST_TEST(cov4.el(2, 3) == 0.0, tt::tolerance(tol));
    BOOST_TEST(cov4.el(3, 0) == 0.0, tt::tolerance(tol));
    BOOST_TEST(cov4.el(3, 1) == 0.0, tt::tolerance(tol));
    BOOST_TEST(cov4.el(3, 2) == 0.0, tt::tolerance(tol));
    BOOST_TEST(cov4.el(3, 3) == 0.08, tt::tolerance(tol));

    BOOST_TEST(cov4.el(4, 0) == 0.005, tt::tolerance(tol));
    BOOST_TEST(cov4.el(4, 1) == 0.015, tt::tolerance(tol));
    BOOST_TEST(cov4.el(4, 2) == 0.01, tt::tolerance(tol));
    BOOST_TEST(cov4.el(4, 3) == 0.02, tt::tolerance(tol));
    BOOST_TEST(cov4.el(4, 4) == 0.005, tt::tolerance(tol));
    BOOST_TEST(cov4.el(4, 5) == -0.005, tt::tolerance(tol));

    BOOST_TEST(cov4.el(5, 0) == -0.005, tt::tolerance(tol));
    BOOST_TEST(cov4.el(5, 1) == -0.015, tt::tolerance(tol));
    BOOST_TEST(cov4.el(5, 2) == -0.01, tt::tolerance(tol));
    BOOST_TEST(cov4.el(5, 3) == -0.02, tt::tolerance(tol));
    BOOST_TEST(cov4.el(5, 4) == -0.005, tt::tolerance(tol));
    BOOST_TEST(cov4.el(5, 5) == 0.005, tt::tolerance(tol));
  };

  void test_fill_noise_vector(bool R_is_diagonal)
  {
    if (R_is_diagonal)
    {
      boost::property_tree::ptree solver_settings_database;
      DataAssimilator da(MPI_COMM_WORLD, MPI_COMM_WORLD, 0,
                         solver_settings_database);

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
      BOOST_TEST(R(0, 0) == Rtest(0, 0), tt::tolerance(tol));
      BOOST_TEST(R(1, 1) == Rtest(1, 1), tt::tolerance(tol));
      BOOST_TEST(R(2, 2) == Rtest(2, 2), tt::tolerance(tol));
    }
    else
    {
      boost::property_tree::ptree solver_settings_database;
      DataAssimilator da(MPI_COMM_WORLD, MPI_COMM_WORLD, 0,
                         solver_settings_database);

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
      BOOST_TEST(R(0, 0) == Rtest(0, 0), tt::tolerance(tol));
      BOOST_TEST(R(1, 0) == Rtest(1, 0), tt::tolerance(tol));
      BOOST_TEST(R(1, 1) == Rtest(1, 1), tt::tolerance(tol));
      BOOST_TEST(R(0, 1) == Rtest(0, 1), tt::tolerance(tol));
      BOOST_TEST(R(2, 2) == Rtest(2, 2), tt::tolerance(tol));
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
    boost::optional<boost::property_tree::ptree const &>
        units_optional_database;
    adamantine::Geometry<2> geometry(communicator, database,
                                     units_optional_database);
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

    std::pair<std::vector<int>, std::vector<int>> expt_to_dof_mapping;
    expt_to_dof_mapping.first.resize(2);
    expt_to_dof_mapping.second.resize(2);
    expt_to_dof_mapping.first[0] = 0;
    expt_to_dof_mapping.first[1] = 1;
    expt_to_dof_mapping.second[0] = 1;
    expt_to_dof_mapping.second[1] = 3;

    boost::property_tree::ptree solver_settings_database;
    DataAssimilator da(communicator, communicator, 0, solver_settings_database);
    da._sim_size = sim_size;
    da._parameter_size = 0;
    da._expt_size = expt_size;
    da._num_ensemble_members = 3;

    da.update_covariance_sparsity_pattern<2>(dof_handler, 0);
    da.update_dof_mapping<2>(expt_to_dof_mapping);

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
    da.update_ensemble(augmented_state_ensemble, expt_vec, R);

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
      BOOST_TEST(std::abs(expt_vec[0] - sim_at_expt_pt_1_after[member]) <=
                 std::abs(expt_vec[0] - sim_at_expt_pt_1_before[member]));
      BOOST_TEST(std::abs(expt_vec[1] - sim_at_expt_pt_2_after[member]) <=
                 std::abs(expt_vec[1] - sim_at_expt_pt_2_before[member]));
    }
  };

  void test_update_ensemble_augmented()
  {
    // Create the DoF mapping
    MPI_Comm communicator = MPI_COMM_WORLD;

    boost::property_tree::ptree database;
    database.put("import_mesh", false);
    database.put("length", 1);
    database.put("length_divisions", 1);
    database.put("height", 1);
    database.put("height_divisions", 1);
    boost::optional<boost::property_tree::ptree const &>
        units_optional_database;
    adamantine::Geometry<2> geometry(communicator, database,
                                     units_optional_database);
    dealii::parallel::distributed::Triangulation<2> const &tria =
        geometry.get_triangulation();

    dealii::FE_Q<2> fe(1);
    dealii::DoFHandler<2> dof_handler(tria);
    dof_handler.distribute_dofs(fe);

    int sim_size = 4;
    int parameter_size = 2;
    int expt_size = 2;

    std::vector<double> expt_vec(2);
    expt_vec[0] = 2.5;
    expt_vec[1] = 9.5;

    std::pair<std::vector<int>, std::vector<int>> expt_to_dof_mapping;
    expt_to_dof_mapping.first.resize(2);
    expt_to_dof_mapping.second.resize(2);
    expt_to_dof_mapping.first[0] = 0;
    expt_to_dof_mapping.first[1] = 1;
    expt_to_dof_mapping.second[0] = 1;
    expt_to_dof_mapping.second[1] = 3;

    boost::property_tree::ptree solver_settings_database;
    DataAssimilator da(communicator, communicator, 0, solver_settings_database);
    da._sim_size = sim_size;
    da._parameter_size = parameter_size;
    da._expt_size = expt_size;
    da._num_ensemble_members = 3;

    da.update_covariance_sparsity_pattern<2>(dof_handler, parameter_size);
    da.update_dof_mapping<2>(expt_to_dof_mapping);

    // Create the simulation data
    std::vector<dealii::LA::distributed::BlockVector<double>>
        augmented_state_ensemble(3);

    augmented_state_ensemble[0].reinit(2);
    augmented_state_ensemble[0].block(0).reinit(sim_size);
    augmented_state_ensemble[0].block(0)(0) = 1.0;
    augmented_state_ensemble[0].block(0)(1) = 3.0;
    augmented_state_ensemble[0].block(0)(2) = 6.0;
    augmented_state_ensemble[0].block(0)(3) = 9.0;
    augmented_state_ensemble[1].reinit(2);
    augmented_state_ensemble[1].block(0).reinit(sim_size);
    augmented_state_ensemble[1].block(0)(0) = 1.5;
    augmented_state_ensemble[1].block(0)(1) = 3.2;
    augmented_state_ensemble[1].block(0)(2) = 6.3;
    augmented_state_ensemble[1].block(0)(3) = 9.7;
    augmented_state_ensemble[2].reinit(2);
    augmented_state_ensemble[2].block(0).reinit(sim_size);
    augmented_state_ensemble[2].block(0)(0) = 1.1;
    augmented_state_ensemble[2].block(0)(1) = 3.1;
    augmented_state_ensemble[2].block(0)(2) = 6.1;
    augmented_state_ensemble[2].block(0)(3) = 9.1;

    augmented_state_ensemble[0].block(1).reinit(parameter_size);
    augmented_state_ensemble[0].block(1)(0) = 1.0;
    augmented_state_ensemble[0].block(1)(0) = 5.0;
    augmented_state_ensemble[1].block(1).reinit(parameter_size);
    augmented_state_ensemble[1].block(1)(0) = 1.2;
    augmented_state_ensemble[1].block(1)(0) = 4.5;
    augmented_state_ensemble[2].block(1).reinit(parameter_size);
    augmented_state_ensemble[2].block(1)(0) = 1.4;
    augmented_state_ensemble[2].block(1)(0) = 5.5;

    augmented_state_ensemble[0].collect_sizes();
    augmented_state_ensemble[1].collect_sizes();
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
    da.update_ensemble(augmented_state_ensemble, expt_vec, R);

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
      BOOST_TEST(std::abs(expt_vec[0] - sim_at_expt_pt_1_after[member]) <=
                 std::abs(expt_vec[0] - sim_at_expt_pt_1_before[member]));
      BOOST_TEST(std::abs(expt_vec[1] - sim_at_expt_pt_2_after[member]) <=
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
  dat.test_update_ensemble_augmented();
}
} // namespace adamantine
