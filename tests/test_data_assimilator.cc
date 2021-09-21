/* Copyright (c) 2016 - 2021, the adamantine authors.
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
  bool testCalcKalmanGain()
  {
    bool pass = true;

    // Create the DoF mapping
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

    int sim_size = 5;
    int expt_size = 2;

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

    DataAssimilator<2, dealii::Vector<double>> da;
    da.updateDofMapping(dof_handler, expt_vec.size(), indices_and_offsets);

    // Create the simulation data
    std::vector<dealii::Vector<double>> data;
    dealii::Vector<double> sample1({1., 3., 6., 9., 11.});
    data.push_back(sample1);
    dealii::Vector<double> sample2({1.5, 3.2, 6.3, 9.7, 11.9});
    data.push_back(sample2);
    dealii::Vector<double> sample3({1.1, 3.1, 6.1, 9.1, 11.1});
    data.push_back(sample3);

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
    for (int sample = 0; sample < perturbed_innovation.size(); ++sample)
    {
      perturbed_innovation[sample].reinit(expt_size);
      dealii::Vector<double> temp = da.calcHx(sim_size, data[sample]);
      for (int i = 0; i < expt_size; ++i)
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
    std::vector<dealii::Vector<double>> forcast_shift =
        da.applyKalmanGain(data, expt_size, R, perturbed_innovation);

    // Check output
    std::cout << "After applying the Kalman Gain:" << std::endl;
    for (int sample = 0; sample < forcast_shift.size(); ++sample)
    {
      for (int i = 0; i < sim_size; ++i)
      {
        std::cout << forcast_shift[sample][i] << " ";
      }
      std::cout << std::endl;
    }

    double tol = 1.0e-8;

    // Reference solution calculated using Python
    if (std::abs(forcast_shift[0][0] - 0.21352564) > tol)
      pass = false;
    if (std::abs(forcast_shift[0][1] + 0.14600986) > tol)
      pass = false;
    if (std::abs(forcast_shift[0][2] + 0.02616469) > tol)
      pass = false;
    if (std::abs(forcast_shift[0][3] - 0.45321598) > tol)
      pass = false;
    if (std::abs(forcast_shift[0][4] - 0.69290631) > tol)
      pass = false;
    if (std::abs(forcast_shift[1][0] + 0.27786325) > tol)
      pass = false;
    if (std::abs(forcast_shift[1][1] + 0.32946285) > tol)
      pass = false;
    if (std::abs(forcast_shift[1][2] + 0.31226298) > tol)
      pass = false;
    if (std::abs(forcast_shift[1][3] + 0.24346351) > tol)
      pass = false;
    if (std::abs(forcast_shift[1][4] + 0.20906377) > tol)
      pass = false;
    if (std::abs(forcast_shift[2][0] - 0.12767094) > tol)
      pass = false;
    if (std::abs(forcast_shift[2][1] + 0.20319395) > tol)
      pass = false;
    if (std::abs(forcast_shift[2][2] + 0.09290565) > tol)
      pass = false;
    if (std::abs(forcast_shift[2][3] - 0.34824753) > tol)
      pass = false;
    if (std::abs(forcast_shift[2][4] - 0.56882413) > tol)
      pass = false;

    return pass;
  };

  bool testUpdateDofMapping()
  {
    bool pass = true;

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

    DataAssimilator<2, dealii::Vector<double>> da;
    da.updateDofMapping(dof_handler, expt_size, indices_and_offsets);

    BOOST_CHECK(da._expt_to_dof_mapping.first[0] == 0);
    BOOST_CHECK(da._expt_to_dof_mapping.first[1] == 1);
    BOOST_CHECK(da._expt_to_dof_mapping.first[2] == 2);
    BOOST_CHECK(da._expt_to_dof_mapping.second[0] == 0);
    BOOST_CHECK(da._expt_to_dof_mapping.second[1] == 1);
    BOOST_CHECK(da._expt_to_dof_mapping.second[2] == 3);
  };

  bool testCalcH()
  {
    bool pass = true;

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

    DataAssimilator<2, dealii::Vector<double>> da;
    da.updateDofMapping(dof_handler, expt_size, indices_and_offsets);

    dealii::SparsityPattern pattern(expt_size, sim_size, expt_size);

    dealii::SparseMatrix<double> H = da.calcH(pattern);

    double tol = 1e-12;
    for (int i = 0; i < expt_size; ++i)
    {
      for (int j = 0; j < sim_size; ++j)
      {
        if (i == 0 && j == 0)
        {
          std::cout << "1" << std::endl;
          if (std::abs(H(i, j) - 1.0) > tol)
          {
            pass = false;
          }
        }
        else if (i == 1 && j == 1)
        {
          std::cout << "2" << std::endl;
          if (std::abs(H(i, j) - 1.0) > tol)
          {
            pass = false;
          }
        }
        else if (i == 2 && j == 3)
        {
          std::cout << "3" << std::endl;
          if (std::abs(H(i, j) - 1.0) > tol)
          {
            pass = false;
          }
        }
        else
        {
          std::cout << "4" << std::endl;
          if (std::abs(H.el(i, j)) > tol)
          {
            pass = false;
          }
        }
      }
    }

    return pass;
  };
  bool testCalcHx()
  {

    bool pass = false;

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

    dealii::Vector<double> sim_vec(dof_handler.n_dofs());
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

    DataAssimilator<2, dealii::Vector<double>> da;
    da.updateDofMapping(dof_handler, expt_vec.size(), indices_and_offsets);
    dealii::Vector<double> Hx = da.calcHx(sim_size, sim_vec);

    double tol = 1e-12;
    if (std::abs(Hx(0) - 2.0) < tol && std::abs(Hx(1) - 4.0) < tol &&
        std::abs(Hx(2) - 7.0) < tol)
    {
      pass = true;
    }

    return pass;
  };

  bool testCalcSampleCovarianceDense()
  {
    double tol = 1e-12;

    // Trivial case of identical vectors, covariance should be the zero matrix
    bool pass1 = true;
    std::vector<dealii::Vector<double>> data1;
    dealii::Vector<double> sample1({1, 3, 5, 7});
    data1.push_back(sample1);
    data1.push_back(sample1);
    data1.push_back(sample1);

    dealii::FullMatrix<double> cov(4);

    DataAssimilator<2, dealii::Vector<double>> da;
    da.calcSampleCovarianceDense(data1, cov);

    // Check results
    for (unsigned int i = 0; i < 4; ++i)
    {
      for (unsigned int j = 0; j < 4; ++j)
      {
        if (std::abs(cov(i, j)) > tol)
        {
          pass1 = false;
        }
      }
    }

    // Non-trivial case, using NumPy solution as the reference
    bool pass2 = true;
    std::vector<dealii::Vector<double>> data2;
    dealii::Vector<double> sample21({1., 3., 6., 9., 11.});
    data2.push_back(sample21);
    dealii::Vector<double> sample22({1.5, 3.2, 6.3, 9.7, 11.9});
    data2.push_back(sample22);
    dealii::Vector<double> sample23({1.1, 3.1, 6.1, 9.1, 11.1});
    data2.push_back(sample23);

    dealii::FullMatrix<double> cov2(5);
    da.calcSampleCovarianceDense(data2, cov2);

    if (std::abs(cov2(0, 0) - 0.07) > tol)
      pass2 = false;
    if (std::abs(cov2(1, 0) - 0.025) > tol)
      pass2 = false;
    if (std::abs(cov2(2, 0) - 0.04) > tol)
      pass2 = false;
    if (std::abs(cov2(3, 0) - 0.1) > tol)
      pass2 = false;
    if (std::abs(cov2(4, 0) - 0.13) > tol)
      pass2 = false;
    if (std::abs(cov2(0, 1) - 0.025) > tol)
      pass2 = false;
    if (std::abs(cov2(1, 1) - 0.01) > tol)
      pass2 = false;
    if (std::abs(cov2(2, 1) - 0.015) > tol)
      pass2 = false;
    if (std::abs(cov2(3, 1) - 0.035) > tol)
      pass2 = false;
    if (std::abs(cov2(4, 1) - 0.045) > tol)
      pass2 = false;
    if (std::abs(cov2(0, 2) - 0.04) > tol)
      pass2 = false;
    if (std::abs(cov2(1, 2) - 0.015) > tol)
      pass2 = false;
    if (std::abs(cov2(2, 2) - 0.02333333333333) > tol)
      pass2 = false;
    if (std::abs(cov2(3, 2) - 0.05666666666667) > tol)
      pass2 = false;
    if (std::abs(cov2(4, 2) - 0.07333333333333) > tol)
      pass2 = false;
    if (std::abs(cov2(0, 3) - 0.1) > tol)
      pass2 = false;
    if (std::abs(cov2(1, 3) - 0.035) > tol)
      pass2 = false;
    if (std::abs(cov2(2, 3) - 0.05666666666667) > tol)
      pass2 = false;
    if (std::abs(cov2(3, 3) - 0.14333333333333) > tol)
      pass2 = false;
    if (std::abs(cov2(4, 3) - 0.18666666666667) > tol)
      pass2 = false;
    if (std::abs(cov2(0, 4) - 0.13) > tol)
      pass2 = false;
    if (std::abs(cov2(1, 4) - 0.045) > tol)
      pass2 = false;
    if (std::abs(cov2(2, 4) - 0.07333333333333) > tol)
      pass2 = false;
    if (std::abs(cov2(3, 4) - 0.18666666666667) > tol)
      pass2 = false;
    if (std::abs(cov2(4, 4) - 0.24333333333333) > tol)
      pass2 = false;

    bool pass = pass1 && pass2;

    return pass;
  };

  bool testFillNoiseVector()
  {
    bool pass = true;

    DataAssimilator<2, dealii::Vector<double>> da;

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
      da.fillNoiseVector(ensemble_member, R);
      data.push_back(ensemble_member);
    }

    dealii::FullMatrix<double> Rtest(3);
    da.calcSampleCovarianceDense(data, Rtest);

    double tol = 20.; // Loose 20% tolerance because this is a statistical check
    BOOST_CHECK_CLOSE(R(0, 0), Rtest(0, 0), tol);
    BOOST_CHECK_CLOSE(R(1, 0), Rtest(1, 0), tol);
    BOOST_CHECK_CLOSE(R(1, 1), Rtest(1, 1), tol);
    BOOST_CHECK_CLOSE(R(0, 1), Rtest(0, 1), tol);
    BOOST_CHECK_CLOSE(R(2, 2), Rtest(2, 2), tol);

    return pass;
  };

  bool testUpdateEnsemble()
  {
    bool pass = true;

    // Create the DoF mapping
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

    int sim_size = 5;
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

    DataAssimilator<2, dealii::Vector<double>> da;
    da.updateDofMapping(dof_handler, expt_vec.size(), indices_and_offsets);

    // Create the simulation data
    std::vector<dealii::Vector<double>> data;
    dealii::Vector<double> sample1({1., 3., 6., 9., 11.});
    data.push_back(sample1);
    dealii::Vector<double> sample2({1.5, 3.2, 6.3, 9.7, 11.9});
    data.push_back(sample2);
    dealii::Vector<double> sample3({1.1, 3.1, 6.1, 9.1, 11.1});
    data.push_back(sample3);

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
    sim_at_expt_pt_1_before.push_back(data[0][1]);
    sim_at_expt_pt_1_before.push_back(data[1][1]);
    sim_at_expt_pt_1_before.push_back(data[2][1]);

    std::vector<double> sim_at_expt_pt_2_before(3);
    sim_at_expt_pt_2_before.push_back(data[0][3]);
    sim_at_expt_pt_2_before.push_back(data[1][3]);
    sim_at_expt_pt_2_before.push_back(data[2][3]);

    // Update the simulation data
    da.updateEnsemble(data, expt_vec, indices_and_offsets, R);

    // Save the data at the observation points after assimilation
    std::vector<double> sim_at_expt_pt_1_after(3);
    sim_at_expt_pt_1_after.push_back(data[0][1]);
    sim_at_expt_pt_1_after.push_back(data[1][1]);
    sim_at_expt_pt_1_after.push_back(data[2][1]);

    std::vector<double> sim_at_expt_pt_2_after(3);
    sim_at_expt_pt_2_after.push_back(data[0][3]);
    sim_at_expt_pt_2_after.push_back(data[1][3]);
    sim_at_expt_pt_2_after.push_back(data[2][3]);

    // Check the solution
    // The observed points should get closer to the experimental values
    // Large entries in R could make these fail spuriously
    for (int member = 0; member < 3; ++member)
    {
      if (std::abs(expt_vec[0] - sim_at_expt_pt_1_after[member]) >
          std::abs(expt_vec[0] - sim_at_expt_pt_1_before[member]))
        pass = false;
      if (std::abs(expt_vec[1] - sim_at_expt_pt_2_after[member]) >
          std::abs(expt_vec[1] - sim_at_expt_pt_2_before[member]))
        pass = false;
    }

    return pass;
  };
};

BOOST_AUTO_TEST_CASE(data_assimilator)
{

  DataAssimilatorTester dat;
  bool passKalmanGain = dat.testCalcKalmanGain();
  BOOST_CHECK(passKalmanGain);

  bool passUpdateDofMapping = dat.testUpdateDofMapping();
  BOOST_CHECK(passUpdateDofMapping);

  bool passHx = dat.testCalcHx();
  BOOST_CHECK(passHx);

  bool passCovDense = dat.testCalcSampleCovarianceDense();
  BOOST_CHECK(passCovDense);

  bool passFillNoiseVector = dat.testFillNoiseVector();
  BOOST_CHECK(passFillNoiseVector);

  bool passCalcH = dat.testCalcH();
  BOOST_CHECK(passCalcH);

  bool passUpdateEnsemble = dat.testUpdateEnsemble();
  BOOST_CHECK(passUpdateEnsemble);
}
} // namespace adamantine
