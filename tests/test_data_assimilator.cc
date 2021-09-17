/* Copyright (c) 2016 - 2021, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#define BOOST_TEST_MODULE DataAssimilator

#include <DataAssimilator.hh>

#include <deal.II/lac/la_parallel_vector.h>

#include "main.cc"

namespace adamantine
{
class DataAssimilatorTester
{
public:
  bool testCalcKalmanGain() { return true; };
  bool testCalcHx()
  {
    /*
   bool pass = false;

   int sim_size = 5;
   int expt_size = 3;

   dealii::Vector<double> sim_vec(sim_size);
   sim_vec(0) = 2.0;
   sim_vec(1) = 4.0;
   sim_vec(2) = 5.0;
   sim_vec(3) = 7.0;
   sim_vec(4) = 8.0;

   dealii::Vector<double> expt_vec(expt_size);
   expt_vec(0) = 2.5;
   expt_vec(1) = 4.5;
   expt_vec(2) = 8.5;

   std::pair<std::vector<int>, std::vector<int>> expt_to_sim_mapping;
   expt_to_sim_mapping.first.resize(3);
   expt_to_sim_mapping.second.resize(3);
   expt_to_sim_mapping.first[0] = 0;
   expt_to_sim_mapping.first[1] = 1;
   expt_to_sim_mapping.first[2] = 3;
   expt_to_sim_mapping.second[0] = 0;
   expt_to_sim_mapping.second[1] = 1;
   expt_to_sim_mapping.second[2] = 4;

   DataAssimilator<2, dealii::Vector<double>> da;

   dealii::Vector<double> Hx = da.calcHx(sim_vec, expt_to_sim_mapping);

   double tol = 1e-12;
   if (std::abs(Hx(0) - 2.0) < tol && std::abs(Hx(1) - 4.0) < tol &&
       std::abs(Hx(2) - 8.0) < tol)
   {
     pass = true;
   }

   */
    bool pass = true;

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

    // cov.print(std::cout);

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
};

BOOST_AUTO_TEST_CASE(data_assimilator)
{

  DataAssimilatorTester dat;
  bool passKalmanGain = dat.testCalcKalmanGain();
  BOOST_CHECK(passKalmanGain);

  bool passHx = dat.testCalcHx();
  BOOST_CHECK(passHx);

  bool passCovDense = dat.testCalcSampleCovarianceDense();
  BOOST_CHECK(passCovDense);

  bool passFillNoiseVector = dat.testFillNoiseVector();
  BOOST_CHECK(passFillNoiseVector);
}
} // namespace adamantine
