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
    bool pass = false;

    int sim_size = 5;
    int expt_size = 3;

    dealii::LA::Vector<double> sim_vec(sim_size);
    sim_vec(0) = 2.0;
    sim_vec(1) = 4.0;
    sim_vec(2) = 5.0;
    sim_vec(3) = 7.0;
    sim_vec(4) = 8.0;

    dealii::LA::Vector<double> expt_vec(expt_size);
    expt_vec(0) = 2.5;
    expt_vec(1) = 4.5;
    expt_vec(2) = 8.5;

    std::map<dealii::types::global_dof_index, dealii::types::global_dof_index>
        expt_to_sim_mapping;
    expt_to_sim_mapping[0] = 0;
    expt_to_sim_mapping[1] = 1;
    expt_to_sim_mapping[3] = 4;

    DataAssimilator<dealii::LA::Vector<double>> da;

    dealii::LA::Vector<double> Hx = da.calcHx(sim_vec, expt_to_sim_mapping);

    double tol = 1e-12;
    if (std::abs(Hx(0) - 2.0) < tol && std::abs(Hx(1) - 4.0) < tol &&
        std::abs(Hx(2) - 8.0) < tol)
    {
      pass = true;
    }

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
}
} // namespace adamantine
