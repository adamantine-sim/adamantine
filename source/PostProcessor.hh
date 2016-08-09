/* Copyright (c) 2016, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#ifndef _POST_PROCESSOR_HH_
#define _POST_PROCESSOR_HH_

#include "types.hh"
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/numerics/data_out.h>
#include <boost/mpi/communicator.hpp>
#include <boost/property_tree/ptree.hpp>

namespace adamantine
{
template <int dim>
class PostProcessor
{
public:
  PostProcessor(boost::mpi::communicator &communicator,
                boost::property_tree::ptree const &database,
                dealii::DoFHandler<dim> &dof_handler);

  void output_pvtu(unsigned int cycle, unsigned int n_time_step, double time,
                   dealii::LA::distributed::Vector<double> const &solution);

  void output_pvd();

private:
  boost::mpi::communicator _communicator;
  std::string _filename;
  std::vector<std::pair<double, std::string>> _times_filenames;
  dealii::DataOut<dim> _data_out;
  dealii::DoFHandler<dim> &_dof_handler;
};
}
#endif
