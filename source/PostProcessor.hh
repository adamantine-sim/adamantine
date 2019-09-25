/* Copyright (c) 2016 - 2017, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#ifndef POST_PROCESSOR_HH
#define POST_PROCESSOR_HH

#include "MaterialProperty.hh"
#include "types.hh"

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/numerics/data_out.h>

#include <boost/property_tree/ptree.hpp>

namespace adamantine
{
/**
 * This class outputs the results using the vtu format.
 */
template <int dim>
class PostProcessor
{
public:
  /**
   * Constructor.
   * \param database requires the following entries:
   *   - <B>file_name</B>: string
   */
  PostProcessor(MPI_Comm &communicator,
                boost::property_tree::ptree const &database,
                dealii::DoFHandler<dim> &dof_handler,
                std::shared_ptr<MaterialProperty<dim>> material_properties);

  /**
   * Output the different vtu and pvtu files.
   */
  void output_pvtu(unsigned int cycle, unsigned int n_time_step, double time,
                   dealii::LA::distributed::Vector<double> const &solution);

  /**
   * Output the pvd file for Paraview.
   */
  void output_pvd();

private:
  /**
   * Compute the temperature given the enthalpy.
   */
  dealii::LA::distributed::Vector<double>
  compute_temperature(dealii::LA::distributed::Vector<double> const &enthalpy);

  /**
   * MPI communicator.
   */
  MPI_Comm _communicator;
  /**
   * Root of the different output files.
   */
  std::string _filename;
  /**
   * Vector of pair of time and pvtu file.
   */
  std::vector<std::pair<double, std::string>> _times_filenames;
  /**
   * DataOut associated with the post-processing.
   */
  dealii::DataOut<dim> _data_out;
  /**
   * DoFHandler associated with the simulation.
   */
  dealii::DoFHandler<dim> &_dof_handler;
  /**
   * Material properties associated with the simulation.
   */
  std::shared_ptr<MaterialProperty<dim>> _material_properties;
};
} // namespace adamantine
#endif
