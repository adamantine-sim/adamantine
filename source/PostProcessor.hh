/* Copyright (c) 2016 - 2022, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#ifndef POST_PROCESSOR_HH
#define POST_PROCESSOR_HH

#include "MaterialProperty.hh"
#include "types.hh"

#include <deal.II/base/types.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/numerics/data_out.h>

#include <boost/property_tree/ptree.hpp>

#include <unordered_map>

namespace adamantine
{
/**
 * Helper class to output the strain given the displacement.
 */
template <int dim>
class StrainPostProcessor : public dealii::DataPostprocessorTensor<dim>
{
public:
  StrainPostProcessor();

  void evaluate_vector_field(
      const dealii::DataPostprocessorInputs::Vector<dim> &displacement_data,
      std::vector<dealii::Vector<double>> &strain) const override;
};

/**
 * This class outputs the results using the vtu format.
 */
template <int dim>
class PostProcessor
{
public:
  /**
   * Constructor takes the DoFHandler of the thermal or the mechanical
   * simulation.
   */
  PostProcessor(MPI_Comm const &communicator,
                boost::property_tree::ptree const &database,
                dealii::DoFHandler<dim> &dof_handler,
                int ensemble_member_index = -1);

  /**
   * Constructor takes the DoFHandlers of the thermal and the mechanical
   * simulations.
   */
  PostProcessor(MPI_Comm const &communicator,
                boost::property_tree::ptree const &database,
                dealii::DoFHandler<dim> &thermal_dof_handler,
                dealii::DoFHandler<dim> &mechanical_dof_handler,
                int ensemble_member_index = -1);

  /**
   * Write the different vtu and pvtu files for a thermal problem.
   */
  void write_thermal_output(
      unsigned int cycle, unsigned int time_step, double time,
      dealii::LA::distributed::Vector<double> const &temperature,
      MemoryBlockView<double, dealii::MemorySpace::Host> state,
      std::unordered_map<dealii::types::global_dof_index, unsigned int> const
          &dofs_map,
      dealii::DoFHandler<dim> const &material_dof_handler);

  /**
   * Write the different vtu and pvtu files for a mechanical problem.
   */
  void write_mechanical_output(
      unsigned int cycle, unsigned int time_step, double time,
      dealii::LA::distributed::Vector<double> const &displacement,
      MemoryBlockView<double, dealii::MemorySpace::Host> state,
      std::unordered_map<dealii::types::global_dof_index, unsigned int> const
          &dofs_map,
      dealii::DoFHandler<dim> const &material_dof_handler);

  /**
   * Write the different vtu and pvtu files for themo-mechanical problems.
   */
  void write_output(unsigned int cycle, unsigned int time_step, double time,
                    dealii::LA::distributed::Vector<double> const &temperature,
                    dealii::LA::distributed::Vector<double> const &displacement,
                    MemoryBlockView<double, dealii::MemorySpace::Host> state,
                    std::unordered_map<dealii::types::global_dof_index,
                                       unsigned int> const &dofs_map,
                    dealii::DoFHandler<dim> const &material_dof_handler);

  /**
   * Write the pvd file for Paraview.
   */
  void write_pvd() const;

private:
  /**
   * Fill _data_out with thermal data.
   */
  void
  thermal_dataout(dealii::LA::distributed::Vector<double> const &temperature);
  /**
   * Fill _data_out with mechanical data.
   */
  void mechanical_dataout(
      dealii::LA::distributed::Vector<double> const &displacement,
      StrainPostProcessor<dim> const &strain);
  /**
   * Fill _data_out with material data.
   */
  void material_dataout(
      MemoryBlockView<double, dealii::MemorySpace::Host> state,
      std::unordered_map<dealii::types::global_dof_index, unsigned int> const
          &dofs_map,
      dealii::DoFHandler<dim> const &material_dof_handler);
  /**
   * Fill _data_out with subdomain data.
   */
  void subdomain_dataout();
  /**
   * Write pvtu file.
   */
  void write_pvtu(unsigned int cycle, unsigned int time_step, double time);

  /**
   * MPI communicator.
   */
  MPI_Comm _communicator;
  /**
   * Flag is true if we output the results of the thermal simulation.
   */
  bool _thermal_output;
  /**
   * Flag is true if we output the results of the mechanical simulation.
   */
  bool _mechanical_output;
  /**
   * Prefix of the different output files.
   */
  std::string _filename_prefix;
  /**
   * Vector of pair of time and pvtu file.
   */
  std::vector<std::pair<double, std::string>> _times_filenames;
  /**
   * DataOut associated with the post-processing.
   */
  dealii::DataOut<dim> _data_out;
  /**
   * DoFHandler associated with the thermal simulation.
   */
  dealii::DoFHandler<dim> *_thermal_dof_handler = nullptr;
  /**
   * DoFHandler associated with the mechanical simulation.
   */
  dealii::DoFHandler<dim> *_mechanical_dof_handler = nullptr;
};
} // namespace adamantine
#endif
