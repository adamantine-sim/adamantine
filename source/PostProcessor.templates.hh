/* Copyright (c) 2016, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#ifndef _POST_PROCESSOR_TEMPLATES_HH_
#define _POST_PROCESSOR_TEMPLATES_HH_

#include "PostProcessor.hh"
#include <deal.II/grid/filtered_iterator.h>
#include <fstream>

namespace adamantine
{
template <int dim>
PostProcessor<dim>::PostProcessor(
    boost::mpi::communicator &communicator,
    boost::property_tree::ptree const &database,
    dealii::DoFHandler<dim> &dof_handler,
    std::shared_ptr<MaterialProperty<dim>> material_properties)
    : _communicator(communicator), _dof_handler(dof_handler),
      _material_properties(material_properties)
{
  _filename = database.get<std::string>("file_name");
}

template <int dim>
void PostProcessor<dim>::output_pvtu(
    unsigned int cycle, unsigned int time_step, double time,
    dealii::LA::distributed::Vector<double> const &solution)
{
  // Add the DoFHandler and the enthalpy.
  _data_out.clear();
  _data_out.attach_dof_handler(_dof_handler);
  solution.update_ghost_values();
  _data_out.add_data_vector(solution, "enthalpy");

  // Add the temperature.
  dealii::LA::distributed::Vector<double> temperature =
      compute_temperature(solution);
  _data_out.add_data_vector(temperature, "temperature");

  // Add MaterialState ratio
  std::array<dealii::LA::distributed::Vector<double>,
             static_cast<unsigned int>(MaterialState::SIZE)> state =
      _material_properties->get_state();
  _data_out.add_data_vector(
      state[static_cast<unsigned int>(MaterialState::powder)], "powder");
  _data_out.add_data_vector(
      state[static_cast<unsigned int>(MaterialState::liquid)], "liquid");
  _data_out.add_data_vector(
      state[static_cast<unsigned int>(MaterialState::solid)], "solid");

  // Add the subdomain IDs.
  dealii::types::subdomain_id subdomain_id =
      _dof_handler.get_triangulation().locally_owned_subdomain();
  dealii::Vector<float> subdomain(
      _dof_handler.get_triangulation().n_active_cells());
  for (unsigned int i = 0; i < subdomain.size(); ++i)
    subdomain[i] = subdomain_id;
  _data_out.add_data_vector(subdomain, "subdomain");

  // Output the data.
  _data_out.build_patches();
  std::string local_filename =
      _filename + "." + dealii::Utilities::int_to_string(cycle, 2) + "." +
      dealii::Utilities::int_to_string(time_step, 6) + "." +
      dealii::Utilities::int_to_string(subdomain_id, 6);
  std::ofstream output((local_filename + ".vtu").c_str());
  dealii::DataOutBase::VtkFlags flags(time, cycle);
  _data_out.set_flags(flags);
  _data_out.write_vtu(output);

  // Output the master record.
  if (_communicator.rank() == 0)
  {
    std::vector<std::string> filenames;
    for (int i = 0; i < _communicator.size(); ++i)
    {
      std::string local_name =
          _filename + "." + dealii::Utilities::int_to_string(cycle, 2) + "." +
          dealii::Utilities::int_to_string(time_step, 6) + "." +
          dealii::Utilities::int_to_string(i, 6) + ".vtu";
      filenames.push_back(local_name);
    }
    std::string master_filename =
        _filename + "." + dealii::Utilities::int_to_string(cycle, 2) + "." +
        dealii::Utilities::int_to_string(time_step, 6) + ".pvtu";
    std::ofstream master_output(master_filename.c_str());
    _data_out.write_pvtu_record(master_output, filenames);

    // Associate the time to the time step.
    _times_filenames.push_back(
        std::pair<double, std::string>(time, master_filename));
  }
}

template <int dim>
void PostProcessor<dim>::output_pvd()
{
  std::ofstream output(_filename + ".pvd");
  dealii::DataOutBase::write_pvd_record(output, _times_filenames);
}

// TODO remove this function
template <int dim>
dealii::LA::distributed::Vector<double> PostProcessor<dim>::compute_temperature(
    dealii::LA::distributed::Vector<double> const &enthalpy)
{
  dealii::LA::distributed::Vector<double> temperature(
      enthalpy.get_partitioner());
  // This is not used for now because the material properties are independent of
  // the temperatures.
  dealii::LA::distributed::Vector<double> state;

  // TODO the computation does not work if there is a phase change
  unsigned int const dofs_per_cell = _dof_handler.get_fe().dofs_per_cell;
  std::vector<dealii::types::global_dof_index> local_dof_indices(dofs_per_cell);
  for (auto cell :
       dealii::filter_iterators(_dof_handler.active_cell_iterators(),
                                dealii::IteratorFilters::LocallyOwnedCell()))
  {
    double const enthalpy_to_temp =
        1. / (_material_properties->get(cell, Property::density, state) *
              _material_properties->get(cell, Property::specific_heat, state));

    cell->get_dof_indices(local_dof_indices);

    for (unsigned int i = 0; i < dofs_per_cell; ++i)
      temperature[local_dof_indices[i]] =
          enthalpy_to_temp * enthalpy[local_dof_indices[i]];
  }

  return temperature;
}
}

#endif
