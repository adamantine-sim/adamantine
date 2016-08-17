/* Copyright (c) 2016, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#ifndef _POST_PROCESSOR_TEMPLATES_HH_
#define _POST_PROCESSOR_TEMPLATES_HH_

#include "PostProcessor.hh"
#include <fstream>

namespace adamantine
{
template <int dim>
PostProcessor<dim>::PostProcessor(boost::mpi::communicator &communicator,
                                  boost::property_tree::ptree const &database,
                                  dealii::DoFHandler<dim> &dof_handler)
    : _communicator(communicator), _dof_handler(dof_handler)
{
  _filename = database.get<std::string>("file_name");
}

template <int dim>
void PostProcessor<dim>::output_pvtu(
    unsigned int cycle, unsigned int time_step, double time,
    dealii::LA::distributed::Vector<double> const &solution)
{
  // Add the DoFHandler and the temperature.
  _data_out.clear();
  _data_out.attach_dof_handler(_dof_handler);
  solution.update_ghost_values();
  _data_out.add_data_vector(solution, "temperature");

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
    for (unsigned int i = 0; i < _communicator.size(); ++i)
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
  _data_out.write_pvd_record(output, _times_filenames);
}
}

#endif
