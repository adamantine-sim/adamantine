/* Copyright (c) 2016, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#include "PostProcessor.hh"

#include "instantiation.hh"

#include <deal.II/grid/filtered_iterator.h>

#include <fstream>

namespace adamantine
{
template <int dim>
PostProcessor<dim>::PostProcessor(
    MPI_Comm &communicator, boost::property_tree::ptree const &database,
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

  // Add MaterialState ratio. We need to copy the data because state is attached
  // to a different DoFHandler.
  std::array<dealii::LA::distributed::Vector<double>,
             static_cast<unsigned int>(MaterialState::SIZE)>
      state = _material_properties->get_state();
  unsigned int const n_active_cells =
      _dof_handler.get_triangulation().n_active_cells();
  dealii::DoFHandler<dim> const &material_dof_handler =
      _material_properties->get_dof_handler();
  dealii::Vector<double> powder(n_active_cells);
  dealii::Vector<double> liquid(n_active_cells);
  dealii::Vector<double> solid(n_active_cells);
  unsigned int constexpr powder_index =
      static_cast<unsigned int>(MaterialState::powder);
  unsigned int constexpr liquid_index =
      static_cast<unsigned int>(MaterialState::liquid);
  unsigned int constexpr solid_index =
      static_cast<unsigned int>(MaterialState::solid);
  auto mp_cell = material_dof_handler.begin_active();
  auto mp_end_cell = material_dof_handler.end();
  std::vector<dealii::types::global_dof_index> mp_dof_indices(1);
  for (unsigned int i = 0; mp_cell != mp_end_cell; ++i, ++mp_cell)
    if (mp_cell->is_locally_owned())
    {
      mp_cell->get_dof_indices(mp_dof_indices);
      dealii::types::global_dof_index const mp_dof_index = mp_dof_indices[0];
      powder[i] = state[powder_index][mp_dof_index];
      liquid[i] = state[liquid_index][mp_dof_index];
      solid[i] = state[solid_index][mp_dof_index];
    }
  _data_out.add_data_vector(powder, "powder");
  _data_out.add_data_vector(liquid, "liquid");
  _data_out.add_data_vector(solid, "solid");

  // Add the subdomain IDs.
  dealii::types::subdomain_id subdomain_id =
      _dof_handler.get_triangulation().locally_owned_subdomain();
  dealii::Vector<float> subdomain(n_active_cells);
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
  unsigned int rank = dealii::Utilities::MPI::this_mpi_process(_communicator);
  if (rank == 0)
  {
    std::vector<std::string> filenames;
    unsigned int comm_size =
        dealii::Utilities::MPI::n_mpi_processes(_communicator);
    for (unsigned int i = 0; i < comm_size; ++i)
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
} // namespace adamantine

INSTANTIATE_DIM(PostProcessor)
