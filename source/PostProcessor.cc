/* Copyright (c) 2016 - 2022, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#include <PostProcessor.hh>
#include <instantiation.hh>

#include <deal.II/grid/filtered_iterator.h>

#include <fstream>
#include <unordered_map>

namespace adamantine
{
template <int dim>
StrainPostProcessor<dim>::StrainPostProcessor()
    : dealii::DataPostprocessorTensor<dim>("strain", dealii::update_gradients)
{
}

template <int dim>
void StrainPostProcessor<dim>::evaluate_vector_field(
    const dealii::DataPostprocessorInputs::Vector<dim> &displacement_data,
    std::vector<dealii::Vector<double>> &strain) const
{
  for (unsigned int i = 0; i < displacement_data.solution_gradients.size(); ++i)
  {
    for (unsigned int j = 0; j < dim; ++j)
    {
      for (unsigned int k = 0; k < dim; ++k)
      {
        // strain = (\nabla u + (\nabla u)^T)/2
        strain[i][dealii::Tensor<2, dim>::component_to_unrolled_index(
            dealii::TableIndices<2>(j, k))] =
            (displacement_data.solution_gradients[i][j][k] +
             displacement_data.solution_gradients[i][k][j]) /
            2.;
      }
    }
  }
}

template <int dim>
PostProcessor<dim>::PostProcessor(MPI_Comm const &communicator,
                                  boost::property_tree::ptree const &database,
                                  dealii::DoFHandler<dim> &dof_handler,
                                  int ensemble_member_index)
    : _communicator(communicator)
{
  // This is internal data we embed in the database that is intended to be set
  // in the application (not by the user in the input file)
  _thermal_output = database.get("thermal_output", false);
  _mechanical_output = database.get("mechanical_output", false);
  ASSERT(_thermal_output != _mechanical_output, "Internal error");
  if (_thermal_output)
  {
    _thermal_dof_handler = &dof_handler;
  }
  else
  {
    _mechanical_dof_handler = &dof_handler;
  }

  // PropertyTreeInput post_processor.file_name
  _filename_prefix = database.get<std::string>("filename_prefix");
  if (ensemble_member_index >= 0)
  {
    _filename_prefix =
        _filename_prefix + "_m" + std::to_string(ensemble_member_index);
  }

  // PropertyTreeInput post_processor.additional_output_refinement
  _additional_output_refinement =
      database.get<unsigned int>("additional_output_refinement", 0);
}

template <int dim>
PostProcessor<dim>::PostProcessor(
    MPI_Comm const &communicator, boost::property_tree::ptree const &database,
    dealii::DoFHandler<dim> &thermal_dof_handler,
    dealii::DoFHandler<dim> &mechanical_dof_handler, int ensemble_member_index)
    : _communicator(communicator)
{
  _thermal_output = database.get("thermal_output", false);
  _mechanical_output = database.get("mechanical_output", false);
  ASSERT(_thermal_output, "Internal error");
  ASSERT(_mechanical_output, "Internal error");
  _thermal_dof_handler = &thermal_dof_handler;
  _mechanical_dof_handler = &mechanical_dof_handler;

  // PropertyTreeInput post_processor.filename_prefix
  _filename_prefix = database.get<std::string>("filename_prefix");
  if (ensemble_member_index >= 0)
  {
    _filename_prefix =
        _filename_prefix + "_m" + std::to_string(ensemble_member_index);
  }

  // PropertyTreeInput post_processor.additional_output_refinement
  _additional_output_refinement =
      database.get<unsigned int>("additional_output_refinement", 0);
}

template <int dim>
void PostProcessor<dim>::write_thermal_output(
    unsigned int cycle, unsigned int time_step, double time,
    dealii::LA::distributed::Vector<double> const &temperature,
    MemoryBlockView<double, dealii::MemorySpace::Host> state,
    std::unordered_map<dealii::types::global_dof_index, unsigned int> const
        &dofs_map,
    dealii::DoFHandler<dim> const &material_dof_handler)
{
  ASSERT(_thermal_dof_handler != nullptr, "Internal Error");
  _data_out.clear();
  thermal_dataout(temperature);
  material_dataout(state, dofs_map, material_dof_handler);
  subdomain_dataout();
  write_pvtu(cycle, time_step, time);
}

template <int dim>
void PostProcessor<dim>::write_mechanical_output(
    unsigned int cycle, unsigned int time_step, double time,
    dealii::LA::distributed::Vector<double> const &displacement,
    MemoryBlockView<double, dealii::MemorySpace::Host> state,
    std::unordered_map<dealii::types::global_dof_index, unsigned int> const
        &dofs_map,
    dealii::DoFHandler<dim> const &material_dof_handler)
{
  ASSERT(_mechanical_dof_handler != nullptr, "Internal Error");
  _data_out.clear();
  // We need the StrainPostProcessor to live until write_pvtu is done
  StrainPostProcessor<dim> strain;
  mechanical_dataout(displacement, strain);
  material_dataout(state, dofs_map, material_dof_handler);
  subdomain_dataout();
  write_pvtu(cycle, time_step, time);
}

template <int dim>
void PostProcessor<dim>::write_output(
    unsigned int cycle, unsigned int time_step, double time,
    dealii::LA::distributed::Vector<double> const &temperature,
    dealii::LA::distributed::Vector<double> const &displacement,
    MemoryBlockView<double, dealii::MemorySpace::Host> state,
    std::unordered_map<dealii::types::global_dof_index, unsigned int> const
        &dofs_map,
    dealii::DoFHandler<dim> const &material_dof_handler)
{
  ASSERT(_thermal_dof_handler != nullptr, "Internal Error");
  ASSERT(_mechanical_dof_handler != nullptr, "Internal Error");
  _data_out.clear();
  thermal_dataout(temperature);
  // We need the StrainPostProcessor to live until write_pvtu is done
  StrainPostProcessor<dim> strain;
  mechanical_dataout(displacement, strain);
  material_dataout(state, dofs_map, material_dof_handler);
  subdomain_dataout();
  write_pvtu(cycle, time_step, time);
}

template <int dim>
void PostProcessor<dim>::write_pvd() const
{
  std::ofstream output(_filename_prefix + ".pvd");
  dealii::DataOutBase::write_pvd_record(output, _times_filenames);
}

template <int dim>
void PostProcessor<dim>::thermal_dataout(
    dealii::LA::distributed::Vector<double> const &temperature)
{
  temperature.update_ghost_values();
  _data_out.add_data_vector(*_thermal_dof_handler, temperature, "temperature");
}

template <int dim>
void PostProcessor<dim>::mechanical_dataout(
    dealii::LA::distributed::Vector<double> const &displacement,
    StrainPostProcessor<dim> const &strain)
{
  // Add the displacement to the output
  displacement.update_ghost_values();
  std::vector<std::string> displacement_names(dim, "displacement");
  std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation>
      displacement_data_component_interpretation(
          dim,
          dealii::DataComponentInterpretation::component_is_part_of_vector);
  _data_out.add_data_vector(*_mechanical_dof_handler, displacement,
                            displacement_names,
                            displacement_data_component_interpretation);

  // Add the strain tensor to the output
  _data_out.add_data_vector(*_mechanical_dof_handler, displacement, strain);

  // TODO add the stress tensor
}

template <int dim>
void PostProcessor<dim>::material_dataout(
    MemoryBlockView<double, dealii::MemorySpace::Host> state,
    std::unordered_map<dealii::types::global_dof_index, unsigned int> const
        &dofs_map,
    dealii::DoFHandler<dim> const &material_dof_handler)
{
  unsigned int const n_active_cells =
      material_dof_handler.get_triangulation().n_active_cells();
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
      dealii::types::global_dof_index const mp_dof_index =
          dofs_map.at(mp_dof_indices[0]);
      powder[i] = state(powder_index, mp_dof_index);
      liquid[i] = state(liquid_index, mp_dof_index);
      solid[i] = state(solid_index, mp_dof_index);
    }
  _data_out.add_data_vector(powder, "powder");
  _data_out.add_data_vector(liquid, "liquid");
  _data_out.add_data_vector(solid, "solid");
}

template <int dim>
void PostProcessor<dim>::subdomain_dataout()
{
  dealii::DoFHandler<dim> *dof_handler =
      (_thermal_dof_handler) ? _thermal_dof_handler : _mechanical_dof_handler;
  unsigned int const n_active_cells =
      dof_handler->get_triangulation().n_active_cells();
  dealii::types::subdomain_id subdomain_id =
      dof_handler->get_triangulation().locally_owned_subdomain();
  dealii::Vector<float> subdomain(n_active_cells);
  for (unsigned int i = 0; i < subdomain.size(); ++i)
    subdomain[i] = subdomain_id;
  _data_out.add_data_vector(subdomain, "subdomain");
}

template <int dim>
void PostProcessor<dim>::write_pvtu(unsigned int cycle, unsigned int time_step,
                                    double time)
{
  dealii::DoFHandler<dim> *dof_handler =
      (_thermal_dof_handler) ? _thermal_dof_handler : _mechanical_dof_handler;
  dealii::types::subdomain_id subdomain_id =
      dof_handler->get_triangulation().locally_owned_subdomain();
  _data_out.build_patches(2);
  std::string local_filename =
      _filename_prefix + "." + dealii::Utilities::int_to_string(cycle, 2) +
      "." + dealii::Utilities::int_to_string(time_step, 6) + "." +
      dealii::Utilities::int_to_string(subdomain_id, 6);
  std::ofstream output((local_filename + ".vtu").c_str());
  dealii::DataOutBase::VtkFlags flags(time, cycle);
  _data_out.set_flags(flags);
  _data_out.write_vtu(output);

  unsigned int rank = dealii::Utilities::MPI::this_mpi_process(_communicator);
  if (rank == 0)
  {
    std::vector<std::string> filenames;
    unsigned int comm_size =
        dealii::Utilities::MPI::n_mpi_processes(_communicator);
    for (unsigned int i = 0; i < comm_size; ++i)
    {
      std::string local_name =
          _filename_prefix + "." + dealii::Utilities::int_to_string(cycle, 2) +
          "." + dealii::Utilities::int_to_string(time_step, 6) + "." +
          dealii::Utilities::int_to_string(i, 6) + ".vtu";
      filenames.push_back(local_name);
    }
    std::string pvtu_filename =
        _filename_prefix + "." + dealii::Utilities::int_to_string(cycle, 2) +
        "." + dealii::Utilities::int_to_string(time_step, 6) + ".pvtu";
    std::ofstream pvtu_output(pvtu_filename.c_str());
    _data_out.write_pvtu_record(pvtu_output, filenames);

    // Associate the time to the time step.
    _times_filenames.push_back(
        std::pair<double, std::string>(time, pvtu_filename));
  }
}
} // namespace adamantine

INSTANTIATE_DIM(PostProcessor)
