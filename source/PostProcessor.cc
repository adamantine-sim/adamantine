/* Copyright (c) 2016 - 2023, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#include <PostProcessor.hh>
#include <instantiation.hh>

#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/numerics/data_component_interpretation.h>

#include <fstream>
#include <unordered_map>

namespace adamantine
{
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
void PostProcessor<dim>::write_pvd() const
{
  std::ofstream output(_filename_prefix + ".pvd");
  dealii::DataOutBase::write_pvd_record(output, _times_filenames);
}

template <int dim>
dealii::Vector<double> PostProcessor<dim>::get_stress_norm(
    std::vector<std::vector<dealii::SymmetricTensor<2, dim>>> const
        &stress_tensor)
{
  dealii::Vector<double> norm(
      _mechanical_dof_handler->get_triangulation().n_active_cells());
  unsigned int const n_quad_pts =
      stress_tensor.size() > 0 ? stress_tensor[0].size() : 0;
  unsigned int cell_id = 0;
  for (auto const &cell :
       _mechanical_dof_handler->active_cell_iterators() |
           dealii::IteratorFilters::ActiveFEIndexEqualTo(0, true))
  {
    dealii::SymmetricTensor<2, dim> accumulated_stress;
    for (unsigned int q = 0; q < n_quad_pts; ++q)
    {
      accumulated_stress += stress_tensor[cell_id][q];
    }

    norm(cell->active_cell_index()) = (accumulated_stress / n_quad_pts).norm();

    ++cell_id;
  }

  return norm;
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
    StrainPostProcessor<dim> const &strain,
    std::vector<std::vector<dealii::SymmetricTensor<2, dim>>> const
        &stress_tensor)
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

  // Add the stress tensor
  dealii::Vector<double> stress_norm = get_stress_norm(stress_tensor);
  _data_out.add_data_vector(stress_norm, "stress");
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
void PostProcessor<dim>::write_pvtu(unsigned int time_step, double time)
{
  dealii::DoFHandler<dim> *dof_handler =
      (_thermal_dof_handler) ? _thermal_dof_handler : _mechanical_dof_handler;
  dealii::types::subdomain_id subdomain_id =
      dof_handler->get_triangulation().locally_owned_subdomain();
  _data_out.build_patches(_additional_output_refinement);
  std::string local_filename = _filename_prefix + "." +
                               dealii::Utilities::to_string(time_step) + "." +
                               dealii::Utilities::to_string(subdomain_id);
  std::ofstream output((local_filename + ".vtu").c_str());
  dealii::DataOutBase::VtkFlags flags(time);
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
      std::string local_name = _filename_prefix + "." +
                               dealii::Utilities::to_string(time_step) + "." +
                               dealii::Utilities::to_string(i) + ".vtu";
      filenames.push_back(local_name);
    }
    std::string pvtu_filename = _filename_prefix + "." +
                                dealii::Utilities::to_string(time_step) +
                                ".pvtu";
    std::ofstream pvtu_output(pvtu_filename.c_str());
    _data_out.write_pvtu_record(pvtu_output, filenames);

    // Associate the time to the time step.
    _times_filenames.push_back(
        std::pair<double, std::string>(time, pvtu_filename));
  }
}
} // namespace adamantine

INSTANTIATE_DIM(PostProcessor)
