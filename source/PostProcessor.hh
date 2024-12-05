/* SPDX-FileCopyrightText: Copyright (c) 2016 - 2024, the adamantine authors.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#ifndef POST_PROCESSOR_HH
#define POST_PROCESSOR_HH

#include <MaterialProperty.hh>
#include <types.hh>

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
  using kokkos_host = dealii::MemorySpace::Host::kokkos_space;
  using kokkos_default = dealii::MemorySpace::Default::kokkos_space;

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
  template <typename LayoutType>
  void write_thermal_output(
      unsigned int time_step, double time,
      dealii::LA::distributed::Vector<double> const &temperature,
      Kokkos::View<double **, LayoutType, kokkos_host> state,
      std::unordered_map<dealii::types::global_dof_index, unsigned int> const
          &dofs_map,
      dealii::DoFHandler<dim> const &material_dof_handler);

  /**
   * Write the different vtu and pvtu files for a mechanical problem.
   */
  template <typename LayoutType>
  void write_mechanical_output(
      unsigned int time_step, double time,
      dealii::LA::distributed::Vector<double> const &displacement,
      std::vector<std::vector<dealii::SymmetricTensor<2, dim>>> const
          &stress_tensor,
      Kokkos::View<double **, LayoutType, kokkos_host> state,
      std::unordered_map<dealii::types::global_dof_index, unsigned int> const
          &dofs_map,
      dealii::DoFHandler<dim> const &material_dof_handler);

  /**
   * Write the different vtu and pvtu files for themo-mechanical problems.
   */
  template <typename LayoutType>
  void
  write_output(unsigned int time_step, double time,
               dealii::LA::distributed::Vector<double> const &temperature,
               dealii::LA::distributed::Vector<double> const &displacement,
               std::vector<std::vector<dealii::SymmetricTensor<2, dim>>> const
                   &stress_tensor,
               Kokkos::View<double **, LayoutType, kokkos_host> state,
               std::unordered_map<dealii::types::global_dof_index,
                                  unsigned int> const &dofs_map,
               dealii::DoFHandler<dim> const &material_dof_handler);

  /**
   * Write the pvd file for Paraview.
   */
  void write_pvd() const;

private:
  /**
   * Compute the norm of the stress.
   */
  dealii::Vector<double> get_stress_norm(
      std::vector<std::vector<dealii::SymmetricTensor<2, dim>>> const
          &stress_tensor);
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
      StrainPostProcessor<dim> const &strain,
      std::vector<std::vector<dealii::SymmetricTensor<2, dim>>> const
          &stress_tensor);
  /**
   * Fill _data_out with material data.
   */
  template <typename LayoutType>
  void material_dataout(Kokkos::View<double **, LayoutType, kokkos_host> state,
                        std::unordered_map<dealii::types::global_dof_index,
                                           unsigned int> const &dofs_map,
                        dealii::DoFHandler<dim> const &material_dof_handler);
  /**
   * Fill _data_out with subdomain data.
   */
  void subdomain_dataout();
  /**
   * Write pvtu file.
   */
  void write_pvtu(unsigned int time_step, double time);

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
  /**
   * Additional levels of refinement for the output.
   */
  unsigned int _additional_output_refinement;
};

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
template <typename LayoutType>
void PostProcessor<dim>::write_thermal_output(
    unsigned int time_step, double time,
    dealii::LA::distributed::Vector<double> const &temperature,
    Kokkos::View<double **, LayoutType, kokkos_host> state,
    std::unordered_map<dealii::types::global_dof_index, unsigned int> const
        &dofs_map,
    dealii::DoFHandler<dim> const &material_dof_handler)
{
  ASSERT(_thermal_dof_handler != nullptr, "Internal Error");
  _data_out.clear();
  thermal_dataout(temperature);
  material_dataout(state, dofs_map, material_dof_handler);
  subdomain_dataout();
  write_pvtu(time_step, time);
}

template <int dim>
template <typename LayoutType>
void PostProcessor<dim>::write_mechanical_output(
    unsigned int time_step, double time,
    dealii::LA::distributed::Vector<double> const &displacement,
    std::vector<std::vector<dealii::SymmetricTensor<2, dim>>> const
        &stress_tensor,
    Kokkos::View<double **, LayoutType, kokkos_host> state,
    std::unordered_map<dealii::types::global_dof_index, unsigned int> const
        &dofs_map,
    dealii::DoFHandler<dim> const &material_dof_handler)
{
  ASSERT(_mechanical_dof_handler != nullptr, "Internal Error");
  _data_out.clear();
  // We need the StrainPostProcessor to live until write_pvtu is done
  StrainPostProcessor<dim> strain;
  mechanical_dataout(displacement, strain, stress_tensor);
  material_dataout(state, dofs_map, material_dof_handler);
  subdomain_dataout();
  write_pvtu(time_step, time);
}

template <int dim>
template <typename LayoutType>
void PostProcessor<dim>::write_output(
    unsigned int time_step, double time,
    dealii::LA::distributed::Vector<double> const &temperature,
    dealii::LA::distributed::Vector<double> const &displacement,
    std::vector<std::vector<dealii::SymmetricTensor<2, dim>>> const
        &stress_tensor,
    Kokkos::View<double **, LayoutType, kokkos_host> state,
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
  mechanical_dataout(displacement, strain, stress_tensor);
  material_dataout(state, dofs_map, material_dof_handler);
  subdomain_dataout();
  write_pvtu(time_step, time);
}

template <int dim>
template <typename LayoutType>
void PostProcessor<dim>::material_dataout(
    Kokkos::View<double **, LayoutType, kokkos_host> state,
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
      static_cast<unsigned int>(SolidLiquidPowder::State::powder);
  unsigned int constexpr liquid_index =
      static_cast<unsigned int>(SolidLiquidPowder::State::liquid);
  unsigned int constexpr solid_index =
      static_cast<unsigned int>(SolidLiquidPowder::State::solid);
  auto mp_cell = material_dof_handler.begin_active();
  auto mp_end_cell = material_dof_handler.end();
  std::vector<dealii::types::global_dof_index> mp_dof_indices(1);
  for (unsigned int i = 0; mp_cell != mp_end_cell; ++i, ++mp_cell)
    if (mp_cell->is_locally_owned())
    {
      mp_cell->get_dof_indices(mp_dof_indices);
      dealii::types::global_dof_index const mp_dof_index =
          dofs_map.at(mp_dof_indices[0]);
      solid[i] = state(solid_index, mp_dof_index);
      liquid[i] = liquid_index < state.extent(0)
                      ? state(liquid_index, mp_dof_index)
                      : 0.;
      powder[i] = powder_index < state.extent(0)
                      ? state(powder_index, mp_dof_index)
                      : 0;
    }
  _data_out.add_data_vector(powder, "powder");
  _data_out.add_data_vector(liquid, "liquid");
  _data_out.add_data_vector(solid, "solid");
}
} // namespace adamantine
#endif
