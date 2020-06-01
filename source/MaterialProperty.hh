/* Copyright (c) 2016 - 2020, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#ifndef MATERIAL_PROPERTY_HH
#define MATERIAL_PROPERTY_HH

#include <types.hh>
#include <utils.hh>

#include <deal.II/base/function_parser.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/la_vector.h>

#include <boost/property_tree/ptree.hpp>

#include <array>
#include <limits>
#include <unordered_map>

namespace adamantine
{
/**
 * This class stores the material properties for all the materials
 */
template <int dim>
class MaterialProperty
{
public:
  /**
   * Constructor.
   * \param[in] database requires the following entries:
   *   - <B>n_materials</B>: unsigned int in \f$(0,\infty)\f$
   *   - <B>material_X</B>: property tree associated with material_X
   *   where X is a number
   *   - <B>material_X.Y</B>: property_tree where Y is either liquid, powder, or
   *   solid [optional]
   *   - <B>material_X.Y.Z</B>: string where Z is either density, specific_heat,
   *   or thermal_conductivity, describe the behavior of the property as a
   *   function of the temperatur (e.g. "2.*T") [optional]
   *   - <B>material.X.A</B>: A is either solidus, liquidus, or latent_heat
   *   [optional]
   */
  MaterialProperty(
      MPI_Comm const &communicator,
      dealii::parallel::distributed::Triangulation<dim> const &tria,
      boost::property_tree::ptree const &database);

  /**
   * Return the value of the given property, for a given cell and a given field
   * state.
   */
  double
  get(typename dealii::Triangulation<dim>::active_cell_iterator const &cell,
      Property prop,
      dealii::LA::distributed::Vector<double> const &field_state) const;

  /**
   * Return the average temperature on every cell given the enthalpy.
   */
  dealii::LA::distributed::Vector<double> enthalpy_to_temperature(
      dealii::DoFHandler<dim> const &enthalpy_dof_handler,
      dealii::LA::distributed::Vector<double> const &enthalpy);

  /**
   * Reinitialize the DoFHandler associated with MaterialProperty and resize the
   * state vectors.
   */
  void reinit_dofs();

  /**
   * Update the material state, i.e, the ratio of liquid, powder, and solid.
   */
  void update_state(dealii::DoFHandler<dim> const &enthalpy_dof_handler,
                    dealii::LA::distributed::Vector<double> const &enthalpy);

  /**
   * Get the array of material state vectors. The order of the different state
   * vectos is given by the MaterialState enum. Each entry in the vector
   * correspond to a cell in the mesh and has a value between 0 and 1. The sum
   * of the states for a given cell is equal to 1.
   */
  std::array<dealii::LA::distributed::Vector<double>,
             static_cast<unsigned int>(MaterialState::SIZE)> &
  get_state();

  double get_state_ratio(
      typename dealii::Triangulation<dim>::active_cell_iterator const &cell,
      MaterialState material_state) const;

  /**
   * Return \f$ -\frac{H_{liquidus}}{\rho C_P} + T_{liquidus} \f$
   */
  double get_liquid_beta(
      typename dealii::Triangulation<dim>::active_cell_iterator const &cell)
      const;

  /**
   * Return \f$ \frac{T_{liquidus}-T_{solidus}}{\mathcal{L}} \f$
   */
  double get_mushy_alpha(
      typename dealii::Triangulation<dim>::active_cell_iterator const &cell)
      const;

  /**
   * Return \f$ -H_{solidus} \frac{T_{liquidus}-T_{solidus}}{\mathcal{L}} +
   * T_{solidus} \f$
   */
  double get_mushy_beta(
      typename dealii::Triangulation<dim>::active_cell_iterator const &cell)
      const;

  /**
   * Return the underlying the DoFHandler.
   */
  dealii::DoFHandler<dim> const &get_dof_handler() const;

private:
  /**
   * Maximum different number of states a given material can be.
   */
  static unsigned int constexpr _n_material_states =
      static_cast<unsigned int>(MaterialState::SIZE);

  /**
   * Number of properties defined.
   */
  static unsigned int constexpr _n_properties =
      static_cast<unsigned int>(Property::SIZE);

  /**
   * Set the values in _state from the values of the user index of the
   * Triangulation.
   */
  void set_state();

  /**
   * Fill the _properties map.
   */
  void fill_properties(boost::property_tree::ptree const &database);

  /**
   * If the density and the specific heat do not depend on the temperature, the
   * relationship between the temperature and the enthalpy can be written by
   * piecewise function \f$ T = \alpha H + \beta \f$. This function computes the
   * constants \f$ \alpha \f$ and \f$ \beta \f$.
   */
  void compute_constants();

  /**
   * Return the index of the dof associated to the cell.
   */
  double get_dof_index(
      typename dealii::Triangulation<dim>::active_cell_iterator const &cell)
      const;

  /**
   * Compute the average of the enthalpy on every cell.
   */
  dealii::LA::distributed::Vector<double> compute_average_enthalpy(
      dealii::DoFHandler<dim> const &enthalpy_dof_handler,
      dealii::LA::distributed::Vector<double> const &enthalpy) const;

  /**
   * MPI communicator.
   */
  MPI_Comm _communicator;
  /**
   * Map of \f$ -\frac{H_{liquidus}}{\rho C_P} + T_{liquidus} \f$ for each
   * material.
   */
  std::unordered_map<dealii::types::material_id, double> _liquid_beta;
  /**
   * Map of \f$ \frac{T_{liquidus}-T_{solidus}}{\mathcal{L}} \f$ for each
   * material.
   */
  std::unordered_map<dealii::types::material_id, double> _mushy_alpha;
  /**
   * Map of \f$ -H_{solidus} \frac{T_{liquidus}-T_{solidus}}{\mathcal{L}} +
   * T_{solidus} \f$ for each material.
   */
  std::unordered_map<dealii::types::material_id, double> _mushy_beta;
  /**
   * Map that stores functions describing the properties of the material.
   */
  std::unordered_map<
      dealii::types::material_id,
      std::array<
          std::array<std::unique_ptr<dealii::FunctionParser<1>>, _n_properties>,
          _n_material_states>>
      _properties;
  /**
   * Array of vector describing the ratio of each state in each cell. Each
   * vector corresponds to a state defined in the MaterialState enum.
   */
  std::array<dealii::LA::distributed::Vector<double>, _n_material_states>
      _state;
  /**
   * Discontinuous piecewise constant finite element.
   */
  dealii::FE_DGQ<dim> _fe;
  /**
   * DoFHandler associated to the _state array.
   */
  dealii::DoFHandler<dim> _mp_dof_handler;
};

template <int dim>
inline std::array<dealii::LA::distributed::Vector<double>,
                  static_cast<unsigned int>(MaterialState::SIZE)> &
MaterialProperty<dim>::get_state()
{
  return _state;
}

template <int dim>
inline double MaterialProperty<dim>::get_state_ratio(
    typename dealii::Triangulation<dim>::active_cell_iterator const &cell,
    MaterialState material_state) const
{
  double const mp_dof_index = get_dof_index(cell);
  unsigned int const mat_state = static_cast<unsigned int>(material_state);

  return _state[mat_state][mp_dof_index];
}

template <int dim>
inline double MaterialProperty<dim>::get_liquid_beta(
    typename dealii::Triangulation<dim>::active_cell_iterator const &cell) const
{
  dealii::types::material_id material_id = cell->material_id();
  auto const liquid_beta = _liquid_beta.find(material_id);
  ASSERT(liquid_beta != _liquid_beta.end(), "Material not found.");

  return liquid_beta->second;
}

template <int dim>
inline double MaterialProperty<dim>::get_mushy_alpha(
    typename dealii::Triangulation<dim>::active_cell_iterator const &cell) const
{
  dealii::types::material_id material_id = cell->material_id();
  auto const mushy_alpha = _mushy_alpha.find(material_id);
  ASSERT(mushy_alpha != _mushy_alpha.end(), "Material not found.");

  return mushy_alpha->second;
}

template <int dim>
inline double MaterialProperty<dim>::get_mushy_beta(
    typename dealii::Triangulation<dim>::active_cell_iterator const &cell) const
{
  dealii::types::material_id material_id = cell->material_id();
  auto const mushy_beta = _mushy_beta.find(material_id);
  ASSERT(mushy_beta != _mushy_beta.end(), "Material not found.");

  return mushy_beta->second;
}

template <int dim>
inline double MaterialProperty<dim>::get_dof_index(
    typename dealii::Triangulation<dim>::active_cell_iterator const &cell) const
{
  // Get a DoFCellAccessor from a Triangulation::active_cell_iterator.
  dealii::DoFAccessor<dim, dealii::DoFHandler<dim>, false> dof_accessor(
      &_mp_dof_handler.get_triangulation(), cell->level(), cell->index(),
      &_mp_dof_handler);
  std::vector<dealii::types::global_dof_index> mp_dof(1.);
  dof_accessor.get_dof_indices(mp_dof);

  return mp_dof[0];
}

template <int dim>
inline dealii::DoFHandler<dim> const &
MaterialProperty<dim>::get_dof_handler() const
{
  return _mp_dof_handler;
}
} // namespace adamantine

#endif
