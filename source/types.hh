/* Copyright (c) 2016 - 2024, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#ifndef TYPES_HH
#define TYPES_HH

#include <Kokkos_NumericTraits.hpp>

#include <array>
#include <string>

namespace dealii
{
namespace LinearAlgebra
{
}

/**
 * Shorten dealii::LinearAlgebra to dealii::LA.
 */
namespace LA = LinearAlgebra;
} // namespace dealii

namespace adamantine
{
/**
 * Enum on the possible material properties that depend on the state of the
 * material.
 */
enum class StateProperty
{
  density,
  specific_heat,
  thermal_conductivity_x,
  thermal_conductivity_y,
  thermal_conductivity_z,
  emissivity,
  radiation_heat_transfer_coef,
  convection_heat_transfer_coef,
  // Mechanical properties only make sense for the solid state. They do not make
  // sense for the powder and the liquid state
  lame_first_parameter,
  lame_second_parameter,
  thermal_expansion_coef,
  // Density is used both by the thermal and the mechanical simulation. We need
  // to duplicate the property because the mechanical code cannot run on the
  // GPU.
  density_s,
  plastic_modulus,
  isotropic_hardening,
  elastic_limit,
  SIZE,
  SIZE_MECHANICAL = 7
};

/**
 * Number of StateProperty defined.
 */
static unsigned int constexpr g_n_state_properties =
    static_cast<unsigned int>(StateProperty::SIZE);

/**
 * Number of mechanical StateProperty.
 */
static unsigned int constexpr g_n_mechanical_state_properties =
    static_cast<unsigned int>(StateProperty::SIZE_MECHANICAL);

/**
 * Number of thermal StateProperty.
 */
static unsigned int constexpr g_n_thermal_state_properties =
    g_n_state_properties - g_n_mechanical_state_properties;

/**
 * Enum on the possible material properties that do not depend on the state of
 * the material.
 */
enum class Property
{
  liquidus,
  solidus,
  latent_heat,
  radiation_temperature_infty,
  convection_temperature_infty,
  SIZE
};

/**
 * Number of Property defined.
 */
static unsigned int constexpr g_n_properties =
    static_cast<unsigned int>(Property::SIZE);

/**
 * Array containing the possible material states.
 */
static std::array<std::string, 3> const material_state_names = {
    {"solid", "liquid", "powder"}};

/**
 * Array continaing the possible material properties that do not depend on the
 * state of the material.
 */
static std::array<std::string, 5> const property_names = {
    {"liquidus", "solidus", "latent_heat", "radiation_temperature_infty",
     "convection_temperature_infty"}};

/**
 * Array containing the possible material properties that depend on the
 * state of the material.
 */
static std::array<std::string, 15> const state_property_names = {
    {"density", "specific_heat", "thermal_conductivity_x",
     "thermal_conductivity_y", "thermal_conductivity_z", "emissivity",
     "radiation_heat_transfer_coef", "convection_heat_transfer_coef",
     "lame_first_parameter", "lame_second_parameter", "thermal_expansion_coef",
     "density", "plastic_modulus", "isotropic_hardening", "elastic_limit"}};

/**
 * Enum on the possible timers.
 */
enum Timing
{
  main,
  refine,
  add_material_search,
  add_material_activate,
  da_experimental_data,
  da_dof_mapping,
  da_covariance_sparsity,
  da_obs_covariance,
  da_update_ensemble,
  evol_time,
  evol_time_eval_th_ph,
  evol_time_J_inv,
  evol_time_update_bound_mat_prop,
  output,
  n_timers
};

/**
 * Structure that stores constants.
 */
struct Constant
{
  /**
   * Stefan-Boltzmann constant. Value from NIST [w/(m^2 k^4)].
   */
  static double constexpr stefan_boltzmann = 5.670374419e-8;
};

/**
 * This structure provides a mapping between the axes x, y, and z and the
 * indices 0, 1, and 2. In 2D, the valid axes are x and z while in 3D x, y, and
 * z are valid.
 */
template <int dim>
struct axis;

// dim == 2 specialization
template <>
struct axis<2>
{
  static unsigned int constexpr x = 0;
  static unsigned int constexpr y =
      Kokkos::Experimental::finite_max_v<unsigned int>;
  static unsigned int constexpr z = 1;
};

// dim == 3 specialization
template <>
struct axis<3>
{
  static unsigned int constexpr x = 0;
  static unsigned int constexpr y = 1;
  static unsigned int constexpr z = 2;
};

/**
 * Enum on the different types of boundary condition supported. Some of them can
 * be combined, for example radiative and convective.
 */
enum BoundaryType
{
  invalid = 0,
  adiabatic = 0x1,
  radiative = 0x2,
  convective = 0x4,
};

/**
 * Global operator which returns an object in which all bits are set which are
 * either set in the first or the second argument. This operator exists since if
 * it did not then the result of the bit-or operator | would be an integer which
 * would in turn trigger a compiler warning when we tried to assign it to an
 * object of type BoundaryType.
 */
inline BoundaryType operator|(const BoundaryType b1, const BoundaryType b2)
{
  return static_cast<BoundaryType>(static_cast<unsigned int>(b1) |
                                   static_cast<unsigned int>(b2));
}

/**
 * Global operator which sets the bits from the second argument also in the
 * first one.
 */
inline BoundaryType &operator|=(BoundaryType &b1, const BoundaryType b2)
{
  b1 = b1 | b2;
  return b1;
}

/**
 * Global operator which returns an object in which all bits are set which are
 * set in the first as well as the second argument. This operator exists since
 * if it did not then the result of the bit-and operator & would be an integer
 * which would in turn trigger a compiler warning when we tried to assign it to
 * an object of type BoundaryType.
 */
inline BoundaryType operator&(const BoundaryType b1, const BoundaryType b2)
{
  return static_cast<BoundaryType>(static_cast<unsigned int>(b1) &
                                   static_cast<unsigned int>(b2));
}

/**
 * Global operator which clears all the bits in the first argument if they are
 * not also set in the second argument.
 */
inline BoundaryType &operator&=(BoundaryType &b1, const BoundaryType b2)
{
  b1 = b1 & b2;
  return b1;
}

} // namespace adamantine

#endif
