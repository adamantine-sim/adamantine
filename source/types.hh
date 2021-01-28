/* Copyright (c) 2016 - 2021, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#ifndef TYPES_HH
#define TYPES_HH

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
 * Enum on the possible materials.
 */
enum class MaterialState
{
  powder,
  solid,
  liquid,
  SIZE
};

/**
 * Enum on the possible material properties that depend on the state of the
 * material.
 */
// TODO add AnisotropicStateProperty
enum class StateProperty
{
  density,
  specific_heat,
  thermal_conductivity,
  SIZE
};

/**
 * Enum on the possible material properties that do not depend on the state of
 * the material.
 */
enum class Property
{
  liquidus,
  solidus,
  latent_heat,
  SIZE
};

/**
 * Enum on the possible timers.
 */
enum Timing
{
  main,
  refine,
  evol_time,
  evol_time_eval_th_ph,
  evol_time_J_inv,
  evol_time_eval_mat_prop
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
  static int constexpr x = 0;
  static int constexpr y = -1;
  static int constexpr z = 1;
};

// dim == 3 specialization
template <>
struct axis<3>
{
  static int constexpr x = 0;
  static int constexpr y = 1;
  static int constexpr z = 2;
};
} // namespace adamantine

#endif
