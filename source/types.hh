/* Copyright (c) 2016 - 2019, the adamantine authors.
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
 * Enum on the possible material properties.
 */
enum class Property
{
  density,
  latent_heat,
  liquidus,
  solidus,
  specific_heat,
  thermal_conductivity,
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
} // namespace adamantine

#endif
