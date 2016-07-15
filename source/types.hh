/* Copyright (c) 2016, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#ifndef _TYPES_HH_
#define _TYPES_HH_

namespace dealii
{
namespace LinearAlgebra
{
}

/**
 * Shorten dealii::LinearAlgebra to dealii::LA.
 */
namespace LA = LinearAlgebra;
}

namespace adamantine
{
/**
 * Enum on the possible materials.
 */
enum MaterialState
{
  powder,
  solid,
  liquid
};

/**
 * Enum of the possible material properties.
 */
enum Property
{
  thermal_conductivity
};
}

#endif
