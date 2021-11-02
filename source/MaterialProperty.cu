/* Copyright (c) 2021, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#include <MaterialProperty.templates.hh>

namespace adamantine
{
template class MaterialProperty<2, dealii::MemorySpace::CUDA>;
template class MaterialProperty<3, dealii::MemorySpace::CUDA>;
} // namespace adamantine
