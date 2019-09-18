/* Copyright (c) 2016, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#include "MaterialProperty.templates.hh"
#include "instantiation.hh"

INSTANTIATE_DIM(MaterialProperty)

namespace adamantine
{
// Instantiate templated function: get
template double MaterialProperty<2>::get(
    dealii::Triangulation<2>::active_cell_iterator const &, Property prop,
    dealii::LA::distributed::Vector<float> const &) const;
template double MaterialProperty<2>::get(
    dealii::Triangulation<2>::active_cell_iterator const &, Property prop,
    dealii::LA::distributed::Vector<double> const &) const;
template double MaterialProperty<3>::get(
    dealii::Triangulation<3>::active_cell_iterator const &, Property prop,
    dealii::LA::distributed::Vector<float> const &) const;
template double MaterialProperty<3>::get(
    dealii::Triangulation<3>::active_cell_iterator const &, Property prop,
    dealii::LA::distributed::Vector<double> const &) const;

// Instantiate templated function: update_state
template void MaterialProperty<2>::update_state(
    dealii::DoFHandler<2> const &,
    dealii::LA::distributed::Vector<float> const &);
template void MaterialProperty<2>::update_state(
    dealii::DoFHandler<2> const &,
    dealii::LA::distributed::Vector<double> const &);
template void MaterialProperty<3>::update_state(
    dealii::DoFHandler<3> const &,
    dealii::LA::distributed::Vector<float> const &);
template void MaterialProperty<3>::update_state(
    dealii::DoFHandler<3> const &,
    dealii::LA::distributed::Vector<double> const &);
} // namespace adamantine
