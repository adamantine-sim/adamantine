/* Copyright (c) 2016, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#ifndef _THERMAL_OPERATOR_TEMPLATES_HH_
#define _THERMAL_OPERATOR_TEMPLATES_HH_

#include "ThermalOperator.hh"
#include <deal.II/base/index_set.h>

namespace adamantine
{
template <int dim, int fe_degree, typename NumberType>
ThermalOperator<dim, fe_degree, NumberType>::ThermalOperator(boost::mpi::communicator &communicator) 
:
_communicator(communicator)
{
}

template <int dim, int fe_degree, typename NumberType>
template <typename QuadratureType>
void ThermalOperator<dim, fe_degree, NumberType>::reinit(dealii::DoFHandler<dim> const &dof_handler,
                                                         dealii::ConstraintMatrix const &constraint_matrix,
                                                         QuadratureType const &quad)
{
  _data(dof_handler, constraint_matrix, quad);
  _inverse_mass_matrix.reinit(_data.get_locally_owned_set, _communicator);
}

template <int dim, int fe_degree, typename NumberType>
void ThermalOperator<dim, fe_degree, NumberType>::clear()
{
  _data.clear();
  _inverse_mass_matrix.reinit(0);
}
}

#endif
