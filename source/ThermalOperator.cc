/* Copyright (c) 2016, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#include "ThermalOperator.templates.hh"

namespace adamantine
{
// Instantiate the templates.
template class ThermalOperator<2, 1, float>;
template class ThermalOperator<2, 2, float>;
template class ThermalOperator<2, 3, float>;
template class ThermalOperator<2, 4, float>;
template class ThermalOperator<2, 5, float>;
template class ThermalOperator<2, 6, float>;
template class ThermalOperator<2, 7, float>;
template class ThermalOperator<2, 8, float>;
template class ThermalOperator<2, 9, float>;
template class ThermalOperator<2, 10, float>;

template void ThermalOperator<2, 1, float>::reinit<dealii::QGauss<1>>(
    dealii::DoFHandler<2> const &, dealii::ConstraintMatrix const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<2, 2, float>::reinit<dealii::QGauss<1>>(
    dealii::DoFHandler<2> const &, dealii::ConstraintMatrix const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<2, 3, float>::reinit<dealii::QGauss<1>>(
    dealii::DoFHandler<2> const &, dealii::ConstraintMatrix const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<2, 4, float>::reinit<dealii::QGauss<1>>(
    dealii::DoFHandler<2> const &, dealii::ConstraintMatrix const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<2, 5, float>::reinit<dealii::QGauss<1>>(
    dealii::DoFHandler<2> const &, dealii::ConstraintMatrix const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<2, 6, float>::reinit<dealii::QGauss<1>>(
    dealii::DoFHandler<2> const &, dealii::ConstraintMatrix const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<2, 7, float>::reinit<dealii::QGauss<1>>(
    dealii::DoFHandler<2> const &, dealii::ConstraintMatrix const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<2, 8, float>::reinit<dealii::QGauss<1>>(
    dealii::DoFHandler<2> const &, dealii::ConstraintMatrix const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<2, 9, float>::reinit<dealii::QGauss<1>>(
    dealii::DoFHandler<2> const &, dealii::ConstraintMatrix const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<2, 10, float>::reinit<dealii::QGauss<1>>(
    dealii::DoFHandler<2> const &, dealii::ConstraintMatrix const &,
    dealii::QGauss<1> const &);

template void ThermalOperator<2, 1, float>::reinit<dealii::QGaussLobatto<1>>(
    dealii::DoFHandler<2> const &, dealii::ConstraintMatrix const &,
    dealii::QGaussLobatto<1> const &);
template void ThermalOperator<2, 2, float>::reinit<dealii::QGaussLobatto<1>>(
    dealii::DoFHandler<2> const &, dealii::ConstraintMatrix const &,
    dealii::QGaussLobatto<1> const &);
template void ThermalOperator<2, 3, float>::reinit<dealii::QGaussLobatto<1>>(
    dealii::DoFHandler<2> const &, dealii::ConstraintMatrix const &,
    dealii::QGaussLobatto<1> const &);
template void ThermalOperator<2, 4, float>::reinit<dealii::QGaussLobatto<1>>(
    dealii::DoFHandler<2> const &, dealii::ConstraintMatrix const &,
    dealii::QGaussLobatto<1> const &);
template void ThermalOperator<2, 5, float>::reinit<dealii::QGaussLobatto<1>>(
    dealii::DoFHandler<2> const &, dealii::ConstraintMatrix const &,
    dealii::QGaussLobatto<1> const &);
template void ThermalOperator<2, 6, float>::reinit<dealii::QGaussLobatto<1>>(
    dealii::DoFHandler<2> const &, dealii::ConstraintMatrix const &,
    dealii::QGaussLobatto<1> const &);
template void ThermalOperator<2, 7, float>::reinit<dealii::QGaussLobatto<1>>(
    dealii::DoFHandler<2> const &, dealii::ConstraintMatrix const &,
    dealii::QGaussLobatto<1> const &);
template void ThermalOperator<2, 8, float>::reinit<dealii::QGaussLobatto<1>>(
    dealii::DoFHandler<2> const &, dealii::ConstraintMatrix const &,
    dealii::QGaussLobatto<1> const &);
template void ThermalOperator<2, 9, float>::reinit<dealii::QGaussLobatto<1>>(
    dealii::DoFHandler<2> const &, dealii::ConstraintMatrix const &,
    dealii::QGaussLobatto<1> const &);
template void ThermalOperator<2, 10, float>::reinit<dealii::QGaussLobatto<1>>(
    dealii::DoFHandler<2> const &, dealii::ConstraintMatrix const &,
    dealii::QGaussLobatto<1> const &);

template class ThermalOperator<2, 1, double>;
template class ThermalOperator<2, 2, double>;
template class ThermalOperator<2, 3, double>;
template class ThermalOperator<2, 4, double>;
template class ThermalOperator<2, 5, double>;
template class ThermalOperator<2, 6, double>;
template class ThermalOperator<2, 7, double>;
template class ThermalOperator<2, 8, double>;
template class ThermalOperator<2, 9, double>;
template class ThermalOperator<2, 10, double>;

template void ThermalOperator<2, 1, double>::reinit<dealii::QGauss<1>>(
    dealii::DoFHandler<2> const &, dealii::ConstraintMatrix const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<2, 2, double>::reinit<dealii::QGauss<1>>(
    dealii::DoFHandler<2> const &, dealii::ConstraintMatrix const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<2, 3, double>::reinit<dealii::QGauss<1>>(
    dealii::DoFHandler<2> const &, dealii::ConstraintMatrix const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<2, 4, double>::reinit<dealii::QGauss<1>>(
    dealii::DoFHandler<2> const &, dealii::ConstraintMatrix const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<2, 5, double>::reinit<dealii::QGauss<1>>(
    dealii::DoFHandler<2> const &, dealii::ConstraintMatrix const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<2, 6, double>::reinit<dealii::QGauss<1>>(
    dealii::DoFHandler<2> const &, dealii::ConstraintMatrix const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<2, 7, double>::reinit<dealii::QGauss<1>>(
    dealii::DoFHandler<2> const &, dealii::ConstraintMatrix const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<2, 8, double>::reinit<dealii::QGauss<1>>(
    dealii::DoFHandler<2> const &, dealii::ConstraintMatrix const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<2, 9, double>::reinit<dealii::QGauss<1>>(
    dealii::DoFHandler<2> const &, dealii::ConstraintMatrix const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<2, 10, double>::reinit<dealii::QGauss<1>>(
    dealii::DoFHandler<2> const &, dealii::ConstraintMatrix const &,
    dealii::QGauss<1> const &);

template void ThermalOperator<2, 1, double>::reinit<dealii::QGaussLobatto<1>>(
    dealii::DoFHandler<2> const &, dealii::ConstraintMatrix const &,
    dealii::QGaussLobatto<1> const &);
template void ThermalOperator<2, 2, double>::reinit<dealii::QGaussLobatto<1>>(
    dealii::DoFHandler<2> const &, dealii::ConstraintMatrix const &,
    dealii::QGaussLobatto<1> const &);
template void ThermalOperator<2, 3, double>::reinit<dealii::QGaussLobatto<1>>(
    dealii::DoFHandler<2> const &, dealii::ConstraintMatrix const &,
    dealii::QGaussLobatto<1> const &);
template void ThermalOperator<2, 4, double>::reinit<dealii::QGaussLobatto<1>>(
    dealii::DoFHandler<2> const &, dealii::ConstraintMatrix const &,
    dealii::QGaussLobatto<1> const &);
template void ThermalOperator<2, 5, double>::reinit<dealii::QGaussLobatto<1>>(
    dealii::DoFHandler<2> const &, dealii::ConstraintMatrix const &,
    dealii::QGaussLobatto<1> const &);
template void ThermalOperator<2, 6, double>::reinit<dealii::QGaussLobatto<1>>(
    dealii::DoFHandler<2> const &, dealii::ConstraintMatrix const &,
    dealii::QGaussLobatto<1> const &);
template void ThermalOperator<2, 7, double>::reinit<dealii::QGaussLobatto<1>>(
    dealii::DoFHandler<2> const &, dealii::ConstraintMatrix const &,
    dealii::QGaussLobatto<1> const &);
template void ThermalOperator<2, 8, double>::reinit<dealii::QGaussLobatto<1>>(
    dealii::DoFHandler<2> const &, dealii::ConstraintMatrix const &,
    dealii::QGaussLobatto<1> const &);
template void ThermalOperator<2, 9, double>::reinit<dealii::QGaussLobatto<1>>(
    dealii::DoFHandler<2> const &, dealii::ConstraintMatrix const &,
    dealii::QGaussLobatto<1> const &);
template void ThermalOperator<2, 10, double>::reinit<dealii::QGaussLobatto<1>>(
    dealii::DoFHandler<2> const &, dealii::ConstraintMatrix const &,
    dealii::QGaussLobatto<1> const &);

template class ThermalOperator<3, 1, float>;
template class ThermalOperator<3, 2, float>;
template class ThermalOperator<3, 3, float>;
template class ThermalOperator<3, 4, float>;
template class ThermalOperator<3, 5, float>;
template class ThermalOperator<3, 6, float>;
template class ThermalOperator<3, 7, float>;
template class ThermalOperator<3, 8, float>;
template class ThermalOperator<3, 9, float>;
template class ThermalOperator<3, 10, float>;

template void ThermalOperator<3, 1, float>::reinit<dealii::QGauss<1>>(
    dealii::DoFHandler<3> const &, dealii::ConstraintMatrix const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<3, 2, float>::reinit<dealii::QGauss<1>>(
    dealii::DoFHandler<3> const &, dealii::ConstraintMatrix const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<3, 3, float>::reinit<dealii::QGauss<1>>(
    dealii::DoFHandler<3> const &, dealii::ConstraintMatrix const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<3, 4, float>::reinit<dealii::QGauss<1>>(
    dealii::DoFHandler<3> const &, dealii::ConstraintMatrix const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<3, 5, float>::reinit<dealii::QGauss<1>>(
    dealii::DoFHandler<3> const &, dealii::ConstraintMatrix const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<3, 6, float>::reinit<dealii::QGauss<1>>(
    dealii::DoFHandler<3> const &, dealii::ConstraintMatrix const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<3, 7, float>::reinit<dealii::QGauss<1>>(
    dealii::DoFHandler<3> const &, dealii::ConstraintMatrix const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<3, 8, float>::reinit<dealii::QGauss<1>>(
    dealii::DoFHandler<3> const &, dealii::ConstraintMatrix const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<3, 9, float>::reinit<dealii::QGauss<1>>(
    dealii::DoFHandler<3> const &, dealii::ConstraintMatrix const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<3, 10, float>::reinit<dealii::QGauss<1>>(
    dealii::DoFHandler<3> const &, dealii::ConstraintMatrix const &,
    dealii::QGauss<1> const &);

template void ThermalOperator<3, 1, float>::reinit<dealii::QGaussLobatto<1>>(
    dealii::DoFHandler<3> const &, dealii::ConstraintMatrix const &,
    dealii::QGaussLobatto<1> const &);
template void ThermalOperator<3, 2, float>::reinit<dealii::QGaussLobatto<1>>(
    dealii::DoFHandler<3> const &, dealii::ConstraintMatrix const &,
    dealii::QGaussLobatto<1> const &);
template void ThermalOperator<3, 3, float>::reinit<dealii::QGaussLobatto<1>>(
    dealii::DoFHandler<3> const &, dealii::ConstraintMatrix const &,
    dealii::QGaussLobatto<1> const &);
template void ThermalOperator<3, 4, float>::reinit<dealii::QGaussLobatto<1>>(
    dealii::DoFHandler<3> const &, dealii::ConstraintMatrix const &,
    dealii::QGaussLobatto<1> const &);
template void ThermalOperator<3, 5, float>::reinit<dealii::QGaussLobatto<1>>(
    dealii::DoFHandler<3> const &, dealii::ConstraintMatrix const &,
    dealii::QGaussLobatto<1> const &);
template void ThermalOperator<3, 6, float>::reinit<dealii::QGaussLobatto<1>>(
    dealii::DoFHandler<3> const &, dealii::ConstraintMatrix const &,
    dealii::QGaussLobatto<1> const &);
template void ThermalOperator<3, 7, float>::reinit<dealii::QGaussLobatto<1>>(
    dealii::DoFHandler<3> const &, dealii::ConstraintMatrix const &,
    dealii::QGaussLobatto<1> const &);
template void ThermalOperator<3, 8, float>::reinit<dealii::QGaussLobatto<1>>(
    dealii::DoFHandler<3> const &, dealii::ConstraintMatrix const &,
    dealii::QGaussLobatto<1> const &);
template void ThermalOperator<3, 9, float>::reinit<dealii::QGaussLobatto<1>>(
    dealii::DoFHandler<3> const &, dealii::ConstraintMatrix const &,
    dealii::QGaussLobatto<1> const &);
template void ThermalOperator<3, 10, float>::reinit<dealii::QGaussLobatto<1>>(
    dealii::DoFHandler<3> const &, dealii::ConstraintMatrix const &,
    dealii::QGaussLobatto<1> const &);

template class ThermalOperator<3, 1, double>;
template class ThermalOperator<3, 2, double>;
template class ThermalOperator<3, 3, double>;
template class ThermalOperator<3, 4, double>;
template class ThermalOperator<3, 5, double>;
template class ThermalOperator<3, 6, double>;
template class ThermalOperator<3, 7, double>;
template class ThermalOperator<3, 8, double>;
template class ThermalOperator<3, 9, double>;
template class ThermalOperator<3, 10, double>;

template void ThermalOperator<3, 1, double>::reinit<dealii::QGauss<1>>(
    dealii::DoFHandler<3> const &, dealii::ConstraintMatrix const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<3, 2, double>::reinit<dealii::QGauss<1>>(
    dealii::DoFHandler<3> const &, dealii::ConstraintMatrix const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<3, 3, double>::reinit<dealii::QGauss<1>>(
    dealii::DoFHandler<3> const &, dealii::ConstraintMatrix const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<3, 4, double>::reinit<dealii::QGauss<1>>(
    dealii::DoFHandler<3> const &, dealii::ConstraintMatrix const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<3, 5, double>::reinit<dealii::QGauss<1>>(
    dealii::DoFHandler<3> const &, dealii::ConstraintMatrix const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<3, 6, double>::reinit<dealii::QGauss<1>>(
    dealii::DoFHandler<3> const &, dealii::ConstraintMatrix const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<3, 7, double>::reinit<dealii::QGauss<1>>(
    dealii::DoFHandler<3> const &, dealii::ConstraintMatrix const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<3, 8, double>::reinit<dealii::QGauss<1>>(
    dealii::DoFHandler<3> const &, dealii::ConstraintMatrix const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<3, 9, double>::reinit<dealii::QGauss<1>>(
    dealii::DoFHandler<3> const &, dealii::ConstraintMatrix const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<3, 10, double>::reinit<dealii::QGauss<1>>(
    dealii::DoFHandler<3> const &, dealii::ConstraintMatrix const &,
    dealii::QGauss<1> const &);

template void ThermalOperator<3, 1, double>::reinit<dealii::QGaussLobatto<1>>(
    dealii::DoFHandler<3> const &, dealii::ConstraintMatrix const &,
    dealii::QGaussLobatto<1> const &);
template void ThermalOperator<3, 2, double>::reinit<dealii::QGaussLobatto<1>>(
    dealii::DoFHandler<3> const &, dealii::ConstraintMatrix const &,
    dealii::QGaussLobatto<1> const &);
template void ThermalOperator<3, 3, double>::reinit<dealii::QGaussLobatto<1>>(
    dealii::DoFHandler<3> const &, dealii::ConstraintMatrix const &,
    dealii::QGaussLobatto<1> const &);
template void ThermalOperator<3, 4, double>::reinit<dealii::QGaussLobatto<1>>(
    dealii::DoFHandler<3> const &, dealii::ConstraintMatrix const &,
    dealii::QGaussLobatto<1> const &);
template void ThermalOperator<3, 5, double>::reinit<dealii::QGaussLobatto<1>>(
    dealii::DoFHandler<3> const &, dealii::ConstraintMatrix const &,
    dealii::QGaussLobatto<1> const &);
template void ThermalOperator<3, 6, double>::reinit<dealii::QGaussLobatto<1>>(
    dealii::DoFHandler<3> const &, dealii::ConstraintMatrix const &,
    dealii::QGaussLobatto<1> const &);
template void ThermalOperator<3, 7, double>::reinit<dealii::QGaussLobatto<1>>(
    dealii::DoFHandler<3> const &, dealii::ConstraintMatrix const &,
    dealii::QGaussLobatto<1> const &);
template void ThermalOperator<3, 8, double>::reinit<dealii::QGaussLobatto<1>>(
    dealii::DoFHandler<3> const &, dealii::ConstraintMatrix const &,
    dealii::QGaussLobatto<1> const &);
template void ThermalOperator<3, 9, double>::reinit<dealii::QGaussLobatto<1>>(
    dealii::DoFHandler<3> const &, dealii::ConstraintMatrix const &,
    dealii::QGaussLobatto<1> const &);
template void ThermalOperator<3, 10, double>::reinit<dealii::QGaussLobatto<1>>(
    dealii::DoFHandler<3> const &, dealii::ConstraintMatrix const &,
    dealii::QGaussLobatto<1> const &);
}
