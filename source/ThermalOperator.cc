/* Copyright (c) 2016, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#include "ThermalOperator.templates.hh"
#include "instantiation.hh"

INSTANTIATE_DIM_FEDEGREE_NUM(TUPLE(ThermalOperator))

// Instantiate the function template.
namespace adamantine
{
template void ThermalOperator<2, 1, float>::setup_dofs<dealii::QGauss<1>>(
    dealii::DoFHandler<2> const &, dealii::AffineConstraints<double> const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<2, 2, float>::setup_dofs<dealii::QGauss<1>>(
    dealii::DoFHandler<2> const &, dealii::AffineConstraints<double> const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<2, 3, float>::setup_dofs<dealii::QGauss<1>>(
    dealii::DoFHandler<2> const &, dealii::AffineConstraints<double> const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<2, 4, float>::setup_dofs<dealii::QGauss<1>>(
    dealii::DoFHandler<2> const &, dealii::AffineConstraints<double> const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<2, 5, float>::setup_dofs<dealii::QGauss<1>>(
    dealii::DoFHandler<2> const &, dealii::AffineConstraints<double> const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<2, 6, float>::setup_dofs<dealii::QGauss<1>>(
    dealii::DoFHandler<2> const &, dealii::AffineConstraints<double> const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<2, 7, float>::setup_dofs<dealii::QGauss<1>>(
    dealii::DoFHandler<2> const &, dealii::AffineConstraints<double> const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<2, 8, float>::setup_dofs<dealii::QGauss<1>>(
    dealii::DoFHandler<2> const &, dealii::AffineConstraints<double> const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<2, 9, float>::setup_dofs<dealii::QGauss<1>>(
    dealii::DoFHandler<2> const &, dealii::AffineConstraints<double> const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<2, 10, float>::setup_dofs<dealii::QGauss<1>>(
    dealii::DoFHandler<2> const &, dealii::AffineConstraints<double> const &,
    dealii::QGauss<1> const &);

template void
    ThermalOperator<2, 1, float>::setup_dofs<dealii::QGaussLobatto<1>>(
        dealii::DoFHandler<2> const &,
        dealii::AffineConstraints<double> const &,
        dealii::QGaussLobatto<1> const &);
template void
    ThermalOperator<2, 2, float>::setup_dofs<dealii::QGaussLobatto<1>>(
        dealii::DoFHandler<2> const &,
        dealii::AffineConstraints<double> const &,
        dealii::QGaussLobatto<1> const &);
template void
    ThermalOperator<2, 3, float>::setup_dofs<dealii::QGaussLobatto<1>>(
        dealii::DoFHandler<2> const &,
        dealii::AffineConstraints<double> const &,
        dealii::QGaussLobatto<1> const &);
template void
    ThermalOperator<2, 4, float>::setup_dofs<dealii::QGaussLobatto<1>>(
        dealii::DoFHandler<2> const &,
        dealii::AffineConstraints<double> const &,
        dealii::QGaussLobatto<1> const &);
template void
    ThermalOperator<2, 5, float>::setup_dofs<dealii::QGaussLobatto<1>>(
        dealii::DoFHandler<2> const &,
        dealii::AffineConstraints<double> const &,
        dealii::QGaussLobatto<1> const &);
template void
    ThermalOperator<2, 6, float>::setup_dofs<dealii::QGaussLobatto<1>>(
        dealii::DoFHandler<2> const &,
        dealii::AffineConstraints<double> const &,
        dealii::QGaussLobatto<1> const &);
template void
    ThermalOperator<2, 7, float>::setup_dofs<dealii::QGaussLobatto<1>>(
        dealii::DoFHandler<2> const &,
        dealii::AffineConstraints<double> const &,
        dealii::QGaussLobatto<1> const &);
template void
    ThermalOperator<2, 8, float>::setup_dofs<dealii::QGaussLobatto<1>>(
        dealii::DoFHandler<2> const &,
        dealii::AffineConstraints<double> const &,
        dealii::QGaussLobatto<1> const &);
template void
    ThermalOperator<2, 9, float>::setup_dofs<dealii::QGaussLobatto<1>>(
        dealii::DoFHandler<2> const &,
        dealii::AffineConstraints<double> const &,
        dealii::QGaussLobatto<1> const &);
template void
    ThermalOperator<2, 10, float>::setup_dofs<dealii::QGaussLobatto<1>>(
        dealii::DoFHandler<2> const &,
        dealii::AffineConstraints<double> const &,
        dealii::QGaussLobatto<1> const &);

template void ThermalOperator<2, 1, double>::setup_dofs<dealii::QGauss<1>>(
    dealii::DoFHandler<2> const &, dealii::AffineConstraints<double> const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<2, 2, double>::setup_dofs<dealii::QGauss<1>>(
    dealii::DoFHandler<2> const &, dealii::AffineConstraints<double> const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<2, 3, double>::setup_dofs<dealii::QGauss<1>>(
    dealii::DoFHandler<2> const &, dealii::AffineConstraints<double> const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<2, 4, double>::setup_dofs<dealii::QGauss<1>>(
    dealii::DoFHandler<2> const &, dealii::AffineConstraints<double> const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<2, 5, double>::setup_dofs<dealii::QGauss<1>>(
    dealii::DoFHandler<2> const &, dealii::AffineConstraints<double> const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<2, 6, double>::setup_dofs<dealii::QGauss<1>>(
    dealii::DoFHandler<2> const &, dealii::AffineConstraints<double> const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<2, 7, double>::setup_dofs<dealii::QGauss<1>>(
    dealii::DoFHandler<2> const &, dealii::AffineConstraints<double> const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<2, 8, double>::setup_dofs<dealii::QGauss<1>>(
    dealii::DoFHandler<2> const &, dealii::AffineConstraints<double> const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<2, 9, double>::setup_dofs<dealii::QGauss<1>>(
    dealii::DoFHandler<2> const &, dealii::AffineConstraints<double> const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<2, 10, double>::setup_dofs<dealii::QGauss<1>>(
    dealii::DoFHandler<2> const &, dealii::AffineConstraints<double> const &,
    dealii::QGauss<1> const &);

template void
    ThermalOperator<2, 1, double>::setup_dofs<dealii::QGaussLobatto<1>>(
        dealii::DoFHandler<2> const &,
        dealii::AffineConstraints<double> const &,
        dealii::QGaussLobatto<1> const &);
template void
    ThermalOperator<2, 2, double>::setup_dofs<dealii::QGaussLobatto<1>>(
        dealii::DoFHandler<2> const &,
        dealii::AffineConstraints<double> const &,
        dealii::QGaussLobatto<1> const &);
template void
    ThermalOperator<2, 3, double>::setup_dofs<dealii::QGaussLobatto<1>>(
        dealii::DoFHandler<2> const &,
        dealii::AffineConstraints<double> const &,
        dealii::QGaussLobatto<1> const &);
template void
    ThermalOperator<2, 4, double>::setup_dofs<dealii::QGaussLobatto<1>>(
        dealii::DoFHandler<2> const &,
        dealii::AffineConstraints<double> const &,
        dealii::QGaussLobatto<1> const &);
template void
    ThermalOperator<2, 5, double>::setup_dofs<dealii::QGaussLobatto<1>>(
        dealii::DoFHandler<2> const &,
        dealii::AffineConstraints<double> const &,
        dealii::QGaussLobatto<1> const &);
template void
    ThermalOperator<2, 6, double>::setup_dofs<dealii::QGaussLobatto<1>>(
        dealii::DoFHandler<2> const &,
        dealii::AffineConstraints<double> const &,
        dealii::QGaussLobatto<1> const &);
template void
    ThermalOperator<2, 7, double>::setup_dofs<dealii::QGaussLobatto<1>>(
        dealii::DoFHandler<2> const &,
        dealii::AffineConstraints<double> const &,
        dealii::QGaussLobatto<1> const &);
template void
    ThermalOperator<2, 8, double>::setup_dofs<dealii::QGaussLobatto<1>>(
        dealii::DoFHandler<2> const &,
        dealii::AffineConstraints<double> const &,
        dealii::QGaussLobatto<1> const &);
template void
    ThermalOperator<2, 9, double>::setup_dofs<dealii::QGaussLobatto<1>>(
        dealii::DoFHandler<2> const &,
        dealii::AffineConstraints<double> const &,
        dealii::QGaussLobatto<1> const &);
template void
    ThermalOperator<2, 10, double>::setup_dofs<dealii::QGaussLobatto<1>>(
        dealii::DoFHandler<2> const &,
        dealii::AffineConstraints<double> const &,
        dealii::QGaussLobatto<1> const &);

template void ThermalOperator<3, 1, float>::setup_dofs<dealii::QGauss<1>>(
    dealii::DoFHandler<3> const &, dealii::AffineConstraints<double> const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<3, 2, float>::setup_dofs<dealii::QGauss<1>>(
    dealii::DoFHandler<3> const &, dealii::AffineConstraints<double> const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<3, 3, float>::setup_dofs<dealii::QGauss<1>>(
    dealii::DoFHandler<3> const &, dealii::AffineConstraints<double> const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<3, 4, float>::setup_dofs<dealii::QGauss<1>>(
    dealii::DoFHandler<3> const &, dealii::AffineConstraints<double> const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<3, 5, float>::setup_dofs<dealii::QGauss<1>>(
    dealii::DoFHandler<3> const &, dealii::AffineConstraints<double> const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<3, 6, float>::setup_dofs<dealii::QGauss<1>>(
    dealii::DoFHandler<3> const &, dealii::AffineConstraints<double> const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<3, 7, float>::setup_dofs<dealii::QGauss<1>>(
    dealii::DoFHandler<3> const &, dealii::AffineConstraints<double> const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<3, 8, float>::setup_dofs<dealii::QGauss<1>>(
    dealii::DoFHandler<3> const &, dealii::AffineConstraints<double> const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<3, 9, float>::setup_dofs<dealii::QGauss<1>>(
    dealii::DoFHandler<3> const &, dealii::AffineConstraints<double> const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<3, 10, float>::setup_dofs<dealii::QGauss<1>>(
    dealii::DoFHandler<3> const &, dealii::AffineConstraints<double> const &,
    dealii::QGauss<1> const &);

template void
    ThermalOperator<3, 1, float>::setup_dofs<dealii::QGaussLobatto<1>>(
        dealii::DoFHandler<3> const &,
        dealii::AffineConstraints<double> const &,
        dealii::QGaussLobatto<1> const &);
template void
    ThermalOperator<3, 2, float>::setup_dofs<dealii::QGaussLobatto<1>>(
        dealii::DoFHandler<3> const &,
        dealii::AffineConstraints<double> const &,
        dealii::QGaussLobatto<1> const &);
template void
    ThermalOperator<3, 3, float>::setup_dofs<dealii::QGaussLobatto<1>>(
        dealii::DoFHandler<3> const &,
        dealii::AffineConstraints<double> const &,
        dealii::QGaussLobatto<1> const &);
template void
    ThermalOperator<3, 4, float>::setup_dofs<dealii::QGaussLobatto<1>>(
        dealii::DoFHandler<3> const &,
        dealii::AffineConstraints<double> const &,
        dealii::QGaussLobatto<1> const &);
template void
    ThermalOperator<3, 5, float>::setup_dofs<dealii::QGaussLobatto<1>>(
        dealii::DoFHandler<3> const &,
        dealii::AffineConstraints<double> const &,
        dealii::QGaussLobatto<1> const &);
template void
    ThermalOperator<3, 6, float>::setup_dofs<dealii::QGaussLobatto<1>>(
        dealii::DoFHandler<3> const &,
        dealii::AffineConstraints<double> const &,
        dealii::QGaussLobatto<1> const &);
template void
    ThermalOperator<3, 7, float>::setup_dofs<dealii::QGaussLobatto<1>>(
        dealii::DoFHandler<3> const &,
        dealii::AffineConstraints<double> const &,
        dealii::QGaussLobatto<1> const &);
template void
    ThermalOperator<3, 8, float>::setup_dofs<dealii::QGaussLobatto<1>>(
        dealii::DoFHandler<3> const &,
        dealii::AffineConstraints<double> const &,
        dealii::QGaussLobatto<1> const &);
template void
    ThermalOperator<3, 9, float>::setup_dofs<dealii::QGaussLobatto<1>>(
        dealii::DoFHandler<3> const &,
        dealii::AffineConstraints<double> const &,
        dealii::QGaussLobatto<1> const &);
template void
    ThermalOperator<3, 10, float>::setup_dofs<dealii::QGaussLobatto<1>>(
        dealii::DoFHandler<3> const &,
        dealii::AffineConstraints<double> const &,
        dealii::QGaussLobatto<1> const &);

template void ThermalOperator<3, 1, double>::setup_dofs<dealii::QGauss<1>>(
    dealii::DoFHandler<3> const &, dealii::AffineConstraints<double> const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<3, 2, double>::setup_dofs<dealii::QGauss<1>>(
    dealii::DoFHandler<3> const &, dealii::AffineConstraints<double> const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<3, 3, double>::setup_dofs<dealii::QGauss<1>>(
    dealii::DoFHandler<3> const &, dealii::AffineConstraints<double> const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<3, 4, double>::setup_dofs<dealii::QGauss<1>>(
    dealii::DoFHandler<3> const &, dealii::AffineConstraints<double> const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<3, 5, double>::setup_dofs<dealii::QGauss<1>>(
    dealii::DoFHandler<3> const &, dealii::AffineConstraints<double> const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<3, 6, double>::setup_dofs<dealii::QGauss<1>>(
    dealii::DoFHandler<3> const &, dealii::AffineConstraints<double> const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<3, 7, double>::setup_dofs<dealii::QGauss<1>>(
    dealii::DoFHandler<3> const &, dealii::AffineConstraints<double> const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<3, 8, double>::setup_dofs<dealii::QGauss<1>>(
    dealii::DoFHandler<3> const &, dealii::AffineConstraints<double> const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<3, 9, double>::setup_dofs<dealii::QGauss<1>>(
    dealii::DoFHandler<3> const &, dealii::AffineConstraints<double> const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<3, 10, double>::setup_dofs<dealii::QGauss<1>>(
    dealii::DoFHandler<3> const &, dealii::AffineConstraints<double> const &,
    dealii::QGauss<1> const &);

template void
    ThermalOperator<3, 1, double>::setup_dofs<dealii::QGaussLobatto<1>>(
        dealii::DoFHandler<3> const &,
        dealii::AffineConstraints<double> const &,
        dealii::QGaussLobatto<1> const &);
template void
    ThermalOperator<3, 2, double>::setup_dofs<dealii::QGaussLobatto<1>>(
        dealii::DoFHandler<3> const &,
        dealii::AffineConstraints<double> const &,
        dealii::QGaussLobatto<1> const &);
template void
    ThermalOperator<3, 3, double>::setup_dofs<dealii::QGaussLobatto<1>>(
        dealii::DoFHandler<3> const &,
        dealii::AffineConstraints<double> const &,
        dealii::QGaussLobatto<1> const &);
template void
    ThermalOperator<3, 4, double>::setup_dofs<dealii::QGaussLobatto<1>>(
        dealii::DoFHandler<3> const &,
        dealii::AffineConstraints<double> const &,
        dealii::QGaussLobatto<1> const &);
template void
    ThermalOperator<3, 5, double>::setup_dofs<dealii::QGaussLobatto<1>>(
        dealii::DoFHandler<3> const &,
        dealii::AffineConstraints<double> const &,
        dealii::QGaussLobatto<1> const &);
template void
    ThermalOperator<3, 6, double>::setup_dofs<dealii::QGaussLobatto<1>>(
        dealii::DoFHandler<3> const &,
        dealii::AffineConstraints<double> const &,
        dealii::QGaussLobatto<1> const &);
template void
    ThermalOperator<3, 7, double>::setup_dofs<dealii::QGaussLobatto<1>>(
        dealii::DoFHandler<3> const &,
        dealii::AffineConstraints<double> const &,
        dealii::QGaussLobatto<1> const &);
template void
    ThermalOperator<3, 8, double>::setup_dofs<dealii::QGaussLobatto<1>>(
        dealii::DoFHandler<3> const &,
        dealii::AffineConstraints<double> const &,
        dealii::QGaussLobatto<1> const &);
template void
    ThermalOperator<3, 9, double>::setup_dofs<dealii::QGaussLobatto<1>>(
        dealii::DoFHandler<3> const &,
        dealii::AffineConstraints<double> const &,
        dealii::QGaussLobatto<1> const &);
template void
    ThermalOperator<3, 10, double>::setup_dofs<dealii::QGaussLobatto<1>>(
        dealii::DoFHandler<3> const &,
        dealii::AffineConstraints<double> const &,
        dealii::QGaussLobatto<1> const &);
} // namespace adamantine
