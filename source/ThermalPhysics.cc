/* Copyright (c) 2016 - 2020, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#include <ThermalPhysics.templates.hh>
#include <instantiation.hh>

// INSTANTIATE_DIM_FEDEGREE_HS_QUAD_HOST(TUPLE(ThermalPhysics))

namespace adamantine
{
template class ThermalPhysics<2, 1, GoldakHeatSource<2>,
                              dealii::MemorySpace::Host, dealii::QGauss<1>>;
template class ThermalPhysics<2, 2, GoldakHeatSource<2>,
                              dealii::MemorySpace::Host, dealii::QGauss<1>>;
template class ThermalPhysics<2, 3, GoldakHeatSource<2>,
                              dealii::MemorySpace::Host, dealii::QGauss<1>>;
template class ThermalPhysics<2, 4, GoldakHeatSource<2>,
                              dealii::MemorySpace::Host, dealii::QGauss<1>>;
template class ThermalPhysics<2, 5, GoldakHeatSource<2>,
                              dealii::MemorySpace::Host, dealii::QGauss<1>>;
template class ThermalPhysics<2, 6, GoldakHeatSource<2>,
                              dealii::MemorySpace::Host, dealii::QGauss<1>>;
template class ThermalPhysics<2, 7, GoldakHeatSource<2>,
                              dealii::MemorySpace::Host, dealii::QGauss<1>>;
template class ThermalPhysics<2, 8, GoldakHeatSource<2>,
                              dealii::MemorySpace::Host, dealii::QGauss<1>>;
template class ThermalPhysics<2, 9, GoldakHeatSource<2>,
                              dealii::MemorySpace::Host, dealii::QGauss<1>>;
template class ThermalPhysics<2, 10, GoldakHeatSource<2>,
                              dealii::MemorySpace::Host, dealii::QGauss<1>>;

template class ThermalPhysics<3, 1, GoldakHeatSource<3>,
                              dealii::MemorySpace::Host, dealii::QGauss<1>>;
template class ThermalPhysics<3, 2, GoldakHeatSource<3>,
                              dealii::MemorySpace::Host, dealii::QGauss<1>>;
template class ThermalPhysics<3, 3, GoldakHeatSource<3>,
                              dealii::MemorySpace::Host, dealii::QGauss<1>>;
template class ThermalPhysics<3, 4, GoldakHeatSource<3>,
                              dealii::MemorySpace::Host, dealii::QGauss<1>>;
template class ThermalPhysics<3, 5, GoldakHeatSource<3>,
                              dealii::MemorySpace::Host, dealii::QGauss<1>>;
template class ThermalPhysics<3, 6, GoldakHeatSource<3>,
                              dealii::MemorySpace::Host, dealii::QGauss<1>>;
template class ThermalPhysics<3, 7, GoldakHeatSource<3>,
                              dealii::MemorySpace::Host, dealii::QGauss<1>>;
template class ThermalPhysics<3, 8, GoldakHeatSource<3>,
                              dealii::MemorySpace::Host, dealii::QGauss<1>>;
template class ThermalPhysics<3, 9, GoldakHeatSource<3>,
                              dealii::MemorySpace::Host, dealii::QGauss<1>>;
template class ThermalPhysics<3, 10, GoldakHeatSource<3>,
                              dealii::MemorySpace::Host, dealii::QGauss<1>>;

template class ThermalPhysics<2, 1, GoldakHeatSource<2>,
                              dealii::MemorySpace::Host,
                              dealii::QGaussLobatto<1>>;
template class ThermalPhysics<2, 2, GoldakHeatSource<2>,
                              dealii::MemorySpace::Host,
                              dealii::QGaussLobatto<1>>;
template class ThermalPhysics<2, 3, GoldakHeatSource<2>,
                              dealii::MemorySpace::Host,
                              dealii::QGaussLobatto<1>>;
template class ThermalPhysics<2, 4, GoldakHeatSource<2>,
                              dealii::MemorySpace::Host,
                              dealii::QGaussLobatto<1>>;
template class ThermalPhysics<2, 5, GoldakHeatSource<2>,
                              dealii::MemorySpace::Host,
                              dealii::QGaussLobatto<1>>;
template class ThermalPhysics<2, 6, GoldakHeatSource<2>,
                              dealii::MemorySpace::Host,
                              dealii::QGaussLobatto<1>>;
template class ThermalPhysics<2, 7, GoldakHeatSource<2>,
                              dealii::MemorySpace::Host,
                              dealii::QGaussLobatto<1>>;
template class ThermalPhysics<2, 8, GoldakHeatSource<2>,
                              dealii::MemorySpace::Host,
                              dealii::QGaussLobatto<1>>;
template class ThermalPhysics<2, 9, GoldakHeatSource<2>,
                              dealii::MemorySpace::Host,
                              dealii::QGaussLobatto<1>>;
template class ThermalPhysics<2, 10, GoldakHeatSource<2>,
                              dealii::MemorySpace::Host,
                              dealii::QGaussLobatto<1>>;

template class ThermalPhysics<3, 1, GoldakHeatSource<3>,
                              dealii::MemorySpace::Host,
                              dealii::QGaussLobatto<1>>;
template class ThermalPhysics<3, 2, GoldakHeatSource<3>,
                              dealii::MemorySpace::Host,
                              dealii::QGaussLobatto<1>>;
template class ThermalPhysics<3, 3, GoldakHeatSource<3>,
                              dealii::MemorySpace::Host,
                              dealii::QGaussLobatto<1>>;
template class ThermalPhysics<3, 4, GoldakHeatSource<3>,
                              dealii::MemorySpace::Host,
                              dealii::QGaussLobatto<1>>;
template class ThermalPhysics<3, 5, GoldakHeatSource<3>,
                              dealii::MemorySpace::Host,
                              dealii::QGaussLobatto<1>>;
template class ThermalPhysics<3, 6, GoldakHeatSource<3>,
                              dealii::MemorySpace::Host,
                              dealii::QGaussLobatto<1>>;
template class ThermalPhysics<3, 7, GoldakHeatSource<3>,
                              dealii::MemorySpace::Host,
                              dealii::QGaussLobatto<1>>;
template class ThermalPhysics<3, 8, GoldakHeatSource<3>,
                              dealii::MemorySpace::Host,
                              dealii::QGaussLobatto<1>>;
template class ThermalPhysics<3, 9, GoldakHeatSource<3>,
                              dealii::MemorySpace::Host,
                              dealii::QGaussLobatto<1>>;
template class ThermalPhysics<3, 10, GoldakHeatSource<3>,
                              dealii::MemorySpace::Host,
                              dealii::QGaussLobatto<1>>;

template class ThermalPhysics<2, 1, ElectronBeamHeatSource<2>,
                              dealii::MemorySpace::Host, dealii::QGauss<1>>;
template class ThermalPhysics<2, 2, ElectronBeamHeatSource<2>,
                              dealii::MemorySpace::Host, dealii::QGauss<1>>;
template class ThermalPhysics<2, 3, ElectronBeamHeatSource<2>,
                              dealii::MemorySpace::Host, dealii::QGauss<1>>;
template class ThermalPhysics<2, 4, ElectronBeamHeatSource<2>,
                              dealii::MemorySpace::Host, dealii::QGauss<1>>;
template class ThermalPhysics<2, 5, ElectronBeamHeatSource<2>,
                              dealii::MemorySpace::Host, dealii::QGauss<1>>;
template class ThermalPhysics<2, 6, ElectronBeamHeatSource<2>,
                              dealii::MemorySpace::Host, dealii::QGauss<1>>;
template class ThermalPhysics<2, 7, ElectronBeamHeatSource<2>,
                              dealii::MemorySpace::Host, dealii::QGauss<1>>;
template class ThermalPhysics<2, 8, ElectronBeamHeatSource<2>,
                              dealii::MemorySpace::Host, dealii::QGauss<1>>;
template class ThermalPhysics<2, 9, ElectronBeamHeatSource<2>,
                              dealii::MemorySpace::Host, dealii::QGauss<1>>;
template class ThermalPhysics<2, 10, ElectronBeamHeatSource<2>,
                              dealii::MemorySpace::Host, dealii::QGauss<1>>;

template class ThermalPhysics<3, 1, ElectronBeamHeatSource<3>,
                              dealii::MemorySpace::Host, dealii::QGauss<1>>;
template class ThermalPhysics<3, 2, ElectronBeamHeatSource<3>,
                              dealii::MemorySpace::Host, dealii::QGauss<1>>;
template class ThermalPhysics<3, 3, ElectronBeamHeatSource<3>,
                              dealii::MemorySpace::Host, dealii::QGauss<1>>;
template class ThermalPhysics<3, 4, ElectronBeamHeatSource<3>,
                              dealii::MemorySpace::Host, dealii::QGauss<1>>;
template class ThermalPhysics<3, 5, ElectronBeamHeatSource<3>,
                              dealii::MemorySpace::Host, dealii::QGauss<1>>;
template class ThermalPhysics<3, 6, ElectronBeamHeatSource<3>,
                              dealii::MemorySpace::Host, dealii::QGauss<1>>;
template class ThermalPhysics<3, 7, ElectronBeamHeatSource<3>,
                              dealii::MemorySpace::Host, dealii::QGauss<1>>;
template class ThermalPhysics<3, 8, ElectronBeamHeatSource<3>,
                              dealii::MemorySpace::Host, dealii::QGauss<1>>;
template class ThermalPhysics<3, 9, ElectronBeamHeatSource<3>,
                              dealii::MemorySpace::Host, dealii::QGauss<1>>;
template class ThermalPhysics<3, 10, ElectronBeamHeatSource<3>,
                              dealii::MemorySpace::Host, dealii::QGauss<1>>;

template class ThermalPhysics<2, 1, ElectronBeamHeatSource<2>,
                              dealii::MemorySpace::Host,
                              dealii::QGaussLobatto<1>>;
template class ThermalPhysics<2, 2, ElectronBeamHeatSource<2>,
                              dealii::MemorySpace::Host,
                              dealii::QGaussLobatto<1>>;
template class ThermalPhysics<2, 3, ElectronBeamHeatSource<2>,
                              dealii::MemorySpace::Host,
                              dealii::QGaussLobatto<1>>;
template class ThermalPhysics<2, 4, ElectronBeamHeatSource<2>,
                              dealii::MemorySpace::Host,
                              dealii::QGaussLobatto<1>>;
template class ThermalPhysics<2, 5, ElectronBeamHeatSource<2>,
                              dealii::MemorySpace::Host,
                              dealii::QGaussLobatto<1>>;
template class ThermalPhysics<2, 6, ElectronBeamHeatSource<2>,
                              dealii::MemorySpace::Host,
                              dealii::QGaussLobatto<1>>;
template class ThermalPhysics<2, 7, ElectronBeamHeatSource<2>,
                              dealii::MemorySpace::Host,
                              dealii::QGaussLobatto<1>>;
template class ThermalPhysics<2, 8, ElectronBeamHeatSource<2>,
                              dealii::MemorySpace::Host,
                              dealii::QGaussLobatto<1>>;
template class ThermalPhysics<2, 9, ElectronBeamHeatSource<2>,
                              dealii::MemorySpace::Host,
                              dealii::QGaussLobatto<1>>;
template class ThermalPhysics<2, 10, ElectronBeamHeatSource<2>,
                              dealii::MemorySpace::Host,
                              dealii::QGaussLobatto<1>>;

template class ThermalPhysics<3, 1, ElectronBeamHeatSource<3>,
                              dealii::MemorySpace::Host,
                              dealii::QGaussLobatto<1>>;
template class ThermalPhysics<3, 2, ElectronBeamHeatSource<3>,
                              dealii::MemorySpace::Host,
                              dealii::QGaussLobatto<1>>;
template class ThermalPhysics<3, 3, ElectronBeamHeatSource<3>,
                              dealii::MemorySpace::Host,
                              dealii::QGaussLobatto<1>>;
template class ThermalPhysics<3, 4, ElectronBeamHeatSource<3>,
                              dealii::MemorySpace::Host,
                              dealii::QGaussLobatto<1>>;
template class ThermalPhysics<3, 5, ElectronBeamHeatSource<3>,
                              dealii::MemorySpace::Host,
                              dealii::QGaussLobatto<1>>;
template class ThermalPhysics<3, 6, ElectronBeamHeatSource<3>,
                              dealii::MemorySpace::Host,
                              dealii::QGaussLobatto<1>>;
template class ThermalPhysics<3, 7, ElectronBeamHeatSource<3>,
                              dealii::MemorySpace::Host,
                              dealii::QGaussLobatto<1>>;
template class ThermalPhysics<3, 8, ElectronBeamHeatSource<3>,
                              dealii::MemorySpace::Host,
                              dealii::QGaussLobatto<1>>;
template class ThermalPhysics<3, 9, ElectronBeamHeatSource<3>,
                              dealii::MemorySpace::Host,
                              dealii::QGaussLobatto<1>>;
template class ThermalPhysics<3, 10, ElectronBeamHeatSource<3>,
                              dealii::MemorySpace::Host,
                              dealii::QGaussLobatto<1>>;
} // namespace adamantine
