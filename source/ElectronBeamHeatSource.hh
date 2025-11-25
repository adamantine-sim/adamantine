/* SPDX-FileCopyrightText: Copyright (c) 2020 - 2025, the adamantine authors.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#ifndef ELECTRON_BEAM_HEAT_SOURCE_HH
#define ELECTRON_BEAM_HEAT_SOURCE_HH

#include <HeatSource.hh>

#include <limits>

namespace adamantine
{
/**
 * A derived class from HeatSource for a model of an electron beam heat source.
 * The form of the heat source model is taken from the following reference:
 * Raghavan et al, Acta Materilia, 112, 2016, pp 303-314.
 */
template <int dim>
class ElectronBeamHeatSource final : public HeatSource<dim>
{
public:
  /**
   * Constructor.
   *  \param[in] beam_database requires the following entries:
   *    - <B>absorption_efficiency</B>: double in \f$[0,1]\f$
   *    - <B>depth</B>: double in \f$[0,\infty)\f$
   *    - <B>diameter</B>: double in \f$[0,\infty)\f$
   *    - <B>max_power</B>: double in \f$[0, \infty)\f$
   *    - <B>input_file</B>: name of the file that contains the scan path
   *      segments
   *  \param[in] units_optional_database may have the following entries:
   *    - <B>heat_source.dimension</B>
   *    - <B>heat_source.power</B>
   */
  ElectronBeamHeatSource(
      boost::property_tree::ptree const &beam_database,
      boost::optional<boost::property_tree::ptree const &> const
          &units_optional_database);

  /**
   * Set the time variable.
   */
  void update_time(double time) final;

  /**
   * Returns the value of an electron beam heat source at a specified point and
   * time.
   */
  double value(dealii::Point<dim> const &point) const final;

  /**
   * Same function as above but it uses vectorized data.
   */
  dealii::VectorizedArray<double>
  value(dealii::Point<dim, dealii::VectorizedArray<double>> const &points)
      const final;

  dealii::BoundingBox<dim>
  get_bounding_box(double const time, double const scaling_factor) const final;

private:
  dealii::Point<3, dealii::VectorizedArray<double>> _beam_center;
  dealii::VectorizedArray<double> _alpha =
      std::numeric_limits<double>::signaling_NaN();
  dealii::VectorizedArray<double> _depth =
      std::numeric_limits<double>::signaling_NaN();
  dealii::VectorizedArray<double> _radius_squared =
      std::numeric_limits<double>::signaling_NaN();
  double const _log_01 = std::log(0.1);
};
} // namespace adamantine

#endif
