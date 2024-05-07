/* Copyright (c) 2020 - 2021, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#ifndef HEAT_SOURCES_HH
#define HEAT_SOURCES_HH

#include <CubeHeatSource.hh>
#include <ElectronBeamHeatSource.hh>
#include <GoldakHeatSource.hh>
#include <utils.hh>

#include <deal.II/base/memory_space.h>

namespace adamantine
{
template <typename MemorySpace, int dim>
class HeatSources
{
public:
  /**
   * Default constructor creating empty object.
   */
  HeatSources() = default;

  /**
   * Constructor.
   */
  HeatSources(boost::property_tree::ptree const &source_database);

  HeatSources<dealii::MemorySpace::Host, dim> copy_to_host() const
  {
    return {Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                _electron_beam_heat_sources),
            Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                _cube_heat_sources),
            Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                _goldak_heat_sources)};
  }

  /**
   * Set the time variable.
   */
  KOKKOS_FUNCTION void update_time(double time);

  /**
   * Compute the cumulative heat source at a given point at a given time given
   * the current height of the object being manufactured.
   */
  KOKKOS_FUNCTION double value(dealii::Point<dim> const &point,
                               double const height) const;

  /**
   * Compute the maxiumum heat source at a given point at a given time given the
   * current height of the object being manufactured.
   */
  KOKKOS_FUNCTION double max_value(dealii::Point<dim> const &point,
                                   double const height) const;

  /**
   * Return the scan paths for the heat source.
   */
  std::vector<ScanPath<MemorySpace>> get_scan_paths() const;

  /**
   * (Re)sets the BeamHeatSourceProperties member variable, necessary if the
   * beam parameters vary in time (e.g. due to data assimilation).
   */
  void
  set_beam_properties(boost::property_tree::ptree const &heat_source_database);

  /**
   * Compute the current height of the where the heat source meets the material
   * (i.e. the current scan path height).
   */
  double get_current_height(double time) const;

private:
  Kokkos::View<ElectronBeamHeatSource<dim, MemorySpace> *,
               typename MemorySpace::kokkos_space>
      _electron_beam_heat_sources;
  Kokkos::View<CubeHeatSource<dim, MemorySpace> *,
               typename MemorySpace::kokkos_space>
      _cube_heat_sources;
  Kokkos::View<GoldakHeatSource<dim, MemorySpace> *,
               typename MemorySpace::kokkos_space>
      _goldak_heat_sources;
  Kokkos::View<ScanPathSegment *, typename MemorySpace::kokkos_space>
      _electron_beam_scan_path_segments;
  Kokkos::View<ScanPathSegment *, typename MemorySpace::kokkos_space>
      _goldak_scan_path_segments;
};

template <typename MemorySpace, int dim>
HeatSources<MemorySpace, dim>::HeatSources(
    boost::property_tree::ptree const &source_database)
{
  unsigned int const n_beams = source_database.get<unsigned int>("n_beams");
  std::vector<BeamHeatSourceProperties> goldak_beams;
  std::vector<BeamHeatSourceProperties> electron_beam_beams;
  std::vector<std::vector<ScanPathSegment>> goldak_scan_path_segments;
  std::vector<std::vector<ScanPathSegment>> electron_beam_scan_path_segments;
  std::vector<CubeHeatSource<dim, MemorySpace>> cube_heat_sources;

  for (unsigned int i = 0; i < n_beams; ++i)
  {
    boost::property_tree::ptree const &beam_database =
        source_database.get_child("beam_" + std::to_string(i));
    std::string type = beam_database.get<std::string>("type");
    std::vector<ScanPathSegment> scan_path_segments =
        ScanPath<MemorySpace>::extract_scan_paths(
            beam_database.get<std::string>("scan_path_file"),
            beam_database.get<std::string>("scan_path_file_format"));
    if (type == "goldak")
    {
      goldak_beams.emplace_back(beam_database);
      goldak_scan_path_segments.push_back(scan_path_segments);
    }
    else if (type == "electron_beam")
    {
      electron_beam_beams.emplace_back(beam_database);
      electron_beam_scan_path_segments.push_back(scan_path_segments);
    }
    else if (type == "cube")
    {
      cube_heat_sources.emplace_back(beam_database);
    }
    else
    {
      ASSERT_THROW(false, "Error: Beam type '" +
                              beam_database.get<std::string>("type") +
                              "' not recognized.");
    }
  }

  std::vector<GoldakHeatSource<dim, MemorySpace>> goldak_heat_sources;
  {
    std::vector<int> goldak_scan_path_segment_start(
        goldak_scan_path_segments.size() + 1);
    for (unsigned int i = 1; i < goldak_scan_path_segments.size() + 1; ++i)
      goldak_scan_path_segment_start[i] =
          goldak_scan_path_segment_start[i - 1] +
          goldak_scan_path_segments[i - 1].size();
    _goldak_scan_path_segments = decltype(_goldak_scan_path_segments)(
        Kokkos::view_alloc(Kokkos::WithoutInitializing,
                           "goldak_scan_path_segments"),
        goldak_scan_path_segment_start.back());
    for (unsigned int i = 0; i < goldak_scan_path_segments.size(); ++i)
    {
      auto goldak_scan_path_segment =
          Kokkos::subview(_goldak_scan_path_segments,
                          Kokkos::pair(goldak_scan_path_segment_start[i],
                                       goldak_scan_path_segment_start[i + 1]));
      Kokkos::deep_copy(goldak_scan_path_segment,
                        Kokkos::View<ScanPathSegment *, Kokkos::HostSpace>(
                            goldak_scan_path_segments[i].data(),
                            goldak_scan_path_segments[i].size()));
      goldak_heat_sources.emplace_back(
          goldak_beams[i], ScanPath<MemorySpace>(goldak_scan_path_segment));
    }
  }

  std::vector<ElectronBeamHeatSource<dim, MemorySpace>>
      electron_beam_heat_sources;
  {
    std::vector<int> electron_beam_scan_path_segment_start(
        electron_beam_scan_path_segments.size() + 1);
    for (unsigned int i = 1; i < electron_beam_scan_path_segments.size() + 1;
         ++i)
      electron_beam_scan_path_segment_start[i] =
          electron_beam_scan_path_segment_start[i - 1] +
          electron_beam_scan_path_segments[i - 1].size();
    _electron_beam_scan_path_segments =
        decltype(_electron_beam_scan_path_segments)(
            Kokkos::view_alloc(Kokkos::WithoutInitializing,
                               "electron_beam_scan_path_segments"),
            electron_beam_scan_path_segment_start.back());
    for (unsigned int i = 0; i < electron_beam_scan_path_segments.size(); ++i)
    {
      auto electron_beam_scan_path_segment = Kokkos::subview(
          _electron_beam_scan_path_segments,
          Kokkos::pair(electron_beam_scan_path_segment_start[i],
                       electron_beam_scan_path_segment_start[i + 1]));
      Kokkos::deep_copy(electron_beam_scan_path_segment,
                        Kokkos::View<ScanPathSegment *, Kokkos::HostSpace>(
                            electron_beam_scan_path_segments[i].data(),
                            electron_beam_scan_path_segments[i].size()));
      electron_beam_heat_sources.emplace_back(
          electron_beam_beams[i],
          ScanPath<MemorySpace>(electron_beam_scan_path_segment));
    }
  }

  _goldak_heat_sources = decltype(_goldak_heat_sources)(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "goldak_heat_sources"),
      goldak_heat_sources.size());
  _electron_beam_heat_sources = decltype(_electron_beam_heat_sources)(
      Kokkos::view_alloc(Kokkos::WithoutInitializing,
                         "electron_beam_heat_sources"),
      electron_beam_heat_sources.size());
  _cube_heat_sources = decltype(_cube_heat_sources)(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "cube_heat_sources"),
      cube_heat_sources.size());
  Kokkos::deep_copy(
      _goldak_heat_sources,
      Kokkos::View<GoldakHeatSource<dim, MemorySpace> *, Kokkos::HostSpace>(
          goldak_heat_sources.data(), goldak_heat_sources.size()));
  Kokkos::deep_copy(
      _electron_beam_heat_sources,
      Kokkos::View<ElectronBeamHeatSource<dim, MemorySpace> *,
                   Kokkos::HostSpace>(electron_beam_heat_sources.data(),
                                      electron_beam_heat_sources.size()));
  Kokkos::deep_copy(
      _cube_heat_sources,
      Kokkos::View<CubeHeatSource<dim, MemorySpace> *, Kokkos::HostSpace>(
          cube_heat_sources.data(), cube_heat_sources.size()));
}

template <typename MemorySpace, int dim>
KOKKOS_FUNCTION void HeatSources<MemorySpace, dim>::update_time(double time)
{
  for (unsigned int i = 0; i < _electron_beam_heat_sources.size(); ++i)
    _electron_beam_heat_sources(i).update_time(time);
  for (unsigned int i = 0; i < _cube_heat_sources.size(); ++i)
    _cube_heat_sources(i).update_time(time);
  for (unsigned int i = 0; i < _goldak_heat_sources.size(); ++i)
    _goldak_heat_sources(i).update_time(time);
}

template <typename MemorySpace, int dim>
KOKKOS_FUNCTION double
HeatSources<MemorySpace, dim>::value(dealii::Point<dim> const &point,
                                     double const height) const
{
  double value = 0;
  for (unsigned int i = 0; i < _electron_beam_heat_sources.size(); ++i)
    value += _electron_beam_heat_sources(i).value(point, height);
  for (unsigned int i = 0; i < _cube_heat_sources.size(); ++i)
    value += _cube_heat_sources(i).value(point, height);
  for (unsigned int i = 0; i < _goldak_heat_sources.size(); ++i)
    value += _goldak_heat_sources(i).value(point, height);
  return value;
}

template <typename MemorySpace, int dim>
KOKKOS_FUNCTION double
HeatSources<MemorySpace, dim>::max_value(dealii::Point<dim> const &point,
                                         double const height) const
{
  double value = 0;
  for (unsigned int i = 0; i < _electron_beam_heat_sources.size(); ++i)
    value =
        std::max(value, _electron_beam_heat_sources(i).value(point, height));
  for (unsigned int i = 0; i < _cube_heat_sources.size(); ++i)
    value = std::max(value, _cube_heat_sources(i).value(point, height));
  for (unsigned int i = 0; i < _goldak_heat_sources.size(); ++i)
    value = std::max(value, _goldak_heat_sources(i).value(point, height));
  return value;
}

template <typename MemorySpace, int dim>
std::vector<ScanPath<MemorySpace>>
HeatSources<MemorySpace, dim>::get_scan_paths() const
{
  std::vector<ScanPath<MemorySpace>> scan_paths;
  for (unsigned int i = 0; i < _electron_beam_heat_sources.size(); ++i)
    scan_paths.push_back(_electron_beam_heat_sources(i).get_scan_path());
  for (unsigned int i = 0; i < _cube_heat_sources.size(); ++i)
    scan_paths.push_back(_cube_heat_sources(i).get_scan_path());
  for (unsigned int i = 0; i < _goldak_heat_sources.size(); ++i)
    scan_paths.push_back(_goldak_heat_sources(i).get_scan_path());
  return scan_paths;
}

template <typename MemorySpace, int dim>
double HeatSources<MemorySpace, dim>::get_current_height(double time) const
{
  // Right now this is just the maximum heat source height, which can lead to
  // unexpected behavior for different sources with different heights.
  double temp_height = std::numeric_limits<double>::lowest();
  for (unsigned int i = 0; i < _electron_beam_heat_sources.size(); ++i)
    temp_height = std::max(
        temp_height, _electron_beam_heat_sources(i).get_current_height(time));
  for (unsigned int i = 0; i < _cube_heat_sources.size(); ++i)
    temp_height =
        std::max(temp_height, _cube_heat_sources(i).get_current_height(time));
  for (unsigned int i = 0; i < _goldak_heat_sources.size(); ++i)
    temp_height =
        std::max(temp_height, _goldak_heat_sources(i).get_current_height(time));
  return temp_height;
}

template <typename MemorySpace, int dim>
void HeatSources<MemorySpace, dim>::set_beam_properties(
    boost::property_tree::ptree const &heat_source_database)
{
  unsigned int source_index = 0;

  auto set_properties = [&](auto &source)
  {
    // PropertyTreeInput sources.beam_X
    boost::property_tree::ptree const &beam_database =
        heat_source_database.get_child("beam_" + std::to_string(source_index));

    // PropertyTreeInput sources.beam_X.type
    std::string type = beam_database.get<std::string>("type");

    if (type == "goldak" || type == "electron_beam")
      source.set_beam_properties(beam_database);

    source_index++;
  };

  for (unsigned int i = 0; i < _electron_beam_heat_sources.size(); ++i)
    set_properties(_electron_beam_heat_sources[i]);
  for (unsigned int i = 0; i < _cube_heat_sources.size(); ++i)
    set_properties(_cube_heat_sources[i]);
  for (unsigned int i = 0; i < _goldak_heat_sources.size(); ++i)
    set_properties(_goldak_heat_sources[i]);
}

} // namespace adamantine

#endif
