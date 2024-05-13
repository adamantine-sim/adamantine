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

  /**
   * Return a copy of this instance in the target memory space.
   */
  template <typename TargetMemorySpace>
  HeatSources<TargetMemorySpace, dim>
  copy_to(TargetMemorySpace target_memory_space) const;

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
  friend class HeatSources<dealii::MemorySpace::Default, dim>;
  friend class HeatSources<dealii::MemorySpace::Host, dim>;

  /**
   * Private constructor used by copy_to_host.
   */
  HeatSources(
      Kokkos::View<ElectronBeamHeatSource<dim, MemorySpace> *,
                   typename MemorySpace::kokkos_space>
          electron_beam_heat_sources,
      Kokkos::View<CubeHeatSource<dim> *, typename MemorySpace::kokkos_space>
          cube_heat_sources,
      Kokkos::View<GoldakHeatSource<dim, MemorySpace> *,
                   typename MemorySpace::kokkos_space>
          goldak_heat_sources,
      std::vector<
          Kokkos::View<ScanPathSegment *, typename MemorySpace::kokkos_space>>
          electron_beam_scan_path_segments,
      std::vector<
          Kokkos::View<ScanPathSegment *, typename MemorySpace::kokkos_space>>
          goldak_scan_path_segments);

  Kokkos::View<ElectronBeamHeatSource<dim, MemorySpace> *,
               typename MemorySpace::kokkos_space>
      _electron_beam_heat_sources;
  Kokkos::View<CubeHeatSource<dim> *, typename MemorySpace::kokkos_space>
      _cube_heat_sources;
  Kokkos::View<GoldakHeatSource<dim, MemorySpace> *,
               typename MemorySpace::kokkos_space>
      _goldak_heat_sources;
  std::vector<
      Kokkos::View<ScanPathSegment *, typename MemorySpace::kokkos_space>>
      _electron_beam_scan_path_segments;
  std::vector<
      Kokkos::View<ScanPathSegment *, typename MemorySpace::kokkos_space>>
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
  std::vector<CubeHeatSource<dim>> cube_heat_sources;

  for (unsigned int i = 0; i < n_beams; ++i)
  {
    boost::property_tree::ptree const &beam_database =
        source_database.get_child("beam_" + std::to_string(i));
    std::string type = beam_database.get<std::string>("type");
    if (type == "goldak")
    {
      goldak_beams.emplace_back(beam_database);
      goldak_scan_path_segments.push_back(
          ScanPath<MemorySpace>::extract_scan_paths(
              beam_database.get<std::string>("scan_path_file"),
              beam_database.get<std::string>("scan_path_file_format")));
    }
    else if (type == "electron_beam")
    {
      electron_beam_beams.emplace_back(beam_database);
      electron_beam_scan_path_segments.push_back(
          ScanPath<MemorySpace>::extract_scan_paths(
              beam_database.get<std::string>("scan_path_file"),
              beam_database.get<std::string>("scan_path_file_format")));
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
  for (unsigned int i = 0; i < goldak_scan_path_segments.size(); ++i)
  {
    _goldak_scan_path_segments.emplace_back(
        Kokkos::view_alloc(Kokkos::WithoutInitializing,
                           "goldak_scan_path_segments_" + std::to_string(i)),
        goldak_scan_path_segments[i].size());
    Kokkos::deep_copy(_goldak_scan_path_segments.back(),
                      Kokkos::View<ScanPathSegment *, Kokkos::HostSpace>(
                          goldak_scan_path_segments[i].data(),
                          goldak_scan_path_segments[i].size()));
    goldak_heat_sources.emplace_back(
        goldak_beams[i],
        ScanPath<MemorySpace>(_goldak_scan_path_segments.back()));
  }

  std::vector<ElectronBeamHeatSource<dim, MemorySpace>>
      electron_beam_heat_sources;
  for (unsigned int i = 0; i < electron_beam_scan_path_segments.size(); ++i)
  {
    _electron_beam_scan_path_segments.emplace_back(
        Kokkos::view_alloc(Kokkos::WithoutInitializing,
                           "electron_beam_scan_path_segments_" +
                               std::to_string(i)),
        electron_beam_scan_path_segments[i].size());
    Kokkos::deep_copy(_electron_beam_scan_path_segments.back(),
                      Kokkos::View<ScanPathSegment *, Kokkos::HostSpace>(
                          electron_beam_scan_path_segments[i].data(),
                          electron_beam_scan_path_segments[i].size()));
    electron_beam_heat_sources.emplace_back(
        electron_beam_beams[i],
        ScanPath<MemorySpace>(_electron_beam_scan_path_segments.back()));
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
  Kokkos::deep_copy(_cube_heat_sources,
                    Kokkos::View<CubeHeatSource<dim> *, Kokkos::HostSpace>(
                        cube_heat_sources.data(), cube_heat_sources.size()));
}

template <typename MemorySpace, int dim>
HeatSources<MemorySpace, dim>::HeatSources(
    Kokkos::View<ElectronBeamHeatSource<dim, MemorySpace> *,
                 typename MemorySpace::kokkos_space>
        electron_beam_heat_sources,
    Kokkos::View<CubeHeatSource<dim> *, typename MemorySpace::kokkos_space>
        cube_heat_sources,
    Kokkos::View<GoldakHeatSource<dim, MemorySpace> *,
                 typename MemorySpace::kokkos_space>
        goldak_heat_sources,
    std::vector<
        Kokkos::View<ScanPathSegment *, typename MemorySpace::kokkos_space>>
        electron_beam_scan_path_segments,
    std::vector<
        Kokkos::View<ScanPathSegment *, typename MemorySpace::kokkos_space>>
        goldak_scan_path_segments)
    : _electron_beam_heat_sources(electron_beam_heat_sources),
      _cube_heat_sources(cube_heat_sources),
      _goldak_heat_sources(goldak_heat_sources),
      _electron_beam_scan_path_segments(electron_beam_scan_path_segments),
      _goldak_scan_path_segments(goldak_scan_path_segments)
{
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

template <typename MemorySpace, int dim>
template <typename TargetMemorySpace>
HeatSources<TargetMemorySpace, dim> HeatSources<MemorySpace, dim>::copy_to(
    TargetMemorySpace /*target_memory_space*/) const
{
  if constexpr (std::is_same_v<MemorySpace, TargetMemorySpace>)
    return *this;
  else
  {
    Kokkos::View<ElectronBeamHeatSource<dim, TargetMemorySpace> *,
                 typename TargetMemorySpace::kokkos_space>
        target_electron_beam_heat_sources(
            Kokkos::view_alloc(Kokkos::WithoutInitializing,
                               "electron_beam_heat_sources"),
            _electron_beam_heat_sources.size());
    std::vector<Kokkos::View<ScanPathSegment *,
                             typename TargetMemorySpace::kokkos_space>>
        target_electron_beam_scan_path_segments(
            _electron_beam_scan_path_segments.size());
    {
      for (unsigned int i = 0; i < _electron_beam_scan_path_segments.size();
           ++i)
        target_electron_beam_scan_path_segments[i] =
            Kokkos::create_mirror_view_and_copy(
                Kokkos::HostSpace{}, _electron_beam_scan_path_segments[i]);
      auto target_copy_electron_beam_heat_sources =
          Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                              _electron_beam_heat_sources);
      std::vector<BeamHeatSourceProperties> electron_beam_beams(
          _electron_beam_heat_sources.size());
      for (unsigned int i = 0; i < _electron_beam_heat_sources.size(); ++i)
        electron_beam_beams[i] =
            target_copy_electron_beam_heat_sources(i).get_beam_properties();
      std::vector<ElectronBeamHeatSource<dim, TargetMemorySpace>>
          electron_beam_heat_source_vector;
      for (unsigned int i = 0; i < _electron_beam_heat_sources.size(); ++i)
        electron_beam_heat_source_vector.emplace_back(
            electron_beam_beams[i],
            ScanPath<TargetMemorySpace>(
                target_electron_beam_scan_path_segments[i]));
      Kokkos::deep_copy(
          target_electron_beam_heat_sources,
          Kokkos::View<ElectronBeamHeatSource<dim, TargetMemorySpace> *,
                       Kokkos::HostSpace>(
              electron_beam_heat_source_vector.data(),
              electron_beam_heat_source_vector.size()));
    }

    Kokkos::View<GoldakHeatSource<dim, TargetMemorySpace> *,
                 typename TargetMemorySpace::kokkos_space>
        target_goldak_heat_sources(
            Kokkos::view_alloc(Kokkos::WithoutInitializing,
                               "goldak_heat_sources"),
            _goldak_heat_sources.size());
    std::vector<Kokkos::View<ScanPathSegment *,
                             typename TargetMemorySpace::kokkos_space>>
        target_goldak_scan_path_segments(_goldak_scan_path_segments.size());
    {
      for (unsigned int i = 0; i < _goldak_scan_path_segments.size(); ++i)
        target_goldak_scan_path_segments[i] =
            Kokkos::create_mirror_view_and_copy(
                typename TargetMemorySpace::kokkos_space{},
                _goldak_scan_path_segments[i]);
      auto target_copy_goldak_heat_sources =
          Kokkos::create_mirror_view_and_copy(
              typename TargetMemorySpace::kokkos_space{}, _goldak_heat_sources);
      std::vector<BeamHeatSourceProperties> goldak_beams(
          _goldak_heat_sources.size());
      for (unsigned int i = 0; i < _goldak_heat_sources.size(); ++i)
        goldak_beams[i] =
            target_copy_goldak_heat_sources(i).get_beam_properties();
      std::vector<GoldakHeatSource<dim, TargetMemorySpace>>
          goldak_heat_source_vector;
      for (unsigned int i = 0; i < _goldak_heat_sources.size(); ++i)
        goldak_heat_source_vector.emplace_back(
            goldak_beams[i],
            ScanPath<TargetMemorySpace>(target_goldak_scan_path_segments[i]));
      Kokkos::deep_copy(
          target_goldak_heat_sources,
          Kokkos::View<GoldakHeatSource<dim, TargetMemorySpace> *,
                       Kokkos::HostSpace>(goldak_heat_source_vector.data(),
                                          goldak_heat_source_vector.size()));
    }

    auto target_cube_heat_sources = Kokkos::create_mirror_view_and_copy(
        typename TargetMemorySpace::kokkos_space{}, _cube_heat_sources);

    return {target_electron_beam_heat_sources, target_cube_heat_sources,
            target_goldak_heat_sources, target_electron_beam_scan_path_segments,
            target_goldak_scan_path_segments};
  }
}

} // namespace adamantine

#endif
