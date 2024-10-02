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
template <int dim, typename MemorySpaceType>
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
  template <typename Dummy = MemorySpaceType,
            typename = std::enable_if_t<
                std::is_same_v<Dummy, dealii::MemorySpace::Host>>>
  HeatSources(boost::property_tree::ptree const &source_database);

  /**
   * Return a copy of this instance in the target memory space.
   */
  // template <typename TargetMemorySpaceType>
  // HeatSources<dim, TargetMemorySpaceType>
  // copy_to(TargetMemorySpaceType target_memory_space) const;

  /**
   * Set the time variable.
   */
  void update_time(double time);

  /**
   * Update the scan paths for all heat sources reading again from the
   * respective files. Returns whether any if the scan paths is still active.
   */
  template <typename Dummy = MemorySpaceType>
  std::enable_if_t<std::is_same_v<Dummy, dealii::MemorySpace::Host>, bool>
  update_scan_paths();

  /**
   * Compute the cumulative heat source at a given point at a given time given
   * the current height of the object being manufactured.
   */
  double value(dealii::Point<dim> const &point, double const height) const;

  /**
   * Return the scan paths for the heat source.
   */
  std::vector<ScanPath> get_scan_paths() const;

  /**
   * (Re)set the BeamHeatSourceProperties member variable, necessary if the
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
  Kokkos::View<ElectronBeamHeatSource<dim> *,
               typename MemorySpaceType::kokkos_space>
      _electron_beam_heat_sources;
  Kokkos::View<CubeHeatSource<dim> *, typename MemorySpaceType::kokkos_space>
      _cube_heat_sources;
  Kokkos::View<GoldakHeatSource<dim> *, typename MemorySpaceType::kokkos_space>
      _goldak_heat_sources;
  std::vector<int> _electron_beam_indices;
  std::vector<int> _cube_indices;
  std::vector<int> _goldak_indices;
};

template <int dim, typename MemorySpaceType>
template <typename Dummy>
std::enable_if_t<std::is_same_v<Dummy, dealii::MemorySpace::Host>, bool>
HeatSources<dim, MemorySpaceType>::update_scan_paths()
{
  bool scan_path_end = true;

  for (unsigned int i = 0; i < _goldak_heat_sources.size(); ++i)
  {
    if (!_goldak_heat_sources[i].get_scan_path().is_finished())
    {
      scan_path_end = false;
      // This functions waits for the scan path file to be updated
      // before reading the file.
      _goldak_heat_sources[i].get_scan_path().read_file();
    }
  }

  for (unsigned int i = 0; i < _electron_beam_heat_sources.size(); ++i)
  {
    if (!_electron_beam_heat_sources[i].get_scan_path().is_finished())
    {
      scan_path_end = false;
      // This functions waits for the scan path file to be updated
      // before reading the file.
      _electron_beam_heat_sources[i].get_scan_path().read_file();
    }
  }

  return scan_path_end;
}

template <int dim, typename MemorySpaceType>
template <typename, typename>
HeatSources<dim, MemorySpaceType>::HeatSources(
    boost::property_tree::ptree const &source_database)
{
  unsigned int const n_beams = source_database.get<unsigned int>("n_beams");
  std::vector<GoldakHeatSource<dim>> goldak_heat_sources;
  std::vector<ElectronBeamHeatSource<dim>> electron_beam_heat_sources;
  std::vector<CubeHeatSource<dim>> cube_heat_sources;

  for (unsigned int i = 0; i < n_beams; ++i)
  {
    boost::property_tree::ptree const &beam_database =
        source_database.get_child("beam_" + std::to_string(i));
    std::string type = beam_database.get<std::string>("type");
    if (type == "goldak")
    {
      goldak_heat_sources.emplace_back(beam_database);
      _goldak_indices.push_back(i);
    }
    else if (type == "electron_beam")
    {
      electron_beam_heat_sources.emplace_back(beam_database);
      _electron_beam_indices.push_back(i);
    }
    else if (type == "cube")
    {
      cube_heat_sources.emplace_back(beam_database);
      _cube_indices.push_back(i);
    }
    else
    {
      ASSERT_THROW(false, "Error: Beam type '" +
                              beam_database.get<std::string>("type") +
                              "' not recognized.");
    }
  }
  _goldak_heat_sources = Kokkos::View<GoldakHeatSource<dim> *,
                                      typename MemorySpaceType::kokkos_space>(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "KokkosGoldakHeatSourcs"),
      goldak_heat_sources.size());
  Kokkos::deep_copy(
      _goldak_heat_sources,
      Kokkos::View<GoldakHeatSource<dim> *, Kokkos::HostSpace>(
          goldak_heat_sources.data(), goldak_heat_sources.size()));
  _electron_beam_heat_sources =
      Kokkos::View<ElectronBeamHeatSource<dim> *,
                   typename MemorySpaceType::kokkos_space>(
          Kokkos::view_alloc(Kokkos::WithoutInitializing,
                             "ElectronBeamHeatSources"),
          electron_beam_heat_sources.size());
  Kokkos::deep_copy(
      _electron_beam_heat_sources,
      Kokkos::View<ElectronBeamHeatSource<dim> *, Kokkos::HostSpace>(
          electron_beam_heat_sources.data(),
          electron_beam_heat_sources.size()));
  _cube_heat_sources = Kokkos::View<CubeHeatSource<dim> *,
                                    typename MemorySpaceType::kokkos_space>(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "CubeHeatSources"),
      cube_heat_sources.size());
  Kokkos::deep_copy(_cube_heat_sources,
                    Kokkos::View<CubeHeatSource<dim> *, Kokkos::HostSpace>(
                        cube_heat_sources.data(), cube_heat_sources.size()));
}

template <int dim, typename MemorySpaceType>
void HeatSources<dim, MemorySpaceType>::update_time(double time)
{
  for (unsigned int i = 0; i < _electron_beam_heat_sources.size(); ++i)
    _electron_beam_heat_sources(i).update_time(time);
  for (unsigned int i = 0; i < _cube_heat_sources.size(); ++i)
    _cube_heat_sources(i).update_time(time);
  for (unsigned int i = 0; i < _goldak_heat_sources.size(); ++i)
    _goldak_heat_sources(i).update_time(time);
}

template <int dim, typename MemorySpaceType>
double HeatSources<dim, MemorySpaceType>::value(dealii::Point<dim> const &point,
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

template <int dim, typename MemorySpaceType>
std::vector<ScanPath> HeatSources<dim, MemorySpaceType>::get_scan_paths() const
{
  std::vector<ScanPath> scan_paths;
  for (unsigned int i = 0; i < _electron_beam_heat_sources.size(); ++i)
    scan_paths.push_back(_electron_beam_heat_sources(i).get_scan_path());
  for (unsigned int i = 0; i < _goldak_heat_sources.size(); ++i)
    scan_paths.push_back(_goldak_heat_sources(i).get_scan_path());
  return scan_paths;
}

template <int dim, typename MemorySpaceType>
double HeatSources<dim, MemorySpaceType>::get_current_height(double time) const
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

template <int dim, typename MemorySpaceType>
void HeatSources<dim, MemorySpaceType>::set_beam_properties(
    boost::property_tree::ptree const &heat_source_database)
{
  auto set_properties = [&](auto &source, int const source_index)
  {
    // PropertyTreeInput sources.beam_X
    boost::property_tree::ptree const &beam_database =
        heat_source_database.get_child("beam_" + std::to_string(source_index));

    // PropertyTreeInput sources.beam_X.type
    std::string type = beam_database.get<std::string>("type");

    if (type == "goldak" || type == "electron_beam")
      source.set_beam_properties(beam_database);
  };

  for (unsigned int i = 0; i < _electron_beam_heat_sources.size(); ++i)
    set_properties(_electron_beam_heat_sources[i], _electron_beam_indices[i]);
  for (unsigned int i = 0; i < _goldak_heat_sources.size(); ++i)
    set_properties(_goldak_heat_sources[i], _goldak_indices[i]);
}

} // namespace adamantine

#endif
