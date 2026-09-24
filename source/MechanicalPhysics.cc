/* SPDX-FileCopyrightText: Copyright (c) 2022 - 2026, the adamantine authors.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include <MechanicalPhysics.hh>
#include <instantiation.hh>

#include <deal.II/base/geometry_info.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/types.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/hp/fe_values.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/numerics/vector_tools.h>

#include <boost/serialization/string.hpp>
#include <boost/serialization/vector.hpp>

#ifdef ADAMANTINE_WITH_CALIPER
#include <caliper/cali.h>
#endif

#include <deque>
#include <map>
#include <numeric>
#include <set>
#include <string>
#include <vector>

namespace adamantine
{
namespace DistributedFEIndexComponentsUF
{
class DisjointSet
{
public:
  DisjointSet() = default;

  explicit DisjointSet(const unsigned int n) : parent(n), rank(n, 0)
  {
    std::iota(parent.begin(), parent.end(), 0U);
  }

  unsigned int find(unsigned int x)
  {
    while (parent[x] != x)
    {
      parent[x] = parent[parent[x]];
      x = parent[x];
    }
    return x;
  }

  void unite(unsigned int a, unsigned int b)
  {
    a = find(a);
    b = find(b);

    if (a == b)
      return;

    if (rank[a] < rank[b])
      std::swap(a, b);

    parent[b] = a;

    if (rank[a] == rank[b])
      ++rank[a];
  }

  unsigned int size() const { return parent.size(); }

private:
  std::vector<unsigned int> parent;
  std::vector<unsigned int> rank;
};

struct OwnedCellRecord
{
  std::string cell_id;
  std::string component_rep_id;

  template <class Archive>
  void serialize(Archive &ar, const unsigned int)
  {
    ar &cell_id &component_rep_id;
  }
};

struct LocalComponentRecord
{
  std::string rep_id;
  unsigned int fe_index = 0;
  bool touches_target_boundary = false;

  template <class Archive>
  void serialize(Archive &ar, const unsigned int)
  {
    ar &rep_id &fe_index &touches_target_boundary;
  }
};

struct InterfaceRecord
{
  std::string cell_id_1;
  std::string cell_id_2;

  template <class Archive>
  void serialize(Archive &ar, const unsigned int)
  {
    ar &cell_id_1 &cell_id_2;
  }
};

struct LocalSummary
{
  std::vector<OwnedCellRecord> owned_cells;
  std::vector<LocalComponentRecord> components;
  std::vector<InterfaceRecord> interfaces;

  template <class Archive>
  void serialize(Archive &ar, const unsigned int)
  {
    ar &owned_cells &components &interfaces;
  }
};

template <int dim, int spacedim = dim>
struct ComponentInfo
{
  unsigned int fe_index = 0;
  bool touches_target_boundary = false;

  // Only locally owned cells on this rank.
  std::vector<typename dealii::DoFHandler<dim, spacedim>::active_cell_iterator>
      locally_owned_cells;
};

template <int dim, int spacedim = dim>
struct Result
{
  std::vector<ComponentInfo<dim, spacedim>> components;

  // Only for locally owned cells on this rank.
  std::map<std::string, unsigned int> component_of_locally_owned_cell;
};

template <int dim, int spacedim = dim>
Result<dim, spacedim> find_components(
    const dealii::DoFHandler<dim, spacedim> &dof_handler,
    const std::vector<dealii::types::boundary_id> &target_boundary_ids,
    const MPI_Comm mpi_communicator)
{
  using Cell = typename dealii::DoFHandler<dim, spacedim>::active_cell_iterator;

  // --------------------------------------------------------------------------
  // Step 1: collect all locally owned active cells
  // --------------------------------------------------------------------------
  std::vector<Cell> local_cells;
  std::map<std::string, unsigned int> local_index_of_id;

  for (const Cell &cell : dof_handler.active_cell_iterators())
    if (cell->is_locally_owned())
    {
      const std::string id = cell->id().to_string();
      local_index_of_id[id] = local_cells.size();
      local_cells.push_back(cell);
    }

  DisjointSet local_dsu(local_cells.size());
  std::vector<bool> local_touches_target_boundary(local_cells.size(), false);

  // Same-FE cross-rank adjacencies.
  std::set<std::pair<std::string, std::string>> interface_pairs;

  auto process_neighbor = [&](const unsigned int i, const unsigned int fe_index,
                              const std::string &cell_id, const Cell &neighbor)
  {
    if (!neighbor->is_active())
      return;

    if (neighbor->active_fe_index() != fe_index)
      return;

    if (neighbor->is_locally_owned())
    {
      const auto it = local_index_of_id.find(neighbor->id().to_string());
      if (it != local_index_of_id.end())
        local_dsu.unite(i, it->second);
    }
    else if (neighbor->is_ghost())
    {
      std::string a = cell_id;
      std::string b = neighbor->id().to_string();

      if (b < a)
        std::swap(a, b);

      interface_pairs.emplace(std::move(a), std::move(b));
    }
  };

  // --------------------------------------------------------------------------
  // Step 2: local union-find
  // --------------------------------------------------------------------------
  for (unsigned int i = 0; i < local_cells.size(); ++i)
  {
    const Cell cell = local_cells[i];
    const std::string cell_id = cell->id().to_string();
    const unsigned int fe_idx = cell->active_fe_index();

    for (unsigned int f = 0; f < dealii::GeometryInfo<dim>::faces_per_cell; ++f)
    {
      if (cell->at_boundary(f))
      {
        if (std::find(target_boundary_ids.begin(), target_boundary_ids.end(),
                      cell->face(f)->boundary_id()) !=
            target_boundary_ids.end())
          local_touches_target_boundary[i] = true;

        continue;
      }

      if (cell->neighbor_is_coarser(f))
      {
        process_neighbor(i, fe_idx, cell_id, cell->neighbor(f));
      }
      else
      {
        const auto neighbor = cell->neighbor(f);

        if (neighbor->is_active())
        {
          process_neighbor(i, fe_idx, cell_id, neighbor);
        }
        else
        {
          for (unsigned int subface = 0; subface < cell->face(f)->n_children();
               ++subface)
            process_neighbor(i, fe_idx, cell_id,
                             cell->neighbor_child_on_subface(f, subface));
        }
      }
    }
  }

  // --------------------------------------------------------------------------
  // Step 3: compress local components
  // --------------------------------------------------------------------------
  LocalSummary local_summary;
  std::map<unsigned int, std::string> rep_id_of_root;
  std::map<unsigned int, unsigned int> fe_index_of_root;
  std::map<unsigned int, bool> touches_of_root;
  std::map<std::string, std::string> local_rep_of_owned_cell;

  for (unsigned int i = 0; i < local_cells.size(); ++i)
  {
    const unsigned int root = local_dsu.find(i);

    if (rep_id_of_root.find(root) == rep_id_of_root.end())
    {
      rep_id_of_root[root] = local_cells[root]->id().to_string();
      fe_index_of_root[root] = local_cells[root]->active_fe_index();
    }

    touches_of_root[root] =
        touches_of_root[root] || local_touches_target_boundary[i];
  }

  for (const auto &[root, rep_id] : rep_id_of_root)
    local_summary.components.push_back(
        {rep_id, fe_index_of_root[root], touches_of_root[root]});

  for (unsigned int i = 0; i < local_cells.size(); ++i)
  {
    const std::string cell_id = local_cells[i]->id().to_string();
    const std::string rep_id = rep_id_of_root[local_dsu.find(i)];

    local_summary.owned_cells.push_back({cell_id, rep_id});
    local_rep_of_owned_cell[cell_id] = rep_id;
  }

  for (const auto &p : interface_pairs)
    local_summary.interfaces.push_back({p.first, p.second});

  // --------------------------------------------------------------------------
  // Step 4: gather local summaries
  // --------------------------------------------------------------------------
  const std::vector<LocalSummary> all_summaries =
      dealii::Utilities::MPI::all_gather(mpi_communicator, local_summary);

  // --------------------------------------------------------------------------
  // Step 5: global union-find on local-component representatives
  // --------------------------------------------------------------------------
  std::map<std::string, unsigned int> component_index_of_rep;
  std::vector<std::string> component_reps;
  std::vector<unsigned int> fe_index_of_rep;
  std::vector<bool> touches_of_rep;
  std::map<std::string, std::string> owned_cell_to_rep;

  auto ensure_component_index =
      [&](const std::string &rep_id, const unsigned int fe_index)
  {
    const auto it = component_index_of_rep.find(rep_id);
    if (it != component_index_of_rep.end())
      return it->second;

    const unsigned int idx = component_reps.size();
    component_index_of_rep[rep_id] = idx;
    component_reps.push_back(rep_id);
    fe_index_of_rep.push_back(fe_index);
    touches_of_rep.push_back(false);
    return idx;
  };

  for (const auto &summary : all_summaries)
  {
    for (const auto &comp : summary.components)
    {
      const unsigned int idx =
          ensure_component_index(comp.rep_id, comp.fe_index);

      touches_of_rep[idx] = touches_of_rep[idx] || comp.touches_target_boundary;
    }

    for (const auto &cell : summary.owned_cells)
      owned_cell_to_rep[cell.cell_id] = cell.component_rep_id;
  }

  DisjointSet global_dsu(component_reps.size());

  for (const auto &summary : all_summaries)
    for (const auto &edge : summary.interfaces)
    {
      const auto it_a = owned_cell_to_rep.find(edge.cell_id_1);
      const auto it_b = owned_cell_to_rep.find(edge.cell_id_2);

      if (it_a == owned_cell_to_rep.end() || it_b == owned_cell_to_rep.end())
        continue;

      const unsigned int ia = component_index_of_rep.at(it_a->second);
      const unsigned int ib = component_index_of_rep.at(it_b->second);

      if (fe_index_of_rep[ia] == fe_index_of_rep[ib])
        global_dsu.unite(ia, ib);
    }

  // --------------------------------------------------------------------------
  // Step 6: compact global components and fill local cell lists
  // --------------------------------------------------------------------------
  Result<dim, spacedim> result;
  std::map<unsigned int, unsigned int> compact_id_of_root;

  for (unsigned int i = 0; i < global_dsu.size(); ++i)
  {
    const unsigned int root = global_dsu.find(i);

    auto it = compact_id_of_root.find(root);
    if (it == compact_id_of_root.end())
    {
      const unsigned int cid = result.components.size();
      compact_id_of_root[root] = cid;
      result.components.emplace_back();
      result.components.back().fe_index = fe_index_of_rep[root];
      it = compact_id_of_root.find(root);
    }

    const unsigned int cid = it->second;
    result.components[cid].touches_target_boundary =
        result.components[cid].touches_target_boundary || touches_of_rep[i];
  }

  for (const Cell &cell : dof_handler.active_cell_iterators())
    if (cell->is_locally_owned())
    {
      const std::string cell_id = cell->id().to_string();
      const std::string rep_id = local_rep_of_owned_cell.at(cell_id);

      const unsigned int root =
          global_dsu.find(component_index_of_rep.at(rep_id));
      const unsigned int cid = compact_id_of_root.at(root);

      result.component_of_locally_owned_cell[cell_id] = cid;
      result.components[cid].locally_owned_cells.push_back(cell);
    }

  return result;
}

} // namespace DistributedFEIndexComponentsUF

template <int dim, int n_materials, int p_order, typename MaterialStates,
          typename MemorySpaceType>
MechanicalPhysics<dim, n_materials, p_order, MaterialStates, MemorySpaceType>::
    MechanicalPhysics(
        MPI_Comm const &communicator, unsigned int const fe_degree,
        Geometry<dim> &geometry, Boundary const &boundary,
        MaterialProperty<dim, n_materials, p_order, MaterialStates,
                         MemorySpaceType> &material_properties,
        std::vector<double> const &reference_temperatures)
    : _geometry(geometry), _boundary(boundary),
      _material_properties(material_properties),
      _dof_handler(_geometry.get_triangulation()),
      _reference_temperatures(reference_temperatures),
      _solution_transfer(_dof_handler),
      _closest_quad_point_adaptation(dealii::QGauss<dim>(fe_degree + 1)),
      _cell_data_transfer(
          dynamic_cast<const dealii::parallel::distributed::Triangulation<dim>
                           &>(_dof_handler.get_triangulation()),
          /* transfer_variable_size_data */ false,
          [&](const typename dealii::Triangulation<dim>::cell_iterator &parent,
              const std::vector<std::vector<double>> &parent_values)
          {
            return _closest_quad_point_adaptation.coarse_to_fine(parent,
                                                                 parent_values);
          },
          [&](const typename dealii::Triangulation<dim>::cell_iterator &parent,
              const std::vector<std::vector<std::vector<double>>> &child_values)
          {
            return _closest_quad_point_adaptation.fine_to_coarse(parent,
                                                                 child_values);
          })
{
  // Create the FECollection
  _fe_collection.push_back(
      dealii::FESystem<dim>(dealii::FE_Q<dim>(fe_degree) ^ dim));
  _fe_collection.push_back(
      dealii::FESystem<dim>(dealii::FE_Nothing<dim>() ^ dim));

  // Create the QCollection
  _q_collection.push_back(dealii::QGauss<dim>(fe_degree + 1));
  _q_collection.push_back(dealii::QGauss<dim>(1));

  // Solve the mechanical problem only on the part of the domain that has solid
  // material.
  unsigned int n_active_cells =
      _dof_handler.get_triangulation().n_active_cells();
  for (auto const &cell :
       dealii::filter_iterators(_dof_handler.active_cell_iterators(),
                                dealii::IteratorFilters::LocallyOwnedCell()))
  {
    if (_material_properties.get_state_ratio(
            cell, MaterialStates::State::solid) > 0.99)
    {
      cell->set_active_fe_index(0);
    }
    else
    {
      cell->set_active_fe_index(1);
    }
  }

  // Create the mechanical operator
  _mechanical_operator =
      std::make_unique<MechanicalOperator<dim, n_materials, p_order,
                                          MaterialStates, MemorySpaceType>>(
          communicator, _material_properties, reference_temperatures);

  // Create the data used to compute the stress tensor
  unsigned int const n_quad_pts = _q_collection.max_n_quadrature_points();
  _plastic_internal_variable.reserve(n_active_cells);

  for (auto const &cell : _dof_handler.active_cell_iterators())
  {
    if (cell->is_locally_owned())
    {
      auto elastic_limit = _material_properties.get_mechanical_property(
          cell, StateProperty::elastic_limit);
      _plastic_internal_variable.emplace_back(
          std::vector<double>(n_quad_pts, elastic_limit));
    }
    else
    {
      _plastic_internal_variable.emplace_back(std::vector<double>(
          n_quad_pts, std::numeric_limits<double>::signaling_NaN()));
    }
  }
  _stress.resize(n_active_cells,
                 std::vector<dealii::SymmetricTensor<2, dim>>(n_quad_pts));
  _back_stress.resize(n_active_cells,
                      std::vector<dealii::SymmetricTensor<2, dim>>(n_quad_pts));
  _thermal_stress.resize(n_active_cells, std::vector<double>(n_quad_pts));
}

template <int dim, int n_materials, int p_order, typename MaterialStates,
          typename MemorySpaceType>
void MechanicalPhysics<dim, n_materials, p_order, MaterialStates,
                       MemorySpaceType>::
    setup_dofs(std::vector<std::shared_ptr<BodyForce<dim>>> const &body_forces)
{
  _dof_handler.distribute_dofs(_fe_collection);
  dealii::IndexSet locally_relevant_dofs =
      dealii::DoFTools::extract_locally_relevant_dofs(_dof_handler);
  dealii::IndexSet locally_owned_dofs = _dof_handler.locally_owned_dofs();
  _affine_constraints.reinit(locally_owned_dofs, locally_relevant_dofs);
  dealii::DoFTools::make_hanging_node_constraints(_dof_handler,
                                                  _affine_constraints);

  std::map<dealii::types::boundary_id, const dealii::Function<dim> *>
      boundary_function_map;
  dealii::Functions::ZeroFunction<dim> zero_function(dim);
  auto boundary_ids = _boundary.get_boundary_ids(BoundaryType::clamped);
  for (auto id : boundary_ids)
  {
    boundary_function_map[id] = &zero_function;
  }
  dealii::VectorTools::interpolate_boundary_values(
      _dof_handler, boundary_function_map, _affine_constraints);
  _affine_constraints.close();

  _mechanical_operator->reinit(_dof_handler, _affine_constraints, _q_collection,
                               body_forces);
}

template <int dim, int n_materials, int p_order, typename MaterialStates,
          typename MemorySpaceType>
void MechanicalPhysics<dim, n_materials, p_order, MaterialStates,
                       MemorySpaceType>::
    update_rhs(std::vector<std::shared_ptr<BodyForce<dim>>> const &body_forces)
{
  _mechanical_operator->assemble_rhs(body_forces);
}

template <int dim, int n_materials, int p_order, typename MaterialStates,
          typename MemorySpaceType>
void MechanicalPhysics<dim, n_materials, p_order, MaterialStates,
                       MemorySpaceType>::
    update_rhs(
        dealii::DoFHandler<dim> const &thermal_dof_handler,
        dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host> const
            &temperature,
        std::vector<bool> const &has_melted,
        std::vector<std::shared_ptr<BodyForce<dim>>> const &body_forces)
{
  _thermal_dof_handler = &thermal_dof_handler;
  _temperature = temperature;
  _has_melted = has_melted;
  _mechanical_operator->update_temperature(thermal_dof_handler, temperature,
                                           has_melted);
  _mechanical_operator->assemble_rhs(body_forces);
}

template <int dim, int n_materials, int p_order, typename MaterialStates,
          typename MemorySpaceType>
void MechanicalPhysics<dim, n_materials, p_order, MaterialStates,
                       MemorySpaceType>::prepare_transfer_mpi()
{
  _old_displacement.update_ghost_values();
  _solution_transfer.prepare_for_coarsening_and_refinement(_old_displacement);

  _data_to_transfer.clear();
  unsigned int const n_quad_pts = _q_collection.max_n_quadrature_points();
  unsigned int const n_doubles_per_quad_scalar = 2;
  unsigned int const n_doubles_per_quad_stress =
      dealii::SymmetricTensor<2, dim>::n_independent_components;

  unsigned int const n_doubles_per_quad =
      n_doubles_per_quad_scalar + n_doubles_per_quad_stress * 2;
  std::vector<std::vector<double>> dummy_cell_data(
      n_quad_pts, std::vector<double>(n_doubles_per_quad,
                                      std::numeric_limits<double>::infinity()));
  std::vector<std::vector<double>> cell_data = dummy_cell_data;
  unsigned int cell_id = 0;
  for (auto const &cell : _dof_handler.active_cell_iterators())
  {
    if (cell->is_locally_owned())
    {
      unsigned int const stress_offset = n_doubles_per_quad_scalar;
      unsigned int const back_stress_offset =
          n_doubles_per_quad_scalar + n_doubles_per_quad_stress;

      for (unsigned int quad = 0; quad < n_quad_pts; ++quad)
      {
        std::vector<double> &cell_data_quad = cell_data[quad];
        cell_data_quad[0] = _plastic_internal_variable[cell_id][quad];
        cell_data_quad[1] = _thermal_stress[cell_id][quad];
        for (unsigned int i = 0; i < n_doubles_per_quad_stress; ++i)
        {
          cell_data_quad[stress_offset + i] =
              _stress[cell_id][quad].access_raw_entry(i);
          cell_data_quad[back_stress_offset + i] =
              _back_stress[cell_id][quad].access_raw_entry(i);
        }
      }
      _data_to_transfer.push_back(cell_data);
    }
    else
    {
      _data_to_transfer.push_back(dummy_cell_data);
    }
    ++cell_id;
  }
  _cell_data_transfer.prepare_for_coarsening_and_refinement(_data_to_transfer);
}

template <int dim, int n_materials, int p_order, typename MaterialStates,
          typename MemorySpaceType>
void MechanicalPhysics<dim, n_materials, p_order, MaterialStates,
                       MemorySpaceType>::complete_transfer_mpi()
{
  _dof_handler.distribute_dofs(_fe_collection);

  const dealii::IndexSet locally_relevant_dofs =
      dealii::DoFTools::extract_locally_relevant_dofs(_dof_handler);
  _old_displacement.reinit(_dof_handler.locally_owned_dofs(),
                           locally_relevant_dofs,
#if DEAL_II_VERSION_GTE(9, 7, 0)
                           _dof_handler.get_mpi_communicator()
#else
                           _dof_handler.get_communicator()
#endif
  );
  _solution_transfer.interpolate(_old_displacement);

  auto n_active_cells = _dof_handler.get_triangulation().n_active_cells();
  unsigned int const n_quad_pts = _q_collection.max_n_quadrature_points();

  _plastic_internal_variable.resize(n_active_cells,
                                    std::vector<double>(n_quad_pts));
  _thermal_stress.resize(n_active_cells, std::vector<double>(n_quad_pts));
  _stress.resize(n_active_cells,
                 std::vector<dealii::SymmetricTensor<2, dim>>(n_quad_pts));
  _back_stress.resize(n_active_cells,
                      std::vector<dealii::SymmetricTensor<2, dim>>(n_quad_pts));

  unsigned int const n_doubles_per_quad_scalar = 2;
  unsigned int const n_doubles_per_quad_stress =
      dealii::SymmetricTensor<2, dim>::n_independent_components;

  unsigned int const n_doubles_per_quad =
      n_doubles_per_quad_scalar + n_doubles_per_quad_stress * 2;
  std::vector<std::vector<std::vector<double>>> data_to_unpack(
      n_active_cells, std::vector<std::vector<double>>(
                          n_quad_pts, std::vector<double>(n_doubles_per_quad)));
  _cell_data_transfer.unpack(data_to_unpack);

  unsigned int cell_id = 0;
  for (auto const &cell : _dof_handler.active_cell_iterators())
  {
    if (cell->is_locally_owned())
    {
      unsigned int const stress_offset = n_doubles_per_quad_scalar;
      unsigned int const back_stress_offset =
          n_doubles_per_quad_scalar + n_doubles_per_quad_stress;

      for (unsigned int quad = 0; quad < n_quad_pts; ++quad)
      {
        _plastic_internal_variable[cell_id][quad] =
            data_to_unpack[cell_id][quad][0];
        _thermal_stress[cell_id][quad] = data_to_unpack[cell_id][quad][1];
        for (unsigned int i = 0; i < n_doubles_per_quad_stress; ++i)
        {
          _stress[cell_id][quad].access_raw_entry(i) =
              data_to_unpack[cell_id][quad][stress_offset + i];
          _back_stress[cell_id][quad].access_raw_entry(i) =
              data_to_unpack[cell_id][quad][back_stress_offset + i];
        }
      }
    }
    ++cell_id;
  }
}

template <int dim, int n_materials, int p_order, typename MaterialStates,
          typename MemorySpaceType>
void MechanicalPhysics<dim, n_materials, p_order, MaterialStates,
                       MemorySpaceType>::
    setup_dofs(
        dealii::DoFHandler<dim> const &thermal_dof_handler,
        dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host> const
            &temperature,
        std::vector<bool> const &has_melted, bool rebuild_matrix,
        std::vector<std::shared_ptr<BodyForce<dim>>> const &body_forces)
{
  _thermal_dof_handler = &thermal_dof_handler;
  _temperature = temperature;
  _has_melted = has_melted;
  _mechanical_operator->update_temperature(thermal_dof_handler, temperature,
                                           has_melted);
  // Update the active fe indices, the plastic variables, and the displacement.
  unsigned int const n_quad_pts = _q_collection.max_n_quadrature_points();
  unsigned int cell_id = 0;
  std::vector<std::vector<double>> saved_old_displacement;
  std::vector<std::vector<double>> tmp_plastic_internal_variable;
  std::vector<std::vector<double>> tmp_thermal_stress;
  std::vector<std::vector<dealii::SymmetricTensor<2, dim>>> tmp_stress;
  std::vector<std::vector<dealii::SymmetricTensor<2, dim>>> tmp_back_stress;
  // The number of cells to activate/deactive should be small, so we can
  // already reserve the memory.
  unsigned int const n_dofs_per_cell = _fe_collection.max_dofs_per_cell();
  unsigned int const n_old_active_cells = _plastic_internal_variable.size();
  std::vector<dealii::types::global_dof_index> global_dof_indices(
      n_dofs_per_cell);
  tmp_plastic_internal_variable.reserve(n_old_active_cells);
  tmp_thermal_stress.reserve(n_old_active_cells);
  tmp_stress.reserve(n_old_active_cells);
  tmp_back_stress.reserve(_back_stress.size());
  // First we save _old_displacement if it exists
  if (_old_displacement.size())
  {
    _old_displacement.update_ghost_values();

    std::vector<double> cell_values(n_dofs_per_cell);
    saved_old_displacement.reserve(n_old_active_cells);
    for (auto const &cell : _dof_handler.active_cell_iterators())
    {
      if (cell->is_locally_owned())
      {
        auto fe_index = cell->active_fe_index();
        if (fe_index == 0)
        {
          // The cell contains solid material, we need to save the displacement
          cell->get_dof_indices(global_dof_indices);
          for (unsigned int i = 0; i < n_dofs_per_cell; ++i)
          {
            cell_values[i] = _old_displacement[global_dof_indices[i]];
          }
        }
        else
        {
          // The cell does not contain material or it is liquid. The
          // displacement is ignored.
          cell_values.assign(n_dofs_per_cell, 0.);
        }
        saved_old_displacement.push_back(cell_values);
      }
      else
      {
        saved_old_displacement.push_back(std::vector<double>(n_dofs_per_cell));
      }
    }
  }

  // Now we can update the fe indices and the plastic variables.
  for (auto const &cell : _dof_handler.active_cell_iterators())
  {
    if (cell->is_locally_owned())
    {
      auto current_fe_index = cell->active_fe_index();
      if (_material_properties.get_state_ratio(
              cell, MaterialStates::State::solid) > 0.99)
      {
        // Only enable the cell if it is also enabled for the thermal simulation
        // Get the thermal DoFHandler cell iterator
        dealii::DoFCellAccessor<dim, dim, false> thermal_cell(
            &(_dof_handler.get_triangulation()), cell->level(), cell->index(),
            &thermal_dof_handler);
        auto updated_fe_index = thermal_cell.active_fe_index();
        if (current_fe_index == updated_fe_index)
        {
          // The cells is unchanged, we just copy the plastic variables as-is.
          tmp_plastic_internal_variable.push_back(
              _plastic_internal_variable[cell_id]);
          tmp_thermal_stress.push_back(_thermal_stress[cell_id]);
          tmp_stress.push_back(_stress[cell_id]);
          tmp_back_stress.push_back(_back_stress[cell_id]);
        }
        else
        {
          // The cell has solidified or material has been added. The new cells
          // are initialized with default values.
          auto elastic_limit = _material_properties.get_mechanical_property(
              cell, StateProperty::elastic_limit);
          tmp_plastic_internal_variable.push_back(
              std::vector<double>(n_quad_pts, elastic_limit));
          tmp_thermal_stress.push_back(std::vector<double>(n_quad_pts));
          tmp_stress.push_back(
              std::vector<dealii::SymmetricTensor<2, dim>>(n_quad_pts));
          tmp_back_stress.push_back(
              std::vector<dealii::SymmetricTensor<2, dim>>(n_quad_pts));

          cell->set_active_fe_index(updated_fe_index);
          rebuild_matrix = true;
        }
      }
      else
      {
        if (current_fe_index == 0)
        {
          rebuild_matrix = true;
        }

        // The cell is liquid. We don't need to save the plastic variables.
        cell->set_active_fe_index(1);
        tmp_plastic_internal_variable.push_back(std::vector<double>(
            n_quad_pts, std::numeric_limits<double>::signaling_NaN()));
        tmp_thermal_stress.push_back(std::vector<double>(n_quad_pts));
        tmp_stress.push_back(
            std::vector<dealii::SymmetricTensor<2, dim>>(n_quad_pts));
        tmp_back_stress.push_back(
            std::vector<dealii::SymmetricTensor<2, dim>>(n_quad_pts));
      }
    }
    else
    {
      tmp_plastic_internal_variable.push_back(std::vector<double>(
          n_quad_pts, std::numeric_limits<double>::signaling_NaN()));
      tmp_thermal_stress.push_back(std::vector<double>(n_quad_pts));
      tmp_stress.push_back(
          std::vector<dealii::SymmetricTensor<2, dim>>(n_quad_pts));
      tmp_back_stress.push_back(
          std::vector<dealii::SymmetricTensor<2, dim>>(n_quad_pts));
    }
    ++cell_id;
  }

  // Check if we need to rebuild the matrix
  rebuild_matrix =
      dealii::Utilities::MPI::logical_or(rebuild_matrix,
#if DEAL_II_VERSION_GTE(9, 7, 0)
                                         _dof_handler.get_mpi_communicator()
#else
                                         _dof_handler.get_communicator()
#endif
      );

  if (rebuild_matrix)
  {
    // Ensure that we aren't activating cells that aren't connected to a clamped
    // boundary
    auto boundary_ids = _boundary.get_boundary_ids(BoundaryType::clamped);

    auto result = DistributedFEIndexComponentsUF::find_components(
        _dof_handler, boundary_ids, MPI_COMM_WORLD);

    for (unsigned int c = 0; c < result.components.size(); ++c)
    {
      if (result.components[c].fe_index == 0 &&
          !result.components[c].touches_target_boundary)
      {
        rebuild_matrix = true;
        for (auto &cell : result.components[c].locally_owned_cells)
          cell->set_active_fe_index(1);
      }
    }
  }

  // If we do not need to rebuild the matrix. Update the rhs and exit.
  if (!rebuild_matrix)
  {
    update_rhs(body_forces);
    return;
  }

  _plastic_internal_variable.swap(tmp_plastic_internal_variable);
  _thermal_stress.swap(tmp_thermal_stress);
  _stress.swap(tmp_stress);
  _back_stress.swap(tmp_back_stress);

  setup_dofs(body_forces);

  // Update _old_displacement if necessary
  const dealii::IndexSet locally_relevant_dofs =
      dealii::DoFTools::extract_locally_relevant_dofs(_dof_handler);
  const dealii::IndexSet locally_owned_dofs = _dof_handler.locally_owned_dofs();
  _old_displacement.reinit(locally_owned_dofs, locally_relevant_dofs,
#if DEAL_II_VERSION_GTE(9, 7, 0)
                           _dof_handler.get_mpi_communicator()
#else
                           _dof_handler.get_communicator()
#endif
  );

  if (saved_old_displacement.size())
  {
    cell_id = 0;
    for (auto const &cell : _dof_handler.active_cell_iterators())
    {
      if (cell->is_locally_owned())
      {
        auto fe_index = cell->active_fe_index();
        if (fe_index == 0)
        {
          cell->get_dof_indices(global_dof_indices);
          for (unsigned int i = 0; i < n_dofs_per_cell; ++i)
          {
            if (locally_owned_dofs.is_element(global_dof_indices[i]))
              _old_displacement[global_dof_indices[i]] =
                  saved_old_displacement[cell_id][i];
          }
        }
      }
      ++cell_id;
    }
    _old_displacement.compress(dealii::VectorOperation::insert);
  }
}

template <int dim, int n_materials, int p_order, typename MaterialStates,
          typename MemorySpaceType>
dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host>
MechanicalPhysics<dim, n_materials, p_order, MaterialStates,
                  MemorySpaceType>::solve()
{
#ifdef ADAMANTINE_WITH_CALIPER
  CALI_MARK_BEGIN("solve mechanical system");
#endif

  dealii::IndexSet locally_owned_dofs = _dof_handler.locally_owned_dofs();
  dealii::IndexSet locally_relevant_dofs =
      dealii::DoFTools::extract_locally_relevant_dofs(_dof_handler);
#if DEAL_II_VERSION_GTE(9, 7, 0) && defined(DEAL_II_TRILINOS_WITH_TPETRA)
  using TrilinosVectorType = dealii::LinearAlgebra::TpetraWrappers::Vector<
      double, dealii::MemorySpace::Default>;
#else
  using TrilinosVectorType = dealii::TrilinosWrappers::MPI::Vector;
#endif
  TrilinosVectorType displacement(
      locally_owned_dofs, _mechanical_operator->rhs().get_mpi_communicator());
  TrilinosVectorType rhs_device(
      locally_owned_dofs, _mechanical_operator->rhs().get_mpi_communicator());
  dealii::LinearAlgebra::ReadWriteVector<double> rw_vector(locally_owned_dofs);

  rw_vector.import_elements(_mechanical_operator->rhs(),
                            dealii::VectorOperation::insert);
  rhs_device.import_elements(rw_vector, dealii::VectorOperation::insert);

  // Solve the mechanical problem assuming that the deformation is elastic
  // TODO check that we are computing only difference of the displacement
  // compared to the previous time step!!
  unsigned int const max_iter = _dof_handler.n_dofs() / 10;
  double const tol = 1e-12 * _mechanical_operator->rhs().l2_norm();
  dealii::SolverControl solver_control(max_iter, tol);
  dealii::SolverCG<TrilinosVectorType> cg(solver_control);
  cg.solve(_mechanical_operator->system_matrix(), displacement, rhs_device,
           _mechanical_operator->preconditioner());

  rw_vector.import_elements(displacement, dealii::VectorOperation::insert);
  dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host>
      displacement_host(locally_owned_dofs, locally_relevant_dofs,
                        _mechanical_operator->rhs().get_mpi_communicator());
  displacement_host.import_elements(rw_vector, dealii::VectorOperation::insert);
  _affine_constraints.distribute(displacement_host);

  // Compute the new stress assuming the deformation is elastic.
  // If the stress is under the yield criterion, the deformation is elastic and
  // we are done. Otherwise we need to use the radial return algorithm to
  // compute the plastic deformation.
  dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host>
      incremental_displacement(
          locally_owned_dofs, locally_relevant_dofs,
          _mechanical_operator->rhs().get_mpi_communicator());
  incremental_displacement = displacement_host;
  if (_old_displacement.size() > 0)
  {
    incremental_displacement -= _old_displacement;
  }
  incremental_displacement.update_ghost_values();
  compute_stress(incremental_displacement);

  _old_displacement.swap(displacement_host);

#ifdef ADAMANTINE_WITH_CALIPER
  CALI_MARK_END("solve mechanical system");
#endif

  return _old_displacement;
}

template <int dim, int n_materials, int p_order, typename MaterialStates,
          typename MemorySpaceType>
void MechanicalPhysics<dim, n_materials, p_order, MaterialStates,
                       MemorySpaceType>::
    compute_stress(
        dealii::LA::distributed::Vector<double, dealii::MemorySpace::Host> const
            &displacement)
{
  dealii::hp::FEValues<dim> displacement_hp_fe_values(
      _fe_collection, _q_collection, dealii::update_gradients);
  unsigned int const n_q_points = _q_collection.max_n_quadrature_points();
  std::vector<dealii::SymmetricTensor<2, dim>> strain_tensor(n_q_points);
  std::vector<double> temperature_values(n_q_points);
  const dealii::FEValuesExtractors::Vector displacement_extr(0);
  std::unique_ptr<dealii::hp::FEValues<dim>> temperature_hp_fe_values;
  std::vector<unsigned int> cell_indices;
  // When solving a thermomechanical problem, we create mapping between the
  // active cells of the mechanical problem and the active cells of the thermal
  // problem. Liquid cells are active in the thermal problem but not in the
  // mechanical problem.
  if (!_reference_temperatures.empty())
  {
    _temperature.update_ghost_values();
    temperature_hp_fe_values = std::make_unique<dealii::hp::FEValues<dim>>(
        _thermal_dof_handler->get_fe_collection(), _q_collection,
        dealii::update_values);

    auto &triangulation = _dof_handler.get_triangulation();
    cell_indices.resize(triangulation.n_active_cells());
    unsigned int thermal_cell_index = 0;
    for (auto const &tria_cell :
         triangulation.active_cell_iterators() |
             dealii::IteratorFilters::LocallyOwnedCell())
    {
      dealii::TriaIterator<dealii::DoFCellAccessor<dim, dim, false>>
          temperature_cell(&triangulation, tria_cell->level(),
                           tria_cell->index(), _thermal_dof_handler);
      if (temperature_cell->active_fe_index() == 0)
      {
        dealii::TriaIterator<dealii::DoFCellAccessor<dim, dim, false>>
            displacement_cell(&triangulation, tria_cell->level(),
                              tria_cell->index(), &_dof_handler);
        if (displacement_cell->active_fe_index() == 0)
          cell_indices[displacement_cell->active_cell_index()] =
              thermal_cell_index;
        ++thermal_cell_index;
      }
    }
  }
  unsigned int cell_id = 0;
  for (auto const &cell : _dof_handler.active_cell_iterators())
  {
    if (cell->is_locally_owned() && cell->active_fe_index() == 0)
    {
      // Formulation based on the combined isotropic-kinematic hardening model
      // for J2 plasticity in Chapter 3 of R. Borja, Plasticity: Modeling and
      // Computation, Springer-Verlag, 2013. DOI: 10.1007/978-3-642-38547-6
      //
      // Compute the strain. We get the strain for all the quadrature points at
      // once.
      displacement_hp_fe_values.reinit(cell);
      auto const &fe_values = displacement_hp_fe_values.get_present_fe_values();

      fe_values[displacement_extr].get_function_symmetric_gradients(
          displacement, strain_tensor);

      double reference_temperature = 0.;
      if (!_reference_temperatures.empty())
      {
        auto &triangulation = _dof_handler.get_triangulation();
        dealii::TriaIterator<dealii::DoFCellAccessor<dim, dim, false>>
            temperature_cell(&triangulation, cell->level(), cell->index(),
                             _thermal_dof_handler);
        temperature_hp_fe_values->reinit(temperature_cell);
        auto const &temperature_fe_values =
            temperature_hp_fe_values->get_present_fe_values();
        temperature_fe_values.get_function_values(_temperature,
                                                  temperature_values);
        reference_temperature =
            _has_melted[cell_indices[cell->active_cell_index()]]
                ? _reference_temperatures[temperature_cell->material_id()]
                : _reference_temperatures.back();
      }

      double const lambda = _material_properties.get_mechanical_property(
          cell, StateProperty::lame_first_parameter);
      double const mu = _material_properties.get_mechanical_property(
          cell, StateProperty::lame_second_parameter);
      double const plastic_modulus =
          _material_properties.get_mechanical_property(
              cell, StateProperty::plastic_modulus);
      double const iso_hardening_coef =
          _material_properties.get_mechanical_property(
              cell, StateProperty::isotropic_hardening);
      double const alpha = _material_properties.get_mechanical_property(
          cell, StateProperty::thermal_expansion_coef);
      double const beta = (3. * lambda + 2. * mu) * alpha;
      dealii::SymmetricTensor<4, dim> stiffness_tensor =
          lambda * dealii::outer_product(dealii::unit_symmetric_tensor<dim>(),
                                         dealii::unit_symmetric_tensor<dim>()) +
          2 * mu * dealii::identity_tensor<dim>();
      // Loop over the quadrature points.
      for (auto const q : fe_values.quadrature_point_indices())
      {
        // Compute the trial elastic stress.
        dealii::SymmetricTensor<2, dim> elastic_stress = _stress[cell_id][q];
        elastic_stress += stiffness_tensor * strain_tensor[q];
        if (!_reference_temperatures.empty())
        {
          // Compute the thermal stress due to temperature change since the last
          // time step. The rest of the termal stress in already included in the
          // previous strees.
          double const current_thermal_stress =
              beta * (temperature_values[q] - reference_temperature);
          elastic_stress -=
              (current_thermal_stress - _thermal_stress[cell_id][q]) *
              dealii::unit_symmetric_tensor<dim>();
          _thermal_stress[cell_id][q] = current_thermal_stress;
        }

        auto stress_deviator = dealii::deviator(elastic_stress);
        auto effective_stress = stress_deviator - _back_stress[cell_id][q];
        double const effective_stress_norm = effective_stress.norm();
        if (effective_stress_norm <= _plastic_internal_variable[cell_id][q])
        {
          // The deformation is elastic. We just update the stress with the
          // elastic stress.
          _stress[cell_id][q] = elastic_stress;
        }
        else
        {
          // The deformation is plastic. We need to compute a new stress and
          // update the plastic internal variable and the back stress.
          double plastic_strain_increment =
              (effective_stress_norm - _plastic_internal_variable[cell_id][q]) /
              (2. * mu + plastic_modulus);
          auto plastic_flow_direction =
              effective_stress / effective_stress_norm;
          // Update stress
          _stress[cell_id][q] = elastic_stress - 2. * mu *
                                                     plastic_strain_increment *
                                                     plastic_flow_direction;
          // Update plastic internal variable
          _plastic_internal_variable[cell_id][q] +=
              iso_hardening_coef * plastic_modulus * plastic_strain_increment;
          // Update back stress
          _back_stress[cell_id][q] +=
              (1. - iso_hardening_coef) * plastic_modulus *
              plastic_strain_increment * plastic_flow_direction;
        }
      }
    }
    ++cell_id;
  }
}

} // namespace adamantine

INSTANTIATE_DIM_NMAT_PORDER_MATERIALSTATES_HOST(MechanicalPhysics)
INSTANTIATE_DIM_NMAT_PORDER_MATERIALSTATES_DEVICE(MechanicalPhysics)
