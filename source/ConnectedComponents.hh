/* SPDX-FileCopyrightText: Copyright (c) 2026, the adamantine authors.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#ifndef CONNECTED_COMPONENTS_HH
#define CONNECTED_COMPONENTS_HH

#include <deal.II/base/types.h>
#include <deal.II/base/utilities.h>
#include <deal.II/dofs/dof_handler.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <numeric>
#include <unordered_map>
#include <utility>
#include <vector>

namespace adamantine {
namespace ConnectedComponents {
/**
 * Class implementing the union-find algorithm.
 */
class DisjointSet {
public:
  DisjointSet() = default;

  explicit DisjointSet(const unsigned int n) : _parent(n), _rank(n, 0) {
    std::iota(_parent.begin(), _parent.end(), 0U);
  }

  /**
   * Return the root representing the component for a given node.
   */
  unsigned int find(unsigned int x) {
    while (_parent[x] != x) {
      // path contraction
      _parent[x] = _parent[_parent[x]];
      x = _parent[x];
    }
    return x;
  }

  /**
   * Join two nodes so that they belong to the same component.
   */
  void unite(unsigned int a, unsigned int b) {
    a = find(a);
    b = find(b);

    if (a == b)
      return;

    if (_rank[a] < _rank[b])
      std::swap(a, b);

    _parent[b] = a;

    if (_rank[a] == _rank[b])
      ++_rank[a];
  }

  /**
   * Return the number of nodes.
   */
  unsigned int size() const { return _parent.size(); }

private:
  /**
   * The _parent for every node.
   */
  std::vector<unsigned int> _parent;
  /**
   * The rank of every node.
   */
  std::vector<unsigned int> _rank;
};

/**
 * Data structure storing the cell index together with the index of the
 * representative for the component the cell belongs to.
 */
struct OwnedCellRecord {
  dealii::types::global_cell_index _cell_id =
      dealii::numbers::invalid_unsigned_int;
  unsigned int _component_rep_id = dealii::numbers::invalid_unsigned_int;
};

/**
 * Data structure storing the representative for a component together with
 * information about it touching a boundary we care about.
 */
struct LocalComponentRecord {
  unsigned int _rep_id = dealii::numbers::invalid_unsigned_int;
  bool _touches_target_boundary = false;
};

/**
 * Data structure storing the two cells indices for locally owned cell with a
 * ghost neighbor.
 */
struct InterfaceRecord {
  dealii::types::global_cell_index _cell_id_1 =
      dealii::numbers::invalid_unsigned_int;
  dealii::types::global_cell_index _cell_id_2 =
      dealii::numbers::invalid_unsigned_int;
};

/**
 * Data structure storing information about locally owned cells.
 */
struct LocalSummary {
  std::vector<OwnedCellRecord> _owned_cells;
  std::vector<LocalComponentRecord> _components;
  std::vector<InterfaceRecord> _interfaces;
};

/**
 * Data structure storing all cell indices for a component and whether it
 * touches a boundary we care about.
 */
template <int dim> struct ComponentInfo {
  bool _touches_target_boundary = false;

  // Only locally owned cells on this rank are stored.
  std::vector<typename dealii::DoFHandler<dim>::active_cell_iterator>
      _locally_owned_cells;
};

/**
 * Find connected components of locally owned active cells with the given FE
 * index. Connectivity is defined through face adjacency, including across MPI
 * ranks through ghost neighbors.
 *
 * The returned vector stores, on each rank, only the locally owned cells that
 * belong to each global component together with a flag indicating whether that
 * global component touches one of the target boundaries.
 */
template <int dim>
std::vector<ComponentInfo<dim>> find_components(
    const dealii::DoFHandler<dim> &dof_handler,
    const std::vector<dealii::types::boundary_id> &target_boundary_id,
    const unsigned int target_fe_index, const MPI_Comm mpi_communicator) {
  using Cell = typename dealii::DoFHandler<dim>::active_cell_iterator;
  using GlobalCellIndex = dealii::types::global_cell_index;
  // The cell index is either a 32-bit or a 64-bit type. Since it's the largest
  // type we need to send, we use std::uint64_t as type for the MPI buffers.
  using PackedWord = std::uint64_t;

  // Step 1: collect all locally owned active cells with the requested FE index.
  std::vector<Cell> local_cells;
  local_cells.reserve(dof_handler.get_triangulation().n_active_cells());
  std::unordered_map<GlobalCellIndex, unsigned int> local_index_of_id;

  for (const Cell &cell : dof_handler.active_cell_iterators())
    if (cell->is_locally_owned() &&
        cell->active_fe_index() == target_fe_index) {
      const GlobalCellIndex id = cell->global_active_cell_index();
      local_index_of_id[id] = local_cells.size();
      local_cells.push_back(cell);
    }

  // --------------------------------------------------------------------------
  // Step 2: local union-find
  //
  // We compute local connected components, mark which local cells touch a
  // target boundary, and record same-FE interfaces to ghost neighbors. We also
  // remember which locally owned cells appear in at least one cross-rank
  // interface, since only those owned cells need to be communicated later.
  // --------------------------------------------------------------------------
  DisjointSet local_dsu(local_cells.size());
  std::vector<bool> local_touches_target_boundary(local_cells.size(), false);
  std::vector<bool> local_cell_has_interface(local_cells.size(), false);

  // Same-FE cross-rank adjacencies, stored as sorted cell-id pairs.
  std::vector<std::pair<GlobalCellIndex, GlobalCellIndex>> interface_pairs;

  auto process_neighbor = [&](const unsigned int i, const unsigned int fe_index,
                              const GlobalCellIndex cell_id,
                              const Cell &neighbor) {
    if (!neighbor->is_active())
      return;

    if (neighbor->active_fe_index() != fe_index)
      return;

    if (neighbor->is_locally_owned()) {
      const auto it =
          local_index_of_id.find(neighbor->global_active_cell_index());
      if (it != local_index_of_id.end())
        local_dsu.unite(i, it->second);
    } else if (neighbor->is_ghost()) {
      GlobalCellIndex a = cell_id;
      GlobalCellIndex b = neighbor->global_active_cell_index();

      if (b < a)
        std::swap(a, b);

      interface_pairs.emplace_back(a, b);
      local_cell_has_interface[i] = true;
    }
  };

  for (unsigned int i = 0; i < local_cells.size(); ++i) {
    const Cell cell = local_cells[i];
    const GlobalCellIndex cell_id = cell->global_active_cell_index();
    const unsigned int fe_idx = cell->active_fe_index();

    for (unsigned int f = 0; f < dealii::GeometryInfo<dim>::faces_per_cell;
         ++f) {
      if (cell->at_boundary(f)) {
        if (std::find(target_boundary_id.begin(), target_boundary_id.end(),
                      cell->face(f)->boundary_id()) != target_boundary_id.end())
          local_touches_target_boundary[i] = true;

        continue;
      }

      const auto neighbor = cell->neighbor(f);
      if (cell->neighbor_is_coarser(f)) {
        process_neighbor(i, fe_idx, cell_id, neighbor);
      } else {
        if (neighbor->is_active()) {
          process_neighbor(i, fe_idx, cell_id, neighbor);
        } else {
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
  //
  // local_summary._owned_cells stores all locally owned target-FE cells on this
  // rank. owned_to_send is a strict subset containing only the owned cells that
  // appear in at least one cross-rank interface record.
  // --------------------------------------------------------------------------
  LocalSummary local_summary;
  std::unordered_map<unsigned int, GlobalCellIndex> rep_id_of_root;
  std::unordered_map<unsigned int, bool> touches_of_root;

  for (unsigned int i = 0; i < local_cells.size(); ++i) {
    const unsigned int root = local_dsu.find(i);

    if (rep_id_of_root.find(root) == rep_id_of_root.end())
      rep_id_of_root[root] = local_cells[root]->global_active_cell_index();

    touches_of_root[root] =
        touches_of_root[root] || local_touches_target_boundary[i];
  }

  local_summary._components.reserve(rep_id_of_root.size());
  for (const auto &[root, rep_id] : rep_id_of_root)
    local_summary._components.push_back({rep_id, touches_of_root[root]});

  local_summary._owned_cells.reserve(local_cells.size());

  const unsigned int n_owned_to_send = std::count(
      local_cell_has_interface.begin(), local_cell_has_interface.end(), true);
  std::vector<OwnedCellRecord> owned_to_send(n_owned_to_send);
  unsigned int owned_to_send_pos = 0;

  for (unsigned int i = 0; i < local_cells.size(); ++i) {
    const GlobalCellIndex cell_id = local_cells[i]->global_active_cell_index();
    const GlobalCellIndex rep_id = rep_id_of_root[local_dsu.find(i)];

    local_summary._owned_cells.push_back({cell_id, rep_id});

    if (local_cell_has_interface[i])
      owned_to_send[owned_to_send_pos++] = {cell_id, rep_id};
  }

  AssertDimension(owned_to_send_pos, owned_to_send.size());

  local_summary._interfaces.reserve(interface_pairs.size());
  for (const auto &p : interface_pairs)
    local_summary._interfaces.push_back({p.first, p.second});

  // If running with a single MPI rank, we can stop here and construct the
  // final result directly from the local compression step. This avoids the
  // distributed gathering and global union-find.
  if (dealii::Utilities::MPI::n_mpi_processes(mpi_communicator) == 1) {
    std::vector<ComponentInfo<dim>> result;
    result.reserve(local_summary._components.size());

    // Map representative id -> compact component index.
    std::unordered_map<GlobalCellIndex, unsigned int> cid_of_rep;
    cid_of_rep.reserve(local_summary._components.size());

    for (const auto &c : local_summary._components) {
      const unsigned int cid = static_cast<unsigned int>(result.size());
      cid_of_rep[c._rep_id] = cid;
      result.emplace_back();
      result.back()._touches_target_boundary = c._touches_target_boundary;
    }

    for (unsigned int i = 0; i < local_cells.size(); ++i) {
      const unsigned int root = local_dsu.find(i);
      const GlobalCellIndex rep_id = rep_id_of_root.at(root);
      const unsigned int cid = cid_of_rep.at(rep_id);
      result[cid]._locally_owned_cells.push_back(local_cells[i]);
    }

    return result;
  }

  // --------------------------------------------------------------------------
  // Step 4: gather local summaries (packed, non-blocking) and build global maps
  //
  // Packed layout:
  //   [n_components, comp*(rep_id,touch),
  //    n_owned_to_send, owned*(cell_id,rep_id),
  //    n_interfaces, iface*(cell_id_1,cell_id_2)]
  //
  // The send buffer is allocated from the exact number of packed entries.
  // Cell ids and representative ids are carried as 64-bit words, so no 32-bit
  // truncation is required.
  // --------------------------------------------------------------------------
  const int n_ranks = dealii::Utilities::MPI::n_mpi_processes(mpi_communicator);
  const int my_rank =
      dealii::Utilities::MPI::this_mpi_process(mpi_communicator);

  const std::size_t send_buf_size = 1U + 2U * local_summary._components.size() +
                                    1U + 2U * owned_to_send.size() + 1U +
                                    2U * local_summary._interfaces.size();

  std::vector<PackedWord> send_buf(send_buf_size);
  std::size_t send_pos = 0;

  // components
  send_buf[send_pos++] =
      static_cast<PackedWord>(local_summary._components.size());
  for (const auto &c : local_summary._components) {
    send_buf[send_pos++] = static_cast<PackedWord>(c._rep_id);
    send_buf[send_pos++] =
        static_cast<PackedWord>(c._touches_target_boundary ? 1U : 0U);
  }

  // owned cells
  send_buf[send_pos++] = static_cast<PackedWord>(owned_to_send.size());
  for (const auto &o : owned_to_send) {
    send_buf[send_pos++] = static_cast<PackedWord>(o._cell_id);
    send_buf[send_pos++] = static_cast<PackedWord>(o._component_rep_id);
  }

  // interfaces
  send_buf[send_pos++] =
      static_cast<PackedWord>(local_summary._interfaces.size());
  for (const auto &itf : local_summary._interfaces) {
    send_buf[send_pos++] = static_cast<PackedWord>(itf._cell_id_1);
    send_buf[send_pos++] = static_cast<PackedWord>(itf._cell_id_2);
  }

  AssertDimension(send_pos, send_buf.size());

  const int my_send_count = static_cast<int>(send_buf.size());

  // Gather exact message sizes first so each receive buffer can be sized
  // tightly before posting Irecv.
  std::vector<int> send_counts(n_ranks, 0);
  MPI_Allgather(&my_send_count, 1, MPI_INT, send_counts.data(), 1, MPI_INT,
                mpi_communicator);

  // Post non-blocking receives for each sender (except self).
  std::vector<std::vector<PackedWord>> recv_bufs(n_ranks);
  std::vector<MPI_Request> recv_reqs(n_ranks, MPI_REQUEST_NULL);
  int n_pending = 0;
  for (int r = 0; r < n_ranks; ++r) {
    if (r == my_rank)
      continue;
    const int cnt = send_counts[r];
    if (cnt == 0)
      continue;
    recv_bufs[r].resize(cnt);
    MPI_Irecv(recv_bufs[r].data(), cnt, MPI_UINT64_T, r, 0, mpi_communicator,
              &recv_reqs[r]);
    ++n_pending;
  }

  // Post non-blocking sends to all other ranks.
  std::vector<MPI_Request> send_reqs(n_ranks, MPI_REQUEST_NULL);
  for (int r = 0; r < n_ranks; ++r) {
    if (r == my_rank)
      continue;
    MPI_Isend(send_buf.data(), my_send_count, MPI_UINT64_T, r, 0,
              mpi_communicator, &send_reqs[r]);
  }

  // Prepare maps for components and owned mappings; insert local ones early.
  std::unordered_map<GlobalCellIndex, unsigned int> component_index_of_rep;
  std::vector<GlobalCellIndex> component_reps;
  std::vector<bool> touches_of_rep;

  auto ensure_component_index = [&](const GlobalCellIndex rep_id) {
    const auto it = component_index_of_rep.find(rep_id);
    if (it != component_index_of_rep.end())
      return it->second;
    const unsigned int idx = static_cast<unsigned int>(component_reps.size());
    component_index_of_rep[rep_id] = idx;
    component_reps.push_back(rep_id);
    touches_of_rep.push_back(false);
    return idx;
  };

  for (const auto &comp : local_summary._components) {
    const unsigned int idx = ensure_component_index(comp._rep_id);
    touches_of_rep[idx] = touches_of_rep[idx] || comp._touches_target_boundary;
  }

  // Owned-cell mapping built from local data and completed as remote messages
  // arrive. This temporarily stores representative ids; these are converted to
  // deterministic representative indices after all summaries have been read.
  std::unordered_map<GlobalCellIndex, GlobalCellIndex> owned_cell_to_rep_local;
  owned_cell_to_rep_local.reserve(local_summary._owned_cells.size());
  for (const auto &o : local_summary._owned_cells)
    owned_cell_to_rep_local[o._cell_id] = o._component_rep_id;

  // Process incoming messages as they arrive.
  //
  // We unpack components and owned-cell mappings immediately. Interface records
  // are skipped for now and revisited later from recv_bufs once all ownership
  // information is available.
  while (n_pending > 0) {
    int idx;
    MPI_Waitany(n_ranks, recv_reqs.data(), &idx, MPI_STATUS_IGNORE);
    if (idx == MPI_UNDEFINED)
      break;

    const auto &buf = recv_bufs[idx];
    std::size_t pos = 0;

    const std::size_t ncomps = static_cast<std::size_t>(buf[pos++]);
    for (std::size_t ci = 0; ci < ncomps; ++ci) {
      const GlobalCellIndex rep_id = static_cast<GlobalCellIndex>(buf[pos++]);
      const bool touches = (buf[pos++] != 0);
      const unsigned int id = ensure_component_index(rep_id);
      touches_of_rep[id] = touches_of_rep[id] || touches;
    }

    const std::size_t nowned = static_cast<std::size_t>(buf[pos++]);
    for (std::size_t oi = 0; oi < nowned; ++oi) {
      const GlobalCellIndex cell_id = static_cast<GlobalCellIndex>(buf[pos++]);
      const GlobalCellIndex rep_id = static_cast<GlobalCellIndex>(buf[pos++]);
      owned_cell_to_rep_local[cell_id] = rep_id;
    }

    const std::size_t nifaces = static_cast<std::size_t>(buf[pos++]);
    pos += 2U * nifaces;

    AssertDimension(pos, buf.size());

    recv_reqs[idx] = MPI_REQUEST_NULL;
    --n_pending;
  }

  // Ensure all sends are complete.
  MPI_Waitall(n_ranks, send_reqs.data(), MPI_STATUSES_IGNORE);

  // Build the complete list of interface pairs from the local summary and all
  // received summaries. The buffer is allocated from the exact number of
  // interface records.
  std::size_t n_global_interfaces = local_summary._interfaces.size();
  for (int r = 0; r < n_ranks; ++r) {
    if (r == my_rank || send_counts[r] == 0)
      continue;

    const auto &buf = recv_bufs[r];
    std::size_t pos = 0;

    const std::size_t ncomps = static_cast<std::size_t>(buf[pos++]);
    pos += 2U * ncomps;

    const std::size_t nowned = static_cast<std::size_t>(buf[pos++]);
    pos += 2U * nowned;

    const std::size_t nifaces = static_cast<std::size_t>(buf[pos++]);
    n_global_interfaces += nifaces;

    pos += 2U * nifaces;
    AssertDimension(pos, buf.size());
  }

  std::vector<std::pair<GlobalCellIndex, GlobalCellIndex>>
      global_interface_pairs(n_global_interfaces);
  std::size_t interface_pos = 0;

  for (const auto &itf : local_summary._interfaces)
    global_interface_pairs[interface_pos++] = {itf._cell_id_1, itf._cell_id_2};

  for (int r = 0; r < n_ranks; ++r) {
    if (r == my_rank || send_counts[r] == 0)
      continue;

    const auto &buf = recv_bufs[r];
    std::size_t pos = 0;

    const std::size_t ncomps = static_cast<std::size_t>(buf[pos++]);
    pos += 2U * ncomps;

    const std::size_t nowned = static_cast<std::size_t>(buf[pos++]);
    pos += 2U * nowned;

    const std::size_t nifaces = static_cast<std::size_t>(buf[pos++]);
    for (std::size_t ii = 0; ii < nifaces; ++ii) {
      const GlobalCellIndex a = static_cast<GlobalCellIndex>(buf[pos++]);
      const GlobalCellIndex b = static_cast<GlobalCellIndex>(buf[pos++]);
      global_interface_pairs[interface_pos++] = {a, b};
    }

    AssertDimension(pos, buf.size());
  }

  AssertDimension(interface_pos, global_interface_pairs.size());

  // At this point component_index_of_rep and component_reps are populated
  // but their insertion order depends on message arrival order which may
  // differ across ranks. Reindex deterministically by sorting rep ids so all
  // ranks assign the same indices for the same rep ids.
  std::vector<GlobalCellIndex> reps_sorted;
  reps_sorted.reserve(component_index_of_rep.size());
  for (const auto &kv : component_index_of_rep)
    reps_sorted.push_back(kv.first);
  std::sort(reps_sorted.begin(), reps_sorted.end());

  std::unordered_map<GlobalCellIndex, unsigned int> new_component_index_of_rep;
  new_component_index_of_rep.reserve(reps_sorted.size());
  std::vector<GlobalCellIndex> new_component_reps;
  std::vector<bool> new_touches_of_rep;

  new_component_reps.reserve(reps_sorted.size());
  new_touches_of_rep.reserve(reps_sorted.size());

  for (unsigned int i = 0; i < reps_sorted.size(); ++i) {
    const GlobalCellIndex rep_id = reps_sorted[i];
    const unsigned int old_idx = component_index_of_rep.at(rep_id);
    new_component_index_of_rep[rep_id] = i;
    new_component_reps.push_back(rep_id);
    new_touches_of_rep.push_back(touches_of_rep[old_idx]);
  }

  // Swap in the deterministically-ordered structures.
  component_index_of_rep.swap(new_component_index_of_rep);
  component_reps.swap(new_component_reps);
  touches_of_rep.swap(new_touches_of_rep);

  // Convert cell -> representative-id mappings to cell -> representative-index
  // mappings using the deterministic representative numbering.
  std::unordered_map<GlobalCellIndex, unsigned int> owned_cell_to_rep;
  owned_cell_to_rep.reserve(owned_cell_to_rep_local.size());
  for (const auto &p : owned_cell_to_rep_local) {
    const GlobalCellIndex cell = p.first;
    const GlobalCellIndex rep_id = p.second;
    const unsigned int rep_idx = component_index_of_rep.at(rep_id);
    owned_cell_to_rep[cell] = rep_idx;
  }

  // Build global edges from the collected interface pairs by mapping interface
  // endpoints to representative indices.
  std::vector<std::pair<unsigned int, unsigned int>> global_edges;
  global_edges.reserve(global_interface_pairs.size());
  for (const auto &itf : global_interface_pairs) {
    const auto it_a = owned_cell_to_rep.find(itf.first);
    const auto it_b = owned_cell_to_rep.find(itf.second);
    if (it_a == owned_cell_to_rep.end() || it_b == owned_cell_to_rep.end())
      continue;
    unsigned int ia = it_a->second;
    unsigned int ib = it_b->second;
    if (ib < ia)
      std::swap(ia, ib);
    global_edges.emplace_back(ia, ib);
  }

  // Sort and deduplicate edges before the global union-find.
  std::sort(global_edges.begin(), global_edges.end());
  auto new_end = std::unique(global_edges.begin(), global_edges.end());
  global_edges.erase(new_end, global_edges.end());

  DisjointSet global_dsu(component_reps.size());

  for (const auto &e : global_edges)
    global_dsu.unite(e.first, e.second);

  // --------------------------------------------------------------------------
  // Step 6: compact global components and fill local cell lists
  // --------------------------------------------------------------------------
  std::vector<ComponentInfo<dim>> result;
  std::unordered_map<unsigned int, unsigned int> compact_id_of_root;

  for (unsigned int i = 0; i < global_dsu.size(); ++i) {
    const unsigned int root = global_dsu.find(i);

    auto it = compact_id_of_root.find(root);
    if (it == compact_id_of_root.end()) {
      const unsigned int cid = result.size();
      compact_id_of_root[root] = cid;
      result.emplace_back();
      it = compact_id_of_root.find(root);
    }

    const unsigned int cid = it->second;
    result[cid]._touches_target_boundary =
        result[cid]._touches_target_boundary || touches_of_rep[i];
  }

  for (unsigned int i = 0; i < local_cells.size(); ++i) {
    const GlobalCellIndex rep_id = rep_id_of_root[local_dsu.find(i)];

    const unsigned int root =
        global_dsu.find(component_index_of_rep.at(rep_id));
    const unsigned int cid = compact_id_of_root.at(root);

    result[cid]._locally_owned_cells.push_back(local_cells[i]);
  }

  return result;
}

} // namespace ConnectedComponents
} // namespace adamantine

#endif
