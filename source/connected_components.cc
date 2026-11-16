/* SPDX-FileCopyrightText: Copyright (c) 2026, the adamantine authors.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include <connected_components.hh>

#include <deal.II/base/point.h>
#include <deal.II/base/types.h>
#include <deal.II/base/utilities.h>
#include <deal.II/dofs/dof_handler.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace
{
/**
 * Class implementing the union-find algorithm.
 */
class DisjointSet
{
public:
  DisjointSet() = default;

  explicit DisjointSet(unsigned int const n) : _parent(n), _rank(n, 0)
  {
    std::iota(_parent.begin(), _parent.end(), 0U);
  }

  /**
   * Return the root representing the component for a given node.
   */
  unsigned int find(unsigned int x)
  {
    while (_parent[x] != x)
    {
      // path contraction
      _parent[x] = _parent[_parent[x]];
      x = _parent[x];
    }
    return x;
  }

  /**
   * Join two nodes so that they belong to the same component.
   */
  void unite(unsigned int a, unsigned int b)
  {
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
struct OwnedCellRecord
{
  dealii::types::global_cell_index _cell_id =
      dealii::numbers::invalid_unsigned_int;
  dealii::types::global_cell_index _component_rep_id =
      dealii::numbers::invalid_unsigned_int;
};

/**
 * Data structure storing the representative for a component together with
 * information about it touching a boundary we care about.
 */
struct LocalComponentRecord
{
  dealii::types::global_cell_index _rep_id =
      dealii::numbers::invalid_unsigned_int;
  bool _touches_target_boundary = false;
};

/**
 * Data structure storing the two cells indices for locally owned cell with a
 * ghost neighbor.
 */
struct InterfaceRecord
{
  dealii::types::global_cell_index _cell_id_1 =
      dealii::numbers::invalid_unsigned_int;
  dealii::types::global_cell_index _cell_id_2 =
      dealii::numbers::invalid_unsigned_int;
};

/**
 * Data structure storing information about locally owned cells.
 */
struct LocalSummary
{
  std::vector<OwnedCellRecord> _owned_cells;
  std::vector<LocalComponentRecord> _components;
  std::vector<InterfaceRecord> _interfaces;
};

/**
 * Two cells with the same FE index that share the vertex _point but are not
 * face neighbors.
 */
template <int dim>
struct ContactRecord
{
  dealii::types::global_cell_index _cell_id_1 =
      dealii::numbers::invalid_unsigned_int;
  dealii::types::global_cell_index _cell_id_2 =
      dealii::numbers::invalid_unsigned_int;
  dealii::Point<dim> _point;
};

/**
 * Return true if the points contain at least three distinct, non-collinear
 * points.
 */
template <int dim>
bool has_three_non_collinear_points(std::vector<dealii::Point<dim>> const &pts)
{
  if (pts.size() < 3)
    return false;

  // Find a second point different from the first
  dealii::Point<dim> p1;
  std::size_t pts_idx = 1;
  for (; pts_idx < pts.size(); ++pts_idx)
  {
    if (pts[pts_idx].distance(pts[0]) > 1.e-10 * (1. + pts[0].norm()))
    {
      p1 = pts[pts_idx];
      break;
    }
  }
  if (pts_idx == pts.size())
    return false;

  // If every point lies on the line through the first two, all are collinear.
  auto const d1 = p1 - pts[0];
  for (; pts_idx < pts.size(); ++pts_idx)
  {
    auto const d2 = pts[pts_idx] - pts[0];
    double const dot = d1 * d2;
    double const sin2_scaled = d1.norm_square() * d2.norm_square() - dot * dot;
    if (sin2_scaled > 1e-12 * d1.norm_square() * d2.norm_square())
      return true;
  }
  return false;
}

/**
 * Unite nodes of dsu whose cells touch at three or more non-collinear
 * vertices. The shared vertices are accumulated over all cell pairs of the two
 * components. Uniting components can create new qualifying pairs, so we repeat
 * until nothing changes. node_of_cell maps a cell id to a node of dsu and
 * returns invalid_unsigned_int for unknown cells.
 */
template <int dim, typename NodeOfCell>
void unite_by_vertex_contacts(std::vector<ContactRecord<dim>> const &contacts,
                              DisjointSet &dsu, NodeOfCell const &node_of_cell)
{
  struct Resolved
  {
    unsigned int _a;
    unsigned int _b;
    dealii::Point<dim> _point;
  };

  std::vector<Resolved> resolved;
  resolved.reserve(contacts.size());
  for (auto const &c : contacts)
  {
    unsigned int const a = node_of_cell(c._cell_id_1);
    unsigned int const b = node_of_cell(c._cell_id_2);
    if (a != dealii::numbers::invalid_unsigned_int &&
        b != dealii::numbers::invalid_unsigned_int)
      resolved.push_back({a, b, c._point});
  }

  bool changed = true;
  while (changed)
  {
    changed = false;

    std::vector<std::tuple<unsigned int, unsigned int, unsigned int>> keys;
    keys.reserve(resolved.size());
    for (unsigned int i = 0; i < resolved.size(); ++i)
    {
      unsigned int ra = dsu.find(resolved[i]._a);
      unsigned int rb = dsu.find(resolved[i]._b);
      if (ra == rb)
        continue;
      if (rb < ra)
        std::swap(ra, rb);
      keys.emplace_back(ra, rb, i);
    }
    std::sort(keys.begin(), keys.end());

    for (std::size_t begin = 0; begin < keys.size();)
    {
      std::size_t end = begin;
      std::vector<dealii::Point<dim>> pts;
      while (end < keys.size() &&
             std::get<0>(keys[end]) == std::get<0>(keys[begin]) &&
             std::get<1>(keys[end]) == std::get<1>(keys[begin]))
        pts.push_back(resolved[std::get<2>(keys[end++])]._point);

      unsigned int const ra = std::get<0>(keys[begin]);
      unsigned int const rb = std::get<1>(keys[begin]);
      if (dsu.find(ra) != dsu.find(rb) && has_three_non_collinear_points(pts))
      {
        dsu.unite(ra, rb);
        changed = true;
      }
      begin = end;
    }
  }
}
} // namespace

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
std::vector<adamantine::ConnectedComponents::ComponentInfo<dim>>
adamantine::ConnectedComponents::find_components(
    dealii::DoFHandler<dim> const &dof_handler,
    std::vector<dealii::types::boundary_id> const &target_boundary_id,
    unsigned int const target_fe_index)
{
#if DEAL_II_VERSION_GTE(9, 7, 0)
  MPI_Comm const mpi_communicator = dof_handler.get_mpi_communicator();
#else
  MPI_Comm const mpi_communicator = dof_handler.get_communicator();
#endif

  using Cell = typename dealii::DoFHandler<dim>::active_cell_iterator;
  using GlobalCellIndex = dealii::types::global_cell_index;
  // The cell index is either a 32-bit or a 64-bit type. Since it's the largest
  // type we need to send, we can use it as type for the MPI buffers.
  using PackedWord = GlobalCellIndex;

  // Step 1: collect all locally owned active cells with the requested FE index.
  std::vector<Cell> local_cells;
  local_cells.reserve(dof_handler.get_triangulation().n_active_cells());
  std::unordered_map<GlobalCellIndex, unsigned int> local_index_of_id;

  for (Cell const &cell : dof_handler.active_cell_iterators())
    if (cell->is_locally_owned() && cell->active_fe_index() == target_fe_index)
    {
      GlobalCellIndex const id = cell->global_active_cell_index();
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

  auto process_neighbor = [&](unsigned int const i, unsigned int const fe_index,
                              GlobalCellIndex const cell_id,
                              Cell const &neighbor)
  {
    if (!neighbor->is_active())
      return;

    if (neighbor->active_fe_index() != fe_index)
      return;

    if (neighbor->is_locally_owned())
    {
      auto const it =
          local_index_of_id.find(neighbor->global_active_cell_index());
      if (it != local_index_of_id.end())
        local_dsu.unite(i, it->second);
    }
    else if (neighbor->is_ghost())
    {
      GlobalCellIndex a = cell_id;
      GlobalCellIndex b = neighbor->global_active_cell_index();

      if (b < a)
        std::swap(a, b);

      interface_pairs.emplace_back(a, b);
      local_cell_has_interface[i] = true;
    }
  };

  for (unsigned int i = 0; i < local_cells.size(); ++i)
  {
    Cell const cell = local_cells[i];
    GlobalCellIndex const cell_id = cell->global_active_cell_index();
    unsigned int const fe_idx = cell->active_fe_index();

    for (unsigned int f = 0; f < dealii::GeometryInfo<dim>::faces_per_cell; ++f)
    {
      if (cell->at_boundary(f))
      {
        if (std::find(target_boundary_id.begin(), target_boundary_id.end(),
                      cell->face(f)->boundary_id()) != target_boundary_id.end())
          local_touches_target_boundary[i] = true;

        continue;
      }

      auto const neighbor = cell->neighbor(f);
      if (cell->neighbor_is_coarser(f))
      {
        process_neighbor(i, fe_idx, cell_id, neighbor);
      }
      else
      {
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
  // Step 2b: vertex contacts
  //
  // Components also count as connected if their cells touch at three or more
  // non-collinear vertices. For every vertex we look at the owned and ghost
  // cells with the target FE index. Owned cells are reduced to one cell per
  // local component. Pairs of owned cells in different local components and
  // pairs of an owned and a ghost cell are recorded as contacts.
  // --------------------------------------------------------------------------
  std::vector<ContactRecord<dim>> contacts;
  {
    struct VertexCell
    {
      GlobalCellIndex _cell_id;
      int _local_index; // -1 for ghost cells
    };
    // First gather for every vertex its position and the cells that contain it.
    std::unordered_map<unsigned int, std::vector<VertexCell>> cells_of_vertex;
    std::unordered_map<unsigned int, dealii::Point<dim>> point_of_vertex;

    for (Cell const &cell : dof_handler.active_cell_iterators())
      if ((cell->is_locally_owned() || cell->is_ghost()) &&
          cell->active_fe_index() == target_fe_index)
      {
        int local_index = -1;
        if (cell->is_locally_owned())
          local_index = static_cast<int>(
              local_index_of_id.at(cell->global_active_cell_index()));
        for (unsigned int v = 0;
             v < dealii::GeometryInfo<dim>::vertices_per_cell; ++v)
        {
          cells_of_vertex[cell->vertex_index(v)].push_back(
              {cell->global_active_cell_index(), local_index});
          point_of_vertex[cell->vertex_index(v)] = cell->vertex(v);
        }
      }

    // For every vertex with at least two corresponding cells, only consider one
    // cell per component (unless it's a ghost cell) and run a pairwise match
    // storing the contact if at least one of those cells is locally owned.
    for (auto const &[vertex_index, vertex_cells] : cells_of_vertex)
    {
      if (vertex_cells.size() < 2)
        continue;

      std::vector<VertexCell> reduced;
      std::unordered_map<unsigned int, bool> root_seen;
      for (auto const &vc : vertex_cells)
        if (vc._local_index < 0 ||
            root_seen.emplace(local_dsu.find(vc._local_index), true).second)
          reduced.push_back(vc);

      for (std::size_t i = 0; i < reduced.size(); ++i)
        for (std::size_t j = i + 1; j < reduced.size(); ++j)
        {
          if (reduced[i]._local_index < 0 && reduced[j]._local_index < 0)
            continue;
          contacts.push_back({reduced[i]._cell_id, reduced[j]._cell_id,
                              point_of_vertex.at(vertex_index)});
          if (reduced[i]._local_index >= 0)
            local_cell_has_interface[reduced[i]._local_index] = true;
          if (reduced[j]._local_index >= 0)
            local_cell_has_interface[reduced[j]._local_index] = true;
        }
    }
  }

  // With a single rank all contacts are known, so merge them right away.
  if (dealii::Utilities::MPI::n_mpi_processes(mpi_communicator) == 1)
    unite_by_vertex_contacts<dim>(
        contacts, local_dsu,
        [&](GlobalCellIndex const id)
        {
          auto const it = local_index_of_id.find(id);
          return it == local_index_of_id.end()
                     ? dealii::numbers::invalid_unsigned_int
                     : it->second;
        });

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

  for (unsigned int i = 0; i < local_cells.size(); ++i)
  {
    unsigned int const root = local_dsu.find(i);

    if (rep_id_of_root.find(root) == rep_id_of_root.end())
      rep_id_of_root[root] = local_cells[root]->global_active_cell_index();

    touches_of_root[root] =
        touches_of_root[root] || local_touches_target_boundary[i];
  }

  local_summary._components.reserve(rep_id_of_root.size());
  for (auto const &[root, rep_id] : rep_id_of_root)
    local_summary._components.push_back({rep_id, touches_of_root[root]});

  local_summary._owned_cells.reserve(local_cells.size());

  unsigned int const n_owned_to_send = std::count(
      local_cell_has_interface.begin(), local_cell_has_interface.end(), true);
  std::vector<OwnedCellRecord> owned_to_send(n_owned_to_send);
  unsigned int owned_to_send_pos = 0;

  for (unsigned int i = 0; i < local_cells.size(); ++i)
  {
    GlobalCellIndex const cell_id = local_cells[i]->global_active_cell_index();
    GlobalCellIndex const rep_id = rep_id_of_root[local_dsu.find(i)];

    local_summary._owned_cells.push_back({cell_id, rep_id});

    if (local_cell_has_interface[i])
      owned_to_send[owned_to_send_pos++] = {cell_id, rep_id};
  }

  AssertDimension(owned_to_send_pos, owned_to_send.size());

  local_summary._interfaces.reserve(interface_pairs.size());
  for (auto const &p : interface_pairs)
    local_summary._interfaces.push_back({p.first, p.second});

  // If running with a single MPI rank, we can stop here and construct the
  // final result directly from the local compression step. This avoids the
  // distributed gathering and global union-find.
  if (dealii::Utilities::MPI::n_mpi_processes(mpi_communicator) == 1)
  {
    std::vector<ComponentInfo<dim>> result;
    result.reserve(local_summary._components.size());

    // Map representative id -> compact component index.
    std::unordered_map<GlobalCellIndex, unsigned int> cid_of_rep;
    cid_of_rep.reserve(local_summary._components.size());

    for (auto const &c : local_summary._components)
    {
      unsigned int const cid = static_cast<unsigned int>(result.size());
      cid_of_rep[c._rep_id] = cid;
      result.emplace_back();
      result.back()._touches_target_boundary = c._touches_target_boundary;
    }

    for (unsigned int i = 0; i < local_cells.size(); ++i)
    {
      unsigned int const root = local_dsu.find(i);
      GlobalCellIndex const rep_id = rep_id_of_root.at(root);
      unsigned int const cid = cid_of_rep.at(rep_id);
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
  //    n_interfaces, iface*(cell_id_1,cell_id_2),
  //    n_contacts, contact*(cell_id_1,cell_id_2,vertex coordinates)]
  //
  // The send buffer is allocated from the exact number of packed entries.
  // Cell ids and representative ids are packed as GlobalCellIndex values
  // avoiding any narrowing casts.
  // --------------------------------------------------------------------------
  int const n_ranks = dealii::Utilities::MPI::n_mpi_processes(mpi_communicator);
  int const my_rank =
      dealii::Utilities::MPI::this_mpi_process(mpi_communicator);

  // The vertex coordinates are copied bit-wise into whole packed words.
  constexpr std::size_t coord_words =
      (dim * sizeof(double) + sizeof(PackedWord) - 1) / sizeof(PackedWord);
  constexpr std::size_t contact_words = 2U + coord_words;

  std::size_t const send_buf_size = 1U + 2U * local_summary._components.size() +
                                    1U + 2U * owned_to_send.size() + 1U +
                                    2U * local_summary._interfaces.size() + 1U +
                                    contact_words * contacts.size();

  std::vector<PackedWord> send_buf(send_buf_size);
  std::size_t send_pos = 0;

  // components
  send_buf[send_pos++] = local_summary._components.size();
  for (auto const &c : local_summary._components)
  {
    send_buf[send_pos++] = c._rep_id;
    send_buf[send_pos++] = c._touches_target_boundary;
  }

  // owned cells
  send_buf[send_pos++] = static_cast<PackedWord>(owned_to_send.size());
  for (auto const &o : owned_to_send)
  {
    send_buf[send_pos++] = o._cell_id;
    send_buf[send_pos++] = o._component_rep_id;
  }

  // interfaces
  send_buf[send_pos++] =
      static_cast<PackedWord>(local_summary._interfaces.size());
  for (auto const &itf : local_summary._interfaces)
  {
    send_buf[send_pos++] = itf._cell_id_1;
    send_buf[send_pos++] = itf._cell_id_2;
  }

  // contacts
  send_buf[send_pos++] = static_cast<PackedWord>(contacts.size());
  for (auto const &c : contacts)
  {
    send_buf[send_pos++] = c._cell_id_1;
    send_buf[send_pos++] = c._cell_id_2;
    std::memcpy(&send_buf[send_pos], &c._point[0], dim * sizeof(double));
    send_pos += coord_words;
  }

  AssertDimension(send_pos, send_buf.size());
  AssertIndexRange(send_buf.size() * sizeof(PackedWord),
                   std::numeric_limits<int>::max());

  int const my_send_count = static_cast<int>(send_buf.size());
  int const my_send_n_bytes = my_send_count * sizeof(PackedWord);

  // Gather exact message sizes first so each receive buffer can be sized
  // tightly before posting Irecv.
  std::vector<int> send_counts =
      dealii::Utilities::MPI::all_gather(mpi_communicator, my_send_count);

  // Post non-blocking receives for each sender (except self).
  std::vector<std::vector<PackedWord>> recv_bufs(n_ranks);
  std::vector<MPI_Request> recv_reqs(n_ranks, MPI_REQUEST_NULL);
  int n_pending = 0;
  for (int r = 0; r < n_ranks; ++r)
  {
    if (r == my_rank)
      continue;
    int const cnt = send_counts[r];
    if (cnt == 0)
      continue;
    recv_bufs[r].resize(cnt);
    int const recv_n_bytes = cnt * static_cast<int>(sizeof(PackedWord));
    MPI_Irecv(recv_bufs[r].data(), recv_n_bytes, MPI_BYTE, r, 0,
              mpi_communicator, &recv_reqs[r]);
    ++n_pending;
  }

  // Post non-blocking sends to all other ranks.
  std::vector<MPI_Request> send_reqs(n_ranks, MPI_REQUEST_NULL);
  for (int r = 0; r < n_ranks; ++r)
  {
    if (r == my_rank)
      continue;
    MPI_Isend(send_buf.data(), my_send_n_bytes, MPI_BYTE, r, 0,
              mpi_communicator, &send_reqs[r]);
  }

  // Prepare maps for components and owned mappings; insert local ones early.
  std::unordered_map<GlobalCellIndex, unsigned int> component_index_of_rep;
  std::vector<GlobalCellIndex> component_reps;
  std::vector<bool> touches_of_rep;

  // Helper populating component_index_of_rep and component_reps
  auto ensure_component_index = [&](GlobalCellIndex const rep_id)
  {
    auto const it = component_index_of_rep.find(rep_id);
    if (it != component_index_of_rep.end())
      return it->second;
    unsigned int const idx = static_cast<unsigned int>(component_reps.size());
    component_index_of_rep[rep_id] = idx;
    component_reps.push_back(rep_id);
    touches_of_rep.push_back(false);
    return idx;
  };

  for (auto const &comp : local_summary._components)
  {
    unsigned int const idx = ensure_component_index(comp._rep_id);
    touches_of_rep[idx] = touches_of_rep[idx] || comp._touches_target_boundary;
  }

  // Owned-cell mapping built from local data and completed as remote messages
  // arrive. This temporarily stores representative ids; these are converted to
  // deterministic representative indices after all summaries have been read.
  std::unordered_map<GlobalCellIndex, GlobalCellIndex> owned_cell_to_rep_local;
  owned_cell_to_rep_local.reserve(local_summary._owned_cells.size());
  for (auto const &o : local_summary._owned_cells)
    owned_cell_to_rep_local[o._cell_id] = o._component_rep_id;

  // Process incoming messages as they arrive.
  //
  // We unpack components and owned-cell mappings immediately. Interface records
  // are skipped for now and revisited later from recv_bufs once all ownership
  // information is available.
  while (n_pending > 0)
  {
    int idx;
    MPI_Waitany(n_ranks, recv_reqs.data(), &idx, MPI_STATUS_IGNORE);
    Assert(idx != MPI_UNDEFINED, dealii::ExcInternalError());

    auto const &buf = recv_bufs[idx];
    std::size_t pos = 0;

    std::size_t const ncomps = static_cast<std::size_t>(buf[pos++]);
    for (std::size_t ci = 0; ci < ncomps; ++ci)
    {
      GlobalCellIndex const rep_id = buf[pos++];
      bool const touches = (buf[pos++] != 0);
      unsigned int const id = ensure_component_index(rep_id);
      touches_of_rep[id] = touches_of_rep[id] || touches;
    }

    std::size_t const nowned = static_cast<std::size_t>(buf[pos++]);
    for (std::size_t oi = 0; oi < nowned; ++oi)
    {
      GlobalCellIndex const cell_id = buf[pos++];
      GlobalCellIndex const rep_id = buf[pos++];
      owned_cell_to_rep_local[cell_id] = rep_id;
    }

    std::size_t const nifaces = static_cast<std::size_t>(buf[pos++]);
    pos += 2U * nifaces;

    std::size_t const ncontacts = static_cast<std::size_t>(buf[pos++]);
    pos += contact_words * ncontacts;

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
  for (int r = 0; r < n_ranks; ++r)
  {
    if (r == my_rank || send_counts[r] == 0)
      continue;

    auto const &buf = recv_bufs[r];
    std::size_t pos = 0;

    std::size_t const ncomps = static_cast<std::size_t>(buf[pos++]);
    pos += 2U * ncomps;

    std::size_t const nowned = static_cast<std::size_t>(buf[pos++]);
    pos += 2U * nowned;

    std::size_t const nifaces = static_cast<std::size_t>(buf[pos++]);
    n_global_interfaces += nifaces;

    pos += 2U * nifaces;
    pos += 1U + contact_words * static_cast<std::size_t>(buf[pos]);
    AssertDimension(pos, buf.size());
  }

  std::vector<std::pair<GlobalCellIndex, GlobalCellIndex>>
      global_interface_pairs(n_global_interfaces);
  std::size_t interface_pos = 0;

  for (auto const &itf : local_summary._interfaces)
    global_interface_pairs[interface_pos++] = {itf._cell_id_1, itf._cell_id_2};

  for (int r = 0; r < n_ranks; ++r)
  {
    if (r == my_rank || send_counts[r] == 0)
      continue;

    auto const &buf = recv_bufs[r];
    std::size_t pos = 0;

    std::size_t const ncomps = static_cast<std::size_t>(buf[pos++]);
    pos += 2U * ncomps;

    std::size_t const nowned = static_cast<std::size_t>(buf[pos++]);
    pos += 2U * nowned;

    std::size_t const nifaces = static_cast<std::size_t>(buf[pos++]);
    for (std::size_t ii = 0; ii < nifaces; ++ii)
    {
      GlobalCellIndex const a = buf[pos++];
      GlobalCellIndex const b = buf[pos++];
      global_interface_pairs[interface_pos++] = {a, b};
    }

    std::size_t const ncontacts = static_cast<std::size_t>(buf[pos++]);
    for (std::size_t ci = 0; ci < ncontacts; ++ci)
    {
      ContactRecord<dim> c;
      c._cell_id_1 = buf[pos++];
      c._cell_id_2 = buf[pos++];
      std::memcpy(&c._point[0], &buf[pos], dim * sizeof(double));
      pos += coord_words;
      contacts.push_back(c);
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
  for (auto const &kv : component_index_of_rep)
    reps_sorted.push_back(kv.first);
  std::sort(reps_sorted.begin(), reps_sorted.end());

  std::unordered_map<GlobalCellIndex, unsigned int> new_component_index_of_rep;
  new_component_index_of_rep.reserve(reps_sorted.size());
  std::vector<GlobalCellIndex> new_component_reps;
  std::vector<bool> new_touches_of_rep;

  new_component_reps.reserve(reps_sorted.size());
  new_touches_of_rep.reserve(reps_sorted.size());

  for (unsigned int i = 0; i < reps_sorted.size(); ++i)
  {
    GlobalCellIndex const rep_id = reps_sorted[i];
    unsigned int const old_idx = component_index_of_rep.at(rep_id);
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
  for (auto const &p : owned_cell_to_rep_local)
  {
    GlobalCellIndex const cell = p.first;
    GlobalCellIndex const rep_id = p.second;
    unsigned int const rep_idx = component_index_of_rep.at(rep_id);
    owned_cell_to_rep[cell] = rep_idx;
  }

  // Build global edges from the collected interface pairs by mapping interface
  // endpoints to representative indices.
  std::vector<std::pair<unsigned int, unsigned int>> global_edges;
  global_edges.reserve(global_interface_pairs.size());
  for (auto const &itf : global_interface_pairs)
  {
    auto const it_a = owned_cell_to_rep.find(itf.first);
    auto const it_b = owned_cell_to_rep.find(itf.second);
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

  for (auto const &e : global_edges)
    global_dsu.unite(e.first, e.second);

  // Merge components touching at three or more non-collinear vertices. The
  // contacts contain all local ones and the ones received from other ranks.
  unite_by_vertex_contacts<dim>(
      contacts, global_dsu,
      [&](GlobalCellIndex const id)
      {
        auto const it = owned_cell_to_rep.find(id);
        return it == owned_cell_to_rep.end()
                   ? dealii::numbers::invalid_unsigned_int
                   : it->second;
      });

  // --------------------------------------------------------------------------
  // Step 6: compact global components and fill local cell lists
  // --------------------------------------------------------------------------
  std::vector<ComponentInfo<dim>> result;
  std::unordered_map<unsigned int, unsigned int> compact_id_of_root;

  for (unsigned int i = 0; i < global_dsu.size(); ++i)
  {
    unsigned int const root = global_dsu.find(i);

    auto it = compact_id_of_root.find(root);
    if (it == compact_id_of_root.end())
    {
      unsigned int const cid = result.size();
      compact_id_of_root[root] = cid;
      result.emplace_back();
      it = compact_id_of_root.find(root);
    }

    unsigned int const cid = it->second;
    result[cid]._touches_target_boundary =
        result[cid]._touches_target_boundary || touches_of_rep[i];
  }

  for (unsigned int i = 0; i < local_cells.size(); ++i)
  {
    GlobalCellIndex const rep_id = rep_id_of_root[local_dsu.find(i)];

    unsigned int const root =
        global_dsu.find(component_index_of_rep.at(rep_id));
    unsigned int const cid = compact_id_of_root.at(root);

    result[cid]._locally_owned_cells.push_back(local_cells[i]);
  }

  return result;
}

// Explicit template instantiations

template std::vector<adamantine::ConnectedComponents::ComponentInfo<2>>
adamantine::ConnectedComponents::find_components(
    dealii::DoFHandler<2> const &dof_handler,
    std::vector<dealii::types::boundary_id> const &target_boundary_id,
    unsigned int const target_fe_index);
template std::vector<adamantine::ConnectedComponents::ComponentInfo<3>>
adamantine::ConnectedComponents::find_components(
    dealii::DoFHandler<3> const &dof_handler,
    std::vector<dealii::types::boundary_id> const &target_boundary_id,
    unsigned int const target_fe_index);
