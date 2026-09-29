/* SPDX-FileCopyrightText: Copyright (c) 2025, the adamantine authors.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#ifndef CONNECTED_COMPONENTS_HH
#define CONNECTED_COMPONENTS_HH

//#include <deal.II/distributed/tria.h>
//#include <deal.II/base/geometry_info.h>
//#include <deal.II/base/types.h>
//#include <deal.II/dofs/dof_handler.h>
//#include <deal.II/grid/grid_generator.h>
//#include <deal.II/fe/fe_dgq.h>
//#include <deal.II/base/conditional_ostream.h>

//#include <deque>
//#include <vector>
//#include <unordered_map>
//#include <unordered_set>
//#include <functional>
//#include <type_traits>
//#include <cstdint>
//#include <algorithm>

namespace adamantine {
namespace ConnectedComponents
{
  class DisjointSet
  {
  public:
    DisjointSet() = default;

    explicit DisjointSet(const unsigned int n)
      : parent(n), rank(n, 0)
    {
      std::iota(parent.begin(), parent.end(), 0U);
    }

    unsigned int
    find(unsigned int x)
    {
      while (parent[x] != x)
        {
          parent[x] = parent[parent[x]];
          x         = parent[x];
        }
      return x;
    }

    void
    unite(unsigned int a, unsigned int b)
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

    unsigned int
    size() const
    {
      return parent.size();
    }

  private:
    std::vector<unsigned int> parent;
    std::vector<unsigned int> rank;
  };

  struct OwnedCellRecord
  {
    unsigned int cell_id = numbers::invalid_unsigned_int;
    unsigned int component_rep_id = numbers::invalid_unsigned_int;

    template <class Archive>
    void
    serialize(Archive &ar, const unsigned int)
    {
      ar &cell_id &component_rep_id;
    }
  };

  struct LocalComponentRecord
  {
    unsigned int  rep_id = numbers::invalid_unsigned_int;
    bool         touches_target_boundary = false;

    template <class Archive>
    void
    serialize(Archive &ar, const unsigned int)
    {
      ar &rep_id &touches_target_boundary;
    }
  };

  struct InterfaceRecord
  {
    unsigned int cell_id_1 = numbers::invalid_unsigned_int;
    unsigned int cell_id_2 = numbers::invalid_unsigned_int;

    template <class Archive>
    void
    serialize(Archive &ar, const unsigned int)
    {
      ar &cell_id_1 &cell_id_2;
    }
  };

  struct LocalSummary
  {
    std::vector<OwnedCellRecord>      owned_cells;
    std::vector<LocalComponentRecord> components;
    std::vector<InterfaceRecord>      interfaces;

    template <class Archive>
    void
    serialize(Archive &ar, const unsigned int)
    {
      ar &owned_cells &components &interfaces;
    }
  };

  template <int dim, int spacedim = dim>
  struct ComponentInfo
  {
    bool         touches_target_boundary = false;

    // Only locally owned cells on this rank.
    std::vector<typename DoFHandler<dim, spacedim>::active_cell_iterator>
      locally_owned_cells;
  };

  template <int dim, int spacedim = dim>
  std::vector<ComponentInfo<dim, spacedim>>
  find_components(const DoFHandler<dim, spacedim> &dof_handler,
                  const std::vector<types::boundary_id>&        target_boundary_id,
                  const unsigned int target_fe_index,
                  const MPI_Comm                   mpi_communicator)
  {
    Kokkos::Timer timer;
    using Cell = typename DoFHandler<dim, spacedim>::active_cell_iterator;

    // Step 1: collect all locally owned active cells
    // --------------------------------------------------------------------------
    std::vector<Cell>                   local_cells;
    local_cells.reserve(dof_handler.get_triangulation().n_cells());
    std::unordered_map<unsigned int, unsigned int> local_index_of_id;

    for (const Cell &cell : dof_handler.active_cell_iterators())
      if (cell->is_locally_owned() && cell->active_fe_index() ==target_fe_index)
        {
          const unsigned int id = cell->active_cell_index();
          local_index_of_id[id] = local_cells.size();
          local_cells.push_back(cell);
        }
    std::cout << "Collect cells: " << timer.seconds() << std::endl;
    timer.reset();

    DisjointSet       local_dsu(local_cells.size());
    std::vector<bool> local_touches_target_boundary(local_cells.size(), false);

    // Same-FE cross-rank adjacencies.
    std::vector<std::pair<unsigned int, unsigned int>> interface_pairs;

    auto process_neighbor =
      [&](const unsigned int i,
          const unsigned int fe_index,
          const unsigned int cell_id,
          const Cell        &neighbor)
      {
        if (!neighbor->is_active())
          return;

        if (neighbor->active_fe_index() != fe_index)
          return;

        if (neighbor->is_locally_owned())
          {
            const auto it = local_index_of_id.find(neighbor->active_cell_index());
            if (it != local_index_of_id.end())
              local_dsu.unite(i, it->second);
          }
        else if (neighbor->is_ghost())
          {
            unsigned int a = cell_id;
            unsigned int b = neighbor->active_cell_index();

            if (b < a)
              std::swap(a, b);

            interface_pairs.emplace_back(a, b);
          }
      };

    // --------------------------------------------------------------------------
    // Step 2: local union-find
    // --------------------------------------------------------------------------

    for (unsigned int i = 0; i < local_cells.size(); ++i)
      {
        const Cell        cell    = local_cells[i];
        const unsigned int cell_id = cell->active_cell_index();
        const unsigned int fe_idx = cell->active_fe_index();

        for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
          {
            if (cell->at_boundary(f))
              {
                if (std::find(target_boundary_id.begin(), target_boundary_id.end(), cell->face(f)->boundary_id()) != target_boundary_id.end())
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
                    for (unsigned int subface = 0;
                         subface < cell->face(f)->n_children();
                         ++subface)
                      process_neighbor(i,
                                       fe_idx,
                                       cell_id,
                                       cell->neighbor_child_on_subface(f,
                                                                       subface));
                  }
              }
          }
      }
  std::cout << "Local union find: " << timer.seconds() << std::endl;
    timer.reset();

    // --------------------------------------------------------------------------
    // Step 3: compress local components
    // --------------------------------------------------------------------------
    LocalSummary                        local_summary;
    std::unordered_map<unsigned int, unsigned int> rep_id_of_root;
    std::unordered_map<unsigned int, unsigned int> fe_index_of_root;
    std::unordered_map<unsigned int, bool>         touches_of_root;
    std::unordered_map<unsigned int, unsigned int> local_rep_of_owned_cell;

    for (unsigned int i = 0; i < local_cells.size(); ++i)
      {
        const unsigned int root = local_dsu.find(i);

        if (rep_id_of_root.find(root) == rep_id_of_root.end())
          {
            rep_id_of_root[root]   = local_cells[root]->active_cell_index();
            fe_index_of_root[root] = local_cells[root]->active_fe_index();
          }

        touches_of_root[root] =
          touches_of_root[root] || local_touches_target_boundary[i];
      }

    for (const auto &[root, rep_id] : rep_id_of_root)
      local_summary.components.push_back(
        {rep_id, touches_of_root[root]});

    for (unsigned int i = 0; i < local_cells.size(); ++i)
      {
        const unsigned int cell_id = local_cells[i]->active_cell_index();
        const unsigned int rep_id  = rep_id_of_root[local_dsu.find(i)];

        local_summary.owned_cells.push_back({cell_id, rep_id});
        local_rep_of_owned_cell[cell_id] = rep_id;
      }

    for (const auto &p : interface_pairs)
      local_summary.interfaces.push_back({p.first, p.second});

  std::cout << "Compress local: " << timer.seconds() << std::endl;
    timer.reset();

    // If running with a single MPI rank, we can stop here and construct the
    // final result directly from the local compression step. This avoids the
    // distributed gathering and global union-find.
    if (Utilities::MPI::n_mpi_processes(mpi_communicator) == 1)
      {
        std::vector<ComponentInfo<dim, spacedim>> result;

        // Map rep_id -> compact component index
        std::unordered_map<unsigned int, unsigned int> cid_of_rep;
        cid_of_rep.reserve(local_summary.components.size() * 2 + 4);

        for (const auto &c : local_summary.components)
          {
            const unsigned int cid = static_cast<unsigned int>(result.size());
            cid_of_rep[c.rep_id] = cid;
            result.emplace_back();
            result.back().touches_target_boundary = c.touches_target_boundary;
          }

        // Fill locally owned cells from local_cells vector using mapping built
        // earlier (local_rep_of_owned_cell).
        for (const Cell &cell : local_cells)
          {
            const unsigned int cell_id = cell->active_cell_index();
            const auto it = local_rep_of_owned_cell.find(cell_id);
            if (it == local_rep_of_owned_cell.end())
              continue;
            const unsigned int rep_id = it->second;
            const unsigned int cid = cid_of_rep.at(rep_id);
            result[cid].locally_owned_cells.push_back(cell);
          }

        return result;
      }

    // --------------------------------------------------------------------------
    // Step 4: gather local summaries (packed, non-blocking) and build global maps
    // --------------------------------------------------------------------------

    const int n_ranks = Utilities::MPI::n_mpi_processes(mpi_communicator);
    const int my_rank = Utilities::MPI::this_mpi_process(mpi_communicator);

    // Pack local summary into a compact uint32 buffer: [C, comp*(rep,fe,touch), O, owned*(cell,rep), I, iface*(a,b)]
    std::vector<uint32_t> send_buf;
    send_buf.reserve(local_summary.components.size() * 3 + local_summary.owned_cells.size() * 2 + local_summary.interfaces.size() * 2 + 3);

    // components
    send_buf.push_back(static_cast<uint32_t>(local_summary.components.size()));
    for (const auto &c : local_summary.components)
      {
        send_buf.push_back(static_cast<uint32_t>(c.rep_id));
        send_buf.push_back(static_cast<uint32_t>(c.touches_target_boundary ? 1U : 0U));
      }

    // owned cells
    send_buf.push_back(static_cast<uint32_t>(local_summary.owned_cells.size()));
    for (const auto &o : local_summary.owned_cells)
      {
        send_buf.push_back(static_cast<uint32_t>(o.cell_id));
        send_buf.push_back(static_cast<uint32_t>(o.component_rep_id));
      }

    // interfaces
    send_buf.push_back(static_cast<uint32_t>(local_summary.interfaces.size()));
    for (const auto &itf : local_summary.interfaces)
      {
        send_buf.push_back(static_cast<uint32_t>(itf.cell_id_1));
        send_buf.push_back(static_cast<uint32_t>(itf.cell_id_2));
      }

    const int my_send_count = static_cast<int>(send_buf.size());

    // Gather sizes non-blocking (then wait) so we can post per-rank receives
    std::vector<int> send_counts(n_ranks, 0);
    MPI_Request sizes_req;
    MPI_Iallgather(&my_send_count, 1, MPI_INT, send_counts.data(), 1, MPI_INT, mpi_communicator, &sizes_req);
    MPI_Wait(&sizes_req, MPI_STATUS_IGNORE);

    // Post non-blocking receives for each sender (except self)
    std::vector<std::vector<uint32_t>> recv_bufs(n_ranks);
    std::vector<MPI_Request> recv_reqs(n_ranks, MPI_REQUEST_NULL);
    int n_pending = 0;
    for (int r = 0; r < n_ranks; ++r)
      {
        if (r == my_rank)
          continue;
        const int cnt = send_counts[r];
        if (cnt == 0)
          continue;
        recv_bufs[r].resize(cnt);
        MPI_Irecv(recv_bufs[r].data(), cnt, MPI_UNSIGNED, r, 0, mpi_communicator, &recv_reqs[r]);
        ++n_pending;
      }

    // Post non-blocking sends to all other ranks that need our buffer
    std::vector<MPI_Request> send_reqs(n_ranks, MPI_REQUEST_NULL);
    for (int r = 0; r < n_ranks; ++r)
      {
        if (r == my_rank)
          continue;
        if (send_counts[r] == 0)
          continue;
        MPI_Isend(send_buf.data(), my_send_count, MPI_UNSIGNED, r, 0, mpi_communicator, &send_reqs[r]);
      }

    // Prepare maps for components and owned mappings; insert local ones early
    std::unordered_map<unsigned int, unsigned int> component_index_of_rep;
    std::vector<unsigned int> component_reps;
    std::vector<bool> touches_of_rep;

    // Global interface pairs gathered from all ranks (we include our own local
    // interfaces and append those unpacked from incoming messages)
    std::vector<std::pair<unsigned int, unsigned int>> global_interface_pairs;
    global_interface_pairs.reserve(local_summary.interfaces.size() + 64);
    for (const auto &p : local_summary.interfaces)
      global_interface_pairs.emplace_back(p.cell_id_1, p.cell_id_2);

    auto ensure_component_index = [&](const unsigned int rep_id) {
      const auto it = component_index_of_rep.find(rep_id);
      if (it != component_index_of_rep.end())
        return it->second;
      const unsigned int idx = static_cast<unsigned int>(component_reps.size());
      component_index_of_rep[rep_id] = idx;
      component_reps.push_back(rep_id);
      touches_of_rep.push_back(false);
      return idx;
    };

    for (const auto &comp : local_summary.components)
      {
        const unsigned int idx = ensure_component_index(comp.rep_id);
        touches_of_rep[idx] = touches_of_rep[idx] || comp.touches_target_boundary;
      }

    // Owned local mapping (rep_id -> cell)
    std::unordered_map<unsigned int, unsigned int> owned_cell_to_rep_local;
    owned_cell_to_rep_local.reserve(local_summary.owned_cells.size() * 2 + 16);
    for (const auto &o : local_summary.owned_cells)
      owned_cell_to_rep_local[o.cell_id] = o.component_rep_id;

    // Process incoming messages as they arrive
    while (n_pending > 0)
      {
        int idx;
        MPI_Waitany(n_ranks, recv_reqs.data(), &idx, MPI_STATUS_IGNORE);
        if (idx == MPI_UNDEFINED)
          break;

        // Unpack buffer from rank idx
        const auto &buf = recv_bufs[idx];
        int pos = 0;
        const uint32_t ncomps = buf[pos++];
        for (uint32_t ci = 0; ci < ncomps; ++ci)
          {
            const unsigned int rep_id = static_cast<unsigned int>(buf[pos++]);
            const bool touches = (buf[pos++] != 0);
            const unsigned int id = ensure_component_index(rep_id);
            touches_of_rep[id] = touches_of_rep[id] || touches;
          }

        const uint32_t nowned = buf[pos++];
        for (uint32_t oi = 0; oi < nowned; ++oi)
          {
            const unsigned int cell_id = static_cast<unsigned int>(buf[pos++]);
            const unsigned int rep_id = static_cast<unsigned int>(buf[pos++]);
            // We store rep indices later (after all components known); temporarily
            // store mapping from cell->rep_id in owned_cell_to_rep_local for now but
            // into a global map below we will convert to indices.
            owned_cell_to_rep_local[cell_id] = rep_id;
          }

        // interfaces: read and append to global_interface_pairs
        const uint32_t nifaces = buf[pos++];
        for (uint32_t ii = 0; ii < nifaces; ++ii)
          {
            const unsigned int a = static_cast<unsigned int>(buf[pos++]);
            const unsigned int b = static_cast<unsigned int>(buf[pos++]);
            global_interface_pairs.emplace_back(a, b);
          }

        // mark this recv as processed
        recv_reqs[idx] = MPI_REQUEST_NULL;
        --n_pending;
      }

    // Ensure all sends are complete
    MPI_Waitall(n_ranks, send_reqs.data(), MPI_STATUSES_IGNORE);

    // At this point component_index_of_rep and component_reps are populated
    // but their insertion order depends on message arrival order which may
    // differ across ranks. Reindex deterministically by sorting rep ids so all
    // ranks assign the same indices for the same rep ids.

    std::vector<unsigned int> reps_sorted;
    reps_sorted.reserve(component_index_of_rep.size());
    for (const auto &kv : component_index_of_rep)
      reps_sorted.push_back(kv.first);
    std::sort(reps_sorted.begin(), reps_sorted.end());

    std::unordered_map<unsigned int, unsigned int> new_component_index_of_rep;
    new_component_index_of_rep.reserve(reps_sorted.size() * 2 + 4);
    std::vector<unsigned int> new_component_reps;
    std::vector<bool> new_touches_of_rep;

    new_component_reps.reserve(reps_sorted.size());
    new_touches_of_rep.reserve(reps_sorted.size());

    for (unsigned int i = 0; i < reps_sorted.size(); ++i)
      {
        const unsigned int rep_id = reps_sorted[i];
        const unsigned int old_idx = component_index_of_rep.at(rep_id);
        new_component_index_of_rep[rep_id] = i;
        new_component_reps.push_back(rep_id);
        new_touches_of_rep.push_back(touches_of_rep[old_idx]);
      }

    // Swap in the deterministically-ordered structures
    component_index_of_rep.swap(new_component_index_of_rep);
    component_reps.swap(new_component_reps);
    touches_of_rep.swap(new_touches_of_rep);

    // Build final owned_cell_to_rep map using the new deterministic rep indices
    std::unordered_map<unsigned int, unsigned int> owned_cell_to_rep;
    owned_cell_to_rep.reserve(owned_cell_to_rep_local.size() * 2 + 16);
    for (const auto &p : owned_cell_to_rep_local)
      {
        const unsigned int cell = p.first;
        const unsigned int rep_id = p.second;
        const unsigned int rep_idx = component_index_of_rep.at(rep_id);
        owned_cell_to_rep[cell] = rep_idx;
      }

    // Map our local interface pairs to component indices using owned_cell_to_rep
    std::vector<std::pair<unsigned int, unsigned int>> global_edges;
    // Build global edges from the collected global interface pairs
    global_edges.reserve(global_interface_pairs.size());
    for (const auto &itf : global_interface_pairs)
      {
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

    // Sort and deduplicate edges
    std::sort(global_edges.begin(), global_edges.end());
    auto new_end = std::unique(global_edges.begin(), global_edges.end());
    global_edges.erase(new_end, global_edges.end());

    std::cout << "Gather locals: " << timer.seconds() << std::endl;
    timer.reset();

    DisjointSet global_dsu(component_reps.size());

    // Unite according to deduplicated global edges
    for (const auto &e : global_edges)
      global_dsu.unite(e.first, e.second);

  std::cout << "Global union-find: " << timer.seconds() << std::endl;
    timer.reset();

    // --------------------------------------------------------------------------
    // Step 6: compact global components and fill local cell lists
    // --------------------------------------------------------------------------
    std::vector<ComponentInfo<dim, spacedim>>               result;
    std::unordered_map<unsigned int, unsigned int> compact_id_of_root;

    for (unsigned int i = 0; i < global_dsu.size(); ++i)
      {
        const unsigned int root = global_dsu.find(i);

        auto it = compact_id_of_root.find(root);
        if (it == compact_id_of_root.end())
          {
            const unsigned int cid = result.size();
            compact_id_of_root[root] = cid;
            result.emplace_back();
            it = compact_id_of_root.find(root);
          }

        const unsigned int cid = it->second;
        result[cid].touches_target_boundary =
          result[cid].touches_target_boundary || touches_of_rep[i];
      }

    for (const Cell &cell : dof_handler.active_cell_iterators())
      if (cell->is_locally_owned() && cell->active_fe_index() == target_fe_index)
        {
          const unsigned int cell_id = cell->active_cell_index();
          const auto it_rep = local_rep_of_owned_cell.find(cell_id);
          if (it_rep == local_rep_of_owned_cell.end())
            continue;

          const unsigned int rep_id = it_rep->second;

          const unsigned int root =
            global_dsu.find(component_index_of_rep.at(rep_id));
          const unsigned int cid = compact_id_of_root.at(root);

          result[cid].locally_owned_cells.push_back(cell);
        }
  std::cout << "Compact global: " << timer.seconds() << std::endl;
    timer.reset();

    return result;
  }

} // namespace DistributedFEIndexComponentsUF
}

#endif
