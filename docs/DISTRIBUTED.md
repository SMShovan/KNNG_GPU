# Distributed NN-Descent (Phase 6) — Architecture and Design Notes

## Overview

Phase 6 introduces distributed-memory parallelism to the KNNG construction
pipeline.  The goal is to understand MPI-based data partitioning, ring
communication, and the two headline optimisations from the
[NEO-DNND paper (Luo et al., 2021)][neodnnd] *before* GPUs are introduced
— so the distributed algorithm can be debugged on familiar CPU semantics.

---

## Communication topology

```
Node A                            Node B
┌─────────────────────┐          ┌─────────────────────┐
│  Rank 0  │  Rank 1  │   MPI    │  Rank 2  │  Rank 3  │
│  shard 0 │  shard 1 │◄────────►│  shard 2 │  shard 3 │
│  [SHM]───┤──[SHM]   │          │  [SHM]───┤──[SHM]   │
└─────────────────────┘          └─────────────────────┘
      intra_comm                       intra_comm
      (MPI-3 SHM)                     (MPI-3 SHM)
```

- **Intra-node:** Ranks on the same physical node share a `MPI_Win_allocate_shared`
  window (Step 43).  Feature-vector reads for intra-node neighbors are
  zero-copy pointer dereferences.
- **Inter-node:** Normal MPI point-to-point or collective operations (Steps 40–42).
  `MPI_Alltoallv` for the feature-request/response protocol.

---

## Point sharding (`ShardedDataset`, Step 39)

Each rank owns a contiguous slice of the `n`-point dataset.

```
compute_shard(global_n, num_ranks, rank)
  → {start, count}
  where start = rank * (global_n / num_ranks)
        count = global_n / num_ranks   (last rank: + global_n % num_ranks)
```

Memory per rank: `count × d × sizeof(float)` for the feature buffer.
Peak working memory: two feature buffers (`own shard` + ring buffer)
during the distributed brute-force ring shift (Step 40).

---

## Distributed brute-force (Step 40)

**Algorithm — ring shift:**

```
ring_buf = own_shard
ring_start = local_start
for step in 0 .. P-1:
    score(local_queries, ring_buf)          → update local top-k
    MPI_Sendrecv(ring_buf → right, ← left)  → advance ring one step
```

After `P` steps every rank has seen all `n` reference points.

| Metric            | Value                        |
|-------------------|------------------------------|
| Communication     | `P-1` rounds of `MPI_Sendrecv` |
| Bytes per round   | `≤ (n/P + remainder) × d × 4` |
| Peak memory       | `2 × (n/P) × d × 4` bytes     |
| Communication vs AllGather | Same total bytes; `O(n/P)` peak vs `O(n)` |

---

## Distributed NN-Descent (Steps 41–43)

### Ownership model

Point `p` is *owned* by the rank whose shard contains `p`.  The rank owns
`p`'s neighbor list; no other rank caches `p`'s feature vector unless it
fetches it during an iteration.

### Per-iteration protocol

```
┌──────────────────────────────────────────────────────────┐
│  1. Snapshot: build new[p] / old[p] for local points      │
│  2. build_remote_requests: collect remote IDs needed      │
│  3. dedup_requests (Step 42): remove duplicates          │
│  4. exchange_features (Alltoall × 3 rounds):             │
│       a. send counts    (MPI_Alltoall)                   │
│       b. send IDs       (MPI_Alltoallv)                  │
│       c. recv features  (MPI_Alltoallv)                  │
│  5. local_join_distributed: compute distances, insert    │
│  6. MPI_Allreduce(SUM) updates → convergence check       │
└──────────────────────────────────────────────────────────┘
```

### Optimisation 1 — Duplicate-request reduction (Step 42)

Without deduplication, a single popular global point `p` that appears in
`m` local neighbor lists generates `m` separate requests for `p`'s
feature vector.

After `dedup_requests`, each target rank receives at most one request per
unique ID from this rank.  The deduplication ratio is:

```
ratio = (raw_count - dedup_count) / raw_count
```

Typical values on real datasets:

| Phase                 | Ratio |
|-----------------------|-------|
| Iteration 1 (random)  | 0.0 – 0.2 |
| Mid-convergence       | 0.3 – 0.6 |
| Near-converged        | 0.6 – 0.9 |

Higher ratio means fewer bytes sent per iteration.

**Reference:** NEO-DNND §3.2 "Duplicate-Request Reduction."

### Optimisation 2 — Intra-node shared-memory replication (Step 43)

For ranks co-located on the same physical node, feature vectors can be
shared through a process-shared memory window (MPI-3 `MPI_Win_allocate_shared`)
rather than sent through MPI point-to-point buffers.

```
ShmRegion region(local_data, local_n, d, MPI_COMM_WORLD);
// All intra-node ranks can now read each other's data directly:
const float* row = region.read_remote_row(intra_r, row_i, d);
```

The intra-node communicator is obtained via `MPI_Comm_split_type` with
the `MPI_COMM_TYPE_SHARED` key.  When `intra_size() == 1`, the rank is
the sole occupant of its node and SHM provides no benefit (the window
covers only its own data).

**Reference:** NEO-DNND §3.3 "Intra-Node Replication."

---

## Performance model

Let:
- `P` = number of MPI ranks
- `n` = total points
- `d` = dimensionality
- `k` = neighbors per point
- `I` = iterations to convergence

| Step | Compute       | Communication                    |
|------|---------------|----------------------------------|
| 40   | O(n²d / P)    | O(n × d × P) total ring bytes   |
| 41   | O(n × k² × I / P) | O(n × d × I) alltoallv bytes |
| 42   | same as 41    | reduced by dedup ratio           |
| 43   | same as 42    | intra-node fetches become 0-copy |

---

## File inventory

| File | Purpose |
|------|---------|
| `include/knng/dist/mpi_env.hpp` | RAII MPI init/finalize |
| `include/knng/dist/sharded_dataset.hpp` | Point-sharded data container |
| `include/knng/dist/brute_force_mpi.hpp` | Ring-based exact KNN |
| `include/knng/dist/nn_descent_mpi.hpp` | Distributed NN-Descent driver |
| `include/knng/dist/request_dedup.hpp` | Deduplication primitive |
| `include/knng/dist/shm_replication.hpp` | MPI-3 SHM window |
| `src/dist/*.cpp` | Implementations of the above |
| `cmake/FindKnngMPI.cmake` | MPI discovery + `knng::mpi_iface` |

---

## Open questions (deferred to Phase 12)

1. **Overlap inter-node communication + local compute.**  The Step 41
   `MPI_Alltoallv` blocks until all feature responses arrive.  Step 94
   will overlap the MPI communication with local distance computation
   using `MPI_Isend`/`MPI_Irecv` + CUDA streams.

2. **Bandwidth-aware replication.**  A point requested by ≥ T ranks in
   the previous iteration could be proactively replicated to reduce
   inter-node traffic.  Threshold T is a hyperparameter.

3. **GPU-aware MPI.**  `MPIX_Query_cuda_support()` detection (Step 87)
   will allow direct GPU-to-GPU sends without staging through host
   pinned memory.

4. **Non-uniform shard sizes.**  When GPU memory constraints differ across
   nodes, the static balanced partitioning may leave some GPUs idle.
   Phase 12 will explore work-stealing rebalancing.

---

[neodnnd]: https://dl.acm.org/doi/10.1145/3448016.3452773
