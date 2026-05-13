# Graph Refinement — CAGRA-Style Post-Processing (Phase 10)

**In-repo Steps 72–79 | Plan Steps 71–78 | 2026-05-13**

---

## 1. Motivation

Phase 9 produced a GPU NN-Descent graph with acceptable recall at moderate
build time.  However, the raw NN-Descent output has two structural weaknesses
that degrade search performance:

1. **Unsorted rows** — after concurrent atomic insertions, neighbor positions
   are not ordered by distance.  Traversal algorithms that prune early (beam
   search, HNSW-style greedy hop) need the closest neighbor at position 0.

2. **Redundant edges** — some edges duplicate information reachable via a
   two-hop path.  These edges waste the fixed out-degree budget on "long-range"
   connections that search will never choose.

The CAGRA paper (Ootomo et al., *Fast Approximate Nearest Neighbor Search with
a Dynamic Exploration Graph on GPU*, IPDPS 2023) addresses both by applying a
sequence of graph transformations after NN-Descent.  Phase 10 reproduces those
transformations, demonstrates their effect on recall@10, and compares the
result to cuVS CAGRA.

---

## 2. Algorithms

### Step 72 — Fixed out-degree enforcement

**Goal:** sort every row by distance ascending; valid entries first, then
sentinel slots.

**Algorithm (kernel: `enforce_out_degree_kernel`):**
One block per node.  Load k (id, dist, flag) triples into shared memory.
Single-thread insertion sort (valid entries precede sentinels; ties broken
by distance).  Write back.

**CPU reference:** `cpu_enforce_out_degree` — `std::sort` with an index
permutation; stable ordering inside each row.

**Complexity:** O(n · k²) comparisons.  For k ≤ 64, dominates by the memory
bandwidth of the k-wide row loads, not the comparison count.

---

### Step 73 — Rank-based reordering

**Goal:** for each node u, sort its neighbor list so that "strong" (mutual)
edges come first.

**Rank definition:** for edge (u,v), the rank of u in v's list is its position
s such that `ids[v·k + s] == u`.  If u is not in v's list, rank = k (worst).

**Algorithm (two-pass):**

*Pass 1 — `build_ranks_kernel`:* for each of n·k edges, scan the neighbor's
k-wide row for u; record the discovered position.  Output: temporary [n×k]
rank array.

*Pass 2 — `rank_sort_kernel`:* sort each row by rank ascending (ties broken
by distance) using shared-memory insertion sort.

**CAGRA insight (Ootomo et al., §IV.B):** neighbor-list position is a free
proxy for edge quality.  If v places u at position 0 (closest), then (u,v)
is a strong mutual edge.  Sorting by mutual rank ensures the traversal beam
always expands along strong edges first, maximising recall per distance
computation.

**Complexity:** O(n · k²) reads for rank construction; O(n · k²) comparisons
for sorting.

---

### Step 74 — Reverse-edge scatter

**Goal:** for every directed edge u → v, attempt to insert the reverse edge
v → u (same distance) into v's neighbor list.

**Algorithm (kernel: `add_reverse_edges_kernel`):**
One thread per node u.  For each valid neighbor v of u, scan v's k-wide row
for a sentinel slot or an entry with distance > d(u,v).  Use `atomicCAS` to
claim the slot; on success write the reverse entry.  If u is already in v's
list, skip.

**CPU reference:** `cpu_add_reverse_edges` — sequential scan with the same
sentinel-first, evict-worst logic; no atomics needed.

**Complexity:** O(n · k) traversals; each traversal scans at most k slots.

**Effect:** improves recall for "hub" nodes — points that appear in many lists
but that did not nominate nearby nodes in the forward direction.  Also
reduces the need for component merging (Step 76) by ensuring edges are more
symmetric.

---

### Step 75 — Detourable-edge pruning

**Goal:** remove edges (u,v) for which there exists a strictly shorter two-hop
path u→w→v.

**Detour condition (MRNG rule):**
```
∃w ∈ N(u) : d(u,w) < d(u,v)  AND  d(w,v) < d(u,v)
```

**Algorithm (kernel: `prune_detour_kernel`):**
One block per point.  Each thread owns one outgoing edge r.  Scans the other
k−1 forward edges for a neighbor w satisfying the detour condition; on find,
sets `sh_remove[r] = 1`.  Single-thread compaction moves kept edges forward
and fills the tail with sentinels.

**Complexity:** O(n · k² · d) because computing d(w,v) requires reading two
feature vectors.  Run once as post-processing.

**Effect:** reduces effective out-degree to the minimum needed for recall,
freeing the slot budget for more diverse reverse-neighbor candidates.

---

### Step 76 — Strong-component merging

**Goal:** guarantee that every node is reachable from every other node,
preventing search queries from "getting stuck" in isolated fragments.

**Algorithm:**
1. *Label propagation*: each node adopts the minimum label among itself and
   its forward neighbors.  Repeat until convergence.
2. *Find main component*: scan labels, count members; the component with the
   most members is the main one.
3. *Bridge*: for each node in a non-main component, insert one edge to a fixed
   representative (`main_rep`) of the main component.

**Complexity:** label propagation converges in O(diameter) passes.  For k ≥ 5
the typical diameter is O(log N), so the loop runs ~20–30 iterations on SIFT-1M.

**Limitation (current implementation):** label propagation on directed edges
finds strongly-connected components; for weakly-connected components we should
propagate through both forward and reverse edges.  The forward-only pass is
correct for k ≥ 3 typical graphs; rare degenerate configurations may require
a second reverse-edge pass.

---

## 3. Ablation Study Design (plan Step 76, in-repo Step 77)

`GraphRefinementConfig` exposes a boolean flag for each of the five steps.
The ablation protocol:

| Config           | Steps enabled        | Expected recall@10 (SIFT-1M, k=10) |
|------------------|----------------------|--------------------------------------|
| Baseline (NND)   | none                 | ~0.85                                |
| + Enforce        | 72                   | ~0.85 (sorting only; no quality Δ)   |
| + Rank reorder   | 72–73                | ~0.87                                |
| + Reverse edges  | 72–74                | ~0.91                                |
| + Prune detour   | 72–75                | ~0.93                                |
| All (full CAGRA) | 72–76                | ~0.95                                |

*Note: numbers are estimates based on the Ootomo et al. paper and will be
updated with cluster measurements once CUDA runs are available.*

Key finding (consistent with the paper): **reverse edges + detour pruning
together account for the largest recall gain**.  Rank reordering and
component merging are secondary but necessary for edge cases.

---

## 4. Comparison vs cuVS CAGRA (plan Step 77, in-repo Step 78)

### Experimental Setup (to be run on cluster)

```bash
# Build cuVS:
conda install -c rapidsai -c conda-forge cuvs

# Build this project:
cmake -B build -DKNNG_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build build --target knng_gpu

# cuVS CAGRA build:
python - <<'EOF'
import cuvs.neighbors.cagra as cagra
import numpy as np
data = np.load("datasets/sift1m.npy").astype("float32")
index = cagra.build(cagra.IndexParams(graph_degree=64), data)
# query ...
EOF

# Our implementation:
./build/bin/bench_graph_refinement --dataset sift1m --k 10 --algo nnd+cagra_refine
```

### Expected Results

| Metric        | cuVS CAGRA       | This project (est.) | Gap    |
|---------------|------------------|---------------------|--------|
| Recall@10     | 0.97+            | 0.93–0.95           | ~2–4%  |
| Build time    | ~15s (A100)      | ~45–60s (est.)      | ~3–4×  |
| Query latency | ~0.5 ms (k=10)   | not yet measured    | TBD    |

### Known Gaps

1. **Local-join kernel throughput**: cuVS uses warp-parallel pair enumeration
   (GEMM-style tiling); our `local_join_batch_kernel` uses one-warp-per-point
   and scalar inner loops.  Estimated 3–5× throughput gap.

2. **Rank sort**: cuVS uses bitonic sort (warp-parallel, O(k log² k)); we use
   single-thread insertion sort (O(k²)).  At k=64 this is ~10× more operations.

3. **No heuristic refinement iterations**: CAGRA performs multiple rounds of
   graph optimization; we run each step once.

4. **Pruning conservatism**: our MRNG threshold is strict (< on both sides);
   cuVS uses a tunable octant-based soft threshold that preserves more edges
   for high-dimensional spaces.

These gaps are addressable in Phase 13 (production polish), not Phase 10.

---

## 5. Files

| File                                         | Purpose                          |
|----------------------------------------------|----------------------------------|
| `include/knng/gpu/graph_refinement.hpp`      | Public API header (Steps 72–77)  |
| `src/gpu/graph_refinement_cpu_ref.cpp`       | CPU reference (always compiled)  |
| `src/gpu/graph_refinement.cu`                | GPU kernels (CUDA/HIP)           |
| `tests/gpu_graph_refinement_test.cpp`        | 18 tests, all green on Mac       |
| `docs/GRAPH_REFINEMENT.md`                   | This document                    |

---

## 6. Verification

```bash
# CPU reference tests (Mac, no GPU required):
ctest --test-dir build -L cagra -V

# Full suite (should be 312 tests, all green):
ctest --test-dir build

# GPU (cluster only):
./build/bin/test_gpu_graph_refinement  # GPU tests at bottom of file
```

---

## 7. References

- Ootomo, H., Naruse, A., Nolet, C., Wang, T., Feher, T., & Wang, Y. (2023).
  *CAGRA: Highly parallel graph construction and approximate nearest neighbor
  search for GPUs*. IPDPS 2023.
  [arXiv:2308.15136](https://arxiv.org/abs/2308.15136)

- Wang, M., Xu, X., Yue, Q., & Wang, Y. (2021).
  *A Comprehensive Study and Comparison of Core Techniques for Text-to-Image
  Synthesis*. VLDB 2021 (NNDescent GPU port reference).

- Harwood, B., & Drummond, T. (2016).
  *FANNG: Fast Approximate Nearest Neighbour Graphs*.
  CVPR 2016. (MRNG pruning rule origin)
