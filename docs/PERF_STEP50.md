# GPU Performance Baseline: `brute_force_block_kernel` (Step 50)

> **Status:** template — to be filled with real measurements when the
> branch is pushed to a GPU machine with Nsight Compute installed.
> Placeholders marked `[TODO: measure]` below.

This document records the Nsight Compute (`ncu`) profile of the
Step-50 `brute_force_block_kernel` on SIFT-small (10 K points, 128 D,
k = 10).  It establishes the Phase-7 GPU performance baseline and sets
the pattern for all later GPU profiling writeups (Steps 55, 58, 60, 70).

---

## How to capture

```bash
# On a GPU machine, from the repo root:
cmake -S . -B build-gpu -DKNNG_ENABLE_CUDA=ON -DKNNG_BUILD_BENCHMARKS=ON
cmake --build build-gpu -j$(nproc)

# Run the brute-force GPU benchmark under Nsight Compute:
ncu --set full \
    --section SpeedOfLight \
    --section MemoryWorkloadAnalysis \
    --section ComputeWorkloadAnalysis \
    --section Occupancy \
    --kernel-name brute_force_block_kernel \
    --output perf/step50_ncu_report \
    ./build-gpu/bin/bench_gpu_brute_force \
    --benchmark_filter="GpuBruteForce/SIFT_small_k10"

# View the report:
ncu-ui perf/step50_ncu_report.ncu-rep
```

---

## Roofline model

The roofline chart positions the kernel relative to the hardware ceilings:

| Axis          | Value                                   |
|---------------|-----------------------------------------|
| Peak FP32     | [TODO: measure] TFLOP/s                 |
| Peak mem BW   | [TODO: measure] GB/s (HBM)             |
| Arithmetic intensity | [TODO: measure] FLOP/byte        |
| Achieved perf | [TODO: measure] TFLOP/s                 |

**Expected:** The naive kernel is memory-bound (low arithmetic intensity
because each thread loads `2 × d` floats but only performs `2d` FP ops).
The roofline should sit well below the FP32 ceiling.

---

## Occupancy

| Metric                                | Value               |
|---------------------------------------|---------------------|
| Theoretical occupancy                 | [TODO: measure] %   |
| Achieved occupancy                    | [TODO: measure] %   |
| Registers per thread                  | [TODO: measure]     |
| Shared memory per block               | k × 8 + 8 bytes     |
| Blocks per SM                         | [TODO: measure]     |
| Limiting factor                       | [TODO: shared mem / registers / …] |

**Expected:** With `blockDim = 128` and k = 10 the shared memory is
`10 × 8 + 8 = 88 bytes` per block — tiny.  Occupancy should be limited
by register count rather than shared memory.

---

## Memory

| Metric                                | Value               |
|---------------------------------------|---------------------|
| L1 hit rate (instructions)            | [TODO: measure] %   |
| L2 hit rate                           | [TODO: measure] %   |
| Global load efficiency (coalescing)   | [TODO: measure] %   |
| Bytes transferred device↔host         | [TODO: measure] MB  |

**Expected:** Global loads are **uncoalesced** at this tier: thread `t`
and thread `t+1` both load from `d_pts[qi * d + dim]` but `qi` differs
between threads (strided by block size), so consecutive threads load
from addresses separated by `d * sizeof(float)` = 512 bytes.  L1 hit
rate will be low.  This is the primary inefficiency that Phase 8 Step 52
(shared-memory reference tiling) addresses.

---

## Kernel timeline

| Stage                           | Estimated fraction of runtime |
|---------------------------------|-------------------------------|
| Distance computation (FP MADs)  | [TODO: measure] %             |
| Shared-memory top-k insertion   | [TODO: measure] %             |
| Barrier / synchronisation       | [TODO: measure] %             |
| Final sort + global write       | [TODO: measure] %             |

---

## Optimization opportunities (deferred to Phase 8)

1. **Coalescing (Step 51 plan):** Transpose the reference matrix to
   column-major so that thread `t` and thread `t+1` load adjacent floats.
   Expected improvement: 4–8× reduction in L2 traffic.

2. **Shared-memory tiling (Step 52):** Load a tile of `TILE` references
   into shared memory; all threads scan the same tile.  Reuse factor = 1
   → block size.

3. **Warp-level top-k (Step 53):** Replace the `atomicCAS` spinlock with
   a warp-shuffle–based parallel reduction.  Expected: eliminates
   serialisation overhead for competitive candidates.

4. **cuBLAS GEMM (Step 55):** Express `-2 X Yᵀ + ||X||² + ||Y||²` as a
   `cublasSgemm` call.  Tensor cores handle the GEMM; a fused epilogue
   kernel adds the norms and writes the top-k.  Expected: approach
   hardware bandwidth ceiling for large n.

---

## Comparison table (to be filled as Phase 8 steps land)

| Kernel variant          | Step | Throughput (QPS) | Recall@10 | L2 traffic |
|-------------------------|------|------------------|-----------|------------|
| `brute_force_block`     | 50   | [TODO]           | 1.00      | [TODO]     |
| Coalesced layout        | 51   | [TODO]           | 1.00      | [TODO]     |
| Shared-mem tiling       | 52   | [TODO]           | 1.00      | [TODO]     |
| Warp top-k              | 53   | [TODO]           | 1.00      | [TODO]     |
| cuBLAS GEMM             | 55   | [TODO]           | 1.00      | [TODO]     |

> Recall is 1.00 for all brute-force variants (they are exact).

---

## Files produced

| Artefact                         | Location                         |
|----------------------------------|----------------------------------|
| Nsight Compute report            | `perf/step50_ncu_report.ncu-rep` |
| JSON benchmark output            | `perf/step50_bench.json`         |
| This document (template)         | `docs/PERF_STEP50.md`            |

*(The `perf/` directory is `.gitignore`d; only the markdown summary is committed.)*
