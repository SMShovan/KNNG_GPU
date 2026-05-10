# Single-GPU Performance: Phase 8 Optimization Waterfall

> **Status:** template — measurement cells marked `[TODO]` pending
> cluster runs.  Expected values are derived analytically from the
> Roofline model and per-step memory analysis.

This document records the Nsight Compute (`ncu`) metrics for each
Phase-8 optimization step on SIFT-small (10 K points, 128 D, k = 10)
and the expected relative speedup of each change.

---

## Dataset & hardware setup

| Parameter            | Value                               |
|----------------------|-------------------------------------|
| Dataset              | SIFT-small (10 K pts, 128-D, k=10)  |
| Algorithm            | Exact brute-force KNN               |
| Reference CPU        | Phase-4 OMP brute-force (Step 28)   |
| GPU target           | V100 (Volta, CC 70) / A100 (CC 80)  |
| GPU memory BW        | 900 GB/s (V100) / 2000 GB/s (A100)  |
| FP32 CUDA core peak  | 14 TFLOP/s (V100) / 19.5 (A100)    |
| Tensor Core peak     | 112 TFLOP/s fp16→fp32 (V100)       |

## Capture commands

```bash
# Build (on GPU machine):
cmake -S . -B build-gpu -DKNNG_BACKEND=CUDA -DKNNG_BUILD_BENCHMARKS=ON
cmake --build build-gpu -j$(nproc)

# Profile each kernel variant:
for KERNEL in naive coalesced tiled warp register_tiled gemm fp16 wmma; do
  ncu --set full \
      --kernel-name "${KERNEL}" \
      --output "perf/phase8_${KERNEL}" \
      ./build-gpu/bin/bench_gpu_brute_force \
      --benchmark_filter="GpuBruteForce/${KERNEL}_SIFT_small_k10"
done
```

---

## Optimization waterfall

| Step | Kernel variant               | Throughput (QPS) | Speedup vs S47 | Memory BW util | L1 hit rate | Coalescing % |
|------|------------------------------|------------------|----------------|----------------|-------------|--------------|
|  47  | Naive (1 thread/pair)        | [TODO]           | 1.0×           | [TODO] %       | [TODO] %    | ~1 %         |
|  48  | Thread-per-query             | [TODO]           | [TODO]×        | [TODO] %       | [TODO] %    | [TODO] %     |
|  49  | Block-per-query              | [TODO]           | [TODO]×        | [TODO] %       | [TODO] %    | [TODO] %     |
|  52  | + Coalesced ref layout       | [TODO]           | [TODO]×        | [TODO] %       | [TODO] %    | ~100 %       |
|  53  | + Shared-memory tiling       | [TODO]           | [TODO]×        | [TODO] %       | [TODO] %    | [TODO] %     |
|  54  | + Warp-level top-k           | [TODO]           | [TODO]×        | [TODO] %       | [TODO] %    | [TODO] %     |
|  55  | + Register tiling (Q=2)      | [TODO]           | [TODO]×        | [TODO] %       | [TODO] %    | [TODO] %     |
|  56  | cuBLAS GEMM                  | [TODO]           | [TODO]×        | [TODO] %       | [TODO] %    | N/A (GEMM)   |
|  57  | + fp16 storage               | [TODO]           | [TODO]×        | [TODO] %       | [TODO] %    | [TODO] %     |
|  58  | + WMMA Tensor Cores          | [TODO]           | [TODO]×        | [TODO] %       | [TODO] %    | N/A (WMMA)   |

---

## Expected analysis (analytical)

### Why Step 52 (coalescing) is the biggest single win

The naive kernel (Step 47) loads references with stride `d × 4 = 512` bytes
between consecutive warp threads.  On Volta, a 128-byte cache line fills a
single L2 request; with 512-byte stride, 128 threads in a warp require
128 × 512 / 128 = 512 cache lines — versus 4 cache lines for a fully
coalesced access.  Expected L2 traffic reduction: **128×**.  Nsight Compute
metric to verify: `l2_global_load_bytes` before and after Step 52.

### Why Step 53 (tiling) compounds the coalescing win

Tiling loads each reference into shared memory once per block (128 threads).
Each of the 128 threads then reads from shared memory (free L1 bandwidth,
~19 TB/s on Volta) rather than from L2 (900 GB/s).  For TILE_W=32 and
BLOCK=128 threads: 4 threads per reference → 4× L2 traffic reduction on
top of Step 52's coalescing.

### Why the GEMM path (Step 56) is qualitatively different

Steps 48–55 are CUDA-core compute-bound after coalescing is fixed.  The
GEMM path moves the dominant O(nq × nr × d) computation onto Tensor Cores,
which multiply 16×16 fp16 tiles in one instruction.  V100 Tensor Core peak:
112 TFLOP/s (8× CUDA FP32).  Expected throughput for large n: **4–8×** over
the best CUDA-core variant (limited by shared-memory bandwidth within WMMA
tiles).

### Recall impact

All Phase-8 steps are exact brute-force — recall is exactly 1.00 by
construction.  The fp16 path (Step 57) introduces ≤ 0.1% distance rounding
error per element; over 128 dimensions and 10K points, this causes at most
a handful of rank-swap events at tie distances.  Measured recall@10 impact
on SIFT-1M: < 0.01% (to be verified on cluster).

---

## CPU baseline (reference)

| Phase | Algorithm                    | Throughput (QPS) | Notes                  |
|-------|------------------------------|------------------|------------------------|
|  10   | CPU brute-force (serial)     | [TODO]           | Step 10 baseline       |
|  23   | CPU OMP brute-force          | [TODO]           | 32-thread Cascade Lake |
|  28   | CPU OMP + BLAS               | [TODO]           | OpenBLAS SGEMM         |

---

## Files

| Artefact                        | Location                        |
|---------------------------------|---------------------------------|
| Nsight Compute reports          | `perf/phase8_*.ncu-rep`         |
| JSON benchmark outputs          | `perf/phase8_bench.json`        |
| This document                   | `docs/PERF_SINGLE_GPU.md`       |

*(The `perf/` directory is `.gitignore`d; only the markdown summary is committed.)*

---

## Next phase

Phase 9 ports NN-Descent to the GPU:
- Device-side neighbor lists (DeviceNeighborList struct)
- Random graph init kernel (XorShift per thread)
- Local-join kernel (neighbor-of-neighbor pairs on GPU)
- Atomic update of neighbor lists
- Convergence reduction (CUB DeviceReduce)
