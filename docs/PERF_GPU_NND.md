# GPU NN-Descent: Recall vs Time Pareto Plot

> **Status:** template — measurement cells marked `[TODO]` pending
> GPU-cluster runs.  Expected values are derived from the CPU NN-Descent
> results (Phase 5 Step 36) scaled by GPU speedup factors from Phase 8.

This document records the recall@k vs build-time Pareto curve for the
GPU NN-Descent builder (Phase 9) and compares it against:
1. CPU NN-Descent (Phase 5, Step 36 — OpenMP parallel)
2. GPU brute-force KNN (Phase 8, Step 56 — cuBLAS GEMM)

---

## Datasets & hardware

| Dataset       | n       | d   | k  |
|---------------|---------|-----|----|
| SIFT-small    | 10K     | 128 | 10 |
| SIFT-1M       | 1M      | 128 | 10 |
| GIST-1M       | 1M      | 960 | 10 |
| Fashion-MNIST | 60K     | 784 | 10 |

| System          | GPU           | CPU                    |
|-----------------|---------------|------------------------|
| Dev cluster     | NVIDIA V100   | Intel Xeon (32 cores)  |
| Target system   | AMD MI350A    | same host cores        |

---

## Capture commands

```bash
# Build with GPU enabled:
cmake -S . -B build-gpu -DKNNG_BACKEND=CUDA -DKNNG_BUILD_BENCHMARKS=ON
cmake --build build-gpu -j$(nproc)

# CPU NN-Descent baseline (Phase 5):
./build-gpu/bin/bench_nn_descent \
  --benchmark_filter="NNDescent/SIFT1M" \
  --benchmark_out=perf/cpu_nnd_sift1m.json

# GPU brute-force baseline (Phase 8):
./build-gpu/bin/bench_gpu_brute_force \
  --benchmark_filter="GpuBruteForce/cuBLAS_SIFT1M" \
  --benchmark_out=perf/gpu_bf_sift1m.json

# GPU NN-Descent (Phase 9):
./build-gpu/bin/bench_gpu_nn_descent \
  --benchmark_filter="GpuNND/SIFT1M" \
  --benchmark_out=perf/gpu_nnd_sift1m.json
```

---

## Pareto table: SIFT-1M (n=1M, d=128, k=10)

| Algorithm                      | Build time | Recall@10 | Notes                       |
|-------------------------------|------------|-----------|------------------------------|
| CPU brute-force (Step 10)      | [TODO]     | 1.000     | 1-thread reference           |
| CPU OMP brute-force (Step 28)  | [TODO]     | 1.000     | 32 threads + BLAS            |
| CPU NN-Descent rho=1.0 (S36)   | [TODO]     | [TODO]    | 50 iters, delta=0.001        |
| CPU NN-Descent rho=0.5 (S36)   | [TODO]     | [TODO]    | faster, lower recall         |
| GPU brute-force cuBLAS (S56)   | [TODO]     | 1.000     | exact                        |
| GPU NND naive (S64)            | [TODO]     | [TODO]    | 50 iters, delta=0.001        |
| GPU NND + spinlock (S66)       | [TODO]     | [TODO]    | same, parallel update        |
| GPU NND + sampling rho=0.5 (S68)| [TODO]    | [TODO]    | tradeoff: faster, lower R@10 |
| GPU NND fp16 (S70)             | [TODO]     | [TODO]    | halved distance memory       |

---

## Expected speedup analysis

### GPU NN-Descent vs CPU NN-Descent

The local-join kernel (Step 64) parallelises the per-point pair enumeration:
- CPU: outer loop serial over n points, inner k² pairs per point → O(n k²) serial.
- GPU: one block per point, 64 threads split the k² pairs → 64× parallel inner loop.

For SIFT-1M with k=10 (100 pairs per point, 1M blocks):
- Expected GPU speedup over serial CPU: ~20–60× (limited by global memory latency).
- Against CPU OpenMP (32 threads): expected 2–8× advantage.

### GPU brute-force vs GPU NN-Descent

GPU brute-force (exact, O(n² d) work) is slower than GPU NN-Descent
(approximate, O(n k² iter) work with iter ≈ 10–20) for large n:

| n    | GPU BF time (estimated) | GPU NND time (estimated) | NND recall@10 |
|------|------------------------|--------------------------|---------------|
| 10K  | [TODO]                 | [TODO]                   | [TODO]        |
| 1M   | [TODO]                 | [TODO]                   | [TODO]        |

The crossover point (where NND is faster) is expected at n ≈ 50K for k=10.

---

## Recall vs time Pareto curve

```
Recall@10
  1.00 ┤ GPU BF ●                     CPU BF ●
  0.99 ┤
  0.98 ┤                                 CPU NND rho=1.0 ○
  0.97 ┤         GPU NND fp16 ○
  0.96 ┤   GPU NND rho=1.0 ○
  0.95 ┤
  0.90 ┤   GPU NND rho=0.5 ○  CPU NND rho=0.5 ○
  0.85 ┤
       └─────────────────────────────────────────── Build time (s)
        0.1      1       10      100    1000
```

*(Schematic — actual positions to be filled from cluster measurements.)*

---

## Recall impact of fp16 distances (Step 70)

Expected recall impact on SIFT-1M: < 0.01% degradation.  Mechanism:
fp16 rounding introduces ≤0.1% relative distance error; for sorted
k-neighbor lists, rank swaps occur only at near-equal distances.  The
probability of a rank swap ≈ `k × ε / d_gap` where `d_gap` is the
typical gap between the k-th and (k+1)-th neighbor — for SIFT-1M this
gap is ~5% of the k-th distance, so `ε ≈ 0.001` causes swap probability
≈ `10 × 0.001 / 0.05 ≈ 0.2%` of points — on 1M points that is ~2K
affected points, each with at most 1 rank swap → recall impact < 0.002.

---

## Key Nsight Compute metrics for local_join_kernel

| Metric                      | Expected value  |
|-----------------------------|-----------------|
| Achieved occupancy          | [TODO] %        |
| Shared memory per block     | 2k + 12 bytes   |
| Warp divergence             | [TODO] %        |
| Global load coalescing      | [TODO] %        |
| Spinlock contention cycles  | [TODO] %        |

Primary bottleneck prediction: global memory latency for distance
computations (d=128 floats per pair × 2 pairs per thread).
Optimization deferred to Phase 12: load reference features into shared
memory before the pair-enumeration loop.

---

## Files produced by cluster runs

| Artefact                       | Location                          |
|--------------------------------|-----------------------------------|
| CPU NND benchmark JSON         | `perf/cpu_nnd_*.json`             |
| GPU NND benchmark JSON         | `perf/gpu_nnd_*.json`             |
| Nsight Compute report          | `perf/local_join_kernel.ncu-rep`  |
| Pareto plot (Python)           | `tools/plot_bench.py`             |
| This document                  | `docs/PERF_GPU_NND.md`            |

*(The `perf/` directory is `.gitignore`d — only this markdown is committed.)*

---

## Next phase

Phase 10 — CAGRA-style graph refinement (Steps 72–79):
- Fixed out-degree enforcement
- Rank-based reordering (Ootomo et al. 2023 CAGRA key insight)
- Reverse edge merging
- Detour edge pruning
- Strong-component merging
