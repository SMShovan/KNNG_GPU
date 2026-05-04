# NUMA awareness in `knng_cpu`

Phase 4's parallel CPU builders run on workloads where the
dataset is large enough that *which physical memory the buffer
sits on* matters. On a Linux multi-socket host (e.g. 2× EPYC,
4× Xeon), DRAM is partitioned into NUMA *nodes*: one node per
socket, each with its own controllers and bandwidth. A worker
thread reading a page that belongs to a remote node pays the
inter-socket interconnect latency on every L3 miss — typically
1.3–2.0× the local-DRAM latency. At SIFT1M scale (`n*d ≈ 128M`
floats == 512 MB), the *entire* dataset cannot fit in a single
socket's L3, so the read pattern is dominated by DRAM and the
NUMA layout becomes the bottleneck.

This page documents the project's NUMA story and the two pieces
of infrastructure Step 26 ships:

  1. `knng::cpu::first_touch` — a cross-platform helper that
     binds buffer pages to the workers that will later read
     them.
  2. `tools/run_bench_numa.sh` — a Linux-aware bench wrapper
     that wraps `numactl --interleave=all` when the host has it.

## Why first-touch is the right primitive

Linux uses a *first-touch* allocation policy by default: when
a `malloc`-ed page is first written, the kernel allocates it
on the NUMA node where the writing thread is running. A
single-threaded loader that fills `Dataset::data` puts every
page on whichever node the main thread happened to be on; an
8-thread parallel-for that later reads the dataset hits remote
DRAM 7/8ths of the time.

The fix is to *first-touch the dataset from the same parallel
schedule the algorithm uses*. The schedule that Step 24+ use is
`#pragma omp parallel for schedule(static)`, which divides
iterations into contiguous blocks. If the first-touch pass
walks the buffer with the same partition, each block's pages
bind to the worker that will later read them — and the bench's
strong-scaling number stops being dominated by remote-DRAM
stalls.

`knng::cpu::first_touch` implements exactly this:

```cpp
namespace knng::cpu {
void first_touch(float* data, std::size_t n_elements,
                 int num_threads = 0) noexcept;
}
```

It walks the buffer in a `#pragma omp parallel for
schedule(static)` loop, writing one float per OS page (page
size queried at runtime via `sysconf(_SC_PAGESIZE)`). The
write is `data[i] = data[i]` so contents are preserved. Cost:
~0.3 s on SIFT1M on a quiescent Linux server, dominated by
the page-fault rate.

The convention for every parallel CPU builder, from Step 24
onwards: after populating the dataset (loader, synthetic
generator, anything that wrote to `ds.data`), call
`knng::cpu::first_touch(ds.data_ptr(), ds.size(), num_threads)`
*before* the first parallel read. The norms vector and the
output `Knng` adjacency are similar — large, parallel-read,
warrant first-touching.

## What about macOS / Apple Silicon?

Apple Silicon SoCs have a *single* unified memory pool — there
is no remote-DRAM penalty to avoid. `first_touch` runs on
macOS too (the function works identically), but the only side
effect is a cache-warm pass; pages are not redistributed.
Calling `is_numa_relevant_platform()` returns `false` on
macOS, `true` on Linux. The flag is informational; the
function does not need to be guarded.

## `tools/run_bench_numa.sh`

When `numactl` is available on the host, the wrapper invokes:

```sh
numactl --interleave=all build/bin/bench_brute_force [args...]
```

The `--interleave=all` policy spreads pages across every NUMA
node round-robin. It is the *baseline* the bench reports — a
NUMA-blind run on a multi-socket host without this would lie
about strong-scaling efficiency by reporting numbers
dominated by remote-DRAM stalls on whichever node the bench
process happened to land on.

On macOS or any host without `numactl`, the wrapper falls
through to the plain bench invocation and prints a one-line
note to stderr.

Note that `--interleave=all` is the *spread-everything-evenly*
fix; the *correct-locality* fix is `first_touch` from inside
the algorithm. The wrapper exists as the simpler-but-coarser
guarantee for ad-hoc bench runs that have not invoked
`first_touch` themselves.

## Open questions

- **libnuma probe.** `is_numa_relevant_platform()` currently
  returns `true` on every Linux build. A future refinement
  will swap the constant for `numa_available() &&
  numa_max_node() > 0` so a single-socket Linux host (a
  developer laptop, a CI runner) reports `false`. The cost
  is one new optional dependency (libnuma); the benefit is
  more honest scaling-writeup numbers.
- **Per-builder first-touch wiring.** Step 26 ships the
  helper; integrating it into the bench harness's setup phase
  lands in Step 29's scaling writeup, where the
  `first_touch`-on-vs-off comparison is the artefact.
- **`numactl --membind=N` for single-socket runs.** When a
  developer wants to stress-test a single socket of a
  multi-socket host, `--membind=N` pins pages to node `N`
  exclusively. The wrapper does not expose this knob; a
  future revision can grow a `--membind=N` flag if the
  use case actually arises.
