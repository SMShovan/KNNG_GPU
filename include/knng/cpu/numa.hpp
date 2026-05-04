#pragma once

/// @file
/// @brief Cross-platform first-touch helper for NUMA-aware buffers.
///
/// On a Linux multi-socket system, the kernel allocates each page
/// of a `malloc`-ed buffer on the NUMA node *where the page is
/// first written*. A buffer that is allocated on one socket but
/// later read by a worker on a different socket pays a remote-DRAM
/// latency (~1.5–2× the local-DRAM latency on EPYC, ~1.3× on Xeon)
/// every time the prefetcher pulls a fresh cache line. The fix is
/// to "first-touch" the buffer from the workers that will later
/// read it, using the *same* parallel schedule the algorithm uses,
/// so each page binds to the right node before the timed loop.
///
/// On macOS / Apple Silicon — a single-NUMA-domain SoC — there is
/// no remote-DRAM latency to avoid; `first_touch` here is a
/// (cheap) cache warm-up rather than a node-binding redistribution.
/// We keep the function on every platform so an algorithm written
/// against it ports to Linux without an `#ifdef`.
///
/// Convention: every parallel CPU algorithm that streams a large
/// buffer (the `Dataset::data` row-major float array, the norms
/// vector, any output `Knng` adjacency) calls `first_touch` on it
/// after population, with the same `num_threads` it will later use
/// for the timed loop. Step 25's per-thread scratch already lives
/// on the worker's stack and does not need first-touching.

#include <cstddef>

namespace knng::cpu {

/// Walk `n_elements` floats starting at `data`, writing every
/// page-sized stride in parallel. On Linux + OpenMP, the
/// `schedule(static)` partition matches the partition Step 24's
/// builders later use, so each page is first-touched by the
/// worker that will later read it.
///
/// @param data Pointer to the start of the buffer. Must point to
///        at least `n_elements` valid floats.
/// @param n_elements Number of float elements in the buffer.
/// @param num_threads If > 0, sets the OpenMP team size for the
///        first-touch pass; should equal the team size of the
///        timed loop that follows. If 0 (default), uses the
///        runtime's default.
///
/// The pass *re-writes* every touched float to itself — a
/// `data[i] = data[i]` no-op — so the buffer's contents are
/// preserved. Cost: one read + one write per page (~4 KB on
/// every supported platform), bounded by the OS page faulting
/// rate. For SIFT1M (`n*d == 128M floats == 512 MB`) the
/// first-touch pass takes ~0.3 s on a quiescent Linux server.
void first_touch(float* data, std::size_t n_elements,
                 int num_threads = 0) noexcept;

/// True iff the build links a platform with NUMA semantics worth
/// the first-touch effort. Apple Silicon SoCs report `false` here
/// (single domain); a Linux multi-socket host reports `true`. The
/// flag is informational — `first_touch` runs unconditionally on
/// every platform — but Step 29's scaling writeup uses it to
/// decide whether to run the `numactl --interleave=all` companion.
[[nodiscard]] bool is_numa_relevant_platform() noexcept;

} // namespace knng::cpu
