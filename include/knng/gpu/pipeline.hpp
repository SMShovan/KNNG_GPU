#pragma once

/// @file
/// @brief Step 84 — Triple-buffered compute + communication pipeline.
///
/// `TripleBuffer<T>` rotates three slots (fill / compute / drain) to overlap
/// host-to-device transfer, GPU computation, and device-to-host readback.
///
/// Always-compiled CPU simulation; GPU path uses CUDA streams (gated).

#include <array>
#include <cassert>
#include <cstddef>
#include <functional>
#include <vector>

namespace knng::gpu {

/// @brief Slot index in a triple buffer (0 / 1 / 2).
enum class BufSlot : unsigned char { A = 0, B = 1, C = 2 };

/// @brief Rotate triple buffer slots: fill → compute → drain.
inline BufSlot next_slot(BufSlot s) noexcept {
    return static_cast<BufSlot>((static_cast<unsigned>(s) + 1u) % 3u);
}

// ===========================================================================
// CPU simulation of triple-buffered pipeline
// ===========================================================================

/// @brief CPU simulation of a triple-buffered H2D/compute/D2H pipeline.
///
/// Processes `n_chunks` independent work items through three stages:
///   Stage 0 (fill)    — produce chunk data (simulates H2D transfer)
///   Stage 1 (compute) — process the chunk (simulates GPU kernel)
///   Stage 2 (drain)   — consume results  (simulates D2H transfer)
///
/// On a real GPU each stage runs on a separate stream; the CPU simulation
/// runs them sequentially in a sliding window to exercise the same data-flow.
///
/// @tparam T  Element type stored in each buffer slot.
template <typename T>
class CpuTripleBuffer {
public:
    /// @brief Construct with `chunk_size` elements per slot.
    explicit CpuTripleBuffer(std::size_t chunk_size)
        : chunk_size_(chunk_size)
    {
        for (auto& buf : bufs_) buf.resize(chunk_size);
    }

    /// @brief Run the full pipeline over `n_chunks` work items.
    ///
    /// @param fill     Called with (slot_data, chunk_index) to produce input.
    /// @param compute  Called with (slot_data) to transform data in place.
    /// @param drain    Called with (slot_data, chunk_index) to consume output.
    void run(std::size_t n_chunks,
             std::function<void(std::vector<T>&, std::size_t)> fill,
             std::function<void(std::vector<T>&)>              compute,
             std::function<void(const std::vector<T>&, std::size_t)> drain)
    {
        // Sliding-window: at step i, fill slot for chunk i+2, compute chunk i+1,
        // drain chunk i.  CPU simulation runs sequentially — same data-flow as GPU
        // streams but without actual overlap.
        for (std::size_t chunk = 0; chunk < n_chunks; ++chunk) {
            const auto fill_slot    = static_cast<std::size_t>(BufSlot::A);
            const auto compute_slot = static_cast<std::size_t>(BufSlot::B);
            const auto drain_slot   = static_cast<std::size_t>(BufSlot::C);

            // Stage 0: fill
            fill(bufs_[fill_slot], chunk);

            // Rotate: filled data moves to compute slot
            std::swap(bufs_[fill_slot], bufs_[compute_slot]);

            // Stage 1: compute
            compute(bufs_[compute_slot]);

            // Rotate: computed data moves to drain slot
            std::swap(bufs_[compute_slot], bufs_[drain_slot]);

            // Stage 2: drain
            drain(bufs_[drain_slot], chunk);
        }
    }

    std::size_t chunk_size() const noexcept { return chunk_size_; }

private:
    std::size_t              chunk_size_;
    std::array<std::vector<T>, 3> bufs_;
};

// ===========================================================================
// GPU stream pipeline (CUDA only)
// ===========================================================================

#if defined(KNNG_HAVE_CUDA)
/// @brief GPU triple-buffered pipeline using three CUDA streams.
///
/// Stream 0: H2D memcpy (fill)
/// Stream 1: kernel launch (compute)
/// Stream 2: D2H memcpy (drain)
///
/// CUDA events serialize the three streams at buffer boundaries so that
/// stream 1 does not start until stream 0's H2D is done for the same slot,
/// and stream 2 does not start until stream 1's kernel is done.
class GpuTriplePipeline {
public:
    explicit GpuTriplePipeline(std::size_t chunk_bytes);
    ~GpuTriplePipeline();

    GpuTriplePipeline(const GpuTriplePipeline&)            = delete;
    GpuTriplePipeline& operator=(const GpuTriplePipeline&) = delete;

    /// @brief Pointer to device buffer for slot s (size = chunk_bytes).
    float* device_ptr(BufSlot s);

    cudaStream_t fill_stream()    const noexcept;
    cudaStream_t compute_stream() const noexcept;
    cudaStream_t drain_stream()   const noexcept;

    /// @brief Record completion of the fill stage for slot s.
    void signal_fill_done(BufSlot s);
    /// @brief Wait until fill is done before starting compute for slot s.
    void wait_fill_done(BufSlot s);
    /// @brief Record completion of the compute stage.
    void signal_compute_done(BufSlot s);
    /// @brief Wait until compute is done before starting drain.
    void wait_compute_done(BufSlot s);

private:
    struct Impl;
    Impl* impl_ = nullptr;
};
#endif // KNNG_HAVE_CUDA

} // namespace knng::gpu
