/// @file pipeline.cu
/// @brief GpuTriplePipeline — three-stream H2D/compute/D2H overlap.
///
/// ## Why triple-buffering?
///
/// A single-stream pipeline processes one chunk at a time:
///   Time: |─H2D─|─kernel─|─D2H─|─H2D─|─kernel─|─D2H─|
/// GPU and PCIe are idle 2/3 of the time.
///
/// Triple-buffering uses three independent streams and three device buffer
/// slots (A, B, C) so all three stages run concurrently:
///
///   Stream 0 (fill):    ──H2D(A)──H2D(B)──H2D(C)──H2D(A)──
///   Stream 1 (compute): ────────kernel(A)──kernel(B)──kernel(C)──
///   Stream 2 (drain):   ──────────────────D2H(A)────D2H(B)──
///
/// The PCIe bus is fully utilized (both directions overlap via separate
/// streams), and the SM executes continuously between H2D completions.
///
/// ## CUDA event synchronization
///
/// CUDA streams run independently by default — no ordering guarantee.
/// `cudaEventRecord` stamps a lightweight event in a stream's work queue.
/// `cudaStreamWaitEvent` injects a dependency: the waiting stream will not
/// advance past that point until the event fires.
///
/// For slot s:
///   1. fill_stream:    [H2D(s)]  → signal_fill_done(s)
///   2. compute_stream: wait_fill_done(s) → [kernel(s)] → signal_compute_done(s)
///   3. drain_stream:   wait_compute_done(s) → [D2H(s)]
///
/// `cudaEventDisableTiming` creates zero-overhead events (no timestamp
/// recording) — perfect for synchronization-only use cases.
///
/// ## Buffer rotation
///
/// Slots rotate: A→compute while B→fill, then B→compute while C→fill, etc.
/// The host scheduler drives the rotation; the GPU events enforce the
/// ordering without any host-side blocking.

#include <knng/gpu/pipeline.hpp>
#include <knng/gpu/backend.hpp>

#if defined(KNNG_HAVE_CUDA)

#include <cuda_runtime.h>
#include <stdexcept>

namespace knng::gpu {

// ---------------------------------------------------------------------------
// Pimpl: owns all CUDA handles for the three-stream pipeline.
// ---------------------------------------------------------------------------

struct GpuTriplePipeline::Impl {
    // Three device buffer slots — one per pipeline stage in flight.
    float* d_bufs[3] = {nullptr, nullptr, nullptr};

    // Stream 0: H2D (host-to-device) transfers — "fill"
    // Stream 1: GPU kernel computation — "compute"
    // Stream 2: D2H (device-to-host) transfers — "drain"
    cudaStream_t streams[3] = {};

    // fill_done[s]    fires when H2D for slot s is complete.
    // compute_done[s] fires when the kernel for slot s is complete.
    //
    // cudaEventDisableTiming avoids the ~2 µs overhead that timer events
    // incur on every record/query cycle.  For synchronization-only events
    // (not used with cudaEventElapsedTime) this flag is always correct.
    cudaEvent_t fill_done   [3] = {};
    cudaEvent_t compute_done[3] = {};

    std::size_t chunk_bytes = 0;
};

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

GpuTriplePipeline::GpuTriplePipeline(std::size_t chunk_bytes)
    : impl_(new Impl{})
{
    impl_->chunk_bytes = chunk_bytes;

    for (int s = 0; s < 3; ++s) {
        // Allocate one device buffer slot of chunk_bytes each.
        GPU_CHECK(cudaMalloc(&impl_->d_bufs[s], chunk_bytes));

        // Non-blocking streams: do not synchronize with the default stream
        // (stream 0).  This lets all three run truly concurrently without
        // accidental serialization at cudaStreamSynchronize(0).
        GPU_CHECK(cudaStreamCreateWithFlags(
            &impl_->streams[s], cudaStreamNonBlocking));

        // DisableTiming saves ~2 µs per record on Ampere+ hardware.
        const unsigned flags = cudaEventDisableTiming;
        GPU_CHECK(cudaEventCreateWithFlags(&impl_->fill_done   [s], flags));
        GPU_CHECK(cudaEventCreateWithFlags(&impl_->compute_done[s], flags));
    }
}

// ---------------------------------------------------------------------------
// Destructor — destroy in reverse creation order (events before streams)
// ---------------------------------------------------------------------------

GpuTriplePipeline::~GpuTriplePipeline() {
    if (!impl_) return;
    for (int s = 0; s < 3; ++s) {
        if (impl_->fill_done   [s]) cudaEventDestroy(impl_->fill_done   [s]);
        if (impl_->compute_done[s]) cudaEventDestroy(impl_->compute_done[s]);
        if (impl_->streams     [s]) cudaStreamDestroy(impl_->streams     [s]);
        if (impl_->d_bufs      [s]) cudaFree(impl_->d_bufs[s]);
    }
    delete impl_;
    impl_ = nullptr;
}

// ---------------------------------------------------------------------------
// Accessors
// ---------------------------------------------------------------------------

float* GpuTriplePipeline::device_ptr(BufSlot s) {
    return impl_->d_bufs[static_cast<unsigned>(s)];
}

cudaStream_t GpuTriplePipeline::fill_stream()    const noexcept
    { return impl_->streams[0]; }

cudaStream_t GpuTriplePipeline::compute_stream() const noexcept
    { return impl_->streams[1]; }

cudaStream_t GpuTriplePipeline::drain_stream()   const noexcept
    { return impl_->streams[2]; }

// ---------------------------------------------------------------------------
// Synchronization: signal/wait pairs per stage.
//
// The naming follows a producer/consumer model:
//   "signal" = producer records an event in its stream (non-blocking).
//   "wait"   = consumer injects a dependency on that event in its stream.
//
// cudaStreamWaitEvent is NON-BLOCKING on the host: it inserts an
// ordering constraint into the GPU's work queue without stopping the CPU.
// ---------------------------------------------------------------------------

/// @brief Record "H2D for slot s is done" in the fill stream.
void GpuTriplePipeline::signal_fill_done(BufSlot s) {
    GPU_CHECK(cudaEventRecord(
        impl_->fill_done[static_cast<unsigned>(s)],
        impl_->streams[0]));
}

/// @brief Block the compute stream until H2D for slot s has finished.
void GpuTriplePipeline::wait_fill_done(BufSlot s) {
    // Flag 0 = reserved for future use; always pass 0.
    GPU_CHECK(cudaStreamWaitEvent(
        impl_->streams[1],
        impl_->fill_done[static_cast<unsigned>(s)], 0));
}

/// @brief Record "kernel for slot s is done" in the compute stream.
void GpuTriplePipeline::signal_compute_done(BufSlot s) {
    GPU_CHECK(cudaEventRecord(
        impl_->compute_done[static_cast<unsigned>(s)],
        impl_->streams[1]));
}

/// @brief Block the drain stream until the kernel for slot s has finished.
void GpuTriplePipeline::wait_compute_done(BufSlot s) {
    GPU_CHECK(cudaStreamWaitEvent(
        impl_->streams[2],
        impl_->compute_done[static_cast<unsigned>(s)], 0));
}

} // namespace knng::gpu

#endif // KNNG_HAVE_CUDA
