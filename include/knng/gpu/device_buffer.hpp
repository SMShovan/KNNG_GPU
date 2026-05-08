#pragma once

/// @file
/// @brief RAII wrapper for a typed device-side (or stub heap-side) buffer.
///
/// `DeviceBuffer<T>` owns a contiguous allocation of `T` on the GPU
/// device (or on the host heap when the CPU stub backend is active).
/// It follows move-only semantics — copying device memory requires an
/// explicit `copy_to_host` / `copy_from_host` round-trip, which is
/// intentionally verbose to make host↔device transfers visible at
/// the call site.
///
/// ### Invariants
/// - A default-constructed (or moved-from) buffer has `data() == nullptr`
///   and `size() == 0`.
/// - `resize(n)` frees the existing allocation and allocates a fresh
///   `n * sizeof(T)` block; existing contents are **not** preserved.
/// - All `GPU_MALLOC` / `GPU_FREE` calls go through `backend.hpp` so
///   the buffer is HIP-portable without source changes.
///
/// @tparam T  Element type. Must be trivially copyable (the CUDA runtime
///            only guarantees byte-for-byte transfer; constructors are
///            not called on device memory).

#include <knng/gpu/backend.hpp>

#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace knng::gpu {

template <typename T>
class DeviceBuffer
{
    static_assert(std::is_trivially_copyable_v<T>,
                  "DeviceBuffer<T> requires T to be trivially copyable");

public:
    // -----------------------------------------------------------------------
    // Construction / destruction
    // -----------------------------------------------------------------------

    /// @brief Default-construct an empty buffer (null pointer, zero size).
    DeviceBuffer() noexcept = default;

    /// @brief Allocate a buffer for @p n elements.
    ///
    /// @param n  Number of elements; must be > 0.
    /// @throws std::invalid_argument if n == 0.
    explicit DeviceBuffer(std::size_t n) { resize(n); }

    /// @brief Allocate and initialise from a host vector.
    ///
    /// Equivalent to `DeviceBuffer(v.size())` followed by `copy_from_host(v)`.
    explicit DeviceBuffer(const std::vector<T>& v)
    {
        resize(v.size());
        copy_from_host(v.data(), v.size());
    }

    ~DeviceBuffer() { free(); }

    // Non-copyable: device-to-device copies require explicit GPU_MEMCPY_D2D.
    DeviceBuffer(const DeviceBuffer&)            = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    // -----------------------------------------------------------------------
    // Move
    // -----------------------------------------------------------------------

    /// @brief Move-construct: steals the allocation from @p other.
    DeviceBuffer(DeviceBuffer&& other) noexcept
        : data_(other.data_), size_(other.size_)
    {
        other.data_ = nullptr;
        other.size_ = 0;
    }

    /// @brief Move-assign: frees the current allocation, then steals from @p other.
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept
    {
        if (this != &other) {
            free();
            data_       = other.data_;
            size_       = other.size_;
            other.data_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    // -----------------------------------------------------------------------
    // Capacity
    // -----------------------------------------------------------------------

    /// @brief Number of elements currently allocated.
    [[nodiscard]] std::size_t size() const noexcept { return size_; }

    /// @brief Size in bytes of the allocation.
    [[nodiscard]] std::size_t bytes() const noexcept { return size_ * sizeof(T); }

    /// @brief True iff the buffer holds no allocation.
    [[nodiscard]] bool empty() const noexcept { return size_ == 0; }

    // -----------------------------------------------------------------------
    // Access
    // -----------------------------------------------------------------------

    /// @brief Raw device (or stub-host) pointer.
    [[nodiscard]] T*       data()       noexcept { return data_; }
    /// @brief Raw device (or stub-host) const pointer.
    [[nodiscard]] const T* data() const noexcept { return data_; }

    // -----------------------------------------------------------------------
    // Resize
    // -----------------------------------------------------------------------

    /// @brief Free the existing allocation and allocate @p n elements.
    ///
    /// Existing contents are not preserved.
    ///
    /// @param n  New element count; must be > 0.
    /// @throws std::invalid_argument if n == 0.
    void resize(std::size_t n)
    {
        if (n == 0) {
            throw std::invalid_argument("DeviceBuffer::resize: n must be > 0");
        }
        free();
        GPU_MALLOC(data_, n * sizeof(T));
        size_ = n;
    }

    // -----------------------------------------------------------------------
    // Host ↔ device transfer
    // -----------------------------------------------------------------------

    /// @brief Copy @p n host elements pointed to by @p src into device memory.
    ///
    /// @pre `n <= size()`.
    void copy_from_host(const T* src, std::size_t n)
    {
        if (n > size_) {
            throw std::out_of_range("DeviceBuffer::copy_from_host: n > size()");
        }
        GPU_MEMCPY_H2D(data_, src, n * sizeof(T));
    }

    /// @brief Copy the entire buffer back to host vector @p dst (resized as needed).
    void copy_to_host(std::vector<T>& dst) const
    {
        dst.resize(size_);
        GPU_MEMCPY_D2H(dst.data(), data_, size_ * sizeof(T));
    }

    /// @brief Return device contents as a new host vector.
    [[nodiscard]] std::vector<T> to_host() const
    {
        std::vector<T> v;
        copy_to_host(v);
        return v;
    }

private:
    T*          data_ = nullptr;
    std::size_t size_ = 0;

    void free() noexcept
    {
        if (data_) {
            GPU_FREE(data_);
            data_ = nullptr;
            size_ = 0;
        }
    }
};

} // namespace knng::gpu
