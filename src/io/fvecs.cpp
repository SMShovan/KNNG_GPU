/// @file
/// @brief Implementation of the `.fvecs` / `.ivecs` / `.bvecs` loaders.
///
/// All three loaders share the same shape: open the file, `mmap` it
/// read-only-private, recover `n` from the file size, validate every
/// per-record dim prefix matches the first, and copy the elements
/// into the row-major destination buffer. The shared logic lives in
/// the anonymous-namespace `read_vecs_records` helper template; the
/// public entry points are thin wrappers that pin the element type
/// and choose a destination container.
///
/// Endianness: the file format documents little-endian integers, and
/// every supported development platform (macOS / Linux on x86_64,
/// macOS / Linux on arm64) is little-endian. We do not byte-swap. A
/// future big-endian port would add a swap step in `read_vecs_records`
/// — gated on `std::endian::native` — but that is not on the current
/// roadmap.

#include "knng/io/fvecs.hpp"

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace knng::io {

namespace {

/// RAII wrapper for an `mmap`-ed read-only file. Owns both the file
/// descriptor and the mapping; both are released on destruction. The
/// type is move-only — copying a mapping would require either a
/// reference-count or a second `mmap` of the same fd, neither of
/// which we want.
class MmapFile {
public:
    explicit MmapFile(const std::filesystem::path& path)
    {
        fd_ = ::open(path.c_str(), O_RDONLY);
        if (fd_ < 0) {
            throw std::runtime_error(
                "knng::io: open() failed for '" + path.string()
                + "': " + std::strerror(errno));
        }
        struct ::stat st {};
        if (::fstat(fd_, &st) != 0) {
            const int saved = errno;
            ::close(fd_);
            fd_ = -1;
            throw std::runtime_error(
                "knng::io: fstat() failed for '" + path.string()
                + "': " + std::strerror(saved));
        }
        if (st.st_size <= 0) {
            ::close(fd_);
            fd_ = -1;
            throw std::runtime_error(
                "knng::io: file is empty: '" + path.string() + "'");
        }
        size_ = static_cast<std::size_t>(st.st_size);
        void* p = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
        if (p == MAP_FAILED) {
            const int saved = errno;
            ::close(fd_);
            fd_ = -1;
            throw std::runtime_error(
                "knng::io: mmap() failed for '" + path.string()
                + "': " + std::strerror(saved));
        }
        addr_ = p;
    }

    ~MmapFile()
    {
        if (addr_ != nullptr) {
            ::munmap(addr_, size_);
        }
        if (fd_ >= 0) {
            ::close(fd_);
        }
    }

    MmapFile(const MmapFile&) = delete;
    MmapFile& operator=(const MmapFile&) = delete;
    MmapFile(MmapFile&&) = delete;
    MmapFile& operator=(MmapFile&&) = delete;

    [[nodiscard]] const std::byte* data() const noexcept
    {
        return static_cast<const std::byte*>(addr_);
    }
    [[nodiscard]] std::size_t size() const noexcept { return size_; }

private:
    int         fd_{-1};
    void*       addr_{nullptr};
    std::size_t size_{0};
};

/// Read a *vecs file into a contiguous row-major buffer of `Element`s.
///
/// `out` is resized to `n * d` and filled with the file's records'
/// element payloads (no per-record dim prefix). `n` and `d` are
/// recovered from the file size and the first record's dim prefix
/// respectively. Inconsistent prefixes or non-integral record counts
/// throw `std::runtime_error` so the caller never sees a partially
/// populated buffer.
template <class Element>
void read_vecs_records(const std::filesystem::path& path,
                       std::vector<Element>& out,
                       std::size_t& n_out,
                       std::size_t& d_out)
{
    const MmapFile file(path);
    const std::byte* base = file.data();
    const std::size_t total = file.size();

    constexpr std::size_t prefix_bytes = sizeof(std::int32_t);
    if (total < prefix_bytes) {
        throw std::runtime_error(
            "knng::io: file too small to hold a single record: '"
            + path.string() + "'");
    }

    // Read the first record's dim prefix to discover the dimension.
    std::int32_t first_dim_signed = 0;
    std::memcpy(&first_dim_signed, base, prefix_bytes);
    if (first_dim_signed <= 0) {
        throw std::runtime_error(
            "knng::io: invalid dim prefix " + std::to_string(first_dim_signed)
            + " in '" + path.string() + "'");
    }
    const std::size_t d = static_cast<std::size_t>(first_dim_signed);
    const std::size_t record_bytes = prefix_bytes + d * sizeof(Element);
    if (total % record_bytes != 0) {
        throw std::runtime_error(
            "knng::io: file size " + std::to_string(total)
            + " is not a multiple of record size " + std::to_string(record_bytes)
            + " for '" + path.string() + "'");
    }
    const std::size_t n = total / record_bytes;

    out.resize(n * d);
    for (std::size_t i = 0; i < n; ++i) {
        const std::byte* rec = base + i * record_bytes;
        std::int32_t this_dim_signed = 0;
        std::memcpy(&this_dim_signed, rec, prefix_bytes);
        if (this_dim_signed != first_dim_signed) {
            throw std::runtime_error(
                "knng::io: record " + std::to_string(i)
                + " has dim " + std::to_string(this_dim_signed)
                + " but the first record has dim "
                + std::to_string(first_dim_signed)
                + " in '" + path.string() + "'");
        }
        std::memcpy(out.data() + i * d,
                    rec + prefix_bytes,
                    d * sizeof(Element));
    }

    n_out = n;
    d_out = d;
}

} // namespace

Dataset load_fvecs(const std::filesystem::path& path)
{
    Dataset ds;
    read_vecs_records<float>(path, ds.data, ds.n, ds.d);
    return ds;
}

IvecsData load_ivecs(const std::filesystem::path& path)
{
    IvecsData ds;
    read_vecs_records<std::int32_t>(path, ds.data, ds.n, ds.d);
    return ds;
}

BvecsData load_bvecs(const std::filesystem::path& path)
{
    BvecsData ds;
    read_vecs_records<std::uint8_t>(path, ds.data, ds.n, ds.d);
    return ds;
}

Dataset load_bvecs_as_float(const std::filesystem::path& path)
{
    BvecsData raw = load_bvecs(path);
    Dataset ds(raw.n, raw.d);
    for (std::size_t i = 0; i < raw.data.size(); ++i) {
        ds.data[i] = static_cast<float>(raw.data[i]);
    }
    return ds;
}

} // namespace knng::io
