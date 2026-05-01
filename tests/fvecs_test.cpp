/// @file
/// @brief Unit tests for the `.fvecs` / `.ivecs` / `.bvecs` loaders.
///
/// Each test materialises a tiny binary file in
/// `std::filesystem::temp_directory_path()` matching the on-disk
/// format documented in `knng/io/fvecs.hpp`, hands the path to the
/// loader, and asserts the recovered shape and values. The temp
/// files are removed at the end of every test (success or failure)
/// via a small RAII helper. No network access; no SIFT download
/// required.

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <unistd.h>

#include <gtest/gtest.h>

#include "knng/io/fvecs.hpp"

namespace {

/// RAII wrapper that creates a unique temp-file path at construction
/// and removes the file at destruction. Distinct from
/// `std::tmpfile()` because the loaders take a path, not a `FILE*`.
class TempPath {
public:
    explicit TempPath(const std::string& suffix)
    {
        const auto base = std::filesystem::temp_directory_path();
        // Two-stage uniqueness: PID + a per-instance counter. Sufficient
        // for single-process unit-test parallelism (we never `fork`).
        static std::size_t counter = 0;
        path_ = base / ("knng_test_"
            + std::to_string(::getpid())
            + "_" + std::to_string(counter++)
            + suffix);
    }
    ~TempPath()
    {
        std::error_code ec;
        std::filesystem::remove(path_, ec);
    }

    TempPath(const TempPath&) = delete;
    TempPath& operator=(const TempPath&) = delete;
    TempPath(TempPath&&) = delete;
    TempPath& operator=(TempPath&&) = delete;

    [[nodiscard]] const std::filesystem::path& path() const noexcept { return path_; }

private:
    std::filesystem::path path_;
};

/// Append a single record (dim prefix + payload) to an open binary
/// stream. `Element` is the on-disk element type (`float`,
/// `std::int32_t`, or `std::uint8_t`).
template <class Element>
void write_record(std::ofstream& os, const std::vector<Element>& payload)
{
    const std::int32_t dim_signed = static_cast<std::int32_t>(payload.size());
    os.write(reinterpret_cast<const char*>(&dim_signed), sizeof(dim_signed));
    os.write(reinterpret_cast<const char*>(payload.data()),
             static_cast<std::streamsize>(payload.size() * sizeof(Element)));
}

TEST(LoadFvecs, RoundTripsThreeRecordsOfDimFour)
{
    TempPath tmp(".fvecs");
    {
        std::ofstream os(tmp.path(), std::ios::binary);
        ASSERT_TRUE(os.good());
        write_record<float>(os, {1.0f, 2.0f, 3.0f, 4.0f});
        write_record<float>(os, {5.0f, 6.0f, 7.0f, 8.0f});
        write_record<float>(os, {9.0f, 10.0f, 11.0f, 12.0f});
    }

    const knng::Dataset ds = knng::io::load_fvecs(tmp.path());
    EXPECT_EQ(ds.n, std::size_t{3});
    EXPECT_EQ(ds.d, std::size_t{4});
    ASSERT_EQ(ds.data.size(), std::size_t{12});

    constexpr std::array<float, 12> expected{
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
        9.0f, 10.0f, 11.0f, 12.0f,
    };
    for (std::size_t i = 0; i < expected.size(); ++i) {
        EXPECT_FLOAT_EQ(ds.data[i], expected[i]) << "i=" << i;
    }
}

TEST(LoadIvecs, RoundTripsTwoRecordsOfDimThree)
{
    TempPath tmp(".ivecs");
    {
        std::ofstream os(tmp.path(), std::ios::binary);
        ASSERT_TRUE(os.good());
        write_record<std::int32_t>(os, {10, 20, 30});
        write_record<std::int32_t>(os, {40, 50, 60});
    }

    const knng::io::IvecsData iv = knng::io::load_ivecs(tmp.path());
    EXPECT_EQ(iv.n, std::size_t{2});
    EXPECT_EQ(iv.d, std::size_t{3});
    ASSERT_EQ(iv.data.size(), std::size_t{6});
    EXPECT_EQ(iv.data[0], 10);
    EXPECT_EQ(iv.data[1], 20);
    EXPECT_EQ(iv.data[2], 30);
    EXPECT_EQ(iv.data[3], 40);
    EXPECT_EQ(iv.data[4], 50);
    EXPECT_EQ(iv.data[5], 60);
}

TEST(LoadBvecs, RoundTripsRawBytesAndWidenedFloats)
{
    TempPath tmp(".bvecs");
    {
        std::ofstream os(tmp.path(), std::ios::binary);
        ASSERT_TRUE(os.good());
        write_record<std::uint8_t>(os, {0, 127, 255});
        write_record<std::uint8_t>(os, {1, 2, 3});
    }

    const knng::io::BvecsData raw = knng::io::load_bvecs(tmp.path());
    EXPECT_EQ(raw.n, std::size_t{2});
    EXPECT_EQ(raw.d, std::size_t{3});
    ASSERT_EQ(raw.data.size(), std::size_t{6});
    EXPECT_EQ(raw.data[0], std::uint8_t{0});
    EXPECT_EQ(raw.data[1], std::uint8_t{127});
    EXPECT_EQ(raw.data[2], std::uint8_t{255});

    const knng::Dataset widened = knng::io::load_bvecs_as_float(tmp.path());
    EXPECT_EQ(widened.n, std::size_t{2});
    EXPECT_EQ(widened.d, std::size_t{3});
    EXPECT_FLOAT_EQ(widened.data[0], 0.0f);
    EXPECT_FLOAT_EQ(widened.data[1], 127.0f);
    EXPECT_FLOAT_EQ(widened.data[2], 255.0f);
    EXPECT_FLOAT_EQ(widened.data[5], 3.0f);
}

TEST(LoadFvecs, MissingFileThrows)
{
    const std::filesystem::path bogus =
        std::filesystem::temp_directory_path() / "knng_test_does_not_exist.fvecs";
    std::error_code ec;
    std::filesystem::remove(bogus, ec);  // best effort
    EXPECT_THROW(static_cast<void>(knng::io::load_fvecs(bogus)),
                 std::runtime_error);
}

TEST(LoadFvecs, InconsistentDimPrefixThrows)
{
    TempPath tmp(".fvecs");
    {
        std::ofstream os(tmp.path(), std::ios::binary);
        ASSERT_TRUE(os.good());
        // Two records, but the second declares dim=3 while the first
        // declared dim=4 — and we deliberately size the second
        // record's payload to *also* be three floats so that the file
        // size happens to be consistent with one (4+4*4)+(4+3*4) = 32
        // byte file, making record_bytes math also consistent if the
        // dim prefix were not validated. The loader must catch this.
        write_record<float>(os, {1.0f, 2.0f, 3.0f, 4.0f});
        write_record<float>(os, {5.0f, 6.0f, 7.0f});
    }
    EXPECT_THROW(static_cast<void>(knng::io::load_fvecs(tmp.path())),
                 std::runtime_error);
}

TEST(LoadFvecs, FileSizeNotMultipleOfRecordSizeThrows)
{
    TempPath tmp(".fvecs");
    {
        std::ofstream os(tmp.path(), std::ios::binary);
        ASSERT_TRUE(os.good());
        // Write a complete record (dim=4, four floats) followed by
        // a stray byte that breaks the record-size invariant.
        write_record<float>(os, {1.0f, 2.0f, 3.0f, 4.0f});
        const char garbage = 'X';
        os.write(&garbage, 1);
    }
    EXPECT_THROW(static_cast<void>(knng::io::load_fvecs(tmp.path())),
                 std::runtime_error);
}

TEST(LoadFvecs, EmptyFileThrows)
{
    TempPath tmp(".fvecs");
    {
        std::ofstream os(tmp.path(), std::ios::binary);
        ASSERT_TRUE(os.good());
        // Close immediately — zero bytes on disk.
    }
    EXPECT_THROW(static_cast<void>(knng::io::load_fvecs(tmp.path())),
                 std::runtime_error);
}

} // namespace
