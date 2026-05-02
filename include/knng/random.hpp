#pragma once

/// @file
/// @brief Deterministic XorShift64 RNG used by every randomised
///        algorithm in the project.
///
/// The single source of truth for "give me random bits" — every CPU
/// algorithm that randomises (NN-Descent in Phase 5, sampling in
/// Phase 5, the bench harness's synthetic dataset generator) and
/// every GPU port (Phase 9, where the same XorShift64 will run
/// per-thread on device) consumes this type. Two properties matter:
///
///   1. **Identical bits across CPU and GPU.** XorShift64 is a
///      single 64-bit state with two `^=` and three shifts per
///      step — the exact algorithm that fits in a CUDA / HIP
///      register and that we will replay byte-for-byte from a
///      device kernel. Using `std::mt19937_64` would force a
///      separate GPU implementation and the two would diverge over
///      time.
///   2. **Same seed ⇒ same output.** No `std::random_device`, no
///      thread-local entropy mixing. Determinism is a *feature* —
///      every regression test, every benchmark JSON, every
///      `--seed N` CLI invocation must be reproducible to the bit.
///
/// Quality: the original Marsaglia (2003) XorShift64 with shifts
/// `(13, 7, 17)`. Period `2^64 - 1` (the all-zero state is a fixed
/// point and is rejected at construction). Not cryptographic — that
/// is fine, the RNG is used for sampling and randomized
/// initialisation, not for security boundaries. For per-bit
/// statistical quality we reach for `std::mt19937_64` only when an
/// explicit need arises, which has not happened yet in the plan.
///
/// Convention: every algorithm that randomises takes a `seed`
/// argument. Top-level helpers (e.g. `init_random_graph`,
/// `make_synthetic`) construct one `XorShift64{seed}` and pass it
/// by reference into inner loops; nested functions never construct
/// their own RNG. This makes "same seed ⇒ same graph" a
/// repository-wide invariant rather than a per-file convention.

#include <cstdint>
#include <stdexcept>

namespace knng::random {

/// 64-bit XorShift PRNG. Conforms to the C++ `UniformRandomBitGenerator`
/// concept (the named requirement in `<random>` that lets the type
/// drop in to `std::shuffle`, `std::uniform_int_distribution`, etc.).
class XorShift64 {
public:
    using result_type = std::uint64_t;

    /// Construct from a non-zero seed. The all-zero state is the
    /// XorShift fixed point — the next value would also be zero,
    /// then zero forever — so it is explicitly rejected. Pass any
    /// non-zero `std::uint64_t` and the period is `2^64 - 1`.
    explicit XorShift64(std::uint64_t seed) : state_{seed}
    {
        if (seed == 0) {
            throw std::invalid_argument(
                "knng::random::XorShift64: seed must be non-zero "
                "(0 is the algorithm's fixed point)");
        }
    }

    /// Smallest possible value the generator can return — `1` (the
    /// algorithm cannot return `0` from a valid state). Required by
    /// `UniformRandomBitGenerator`.
    static constexpr result_type min() noexcept { return 1u; }

    /// Largest possible value — `2^64 - 1`. Required by
    /// `UniformRandomBitGenerator`.
    static constexpr result_type max() noexcept
    {
        return static_cast<result_type>(-1);
    }

    /// Advance the state and return the new value. Marsaglia's
    /// (13, 7, 17) shift triple; period 2^64 - 1.
    constexpr result_type operator()() noexcept
    {
        std::uint64_t x = state_;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        state_ = x;
        return x;
    }

    /// Return the current internal state. Useful for snapshot-and-
    /// restore patterns (e.g. parallel-NN-Descent reproducibility,
    /// where each worker thread starts from a known sub-seed and
    /// the test asserts the whole-graph output bit-matches a
    /// previous run).
    [[nodiscard]] constexpr std::uint64_t state() const noexcept
    {
        return state_;
    }

    /// Replace the state. The same non-zero invariant applies.
    void seed(std::uint64_t new_seed)
    {
        if (new_seed == 0) {
            throw std::invalid_argument(
                "knng::random::XorShift64::seed: must be non-zero");
        }
        state_ = new_seed;
    }

    /// Uniform float in `[0, 1)` produced by taking the high 24
    /// bits of one RNG step and dividing by `2^24`. Cheaper and
    /// more cache-friendly than the equivalent
    /// `std::uniform_real_distribution<float>`, and produces the
    /// same bit pattern across CPU and GPU because the arithmetic
    /// is integer-then-cast.
    constexpr float next_float01() noexcept
    {
        // 24 bits is the float significand; using more bits would
        // not buy precision and would round to the same value.
        constexpr std::uint64_t mask24 = (std::uint64_t{1} << 24) - 1;
        constexpr float scale = 1.0f / static_cast<float>(1u << 24);
        const std::uint64_t bits = (operator()() >> 40) & mask24;
        return static_cast<float>(bits) * scale;
    }

    /// Uniform integer in `[0, bound)`. Uses the rejection-free
    /// "Lemire" multiplicative trick — slightly biased, but the
    /// bias is bounded by `bound / 2^64`, which is < 5e-15 for
    /// any `bound <= 2^16`. Good enough for our sampling needs;
    /// when an algorithm later demands exact uniformity, it can
    /// roll its own rejection loop on top of `operator()()`.
    constexpr std::uint64_t next_below(std::uint64_t bound) noexcept
    {
        if (bound <= 1) {
            return 0;
        }
        const __uint128_t product =
            static_cast<__uint128_t>(operator()())
            * static_cast<__uint128_t>(bound);
        return static_cast<std::uint64_t>(product >> 64);
    }

private:
    std::uint64_t state_;
};

/// Convenience wrapper around `XorShift64` for the common pattern
/// "sample one float in `[0, 1)`, then advance state." Stateful
/// because every consumer wants determinism — passing the RNG by
/// reference into nested calls is the project-wide convention.
[[nodiscard]] inline float next_float01(XorShift64& rng) noexcept
{
    return rng.next_float01();
}

} // namespace knng::random
