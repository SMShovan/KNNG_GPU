/// @file
/// @brief Translation unit for `knng::cpu::brute_force_knn`.
///
/// The algorithm itself is a function template defined in
/// `include/knng/cpu/brute_force.hpp` (see that file for the
/// contract). This `.cpp` exists for two reasons:
///
///   1. To explicitly instantiate the template for the two built-in
///      distance functors so that downstream callers using `L2Squared`
///      or `NegativeInnerProduct` link against pre-compiled symbols
///      and pay the parsing / instantiation cost exactly once. Other
///      `Distance`-satisfying functors are still implicitly
///      instantiated at the consumer's site (the explicit
///      instantiations do not preclude implicit ones).
///   2. To give the new `knng_cpu` static library its first real
///      translation unit. Until Step 10 the project shipped only
///      INTERFACE libraries; `knng_cpu` is the first STATIC target
///      and a non-empty `.cpp` is the simplest way to enforce that.

#include "knng/cpu/brute_force.hpp"

namespace knng::cpu {

template Knng brute_force_knn<L2Squared>(
    const Dataset&, std::size_t, L2Squared);

template Knng brute_force_knn<NegativeInnerProduct>(
    const Dataset&, std::size_t, NegativeInnerProduct);

} // namespace knng::cpu
