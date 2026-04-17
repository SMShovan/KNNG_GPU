# src/CMakeLists.txt
#
# The `knng_core` library target.
#
# At Step 3 `knng_core` is a header-only INTERFACE library — every public
# API (types, Distance concept, Knng struct) lives in
# `include/knng/core/*.hpp`. Consumers link `knng::core` to pick up the
# include paths and the C++20 requirement transitively.
#
# When the first non-template implementation lands (a CPU brute-force
# reference builder, quantization helpers, etc.), this target will be
# upgraded from INTERFACE to STATIC and gain real source files. Code
# that links `knng::core` will not need to change.

add_library(knng_core INTERFACE)
target_link_libraries(knng_core INTERFACE knng::headers)
add_library(knng::core ALIAS knng_core)
