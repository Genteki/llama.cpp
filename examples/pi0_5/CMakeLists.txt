# Shared PI0.5 implementation (tensor types, graph builders, forward passes).
add_library(pi0_5-common OBJECT pi0_5-common.cpp)
target_link_libraries(pi0_5-common PUBLIC common mtmd)
target_compile_features(pi0_5-common PUBLIC cxx_std_17)

set(TARGET llama-pi0_5)
add_executable(${TARGET} pi0_5.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE pi0_5-common ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

# Benchmark + profiler (one tool, --mode phase|op|quant|profile).
set(TARGET_BENCH llama-pi0_5-bench)
add_executable(${TARGET_BENCH} pi0_5-bench.cpp)
install(TARGETS ${TARGET_BENCH} RUNTIME)
target_link_libraries(${TARGET_BENCH} PRIVATE pi0_5-common ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET_BENCH} PRIVATE cxx_std_17)
