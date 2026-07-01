# Shared PI0 implementation. Keeps tensor types, graph builders and forward
# passes in a single TU so the executables don't each get their own copy.
add_library(pi0-common OBJECT pi0-common.cpp)
target_link_libraries(pi0-common PUBLIC common mtmd)
target_compile_features(pi0-common PUBLIC cxx_std_17)

set(TARGET llama-pi0)
add_executable(${TARGET} pi0.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE pi0-common ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET_BENCH llama-pi0-bench)
add_executable(${TARGET_BENCH} pi0-bench.cpp)
install(TARGETS ${TARGET_BENCH} RUNTIME)
target_link_libraries(${TARGET_BENCH} PRIVATE pi0-common ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET_BENCH} PRIVATE cxx_std_17)

# Standalone attention-GEMM microbenchmark (batched vs collapsed, prefix shapes).
set(TARGET_ATTN_BENCH llama-pi0-attn-bench)
add_executable(${TARGET_ATTN_BENCH} pi0-attn-bench.cpp)
install(TARGETS ${TARGET_ATTN_BENCH} RUNTIME)
target_link_libraries(${TARGET_ATTN_BENCH} PRIVATE pi0-common ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET_ATTN_BENCH} PRIVATE cxx_std_17)

# q4_0 GEMM B-load-path microbenchmark (image/texture vs global vs LDS).
set(TARGET_GEMM_BENCH llama-pi0-gemm-bench)
add_executable(${TARGET_GEMM_BENCH} pi0-gemm-bench.cpp)
install(TARGETS ${TARGET_GEMM_BENCH} RUNTIME)
target_link_libraries(${TARGET_GEMM_BENCH} PRIVATE pi0-common ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET_GEMM_BENCH} PRIVATE cxx_std_17)

# Pure-OpenCL ALU throughput micro-probe: int8 dp4a (qcom_udot8_acc) vs fp16 mad.
# Links OpenCL directly, so only build it when the OpenCL backend is enabled.
if (GGML_OPENCL)
    find_package(OpenCL REQUIRED)
    set(TARGET_UDOT8_BENCH llama-pi0-udot8-bench)
    add_executable(${TARGET_UDOT8_BENCH} pi0-udot8-bench.cpp)
    install(TARGETS ${TARGET_UDOT8_BENCH} RUNTIME)
    target_link_libraries(${TARGET_UDOT8_BENCH} PRIVATE ${OpenCL_LIBRARIES})
    target_include_directories(${TARGET_UDOT8_BENCH} PRIVATE ${OpenCL_INCLUDE_DIRS})
    target_compile_features(${TARGET_UDOT8_BENCH} PRIVATE cxx_std_17)
endif()
