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
