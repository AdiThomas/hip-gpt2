cmake_minimum_required(VERSION 3.28)  # 3.28+ needed for HIP on NVIDIA

# Pick the backend at configure time:
#   NVIDIA (RTX 4060):  -DCMAKE_HIP_PLATFORM=nvidia -DCMAKE_HIP_ARCHITECTURES=89
#   AMD MI300X:         -DCMAKE_HIP_ARCHITECTURES=gfx942
#   AMD RX 7900 XT/XTX: -DCMAKE_HIP_ARCHITECTURES=gfx1100
#   AMD RX 9070 / XT:   -DCMAKE_HIP_ARCHITECTURES=gfx1201
project(hip_gemm LANGUAGES CXX HIP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_HIP_STANDARD 17)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(GEMM_USE_OPENMP "Parallelize the CPU reference check" ON)

# CPU reference / checker: plain C++, no HIP.
add_library(reference STATIC src/reference.cpp)
target_include_directories(reference PUBLIC include)
if(GEMM_USE_OPENMP)
  find_package(OpenMP)
  if(OpenMP_CXX_FOUND)
    # PRIVATE so -fopenmp doesn't leak into HIP compiles (nvcc rejects it).
    target_link_libraries(reference PRIVATE OpenMP::OpenMP_CXX)
  endif()
endif()

# Every file in src/kernels/ is picked up automatically.
file(GLOB KERNEL_SOURCES CONFIGURE_DEPENDS src/kernels/*.hip)
add_executable(gemm_bench src/bench.hip ${KERNEL_SOURCES})
target_include_directories(gemm_bench PRIVATE include)
target_link_libraries(gemm_bench PRIVATE reference)

# Stamp results with the commit they came from (matters once CI tracks regressions).
execute_process(COMMAND git rev-parse --short HEAD
                WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
                OUTPUT_VARIABLE GIT_SHA OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
if(GIT_SHA)
  target_compile_definitions(gemm_bench PRIVATE GEMM_GIT_SHA="${GIT_SHA}")
endif()

# Host-only unit test of the checker itself (runs without a GPU).
add_executable(test_reference tests/test_reference.cpp)
target_link_libraries(test_reference PRIVATE reference)

enable_testing()
add_test(NAME reference_checker COMMAND test_reference)
foreach(k v0)  # add v1 v2 ... as you write them
  add_test(NAME correctness_${k} COMMAND gemm_bench --kernel ${k} --test)
endforeach()
