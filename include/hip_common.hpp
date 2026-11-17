#pragma once
// Platform detection + error-check macros shared by every component.
// The only place allowed to test __HIP_CPU_RT__ / __HIP_PLATFORM_*__ (SPEC 3.3).
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cstdlib>

#if defined(__HIP_CPU_RT__)
#define HIPGPT2_PLATFORM_CPU 1
#define HIPGPT2_PLATFORM_NAME "cpu"
#elif defined(__HIP_PLATFORM_NVIDIA__)
#define HIPGPT2_PLATFORM_NVIDIA 1
#define HIPGPT2_PLATFORM_NAME "nvidia"
#elif defined(__HIP_PLATFORM_AMD__)
#define HIPGPT2_PLATFORM_AMD 1
#define HIPGPT2_PLATFORM_NAME "amd"
#else
#error "hip-gpt2: unknown HIP platform (expected HIP-CPU, NVIDIA or AMD)"
#endif

#ifndef HIPGPT2_GIT_SHA
#define HIPGPT2_GIT_SHA "unknown"
#endif

// Aborts with file:line and the HIP error string on any HIP API failure.
#define HIP_CHECK(expr)                                                        \
    do {                                                                       \
        hipError_t _e = (expr);                                                \
        if (_e != hipSuccess) {                                                \
            std::fprintf(stderr, "HIP error '%s' at %s:%d: %s\n", #expr,       \
                         __FILE__, __LINE__, hipGetErrorString(_e));           \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

// Host-side invariant check.
#define HIPGPT2_REQUIRE(cond, msg)                                             \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::fprintf(stderr, "check failed: %s at %s:%d\n", (msg),         \
                         __FILE__, __LINE__);                                  \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)
