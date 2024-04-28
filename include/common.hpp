#pragma once

#include <cuda.h>          // Driver API
#include <cuda_runtime.h>  // Runtime API

#include <stdexcept>
#include <type_traits>

#define CUDA_CHECK(errCode) alyx::cudaCheck(errCode, __FILE__, __LINE__, true);
#define CUDA_CHECK_NOTHROW(errCode) alyx::cudaCheck(errCode, __FILE__, __LINE__, false);

namespace alyx::constant {
constexpr int warpSize{32};
}

namespace alyx {
inline void cudaCheck(cudaError_t errCode, const char* fileName, int lineNumber, bool canThrow) {
    if (errCode == cudaError_t::cudaSuccess) return;

    int currentDevice{};
    cudaGetDevice(&currentDevice);
    const char* errName = cudaGetErrorName(errCode);
    const char* errString = cudaGetErrorString(errCode);

    // TODO: Replace with std::format() using a standard compliant compiler
    char errMsg[256];
    std::sprintf(errMsg, "GPU device %d error. %s (%d): %s on line %d in file %s", currentDevice,
                 errName, static_cast<int>(errCode), errString, lineNumber, fileName);
    if (canThrow) throw std::runtime_error(errMsg);
}

__forceinline__ __device__ unsigned getLaneIdx() {
    unsigned laneIdx{};
    asm volatile("mov.u32 %0, %%laneid;" : "=r"(laneIdx));
    return laneIdx;
}

__forceinline__ __device__ unsigned getWarpIdx() { return threadIdx.x / constant::warpSize; }

template <typename T>
requires std::is_integral_v<T>
__forceinline__ __device__ constexpr bool isMultipleOf32(T t) {
    return (t & 31) == 0;
}

template <typename U, typename V>
struct Pair {
    U first;
    V second;
};
}  // namespace alyx