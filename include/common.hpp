#pragma once

#include <cuda.h>          // Driver API
#include <cuda_runtime.h>  // Runtime API

#include <cuda/std/limits>
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

template <typename T>
using SegPair = Pair<T, int>;

// Built-in binary operations
namespace AlyxBinaryOp {
template <typename T>
struct Add {
    __forceinline__ __device__ T operator()(T a, T b) { return a + b; }
    static constexpr T init{static_cast<T>(0)};
};

template <typename T>
struct Multiply {
    __forceinline__ __device__ T operator()(T a, T b) { return a * b; }
    static constexpr T init{static_cast<T>(1)};
};

template <typename T>
struct Max {
    __forceinline__ __device__ T operator()(T a, T b) { return max(a, b); }
    static constexpr T init{static_cast<T>(cuda::std::numeric_limits<T>::min())};
};

template <typename T>
struct Min {
    __forceinline__ __device__ T operator()(T a, T b) { return min(a, b); }
    static constexpr T init{static_cast<T>(cuda::std::numeric_limits<T>::max())};
};

template <typename T>
requires std::is_arithmetic_v<T>
struct SegAdd {
    __forceinline__ __device__ SegPair<T> operator()(SegPair<T> a, SegPair<T> b) {
        SegPair<T> res;
        res.first = a.first * (1 - b.second) + b.first;
        res.second = a.second | b.second;
        return res;
    }
    static constexpr SegPair<T> init{static_cast<T>(0), 1};
};

template <typename T>
requires std::is_arithmetic_v<T>
struct SegMultiply {
    __forceinline__ __device__ SegPair<T> operator()(SegPair<T> a, SegPair<T> b) {
        SegPair<T> res;
        res.first = b.second == 1 ? b.first : a.first * b.first;
        res.second = a.second | b.second;
        return res;
    }
    static constexpr SegPair<T> init{static_cast<T>(1), 1};
};

template <typename T>
struct SegCopy {
    __forceinline__ __device__ SegPair<T> operator()(SegPair<T> a, SegPair<T> b) {
        SegPair<T> res;
        res.first = b.second == 1 ? b.first : a.first;
        res.second = a.second | b.second;
        return res;
    }
    static constexpr SegPair<T> init{static_cast<T>(0), 1};
};
}  // namespace AlyxBinaryOp

template <typename T>
struct Less {
    __forceinline__ __device__ constexpr bool operator()(const T& a, const T& b) const {
        return a < b;
    }
};

template <typename T>
struct Greater {
    __forceinline__ __device__ constexpr bool operator()(const T& a, const T& b) const {
        return a > b;
    }
};

template <typename T>
requires std::is_arithmetic_v<T>
__host__ __device__ void swap(T& a, T& b) {
    T tmp{a};
    a = b;
    b = tmp;
}

template <typename T>
requires std::is_arithmetic_v<T>
__host__ __device__ void swap(T*& a, T*& b) {
    T* tmp{a};
    a = b;
    b = tmp;
}

}  // namespace alyx