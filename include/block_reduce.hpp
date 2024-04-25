#pragma once

#include "common.hpp"

namespace alyx {
template <typename T, typename BinaryOp>
__forceinline__ __device__ T warpReduce(T val, BinaryOp&& binaryOp) {
    unsigned fullMask{0xffff'ffffU};

    // Use bufferfly shuffle for convenience
    val = binaryOp(val, __shfl_xor_sync(fullMask, val, 16));
    val = binaryOp(val, __shfl_xor_sync(fullMask, val, 8));
    val = binaryOp(val, __shfl_xor_sync(fullMask, val, 4));
    val = binaryOp(val, __shfl_xor_sync(fullMask, val, 2));
    val = binaryOp(val, __shfl_xor_sync(fullMask, val, 1));
    return val;
}

template <typename T, typename BinaryOp>
__forceinline__ __device__ T blockReduce(T val, T Init, BinaryOp&& binaryOp) {
    T res = warpReduce(val, binaryOp);

    __align__(sizeof(T)) volatile __shared__ T smem[constant::warpSize];

    auto warpIdx = getWarpIdx();
    auto laneIdx = getLaneIdx();

    // Warp 0 initializes the shared memory
    if (0 == warpIdx) {
        smem[laneIdx] = Init;
    }

    __syncthreads();

    // Lane 0 in each warp writes the results to the shared memory
    if (0 == laneIdx) {
        smem[warpIdx] = res;
    }

    __syncthreads();

    // Warp 0 performs block reduce
    if (0 == warpIdx) {
        res = smem[laneIdx];
        res = warpReduce(res, binaryOp);
    }

    __syncthreads();

    // Thread 0 in the block reports the reduce result.
    // All other threads reports init.
    T ans = (threadIdx.x == 0) ? res : Init;

    return ans;
}
}  // namespace alyx