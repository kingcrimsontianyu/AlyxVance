#pragma once

#include "common.hpp"

namespace alyx {
template <typename T, typename BinaryOp>
__forceinline__ __device__ T warpReduce(T val, BinaryOp&& binaryOp) {
    // Method 1: Use bufferfly shuffle for convenience
    unsigned fullMask{0xffff'ffffU};
    val = binaryOp(val, __shfl_xor_sync(fullMask, val, 16));
    val = binaryOp(val, __shfl_xor_sync(fullMask, val, 8));
    val = binaryOp(val, __shfl_xor_sync(fullMask, val, 4));
    val = binaryOp(val, __shfl_xor_sync(fullMask, val, 2));
    val = binaryOp(val, __shfl_xor_sync(fullMask, val, 1));

    // Method 2:
    // auto laneIdx = getLaneIdx();
    // val = binaryOp(val, __shfl_down_sync(0xffff'ffffU, val, 16));
    // val = binaryOp(val, __shfl_down_sync(0x0000'ffffU, val, 8));
    // val = binaryOp(val, __shfl_down_sync(0x0000'00ffU, val, 4));
    // val = binaryOp(val, __shfl_down_sync(0x0000'000fU, val, 2));
    // val = binaryOp(val, __shfl_down_sync(0x0000'0003U, val, 1));
    return val;
}

template <int blockSize, typename T, typename BinaryOp>
__forceinline__ __device__ T blockReduce(T val, T init, BinaryOp&& binaryOp) {
    static_assert(isMultipleOf32(blockSize));

    T res = warpReduce(val, binaryOp);

    __align__(sizeof(T)) volatile __shared__ T smem[constant::warpSize];

    auto warpIdx = getWarpIdx();
    auto laneIdx = getLaneIdx();

    // Warp 0 initializes the shared memory
    if (0 == warpIdx) {
        smem[laneIdx] = init;
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
    T ans = (threadIdx.x == 0) ? res : init;

    return ans;
}
}  // namespace alyx