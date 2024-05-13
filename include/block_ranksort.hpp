#pragma once

#include "common.hpp"

namespace alyx {

template <int blockSize, typename T, typename Comp>
__forceinline__ __device__ T blockRankSort(T val, Comp&& comp) {
    static_assert(isMultipleOf32(blockSize));

    __shared__ T smem[blockSize];
    smem[threadIdx.x] = val;
    __syncthreads();

    int rank{0};
    for (int i = 0; i < blockSize; ++i) {
        if (i == threadIdx.x) continue;

        if (comp(smem[i], val) || (smem[i] == val && i < threadIdx.x)) {
            ++rank;
        }
    }

    __syncthreads();
    smem[rank] = val;
    __syncthreads();
    return smem[threadIdx.x];
}

template <int blockSize, typename T>
__forceinline__ __device__ T blockRankSort(T val) {
    return blockRankSort<blockSize>(val, Less<T>{});
}

}  // namespace alyx