#pragma once

#include "common.hpp"

// This method is meant to be funny. The block size is required to be 1024,
// and the array size be 32. It is required that only the first lane in each warp pick an element
// and participate in the sorting. This method relies on the flimsy condition that the 32
// warps in the block run concurrently and the spin duration does not have excessive uncertainty. As
// such, this method does not guarantee to work reliably.
//
// Sample usage:
//
// // In device code:
// auto warpIdx = alyx::getWarpIdx();
// auto laneIdx = alyx::getLaneIdx();
// if (laneIdx == 0) {
//     a[warpIdx] = alyx::blockSleepSort<blockSize>(a[warpIdx]);
// }

namespace alyx {

__forceinline__ __device__ void spin(unsigned long long nanoSec) {
    unsigned long long start{};
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(start));
    auto stop{start};
    while (stop - start < nanoSec) {
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(stop));
    }
}

template <int blockSize, typename T>
__forceinline__ __device__ T blockSleepSort(T val) {
    static_assert(blockSize == 1024);

    __shared__ int counter[1];
    volatile __shared__ T smem[constant::warpSize];

    auto warpIdx = getWarpIdx();

    if (warpIdx == 0) {
        *counter = 0;
    }

    __syncthreads();

    auto duration = static_cast<unsigned long long>(1000 * val);
    spin(duration);

    auto offset = atomicAdd(&counter[0], 1);
    smem[offset] = val;
    __syncthreads();
    return smem[warpIdx];
}

}  // namespace alyx