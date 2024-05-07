#pragma once

#include "common.hpp"

namespace alyx {

template <int blockSize, typename T, typename Comp>
__forceinline__ __device__ T blockCountingSort(T val, Comp&& comp) {
    static_assert(isMultipleOf32(blockSize));

    __shared__ T smemA[blockSize];
    __shared__ T smemB[blockSize];

    smemA[threadIdx.x] = val;

    auto* smemIn = smemA;
    auto* smemOut = smemB;

    __syncthreads();

#pragma unroll
    for (int s = 1; s < blockSize; s = s << 1) {
        int mergeGroupSize = s << 1;
        int isLeftGroup = (threadIdx.x & (mergeGroupSize - 1)) < s;
        int groupStartIdx = threadIdx.x & -s;
        int mergeGroupStartIdx = threadIdx.x & -mergeGroupSize;

        int spouseGroupStartIdx = isLeftGroup ? (groupStartIdx + s) : (groupStartIdx - s);
        auto* spouseGroupLeftBound = smemIn + spouseGroupStartIdx;
        auto* spouseGroupRightBound = spouseGroupLeftBound + s - 1;

        int target = smemIn[threadIdx.x];
        int otherGroupRank =
            isLeftGroup ? lowerBound(spouseGroupLeftBound, spouseGroupRightBound, target, comp)
                        : upperBound(spouseGroupLeftBound, spouseGroupRightBound, target, comp);

        int selfGroupRank = threadIdx.x & (s - 1);
        int finalRank{selfGroupRank + otherGroupRank};
        smemOut[mergeGroupStartIdx + finalRank] = target;

        if (s != (blockSize >> 1)) {
            auto* smemInCached = smemIn;
            smemIn = smemOut;
            smemOut = smemInCached;
        }

        __syncthreads();
    }

    return smemOut[threadIdx.x];
}

template <int blockSize, typename T>
__forceinline__ __device__ T blockCountingSort(T val) {
    return blockCountingSort<blockSize>(val, Less<T>{});
}

}  // namespace alyx