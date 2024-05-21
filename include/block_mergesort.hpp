#pragma once

#include "common.hpp"

namespace alyx {

template <typename T, typename Iter, typename Comp>
__forceinline__ __device__ int lowerBound(Iter start, Iter end, const T& target, Comp&& comp) {
    int left = 0;
    int right = end - start;
    int ans{right + 1};

    while (left <= right) {
        int mid = left + ((right - left) >> 1);
        if (comp(*(start + mid), target)) {
            left = mid + 1;
        } else {
            right = mid - 1;
            ans = mid;
        }
    }

    return ans;
}

template <typename T, typename Iter, typename Comp>
__forceinline__ __device__ int upperBound(Iter start, Iter end, const T& target, Comp&& comp) {
    int left = 0;
    int right = end - start;
    int ans{0};

    while (left <= right) {
        int mid = left + ((right - left) >> 1);
        if (comp(target, *(start + mid))) {
            right = mid - 1;
        } else {
            left = mid + 1;
            ans = mid + 1;
        }
    }

    return ans;
}

template <int blockSize, typename T, typename Comp>
__forceinline__ __device__ T blockMergeSort(T val, Comp&& comp) {
    static_assert(isMultipleOf32(blockSize));

    __shared__ T smemA[blockSize];
    __shared__ T smemB[blockSize];

    smemA[threadIdx.x] = val;

    auto* smemIn = smemA;
    auto* smemOut = smemB;

    __syncthreads();

// o o o o o o o o  s mergeGroupSize
// - - - - - - - -  1 2
// --- --- --- ---  2 4
// ------- -------  4 8
// ---------------  8 16
// s: The size of the current group
// mergeGroupSize: The total size of a merged group (merging two adjacent current groups)
// isLeftGroup: Whether the current group is on the left or right in the merge group
//              Left group: (tid % mergeGroupSize) < s
//              Right group: (tid % mergeGroupSize) >= s
// groupStartIdx: The index of the first thread in each group
//                (tid / s * s) = (tid & -s)
// mergeGroupStartIdx: The index of the first thread in each merged group
// spouseGroupStartIdx: The index of the first thread in the other group
//                      within the merged group
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

        // If the current iteration is not the last iteration
        // (where s is blockSize / 2)
        if (s != (blockSize >> 1)) {
            swap(smemIn, smemOut);
        }

        __syncthreads();
    }

    return smemOut[threadIdx.x];
}

template <int blockSize, typename T>
__forceinline__ __device__ T blockMergeSort(T val) {
    return blockMergeSort<blockSize>(val, Less<T>{});
}

}  // namespace alyx