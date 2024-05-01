#pragma once

#include "common.hpp"

namespace alyx {

template <typename T, typename Iter>
__forceinline__ __device__ int lowerBound(Iter start, Iter end, const T& target) {
    int left = 0;
    int right = end - start;
    int ans{right + 1};

    while (left <= right) {
        int mid = left + ((right - left) >> 1);
        if (*(start + mid) < target) {
            left = mid + 1;
        } else {
            right = mid - 1;
            ans = mid;
        }
    }

    return ans;
}

template <typename T, typename Iter>
__forceinline__ __device__ int upperBound(Iter start, Iter end, const T& target) {
    int left = 0;
    int right = end - start;
    int ans{0};

    while (left <= right) {
        int mid = left + ((right - left) >> 1);
        if (*(start + mid) > target) {
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

    volatile __shared__ T smem[blockSize];

    smem[threadIdx.x] = val;

    __syncthreads();

#pragma unroll
    for (int s = 1; s < blockSize; s = s << 1) {
        int isLeftGroup = (threadIdx.x & ((s << 1) - 1)) < s;
        int groupIdx = threadIdx.x / s;

        int tmp{s * groupIdx};
        int startIdx = isLeftGroup ? (tmp + s) : (tmp - s);
        int endIdx = startIdx + s - 1;

        int target = smem[threadIdx.x];
        int otherGroupRank = isLeftGroup ? lowerBound(smem + startIdx, smem + endIdx, target)
                                         : upperBound(smem + startIdx, smem + endIdx, target);

        int selfGroupRank = threadIdx.x & (s - 1);
        int finalRank{selfGroupRank + otherGroupRank};
        smem[finalRank] = target;

        __syncthreads();
    }

    return smem[threadIdx.x];
}

// TODO: Use concepts to constrain TAlyxBinaryOp, as soon as nvcc bug is resolved.
template <int blockSize, typename T>
__forceinline__ __device__ T blockMergeSort(T val) {
    return blockMergeSort<blockSize>(val, [](T a, T b) { return a < b; });
}

}  // namespace alyx