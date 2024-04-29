#pragma once

#include "common.hpp"

namespace alyx {

// See warpScan() for more implementation details.
template <typename T, typename BinaryOp>
__forceinline__ __device__ SegPair<T> warpSegScan(T val, int flag, BinaryOp&& binaryOp) {
    auto laneIdx = getLaneIdx();
    SegPair<T> res{val, flag};

    {
        constexpr int n{5};
        unsigned dstMasks[n] = {0xaaaa'aaaaU, 0x8888'8888U, 0x8080'8080U, 0x8000'8000U,
                                0x8000'0000U};
        unsigned srcMasks[n] = {0x5555'5555U, 0x2222'2222U, 0x0808'0808U, 0x0080'0080U,
                                0x0000'8000U};

        int s{1};
#pragma unroll
        for (int i = 0; i < n; ++i) {
            unsigned workerMask = dstMasks[i] | srcMasks[i];
            int isWorker = (workerMask >> laneIdx) & 1 == 1;
            int isDst = (dstMasks[i] >> laneIdx) & 1 == 1;
            if (isWorker) {
                auto otherVal = __shfl_sync(workerMask, res.first, laneIdx - s);
                auto otherFlag = __shfl_sync(workerMask, res.second, laneIdx - s);
                if (isDst) {
                    res = binaryOp({otherVal, otherFlag}, {res.first, res.second});
                }
            }

            s = s << 1;
        }
    }

    {
        constexpr int m{4};
        unsigned dstMasks[m] = {0x0080'0000U, 0x0808'0800U, 0x2222'2220U, 0x5555'5554U};
        unsigned srcMasks[m] = {0x0000'8000U, 0x0080'8080U, 0x0888'8888U, 0x2aaa'aaaaU};

        int s{8};
#pragma unroll
        for (int i = 0; i < m; ++i) {
            unsigned workerMask = dstMasks[i] | srcMasks[i];
            int isWorker = (workerMask >> laneIdx) & 1 == 1;
            int isDst = (dstMasks[i] >> laneIdx) & 1 == 1;
            if (isWorker) {
                auto otherVal = __shfl_sync(workerMask, res.first, laneIdx - s);
                auto otherFlag = __shfl_sync(workerMask, res.second, laneIdx - s);
                if (isDst) {
                    res = binaryOp({otherVal, otherFlag}, {res.first, res.second});
                }
            }

            s = s >> 1;
        }
    }

    return res;
}

template <int blockSize, typename T, typename BinaryOp>
__forceinline__ __device__ SegPair<T> blockSegScan(T val, int flag, T init, BinaryOp&& binaryOp) {
    static_assert(isMultipleOf32(blockSize));

    auto res = warpSegScan(val, flag, binaryOp);

    volatile __shared__ T smemVal[constant::warpSize];
    volatile __shared__ int smemFlag[constant::warpSize];

    auto warpIdx = getWarpIdx();
    auto laneIdx = getLaneIdx();

    // Warp 0 initializes the shared memory
    if (0 == warpIdx) {
        smemVal[laneIdx] = init;
        smemFlag[laneIdx] = 1;
    }

    __syncthreads();

    // Last lane in each warp writes the increment to the shared memory
    if (constant::warpSize - 1 == laneIdx) {
        smemVal[warpIdx] = res.first;
        smemFlag[warpIdx] = res.second;
    }

    __syncthreads();

    // Warp 0 performs the final scan
    if (0 == warpIdx) {
        auto tmp = warpSegScan(smemVal[laneIdx], smemFlag[warpIdx], binaryOp);
        smemVal[laneIdx] = tmp.first;
        smemFlag[laneIdx] = tmp.second;
    }

    __syncthreads();

    // All threads in each warp reads the increment from the shared memory
    // and adjust their results
    if (warpIdx > 0) {
        SegPair<T> inc{smemVal[warpIdx - 1], smemFlag[warpIdx - 1]};
        res = binaryOp(inc, res);
    }

    return res;
}

// TODO: Use concepts to constrain TAlyxBinaryOp, as soon as nvcc bug is resolved.
template <int blockSize, typename T, typename TAlyxBinaryOp>
__forceinline__ __device__ SegPair<T> blockSegScan(T val, int flag, TAlyxBinaryOp&& alyxBinaryOp) {
    return blockSegScan<blockSize>(val, flag, TAlyxBinaryOp::init.first,
                                   std::forward<TAlyxBinaryOp>(alyxBinaryOp));
}

}  // namespace alyx