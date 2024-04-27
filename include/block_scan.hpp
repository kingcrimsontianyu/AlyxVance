#pragma once

#include "common.hpp"

namespace alyx {

namespace ScanAlgo {
struct Naive {};
struct WorkEfficient1 {};  // Mask is computed at runtime
struct WorkEfficient2 {};  // Mask is predetermined
}  // namespace ScanAlgo

template <typename T, typename BinaryOp>
__forceinline__ __device__ T warpScan(T val, BinaryOp binaryOp, ScanAlgo::Naive) {
    auto laneIdx = getLaneIdx();

    unsigned fullMask{0xffff'ffff};
    for (int s = 1; s <= 16; s = s << 1) {
        int srcIdx = laneIdx - s;

        auto res = __shfl_sync(fullMask, val, srcIdx);
        if (srcIdx >= 0) {
            val = binaryOp(val, res);
        }
    }

    return val;
}

template <typename T, typename BinaryOp>
__forceinline__ __device__ T warpScan(T val, BinaryOp binaryOp, ScanAlgo::WorkEfficient1) {
    auto laneIdx = getLaneIdx();
    unsigned fullMask{0xffff'ffffU};

    // Upsweep
    // i: dst Relative idx
    // g_dst: Lane idx where the data will be copied to
    // g_src: Lane idx where the data will be copied from
    // g_dst = s * i + s - 1
    // g_src = s * i + s - 1 - s / 2 = s * i + s / 2 - 1
    // Worker mask includes src and dst lanes. All worker lanes are active.
    for (int s = 2; s <= 32; s = s << 1) {
        int tmpDst = laneIdx - s + 1;
        int isDst = (tmpDst % s == 0);
        unsigned dstMask = __ballot_sync(fullMask, isDst);

        int tmpSrc = laneIdx - s / 2 + 1;
        int isSrc = (tmpSrc % s == 0);
        unsigned srcMask = __ballot_sync(fullMask, isSrc);

        unsigned workerMask = dstMask | srcMask;

        if (isDst || isSrc) {
            auto res = __shfl_sync(workerMask, val, laneIdx - s / 2);
            if (isDst) val = binaryOp(val, res);
        }
    }

    // Downsweep
    // i: Worker relative idx
    // g_worker = s * i + s - 1
    // Not all worker lanes are active.
    for (int s = 16; s >= 1; s = s >> 1) {
        int tmpWorker = laneIdx - s + 1;
        int isWorker = (tmpWorker % s == 0);
        unsigned workerMask = __ballot_sync(fullMask, isWorker);

        if (isWorker) {
            int workerIdx = tmpWorker / s;
            int nextWorkerIdx = laneIdx + s;
            // For dst lane, its worker relative index is even and non-zero.
            // For src lane, its worker relative index is odd and its next even index should exist,
            // which is g_worker_next = s * (i + 1) + s - 1 = g_worker + s
            int isDst = (workerIdx != 0) && ((workerIdx & 1) == 0);
            int isSrc = ((workerIdx & 1) == 1) && (nextWorkerIdx < 32);

            unsigned dstMask = __ballot_sync(workerMask, isDst);
            unsigned srcMask = __ballot_sync(workerMask, isSrc);
            unsigned comboMask = dstMask | srcMask;

            if (isDst || isSrc) {
                int prevWorkerIdx = laneIdx - s;
                auto res = __shfl_sync(comboMask, val, prevWorkerIdx);
                if (isDst) val = binaryOp(val, res);
            }
        }
    }
    return val;
}

template <typename T, typename BinaryOp>
__forceinline__ __device__ T warpScan(T val, BinaryOp binaryOp, ScanAlgo::WorkEfficient2) {
    auto laneIdx = getLaneIdx();

    // Upsweep
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
                auto res = __shfl_sync(workerMask, val, laneIdx - s);
                if (isDst) val = binaryOp(val, res);
            }

            s = s << 1;
        }
    }

    // Downsweep
    // s  [src, dst] pairs
    // 8  [15, 23]
    // 4  [7, 11] [15, 19] [23, 27]
    // 2  [3, 5] [7, 9] [11, 13] [15, 17] [19, 21] [23, 25] [27, 29]
    // 1  [1, 2] [3, 4] [5, 6] ... [29, 30]
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
                auto res = __shfl_sync(workerMask, val, laneIdx - s);
                if (isDst) val = binaryOp(val, res);
            }

            s = s >> 1;
        }
    }

    return val;
}

template <typename Algo, int blockSize, typename T, typename BinaryOp>
__forceinline__ __device__ T blockScan(T val, T init, BinaryOp&& binaryOp) {
    static_assert(isMultipleOf32(blockSize));

    T res = warpScan(val, binaryOp, Algo{});

    __align__(sizeof(T)) volatile __shared__ T smem[constant::warpSize];

    auto warpIdx = getWarpIdx();
    auto laneIdx = getLaneIdx();

    // Warp 0 initializes the shared memory
    if (0 == warpIdx) {
        smem[laneIdx] = init;
    }

    __syncthreads();

    // Last lane in each warp writes the increment to the shared memory
    if (constant::warpSize - 1 == laneIdx) {
        smem[warpIdx] = res;
    }

    __syncthreads();

    // All threads in each warp reads the increment from the shared memory
    if (warpIdx > 0) {
        auto inc = smem[warpIdx - 1];
        res = binaryOp(res, inc);
    }

    return res;
}

template <int blockSize, typename T, typename BinaryOp>
__forceinline__ __device__ T blockScan(T val, T init, BinaryOp&& binaryOp) {
    return blockScan<ScanAlgo::WorkEfficient2, blockSize, T, BinaryOp>(
        val, init, std::forward<BinaryOp>(binaryOp));
}

}  // namespace alyx