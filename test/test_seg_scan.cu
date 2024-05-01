#include <iostream>
#include <numeric>
#include <vector>

#include "block_seg_scan.hpp"
#include "common.hpp"

template <int blockSize, typename T, typename TAlyxBinaryOp>
__global__ void __launch_bounds__(blockSize)
    blockSegScanKernel(T* a, int* flags, std::size_t numElement, T* b) {
    unsigned gtid = blockDim.x * blockIdx.x + threadIdx.x;

    T val;
    int flag;
    if (gtid >= numElement) {
        val = TAlyxBinaryOp::init.first;
        flag = TAlyxBinaryOp::init.second;
    } else {
        val = a[gtid];
        flag = flags[gtid];
    }

    alyx::SegPair<T> res = alyx::blockSegScan<blockSize>(val, flag, TAlyxBinaryOp{});

    if (gtid < numElement) b[gtid] = res.first;
}

template <typename T, typename TAlyxBinaryOp>
class Test {
public:
    void run() {
        CUDA_CHECK(cudaSetDevice(0));

        constexpr std::size_t gridSize = 2;
        constexpr std::size_t blockSize = 128;
        std::size_t numElement{200};
        std::vector<T> ah(numElement);

        if constexpr (std::is_same_v<TAlyxBinaryOp, alyx::AlyxBinaryOp::SegAdd<T>>) {
            std::fill(ah.begin(), ah.end(), static_cast<T>(1));
        } else if constexpr (std::is_same_v<TAlyxBinaryOp, alyx::AlyxBinaryOp::SegMultiply<T>>) {
            std::fill(ah.begin(), ah.end(), static_cast<T>(2));
        } else if constexpr (std::is_same_v<TAlyxBinaryOp, alyx::AlyxBinaryOp::SegCopy<T>>) {
            std::iota(ah.begin(), ah.end(), static_cast<T>(1));
        }

        std::vector<int> flagsH(numElement, 0);
        for (std::size_t i = 0; i < flagsH.size(); ++i) {
            if (i != 0 && (i % 10) == 0) flagsH[i] = 1;
        }

        T* ad{};
        CUDA_CHECK(cudaMalloc(&ad, numElement * sizeof(T)));
        CUDA_CHECK(
            cudaMemcpy(ad, ah.data(), numElement * sizeof(T), cudaMemcpyKind::cudaMemcpyDefault));

        int* flagsD{};
        CUDA_CHECK(cudaMalloc(&flagsD, numElement * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(flagsD, flagsH.data(), numElement * sizeof(int),
                              cudaMemcpyKind::cudaMemcpyDefault));

        T* bd{};
        CUDA_CHECK(cudaMalloc(&bd, numElement * sizeof(T)));

        blockSegScanKernel<blockSize, T, TAlyxBinaryOp>
            <<<gridSize, blockSize>>>(ad, flagsD, numElement, bd);
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<T> bh(numElement);
        CUDA_CHECK(
            cudaMemcpy(bh.data(), bd, numElement * sizeof(T), cudaMemcpyKind::cudaMemcpyDefault));
        for (std::size_t i = 0; i < bh.size(); ++i) {
            std::cout << bh[i] << " ";
        }
        std::cout << "\n\n";

        CUDA_CHECK(cudaFree(ad));
        CUDA_CHECK(cudaFree(bd));
    }
};

int main() {
    {
        Test<int, alyx::AlyxBinaryOp::SegAdd<int>> t;
        t.run();
    }

    {
        Test<int, alyx::AlyxBinaryOp::SegMultiply<int>> t;
        t.run();
    }

    {
        Test<double, alyx::AlyxBinaryOp::SegCopy<double>> t;
        t.run();
    }

    return 0;
}