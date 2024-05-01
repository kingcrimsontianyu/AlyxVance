#include <iostream>
#include <numeric>
#include <vector>

#include "block_reduce.hpp"
#include "common.hpp"

template <int blockSize, typename T, typename TAlyxBinaryOp>
__global__ void __launch_bounds__(blockSize) blockReduceKernel(T* a, std::size_t numElement, T* b) {
    unsigned gtid = blockDim.x * blockIdx.x + threadIdx.x;

    T val;
    if (gtid >= numElement) {
        val = TAlyxBinaryOp::init;
    } else {
        val = a[gtid];
    }

    T res = alyx::blockReduce<blockSize>(val, TAlyxBinaryOp{});

    if (threadIdx.x == 0) b[blockIdx.x] = res;
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

        if constexpr (std::is_same_v<TAlyxBinaryOp, alyx::AlyxBinaryOp::Add<T>>) {
            std::fill(ah.begin(), ah.end(), static_cast<T>(1));
        } else if constexpr (std::is_same_v<TAlyxBinaryOp, alyx::AlyxBinaryOp::Multiply<T>>) {
            std::fill(ah.begin(), ah.end(), static_cast<T>(2));
        } else if constexpr (std::is_same_v<TAlyxBinaryOp, alyx::AlyxBinaryOp::Max<T>> ||
                             std::is_same_v<TAlyxBinaryOp, alyx::AlyxBinaryOp::Min<T>>) {
            std::iota(ah.begin(), ah.end(), static_cast<T>(1));
        }

        T* ad{};
        CUDA_CHECK(cudaMalloc(&ad, numElement * sizeof(T)));
        CUDA_CHECK(
            cudaMemcpy(ad, ah.data(), numElement * sizeof(T), cudaMemcpyKind::cudaMemcpyDefault));

        T* bd{};
        CUDA_CHECK(cudaMalloc(&bd, gridSize * sizeof(T)));

        blockReduceKernel<blockSize, T, TAlyxBinaryOp><<<gridSize, blockSize>>>(ad, numElement, bd);
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<T> bh(gridSize);
        CUDA_CHECK(
            cudaMemcpy(bh.data(), bd, gridSize * sizeof(T), cudaMemcpyKind::cudaMemcpyDefault));
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
        Test<int, alyx::AlyxBinaryOp::Add<int>> t;
        t.run();
    }

    {
        Test<double, alyx::AlyxBinaryOp::Add<double>> t;
        t.run();
    }

    {
        Test<double, alyx::AlyxBinaryOp::Multiply<double>> t;
        t.run();
    }

    {
        Test<float, alyx::AlyxBinaryOp::Max<float>> t;
        t.run();
    }

    {
        Test<float, alyx::AlyxBinaryOp::Min<float>> t;
        t.run();
    }

    return 0;
}