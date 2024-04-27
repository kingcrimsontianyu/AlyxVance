#include <iostream>
#include <vector>

#include "block_scan.hpp"
#include "common.hpp"

enum class OpType {
    Add,
    Multiply,
    Max,
    Min,
};

template <int blockSize, typename T, OpType opType>
__global__ void __launch_bounds__(blockSize) blockScanKernel(T* a, std::size_t numElement, T* b) {
    unsigned gtid = blockDim.x * blockIdx.x + threadIdx.x;

    T init;
    if constexpr (opType == OpType::Add) {
        init = static_cast<T>(0);
    } else if constexpr (opType == OpType::Multiply) {
        init = static_cast<T>(1);
    } else if constexpr (opType == OpType::Max) {
        init = static_cast<T>(INT_MIN);
    } else if constexpr (opType == OpType::Min) {
        init = static_cast<T>(INT_MAX);
    }

    T val;
    if (gtid >= numElement) {
        val = init;
    } else {
        val = a[gtid];
    }

    T res;
    if constexpr (opType == OpType::Add) {
        res = alyx::blockScan<blockSize>(val, init, [](T a, T b) { return a + b; });
    } else if constexpr (opType == OpType::Multiply) {
        res = alyx::blockScan<blockSize>(val, init, [](T a, T b) { return a * b; });
    } else if constexpr (opType == OpType::Max) {
        res = alyx::blockScan<blockSize>(val, init, [](T a, T b) { return max(a, b); });
    } else if constexpr (opType == OpType::Min) {
        res = alyx::blockScan<blockSize>(val, init, [](T a, T b) { return min(a, b); });
    }

    if (gtid < numElement) b[gtid] = res;
}

template <typename T, OpType opType>
class Test {
public:
    void run() {
        CUDA_CHECK(cudaSetDevice(0));

        constexpr std::size_t gridSize = 2;
        constexpr std::size_t blockSize = 64;
        std::size_t numElement{100};
        std::vector<T> ah(numElement);

        if constexpr (opType == OpType::Add) {
            for (std::size_t i = 0; i < ah.size(); ++i) {
                ah[i] = static_cast<T>(1);
            }
        } else if constexpr (opType == OpType::Multiply) {
            for (std::size_t i = 0; i < ah.size(); ++i) {
                ah[i] = static_cast<T>(2);
            }
        } else if constexpr (opType == OpType::Max || opType == OpType::Min) {
            for (std::size_t i = 0; i < ah.size(); ++i) {
                ah[i] = static_cast<T>(i);
            }

            std::size_t halfSize = ah.size() >> 1;
            for (std::size_t i = 0; i < halfSize; ++i) {
                if ((i & 1) == 1) {
                    std::swap(ah[i], ah[ah.size() - 1 - i]);
                }
            }
        }

        T* ad{};
        CUDA_CHECK(cudaMalloc(&ad, numElement * sizeof(T)));
        CUDA_CHECK(
            cudaMemcpy(ad, ah.data(), numElement * sizeof(T), cudaMemcpyKind::cudaMemcpyDefault));

        T* bd{};
        CUDA_CHECK(cudaMalloc(&bd, numElement * sizeof(T)));

        blockScanKernel<blockSize, T, opType><<<gridSize, blockSize>>>(ad, numElement, bd);
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<T> bh(numElement);
        CUDA_CHECK(
            cudaMemcpy(bh.data(), bd, numElement * sizeof(T), cudaMemcpyKind::cudaMemcpyDefault));
        for (std::size_t i = 0; i < bh.size(); ++i) {
            std::cout << bh[i] << " ";
        }
        std::cout << "\n";

        CUDA_CHECK(cudaFree(ad));
        CUDA_CHECK(cudaFree(bd));
    }
};

int main() {
    {
        Test<int, OpType::Add> t;
        t.run();
    }

    return 0;
}