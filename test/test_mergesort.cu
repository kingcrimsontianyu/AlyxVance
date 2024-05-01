#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#include "block_mergesort.hpp"
#include "common.hpp"

template <int blockSize, typename T>
__global__ void __launch_bounds__(blockSize) blockMergeSortKernel(T* a) {
    unsigned gtid = blockDim.x * blockIdx.x + threadIdx.x;

    a[gtid] = alyx::blockMergeSort<blockSize>(a[gtid]);
}

template <typename T>
class Test {
public:
    void run() {
        CUDA_CHECK(cudaSetDevice(0));

        constexpr std::size_t gridSize = 1;
        constexpr std::size_t blockSize = 32;
        std::size_t numElement{gridSize * blockSize};
        std::vector<T> ah(numElement);

        std::iota(ah.begin(), ah.end(), static_cast<T>(1));

        std::mt19937 rng(2077);
        std::shuffle(ah.begin(), ah.end(), rng);

        T* ad{};
        CUDA_CHECK(cudaMalloc(&ad, numElement * sizeof(T)));
        CUDA_CHECK(
            cudaMemcpy(ad, ah.data(), numElement * sizeof(T), cudaMemcpyKind::cudaMemcpyDefault));

        blockMergeSortKernel<blockSize, T><<<gridSize, blockSize>>>(ad);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(
            cudaMemcpy(ah.data(), ad, gridSize * sizeof(T), cudaMemcpyKind::cudaMemcpyDefault));
        for (std::size_t i = 0; i < ah.size(); ++i) {
            std::cout << ah[i] << " ";
        }
        std::cout << "\n\n";

        CUDA_CHECK(cudaFree(ad));
    }
};

int main() {
    {
        Test<int> t;
        t.run();
    }

    return 0;
}