#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#include "block_sleepsort.hpp"
#include "common.hpp"

template <int blockSize, typename T>
__global__ void __launch_bounds__(blockSize) blockSleepSortKernel(T* a) {
    static_assert(blockSize == 1024);

    auto warpIdx = alyx::getWarpIdx();
    auto laneIdx = alyx::getLaneIdx();

    if (laneIdx == 0) {
        a[warpIdx] = alyx::blockSleepSort<blockSize>(a[warpIdx]);
    }
}

template <typename T>
class Test {
public:
    void run() {
        CUDA_CHECK(cudaSetDevice(0));

        constexpr std::size_t gridSize = 1;
        constexpr std::size_t blockSize = 1024;
        std::size_t numElement{32};
        std::vector<T> ah(numElement);

        std::mt19937 rng(2077);

        std::uniform_int_distribution<int> udist(0, numElement / 2);
        for (auto&& el : ah) {
            el = static_cast<T>(udist(rng));
        }

        std::shuffle(ah.begin(), ah.end(), rng);

        T* ad{};
        CUDA_CHECK(cudaMalloc(&ad, numElement * sizeof(T)));
        CUDA_CHECK(
            cudaMemcpy(ad, ah.data(), numElement * sizeof(T), cudaMemcpyKind::cudaMemcpyDefault));

        blockSleepSortKernel<blockSize><<<gridSize, blockSize>>>(ad);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(
            cudaMemcpy(ah.data(), ad, numElement * sizeof(T), cudaMemcpyKind::cudaMemcpyDefault));
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

    {
        Test<long long> t;
        t.run();
    }

    return 0;
}