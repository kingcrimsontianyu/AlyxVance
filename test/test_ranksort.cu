#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#include "block_ranksort.hpp"
#include "common.hpp"

template <int blockSize, typename T, typename Comp>
__global__ void __launch_bounds__(blockSize) blockRankSortKernel(T* a) {
    unsigned gtid = blockDim.x * blockIdx.x + threadIdx.x;

    a[gtid] = alyx::blockRankSort<blockSize, T, Comp>(a[gtid], Comp{});
}

template <typename T, typename Comp>
class Test {
public:
    void run(bool hasDuplicates) {
        CUDA_CHECK(cudaSetDevice(0));

        constexpr std::size_t gridSize = 1;
        constexpr std::size_t blockSize = 128;
        std::size_t numElement{gridSize * blockSize};
        std::vector<T> ah(numElement);

        std::mt19937 rng(2077);

        if (hasDuplicates) {
            std::uniform_int_distribution<int> udist(0, blockSize / 4);
            for (auto&& el : ah) {
                el = static_cast<T>(udist(rng));
            }
        } else {
            std::iota(ah.begin(), ah.end(), static_cast<T>(1));
        }

        std::shuffle(ah.begin(), ah.end(), rng);

        T* ad{};
        CUDA_CHECK(cudaMalloc(&ad, numElement * sizeof(T)));
        CUDA_CHECK(
            cudaMemcpy(ad, ah.data(), numElement * sizeof(T), cudaMemcpyKind::cudaMemcpyDefault));

        blockRankSortKernel<blockSize, T, Comp><<<gridSize, blockSize>>>(ad);
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
        Test<int, alyx::Less<int>> t;
        t.run(true);
    }

    {
        Test<double, alyx::Greater<int>> t;
        t.run(true);
    }

    {
        Test<int, alyx::Less<int>> t;
        t.run(false);
    }

    {
        Test<double, alyx::Greater<int>> t;
        t.run(false);
    }

    return 0;
}