#include <iostream>
#include <vector>

#include "block_seg_scan.hpp"
#include "common.hpp"

enum class OpType {
    Add,
    Multiply,
    Copy,
};

template <int blockSize, typename T, OpType opType>
__global__ void __launch_bounds__(blockSize)
    blockSegScanKernel(T* a, int* flags, std::size_t numElement, T* b) {
    unsigned gtid = blockDim.x * blockIdx.x + threadIdx.x;

    T initVal;
    int initFlag;
    if constexpr (opType == OpType::Add) {
        initVal = alyx::AlyxBinaryOp::SegAdd<T>::init.first;
        initFlag = alyx::AlyxBinaryOp::SegAdd<T>::init.second;
    } else if constexpr (opType == OpType::Multiply) {
        initVal = alyx::AlyxBinaryOp::SegMultiply<T>::init.first;
        initFlag = alyx::AlyxBinaryOp::SegMultiply<T>::init.second;
    } else if constexpr (opType == OpType::Copy) {
        initVal = alyx::AlyxBinaryOp::SegCopy<T>::init.first;
        initFlag = alyx::AlyxBinaryOp::SegCopy<T>::init.second;
    }

    T val;
    int flag;
    if (gtid >= numElement) {
        val = initVal;
        flag = initFlag;
    } else {
        val = a[gtid];
        flag = flags[gtid];
    }

    alyx::SegPair<T> res;
    if constexpr (opType == OpType::Add) {
        res = alyx::blockSegScan<blockSize>(val, flag, alyx::AlyxBinaryOp::SegAdd<T>{});
    } else if constexpr (opType == OpType::Multiply) {
        res = alyx::blockSegScan<blockSize>(val, flag, alyx::AlyxBinaryOp::SegMultiply<T>{});
    } else if constexpr (opType == OpType::Copy) {
        res = alyx::blockSegScan<blockSize>(val, flag, alyx::AlyxBinaryOp::SegCopy<T>{});
    }

    if (gtid < numElement) b[gtid] = res.first;
}

template <typename T, OpType opType>
class Test {
public:
    void run() {
        CUDA_CHECK(cudaSetDevice(0));

        constexpr std::size_t gridSize = 2;
        constexpr std::size_t blockSize = 128;
        std::size_t numElement{200};
        std::vector<T> ah(numElement);

        if constexpr (opType == OpType::Add) {
            for (std::size_t i = 0; i < ah.size(); ++i) {
                ah[i] = static_cast<T>(1);
            }
        } else if constexpr (opType == OpType::Multiply) {
            for (std::size_t i = 0; i < ah.size(); ++i) {
                ah[i] = static_cast<T>(2);
            }
        } else if constexpr (opType == OpType::Copy) {
            for (std::size_t i = 0; i < ah.size(); ++i) {
                ah[i] = static_cast<T>(i + 1);
            }
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

        blockSegScanKernel<blockSize, T, opType>
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
        Test<int, OpType::Add> t;
        t.run();
    }

    {
        Test<int, OpType::Multiply> t;
        t.run();
    }

    {
        Test<double, OpType::Copy> t;
        t.run();
    }

    return 0;
}