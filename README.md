# AlyxVance (WIP)

Tianyu Liu 2024

------------------------

[AlyxVance](https://half-life.fandom.com/wiki/Alyx_Vance) is a collection of block-wide parallel algorithms implemented in CUDA, intentionally not using much of the [CCCL](https://github.com/NVIDIA/cccl) library. AlyxVance is based on C++20 and CUDA 12 or newer.

This is a side project just for fun, learning, and exploration, through deliberately "reinventing the wheels". Do not use it in production code. Use CCCL instead.

## Currently implemented algorithms

+ [Reduce](include/block_reduce.hpp): Support predefined and user-defined binary operators

+ [Generic scan](include/block_scan.hpp): Support predefined and user-defined binary operators

+ [Generic segmented scan](include/block_seg_scan.hpp): Support predefined and user-defined binary operators

+ [Mergesort](include/block_mergesort.hpp): Support predefined and user-defined comparators

+ [Ranksort](include/block_ranksort.hpp): Support predefined and user-defined comparators
