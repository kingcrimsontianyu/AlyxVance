cmake_minimum_required(VERSION 3.26 FATAL_ERROR)
project(alyxvance LANGUAGES C CXX CUDA)

include(CTest)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(CMAKE_CXX_COMPILER_ID STREQUAL GNU)
    add_compile_options(-fdiagnostics-color=always -m64)
    add_compile_options(-Wuninitialized -Wpedantic -Wextra -Wall -Wshadow)
    add_compile_options(-fvisibility=hidden -fvisibility-inlines-hidden)
endif()

set(ALYX_CUDA_ARCHITECTURES "86-real" CACHE STRING
    "Specify the version of PTX (e.g. 86-virtual) or SASS (e.g. 86-real).")

find_package(CUDAToolkit REQUIRED)

set(ALYX_CUDA_MIN_VERSION "12.0")
if(CUDAToolkit_VERSION VERSION_LESS "${ALYX_CUDA_MIN_VERSION}")
    message(FATAL_ERROR "--> Must upgrade CUDA to ${ALYX_CUDA_MIN_VERSION} or newer.")
endif()
set(ALYX_CUDA_COMPUTE_SANITIZER "${CUDAToolkit_BIN_DIR}/../compute-sanitizer/compute-sanitizer")


function(ALYX_CONFIG_GPU_TARGET myTarget)
    set_property(TARGET ${myTarget}
                PROPERTY CUDA_ARCHITECTURES "${ALYX_CUDA_ARCHITECTURES}")

    target_include_directories(${myTarget} PUBLIC "${PROJECT_SOURCE_DIR}/include")

    target_compile_options(${myTarget}
        # Add -lineinfo to associate SASS with source for Release
        PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v -Xptxas=-warn-spills>
                $<$<AND:$<CONFIG:DEBUG>,$<COMPILE_LANGUAGE:CUDA>>:-g -G>
    )

    if(UNIX AND NOT APPLE)
        target_compile_options(${myTarget}
        PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:-Wno-pedantic>
        )
    endif()
    set_property(TARGET ${myTarget} PROPERTY CUDA_STANDARD 20)
endfunction()

add_subdirectory(example)
add_subdirectory(test)