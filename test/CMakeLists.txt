if(BUILD_TESTING)

# reduce
add_executable(alyx-test-reduce.bin test_reduce.cu)
ALYX_CONFIG_GPU_TARGET(alyx-test-reduce.bin)

add_test(NAME alyx_test_reduce_basic
         WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
         COMMAND ./alyx-test-reduce.bin)

add_test(NAME alyx_test_reduce_memcheck
         WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
         COMMAND ${ALYX_CUDA_COMPUTE_SANITIZER} --tool memcheck --leak-check full ./alyx-test-reduce.bin)

# scan
add_executable(alyx-test-scan.bin test_scan.cu)
ALYX_CONFIG_GPU_TARGET(alyx-test-scan.bin)

add_test(NAME alyx_test_scan_basic
         WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
         COMMAND ./alyx-test-scan.bin)

add_test(NAME alyx_test_scan_memcheck
         WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
         COMMAND ${ALYX_CUDA_COMPUTE_SANITIZER} --tool memcheck --leak-check full ./alyx-test-scan.bin)

# segmented scan
add_executable(alyx-test-seg-scan.bin test_seg_scan.cu)
ALYX_CONFIG_GPU_TARGET(alyx-test-seg-scan.bin)

add_test(NAME alyx_test_seg_scan_basic
         WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
         COMMAND ./alyx-test-seg-scan.bin)

add_test(NAME alyx_test_seg_scan_memcheck
         WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
         COMMAND ${ALYX_CUDA_COMPUTE_SANITIZER} --tool memcheck --leak-check full ./alyx-test-seg-scan.bin)

# merge sort
add_executable(alyx-test-mergesort.bin test_mergesort.cu)
ALYX_CONFIG_GPU_TARGET(alyx-test-mergesort.bin)

add_test(NAME alyx_test_mergesort_basic
         WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
         COMMAND ./alyx-test-mergesort.bin)

add_test(NAME alyx_test_mergesort_memcheck
         WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
         COMMAND ${ALYX_CUDA_COMPUTE_SANITIZER} --tool memcheck --leak-check full ./alyx-test-mergesort.bin)
endif()