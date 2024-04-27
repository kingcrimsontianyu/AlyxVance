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

endif()