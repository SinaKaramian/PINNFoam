include(FetchContent)

FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(googletest)

add_executable(torch_gpu_tests
  gpu_add_test.cpp
)

target_link_libraries(torch_gpu_tests
  PRIVATE
    smoke_lib
    GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(torch_gpu_tests)

