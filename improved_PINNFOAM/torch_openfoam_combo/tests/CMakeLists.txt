include(FetchContent)

option(USE_SYSTEM_GTEST "Use system-installed GTest instead of FetchContent" OFF)

if(USE_SYSTEM_GTEST)
  find_package(GTest REQUIRED)
else()
  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
  )
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
endif()

function(add_smoke_test exe_name source_file)
  add_executable(${exe_name} ${source_file})
  target_compile_features(${exe_name} PRIVATE cxx_std_17)
  target_link_libraries(${exe_name} PRIVATE GTest::gtest_main)
endfunction()

add_smoke_test(test_01_torch_gpu test_01_torch_gpu.cpp)
target_link_libraries(test_01_torch_gpu PRIVATE smoke_torch)
add_test(NAME 01_torch_gpu COMMAND test_01_torch_gpu)

add_smoke_test(test_02_openfoam_cmake test_02_openfoam_cmake.cpp)
target_link_libraries(test_02_openfoam_cmake PRIVATE OpenFOAM::finiteVolume)
if(DEFINED OpenFOAM_LIBRARY_DIR AND OpenFOAM_LIBRARY_DIR)
  set_target_properties(test_02_openfoam_cmake PROPERTIES BUILD_RPATH "${OpenFOAM_LIBRARY_DIR}")
endif()
add_test(NAME 02_openfoam_cmake COMMAND test_02_openfoam_cmake)

add_smoke_test(test_03_integration test_03_integration_torch_openfoam.cpp)
target_link_libraries(test_03_integration PRIVATE smoke_torch OpenFOAM::finiteVolume)
if(DEFINED OpenFOAM_LIBRARY_DIR AND OpenFOAM_LIBRARY_DIR)
  set_target_properties(test_03_integration PROPERTIES BUILD_RPATH "${OpenFOAM_LIBRARY_DIR}")
endif()
add_test(NAME 03_integration_torch_openfoam COMMAND test_03_integration)

set_tests_properties(02_openfoam_cmake PROPERTIES DEPENDS 01_torch_gpu)
set_tests_properties(03_integration_torch_openfoam PROPERTIES DEPENDS "01_torch_gpu;02_openfoam_cmake")
