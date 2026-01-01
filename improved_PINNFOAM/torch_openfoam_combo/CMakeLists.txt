cmake_minimum_required(VERSION 3.20)

project(torch_openfoam_combo LANGUAGES CXX)

option(BUILD_TESTING "Build tests" ON)

# Predictable output folders
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Use modern target-based style, keep standard consistent
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Make CMake find our custom FindOpenFOAM.cmake
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules")

# ---- Dependencies ----
find_package(Torch REQUIRED)
find_package(OpenFOAM REQUIRED)

# Torch sometimes provides TORCH_CXX_FLAGS as a single string; convert to list
if(DEFINED TORCH_CXX_FLAGS AND NOT "${TORCH_CXX_FLAGS}" STREQUAL "")
  separate_arguments(TORCH_CXX_FLAGS_LIST NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
endif()

# ---- Subdirs ----
add_subdirectory(src)
add_subdirectory(apps)

include(CTest)
if(BUILD_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif()
