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

# ---- Torch compatibility target (works whether Torch::Torch exists or not) ----
if(NOT TARGET torch::torch_compat)
  add_library(torch_torch_compat INTERFACE)
  add_library(torch::torch_compat ALIAS torch_torch_compat)

  # Prefer imported target if available; otherwise fall back to variables
  if(TARGET Torch::Torch)
    target_link_libraries(torch_torch_compat INTERFACE Torch::Torch)
  else()
    if(DEFINED TORCH_INCLUDE_DIRS)
      target_include_directories(torch_torch_compat INTERFACE ${TORCH_INCLUDE_DIRS})
    endif()
    if(DEFINED TORCH_LIBRARIES)
      target_link_libraries(torch_torch_compat INTERFACE ${TORCH_LIBRARIES})
    else()
      message(FATAL_ERROR "Torch was found, but TORCH_LIBRARIES is not set and Torch::Torch target is missing.")
    endif()
  endif()

  # Torch sometimes provides TORCH_CXX_FLAGS as a single string; convert/apply
  if(DEFINED TORCH_CXX_FLAGS AND NOT "${TORCH_CXX_FLAGS}" STREQUAL "")
    separate_arguments(_TORCH_CXX_FLAGS_LIST NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
    target_compile_options(torch_torch_compat INTERFACE ${_TORCH_CXX_FLAGS_LIST})
  endif()

  target_compile_features(torch_torch_compat INTERFACE cxx_std_17)
endif()

# ---- Help runtime find libtorch.so on Linux/macOS (optional but helpful) ----
# Infer <libtorch>/lib from Torch_DIR if possible.
if(DEFINED Torch_DIR)
  get_filename_component(_TORCH_PREFIX "${Torch_DIR}/../../.." ABSOLUTE)
  if(EXISTS "${_TORCH_PREFIX}/lib")
    set(CMAKE_BUILD_RPATH "${_TORCH_PREFIX}/lib")
    set(CMAKE_INSTALL_RPATH "${_TORCH_PREFIX}/lib")
  endif()
endif()

# ---- Subdirs ----
add_subdirectory(src)
add_subdirectory(apps)

include(CTest)
if(BUILD_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif()

