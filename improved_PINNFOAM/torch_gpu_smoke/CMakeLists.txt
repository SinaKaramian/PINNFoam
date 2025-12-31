cmake_minimum_required(VERSION 3.20)

project(torch_gpu_smoke LANGUAGES CXX)

option(BUILD_TESTING "Build tests" ON)

# Predictable output folders
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# ---- LibTorch ----
# Provide at configure time:
#   -DCMAKE_PREFIX_PATH=/abs/path/to/libtorch
# or
#   -DTorch_DIR=/abs/path/to/libtorch/share/cmake/Torch
find_package(Torch REQUIRED)

# Torch sometimes provides TORCH_CXX_FLAGS as a single string; convert to list
if(DEFINED TORCH_CXX_FLAGS)
  separate_arguments(TORCH_CXX_FLAGS_LIST NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
endif()

# ---- Your reusable library ----
add_library(smoke_lib
  src/gpu_add.cpp
)

target_include_directories(smoke_lib
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_features(smoke_lib PUBLIC cxx_std_17)

# Prefer imported target if available; otherwise fall back
if(TARGET Torch::Torch)
  target_link_libraries(smoke_lib PUBLIC Torch::Torch)
else()
  target_link_libraries(smoke_lib PUBLIC ${TORCH_LIBRARIES})
endif()

if(DEFINED TORCH_CXX_FLAGS_LIST)
  target_compile_options(smoke_lib PUBLIC ${TORCH_CXX_FLAGS_LIST})
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

# ---- Subprojects ----
add_subdirectory(apps)

if(BUILD_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif()

