add_library(smoke_torch
  gpu_add.cpp
)

target_include_directories(smoke_torch
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

target_compile_features(smoke_torch PUBLIC cxx_std_17)

if(TARGET Torch::Torch)
  target_link_libraries(smoke_torch PUBLIC Torch::Torch)
else()
  target_link_libraries(smoke_torch PUBLIC ${TORCH_LIBRARIES})
endif()

if(DEFINED TORCH_CXX_FLAGS_LIST)
  target_compile_options(smoke_torch PUBLIC ${TORCH_CXX_FLAGS_LIST})
endif()

if(DEFINED Torch_DIR)
  get_filename_component(_TORCH_PREFIX "${Torch_DIR}/../../.." ABSOLUTE)
  if(EXISTS "${_TORCH_PREFIX}/lib")
    set_property(TARGET smoke_torch PROPERTY BUILD_RPATH "${_TORCH_PREFIX}/lib")
  endif()
endif()
