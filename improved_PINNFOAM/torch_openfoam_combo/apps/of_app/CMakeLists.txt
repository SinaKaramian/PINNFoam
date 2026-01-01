add_executable(of_app
  main.C
)

target_compile_features(of_app PRIVATE cxx_std_17)

# Link OpenFOAM + Torch (Torch headers now always visible via torch::torch_compat)
target_link_libraries(of_app PRIVATE
  OpenFOAM::finiteVolume
  torch::torch_compat
)

# Optional runtime convenience: embed OpenFOAM lib path into the binary rpath
if(DEFINED OpenFOAM_LIBRARY_DIR AND OpenFOAM_LIBRARY_DIR)
  set_target_properties(of_app PROPERTIES
    BUILD_RPATH "${OpenFOAM_LIBRARY_DIR}"
  )
endif()

