cmake_minimum_required(VERSION 3.20)
project(openfoam_cmake_demo LANGUAGES CXX)

# Make CMake find our custom FindOpenFOAM.cmake
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules")

# Find OpenFOAM via our module (requires OpenFOAM env sourced, or pass OpenFOAM_ROOT/OpenFOAM_LIBDIR)
find_package(OpenFOAM REQUIRED)

add_executable(of_app
  src/main.C
)

# Optional: mimic common wmake include behavior for local headers (if you add any)
target_include_directories(of_app PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}"
  "${CMAKE_CURRENT_SOURCE_DIR}/lnInclude"
)

# Link to OpenFOAM (this propagates include dirs, defs, options through targets)
target_link_libraries(of_app PRIVATE
  OpenFOAM::finiteVolume
)

# Optional runtime convenience: embed OpenFOAM lib path into the binary rpath
# If you always run inside a sourced OpenFOAM shell, you can remove this.
if(DEFINED OpenFOAM_LIBRARY_DIR AND OpenFOAM_LIBRARY_DIR)
  set_target_properties(of_app PROPERTIES
    BUILD_RPATH "${OpenFOAM_LIBRARY_DIR}"
  )
endif()
