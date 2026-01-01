add_executable(of_app
  main.C
)

target_link_libraries(of_app PRIVATE OpenFOAM::finiteVolume)

if(DEFINED OpenFOAM_LIBRARY_DIR AND OpenFOAM_LIBRARY_DIR)
  set_target_properties(of_app PROPERTIES
    BUILD_RPATH "${OpenFOAM_LIBRARY_DIR}"
  )
endif()
