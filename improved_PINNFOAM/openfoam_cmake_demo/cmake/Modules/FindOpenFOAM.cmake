# cmake/Modules/FindOpenFOAM.cmake
#
# Usage (MODULE mode):
#   list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules")
#   find_package(OpenFOAM REQUIRED)
#
# Targets provided:
#   OpenFOAM::compileOptions  (ALIAS of OpenFOAM_compileOptions)
#   OpenFOAM::OpenFOAM        (IMPORTED)
#   OpenFOAM::meshTools       (IMPORTED)
#   OpenFOAM::finiteVolume    (IMPORTED)

cmake_minimum_required(VERSION 3.20)

include(FindPackageHandleStandardArgs)

# -------------------------------------------------------------------------
# User hints (optional). No hardcoded paths.
# -------------------------------------------------------------------------
set(OpenFOAM_ROOT        "" CACHE PATH "OpenFOAM project dir (WM_PROJECT_DIR)")
set(OpenFOAM_SRC         "" CACHE PATH "OpenFOAM src dir (FOAM_SRC)")
set(OpenFOAM_LIBDIR      "" CACHE PATH "OpenFOAM lib dir (FOAM_LIBBIN)")
set(OpenFOAM_USER_LIBDIR "" CACHE PATH "OpenFOAM user lib dir (FOAM_USER_LIBBIN)")

# -------------------------------------------------------------------------
# Discover root/src/libdir from environment (preferred) or cache hints
# -------------------------------------------------------------------------
set(_of_root "")
if(OpenFOAM_ROOT)
  set(_of_root "${OpenFOAM_ROOT}")
elseif(DEFINED ENV{WM_PROJECT_DIR})
  set(_of_root "$ENV{WM_PROJECT_DIR}")
endif()

set(_of_src "")
if(OpenFOAM_SRC)
  set(_of_src "${OpenFOAM_SRC}")
elseif(DEFINED ENV{FOAM_SRC})
  set(_of_src "$ENV{FOAM_SRC}")
elseif(_of_root)
  set(_of_src "${_of_root}/src")
endif()

set(_of_libdir "")
if(OpenFOAM_LIBDIR)
  set(_of_libdir "${OpenFOAM_LIBDIR}")
elseif(DEFINED ENV{FOAM_LIBBIN})
  set(_of_libdir "$ENV{FOAM_LIBBIN}")
elseif(_of_root AND DEFINED ENV{WM_OPTIONS})
  set(_of_libdir "${_of_root}/platforms/$ENV{WM_OPTIONS}/lib")
endif()

set(_of_user_libdir "")
if(OpenFOAM_USER_LIBDIR)
  set(_of_user_libdir "${OpenFOAM_USER_LIBDIR}")
elseif(DEFINED ENV{FOAM_USER_LIBBIN})
  set(_of_user_libdir "$ENV{FOAM_USER_LIBBIN}")
endif()

# -------------------------------------------------------------------------
# Validate headers (fvCFD.H is a strong signal for finiteVolume)
# -------------------------------------------------------------------------
find_path(OpenFOAM_FV_INCLUDE_DIR
  NAMES fvCFD.H
  HINTS "${_of_src}/finiteVolume/lnInclude"
  NO_DEFAULT_PATH
)

# Include dirs used by many solvers
set(OpenFOAM_INCLUDE_DIRS
  "${_of_src}/finiteVolume/lnInclude"
  "${_of_src}/meshTools/lnInclude"
  "${_of_src}/OpenFOAM/lnInclude"
  "${_of_src}/OSspecific/POSIX/lnInclude"
)

# -------------------------------------------------------------------------
# Libraries (search FOAM_LIBBIN first, then FOAM_USER_LIBBIN)
# -------------------------------------------------------------------------
set(_of_lib_hints "")
if(_of_libdir)
  list(APPEND _of_lib_hints "${_of_libdir}")
endif()
if(_of_user_libdir)
  list(APPEND _of_lib_hints "${_of_user_libdir}")
endif()

find_library(OpenFOAM_OpenFOAM_LIBRARY
  NAMES OpenFOAM
  HINTS ${_of_lib_hints}
  NO_DEFAULT_PATH
)

find_library(OpenFOAM_meshTools_LIBRARY
  NAMES meshTools
  HINTS ${_of_lib_hints}
  NO_DEFAULT_PATH
)

find_library(OpenFOAM_finiteVolume_LIBRARY
  NAMES finiteVolume
  HINTS ${_of_lib_hints}
  NO_DEFAULT_PATH
)

# -------------------------------------------------------------------------
# Definitions: derived from OpenFOAM environment when available (no hardcoding)
# -------------------------------------------------------------------------
set(OpenFOAM_DEFINITIONS "")

# OPENFOAM=#### derived from WM_PROJECT_VERSION if it contains digits
if(DEFINED ENV{WM_PROJECT_VERSION})
  string(REGEX MATCH "([0-9]+)" _of_ver_match "$ENV{WM_PROJECT_VERSION}")
  if(_of_ver_match)
    list(APPEND OpenFOAM_DEFINITIONS "OPENFOAM=${CMAKE_MATCH_1}")
  endif()
endif()

# Precision macro
if(DEFINED ENV{WM_PRECISION_OPTION})
  if("$ENV{WM_PRECISION_OPTION}" STREQUAL "DP")
    list(APPEND OpenFOAM_DEFINITIONS "WM_DP")
  elseif("$ENV{WM_PRECISION_OPTION}" STREQUAL "SP")
    list(APPEND OpenFOAM_DEFINITIONS "WM_SP")
  endif()
endif()

# Label size macro
if(DEFINED ENV{WM_LABEL_SIZE})
  list(APPEND OpenFOAM_DEFINITIONS "WM_LABEL_SIZE=$ENV{WM_LABEL_SIZE}")
endif()

# NoRepository: enable only if you choose to (default ON, but user-controllable)
set(OpenFOAM_USE_NOREPOSITORY ON CACHE BOOL "Define NoRepository (matches common wmake user-app behavior)")
if(OpenFOAM_USE_NOREPOSITORY)
  list(APPEND OpenFOAM_DEFINITIONS "NoRepository")
endif()

# -------------------------------------------------------------------------
# Compile/link flags: take them from the OpenFOAM environment if present
# (no hardcoded warning/optimization flags here).
# -------------------------------------------------------------------------
set(OpenFOAM_COMPILE_OPTIONS "")
if(DEFINED ENV{WM_CXXFLAGS} AND NOT "$ENV{WM_CXXFLAGS}" STREQUAL "")
  set(_of_cxxflags "$ENV{WM_CXXFLAGS}")
  separate_arguments(_of_cxxflags UNIX_COMMAND "${_of_cxxflags}")
  list(APPEND OpenFOAM_COMPILE_OPTIONS ${_of_cxxflags})
endif()

set(OpenFOAM_LINK_OPTIONS "")
if(DEFINED ENV{WM_LDFLAGS} AND NOT "$ENV{WM_LDFLAGS}" STREQUAL "")
  set(_of_ldflags "$ENV{WM_LDFLAGS}")
  separate_arguments(_of_ldflags UNIX_COMMAND "${_of_ldflags}")
  # Put these into link options (not libraries)
  list(APPEND OpenFOAM_LINK_OPTIONS ${_of_ldflags})
endif()

# -------------------------------------------------------------------------
# Standard handling
# -------------------------------------------------------------------------
find_package_handle_standard_args(OpenFOAM
  REQUIRED_VARS
    _of_root
    _of_src
    _of_libdir
    OpenFOAM_FV_INCLUDE_DIR
    OpenFOAM_OpenFOAM_LIBRARY
    OpenFOAM_meshTools_LIBRARY
    OpenFOAM_finiteVolume_LIBRARY
)

# Legacy variables (optional)
set(OpenFOAM_LIBRARY_DIR "${_of_libdir}")
set(OpenFOAM_LIBRARIES
  "${OpenFOAM_finiteVolume_LIBRARY}"
  "${OpenFOAM_meshTools_LIBRARY}"
  "${OpenFOAM_OpenFOAM_LIBRARY}"
)

# -------------------------------------------------------------------------
# Modern targets
# -------------------------------------------------------------------------
if(OpenFOAM_FOUND)

  # Real INTERFACE target (no '::'), then ALIAS it to the namespaced name.
  if(NOT TARGET OpenFOAM_compileOptions)
    add_library(OpenFOAM_compileOptions INTERFACE)
    add_library(OpenFOAM::compileOptions ALIAS OpenFOAM_compileOptions)

    target_include_directories(OpenFOAM_compileOptions INTERFACE ${OpenFOAM_INCLUDE_DIRS})

    if(OpenFOAM_DEFINITIONS)
      target_compile_definitions(OpenFOAM_compileOptions INTERFACE ${OpenFOAM_DEFINITIONS})
    endif()

    if(OpenFOAM_COMPILE_OPTIONS)
      target_compile_options(OpenFOAM_compileOptions INTERFACE ${OpenFOAM_COMPILE_OPTIONS})
    endif()

    if(OpenFOAM_LINK_OPTIONS)
      target_link_options(OpenFOAM_compileOptions INTERFACE ${OpenFOAM_LINK_OPTIONS})
    endif()

    target_compile_features(OpenFOAM_compileOptions INTERFACE cxx_std_17)
  endif()

  # Imported OpenFOAM core library
  if(NOT TARGET OpenFOAM::OpenFOAM)
    add_library(OpenFOAM::OpenFOAM UNKNOWN IMPORTED)
    set_target_properties(OpenFOAM::OpenFOAM PROPERTIES
      IMPORTED_LOCATION "${OpenFOAM_OpenFOAM_LIBRARY}"
    )

    # Use portable dl library variable on platforms that need it
    target_link_libraries(OpenFOAM::OpenFOAM
      INTERFACE
        OpenFOAM::compileOptions
        ${CMAKE_DL_LIBS}
    )

    # libm is typically required on Linux; harmless if not needed elsewhere
    if(UNIX AND NOT APPLE)
      target_link_libraries(OpenFOAM::OpenFOAM INTERFACE m)
    endif()
  endif()

  # meshTools depends on OpenFOAM
  if(NOT TARGET OpenFOAM::meshTools)
    add_library(OpenFOAM::meshTools UNKNOWN IMPORTED)
    set_target_properties(OpenFOAM::meshTools PROPERTIES
      IMPORTED_LOCATION "${OpenFOAM_meshTools_LIBRARY}"
    )
    target_link_libraries(OpenFOAM::meshTools INTERFACE OpenFOAM::OpenFOAM)
  endif()

  # finiteVolume depends on meshTools + OpenFOAM
  if(NOT TARGET OpenFOAM::finiteVolume)
    add_library(OpenFOAM::finiteVolume UNKNOWN IMPORTED)
    set_target_properties(OpenFOAM::finiteVolume PROPERTIES
      IMPORTED_LOCATION "${OpenFOAM_finiteVolume_LIBRARY}"
    )
    target_link_libraries(OpenFOAM::finiteVolume INTERFACE OpenFOAM::meshTools OpenFOAM::OpenFOAM)
  endif()

endif()

