# OpenFOAM + CMake demo (skeleton)

This project is generated as a starting point to build a small OpenFOAM-based application with CMake.

## Expected workflow

1) Source OpenFOAM:
   source <OpenFOAM>/etc/bashrc

2) Configure:
   cmake -S . -B build

   If you cannot or do not want to source OpenFOAM, you may pass hints:
   cmake -S . -B build -DOpenFOAM_ROOT=/path/to/openfoam -DOpenFOAM_LIBDIR=/path/to/platforms/<WM_OPTIONS>/lib

3) Build:
   cmake --build build -j

Notes:
- The generated FindOpenFOAM.cmake creates imported targets:
  OpenFOAM::OpenFOAM, OpenFOAM::meshTools, OpenFOAM::finiteVolume
- The example code is in src/main.C
