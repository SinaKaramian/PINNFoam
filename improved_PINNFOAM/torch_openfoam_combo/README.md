## torch_openfoam_combo

### Requirements
- OpenFOAM environment sourced (WM_PROJECT_DIR, FOAM_SRC, FOAM_LIBBIN, etc.)
- LibTorch (C++ distribution)
- CMake >= 3.20
- C++17 compiler

### Configure example
```bash
source /path/to/OpenFOAM/etc/bashrc

cmake -S . -B build \
  -DCMAKE_PREFIX_PATH=/abs/path/to/libtorch \
  -DBUILD_TESTING=ON
