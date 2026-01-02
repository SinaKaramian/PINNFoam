# PINNFOAM-GPU (Fork)

This repository is a fork of an existing **PINNFOAM** project. The upstream codebase provided a baseline implementation for solving PDE-constrained problems using neural-network-based approaches (e.g., PINNs and related loss formulations).

This fork focuses on **scalability, GPU-native performance, and large-problem practicality**. The key direction is to move from a research-oriented prototype to a **GPU-first, CMake-based, large-scale solver framework** that can handle significantly larger meshes and workloads with fewer host↔device transfers.

---

## Objectives (Roadmap)

This fork is being extended in the following steps:

1. **GPU + CMake build system**
   - Replace legacy build workflows (e.g., `wmake`) with a modern **CMake** toolchain.
   - Ensure the core pipeline runs on **GPU** (CUDA-capable systems), including a clean build + test story.

2. **Discrete numerical method for PDE loss (vs. AD-based PDE residuals)**
   - Replace PDE-loss computation based on automatic differentiation (AD) through the NN with a **discrete numerical formulation** (e.g., finite volume  discrete operator-based residuals).
   - Motivation:
     - Better alignment with standard CFD/engineering discretizations.
     - Improved control of numerical properties and stability.
     - Potentially lower overhead and better scaling for large problems.

3. **AMGx + cuSPARSE integration for very large problems**
   - Introduce GPU-accelerated sparse linear algebra pathways using:
     - **NVIDIA AMGx** for multigrid preconditioning / iterative solves
     - **cuSPARSE** for sparse matrix/vector operations
   - Goal: make training/optimization feasible for **very large systems** (mesh sizes and operator sizes that are not practical with naive approaches).

4. **Native GPU solver to reduce device↔host transfers**
   - Implement a **custom, user-defined GPU-native solver** to improve end-to-end performance.
   - Primary performance target:
     - Reduce data movement between device and host (often the dominant bottleneck).
     - Keep operators, residual evaluation, and solver steps on-device as much as possible.

> Note: This roadmap describes the planned direction. Individual items may be developed behind feature flags and merged incrementally.

---
## Build (CMake + CUDA LibTorch)

### Prerequisites
- CUDA-capable NVIDIA GPU and a working NVIDIA driver
- CUDA Toolkit installed (must provide `include/cuda_runtime.h` and `lib*/libcudart.so`)
- CUDA-enabled LibTorch distribution (not CPU-only)
- CMake 3.20+ and a C++17-capable compiler

### 1) Make CUDA available in your terminal session

### 2) Configure (use `g++`)

Replace the placeholders with paths on your system:

- `Torch_DIR` must point to: `<LIBTORCH_ROOT>/share/cmake/Torch`
- `CUDAToolkit_ROOT` / `CUDA_TOOLKIT_ROOT_DIR` must point to your CUDA Toolkit root

This project supports a build-time precision switch via `PINN_TORCH_FP32`:

- `PINN_TORCH_FP32=1` → FP32 (float)
- `PINN_TORCH_FP32=0` → FP64 (double)

#### FP32 configure

```bash
cmake -S . -B build-fp32 \
  -DCMAKE_CXX_COMPILER=g++ \
  -DTorch_DIR=<LIBTORCH_ROOT>/share/cmake/Torch \
  -DCUDAToolkit_ROOT=<CUDA_TOOLKIT_ROOT> \
  -DCUDA_TOOLKIT_ROOT_DIR=<CUDA_TOOLKIT_ROOT> \
  -DCMAKE_BUILD_TYPE=Release \
  -DPINN_TORCH_FP32=1
```

#### FP64 configure

```bash
cmake -S . -B build-fp64 \
  -DCMAKE_CXX_COMPILER=g++ \
  -DTorch_DIR=<LIBTORCH_ROOT>/share/cmake/Torch \
  -DCUDAToolkit_ROOT=<CUDA_TOOLKIT_ROOT> \
  -DCUDA_TOOLKIT_ROOT_DIR=<CUDA_TOOLKIT_ROOT> \
  -DCMAKE_BUILD_TYPE=Release \
  -DPINN_TORCH_FP32=0
```

### 3) Build

```bash
cmake --build build-fp32 -j   # for FP32 build
cmake --build build-fp64 -j   # for FP64 build
```

### 4) Run all tests (pass/fail)

```bash
ctest --test-dir build-fp32 --output-on-failure   # FP32
ctest --test-dir build-fp64 --output-on-failure   # FP64
```

