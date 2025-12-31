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

