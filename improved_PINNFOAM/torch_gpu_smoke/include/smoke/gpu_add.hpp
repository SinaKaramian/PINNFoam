#pragma once
#include <torch/torch.h>

namespace smoke {

// Adds two scalar floats on CUDA and returns a CUDA tensor.
// Throws std::runtime_error if CUDA is not available.
torch::Tensor add_on_cuda(float a, float b);

} // namespace smoke
