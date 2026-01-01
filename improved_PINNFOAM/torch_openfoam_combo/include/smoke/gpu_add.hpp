#pragma once

#include <torch/torch.h>

namespace smoke {

torch::Tensor add_on_cuda(float a, float b);
torch::Tensor make_cuda_scalar(float v);

}  // namespace smoke
