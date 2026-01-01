#include "smoke/gpu_add.hpp"

namespace smoke {

torch::Tensor make_cuda_scalar(float v) {
  auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
  return torch::full({}, v, opts);
}

torch::Tensor add_on_cuda(float a, float b) {
  auto ta = make_cuda_scalar(a);
  auto tb = make_cuda_scalar(b);
  return ta + tb;
}

} // namespace smoke
