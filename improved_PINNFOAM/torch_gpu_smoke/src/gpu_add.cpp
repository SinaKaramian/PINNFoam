#include "smoke/gpu_add.hpp"
#include <stdexcept>
#include <sstream>

namespace smoke {

torch::Tensor add_on_cuda(float a, float b) {
  if (!torch::cuda::is_available()) {
    std::ostringstream oss;
    oss << "CUDA is NOT available to LibTorch. "
        << "device_count=" << torch::cuda::device_count();
    throw std::runtime_error(oss.str());
  }

  auto opts = torch::TensorOptions()
                .dtype(torch::kFloat32)
                .device(torch::kCUDA);

  auto t1 = torch::tensor({a}, opts);
  auto t2 = torch::tensor({b}, opts);
  auto out = t1 + t2;

  if (!out.is_cuda()) {
    throw std::runtime_error("Internal error: result tensor is not CUDA.");
  }
  return out;
}

} // namespace smoke
