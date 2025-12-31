#include <iostream>
#include "smoke/gpu_add.hpp"

int main() {
  std::cout << "LibTorch CUDA available: "
            << (torch::cuda::is_available() ? "YES" : "NO") << "\n";
  std::cout << "CUDA device count: " << torch::cuda::device_count() << "\n";

  try {
    auto out = smoke::add_on_cuda(2.0f, 3.0f);
    float v = out.to(torch::kCPU).item<float>();

    if (v != 5.0f) {
      std::cerr << "FAIL: expected 5.0, got " << v << "\n";
      return 1;
    }

    std::cout << "SUCCESS: CUDA tensor addition works (2 + 3 = " << v << ")\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "FAIL: " << e.what() << "\n";
    return 1;
  }
}
