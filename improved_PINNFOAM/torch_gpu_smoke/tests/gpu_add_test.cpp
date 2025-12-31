#include <gtest/gtest.h>
#include "smoke/gpu_add.hpp"

TEST(GPUAdd, RunsOnCudaAndComputesCorrectly) {
  ASSERT_TRUE(torch::cuda::is_available())
      << "CUDA must be available for this smoke test. "
      << "device_count=" << torch::cuda::device_count();

  auto out = smoke::add_on_cuda(10.0f, 7.0f);
  EXPECT_TRUE(out.is_cuda());

  float v = out.to(torch::kCPU).item<float>();
  EXPECT_EQ(v, 17.0f);
}
