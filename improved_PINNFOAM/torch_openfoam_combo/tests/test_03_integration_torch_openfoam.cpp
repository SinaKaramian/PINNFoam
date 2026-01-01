#include <gtest/gtest.h>
#include <torch/torch.h>
#include "fvCFD.H"

TEST(Integration, TorchCudaScalarToOpenFOAMScalar)
{
  ASSERT_TRUE(torch::cuda::is_available())
    << "CUDA must be available for this integration test. device_count=" << torch::cuda::device_count();

  const float input = 42.125f;

  auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
  torch::Tensor t = torch::full({}, input, opts);
  ASSERT_TRUE(t.is_cuda());

  float host = t.to(torch::kCPU).item<float>();
  Foam::scalar foamVal = static_cast<Foam::scalar>(host);

  EXPECT_NEAR(static_cast<double>(host), static_cast<double>(input), 1e-6);
  EXPECT_NEAR(static_cast<double>(foamVal), static_cast<double>(input), 1e-6);
}
