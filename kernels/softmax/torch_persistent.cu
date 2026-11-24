// Thin binding to pinned PyTorch v2.7.1 PersistentSoftmax.cuh (BSD notice in provenance).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <ATen/ops/softmax.h>
#include "PersistentSoftmax.cuh"

torch::Tensor stable_softmax(torch::Tensor input) {
  TORCH_CHECK(input.is_cuda() && input.dim() == 2 && input.is_contiguous(),
              "CUDA contiguous rank-2 input required");
  TORCH_CHECK(input.scalar_type() == at::kHalf || input.scalar_type() == at::kFloat,
              "FP16 or FP32 input required");
  TORCH_CHECK(input.size(0) > 0 && input.size(0) <= 65536 && input.size(1) > 0 &&
              input.size(1) <= 16384, "shape outside supported contract");
  const c10::cuda::CUDAGuard guard(input.device());
  if (input.size(1) > 2048) return at::softmax(input, -1, at::kFloat);
  auto output = torch::empty(input.sizes(), input.options().dtype(at::kFloat));
  const int width = static_cast<int>(input.size(1));
  const int rows = static_cast<int>(input.size(0));
  if (input.scalar_type() == at::kHalf)
    dispatch_softmax_forward<at::Half, float, float, false, false>(
        output.data_ptr<float>(), input.data_ptr<at::Half>(), width, width, rows);
  else
    dispatch_softmax_forward<float, float, float, false, false>(
        output.data_ptr<float>(), input.data_ptr<float>(), width, width, rows);
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {m.def("softmax", &stable_softmax);}
