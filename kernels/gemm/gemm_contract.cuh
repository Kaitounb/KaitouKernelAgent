#pragma once
#include <torch/extension.h>

inline void check_gemm_contract(const torch::Tensor& a, const torch::Tensor& b) {
    TORCH_CHECK(a.is_cuda() && a.device() == b.device(), "Expected inputs on the same CUDA device");
    TORCH_CHECK(a.layout() == at::kStrided && b.layout() == at::kStrided &&
                    a.is_contiguous() && b.is_contiguous(),
                "Expected contiguous row-major inputs");
    TORCH_CHECK(a.scalar_type() == at::kHalf && b.scalar_type() == at::kHalf, "Expected FP16 inputs");
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && a.size(1) == b.size(0),
                "Expected A[M,K] and B[K,N]");
    const auto m = a.size(0);
    const auto n = b.size(1);
    const auto k = a.size(1);
    constexpr auto max_dim = 4096;
    TORCH_CHECK(m > 0 && n > 0 && k > 0 && m <= max_dim && n <= max_dim && k <= max_dim,
                "Expected positive dimensions <= 4096");

}
