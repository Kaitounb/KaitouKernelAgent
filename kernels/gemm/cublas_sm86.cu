#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <cublas_v2.h>

#include "gemm_contract.cuh"

torch::Tensor gemm_cublas(const torch::Tensor& a, const torch::Tensor& b) {
    check_gemm_contract(a, b);
    const auto m = a.size(0);
    const auto n = b.size(1);
    const auto k = a.size(1);

    const c10::cuda::CUDAGuard guard(a.device());
    auto output = torch::empty({m, n}, a.options().dtype(at::kFloat));
    auto handle = at::cuda::getCurrentCUDABlasHandle();
    cublasPointerMode_t pointer_mode;
    TORCH_CUDABLAS_CHECK(cublasGetPointerMode(handle, &pointer_mode));
    TORCH_CHECK(pointer_mode == CUBLAS_POINTER_MODE_HOST, "Expected cuBLAS host pointer mode");
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Row-major C is column-major C^T; compute B^T * A^T in that layout.
    TORCH_CUDABLAS_CHECK(cublasGemmEx(
        handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha,
        b.data_ptr<at::Half>(), CUDA_R_16F, n,
        a.data_ptr<at::Half>(), CUDA_R_16F, k,
        &beta, output.data_ptr<float>(), CUDA_R_32F, n,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    return output;
}
