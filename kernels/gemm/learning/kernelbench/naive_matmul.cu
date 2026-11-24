// Derived from KernelBench (MIT, see LICENSE.kernelbench), commit
// 423217d9fda91e0c2d67e4a43bf62f96f6d104f1, model_new_ex_tiled_matmul.py.
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

inline unsigned int cdiv(unsigned int a, unsigned int b) {
    return (a + b - 1) / b;
}

constexpr int BLOCK_SIZE = 16;

__global__ void naive_matmul_kernel(
    float* out, const float* a, const float* b, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // 每个线程独立计算一个输出，边界线程可以在全局读取前返回。
    if (row >= m || col >= n) {
        return;
    }

    float sum = 0.0f;
    for (int q = 0; q < k; ++q) {
        sum += a[row * k + q] * b[q * n + col];
    }
    out[row * n + col] = sum;
}

torch::Tensor naive_matmul_cuda(const torch::Tensor& a, const torch::Tensor& b) {
    CHECK_INPUT(a);
    CHECK_INPUT(b);
    int m = a.size(0);
    int n = b.size(1);
    int k = a.size(1);
    TORCH_CHECK(k == b.size(0), "Size mismatch");

    auto output = torch::empty({m, n}, a.options());
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks(cdiv(n, threads.x), cdiv(m, threads.y));
    naive_matmul_kernel<<<blocks, threads>>>(
        output.data_ptr<float>(), a.data_ptr<float>(), b.data_ptr<float>(), m, n, k);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
