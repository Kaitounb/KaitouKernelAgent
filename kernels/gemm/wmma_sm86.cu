/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

// Adapted from CUDA Samples v13.0 simple_wmma_gemm, commit
// 3f1c50965017932fc81e6d94a3fc9e04c105b312. See provenance/adaptations/.
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cstdint>

namespace wmma = nvcuda::wmma;

template<int WARPS>
__global__ void gemm_wmma(const __half* a, const __half* b, float* out,
                          int m, int n, int k) {
    const int warp_m = blockIdx.x * WARPS + threadIdx.x / 32;
    const int warp_n = blockIdx.y;
    if (warp_m * 16 >= m) return; // warp-uniform; no block synchronization
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> af;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> bf;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    wmma::fill_fragment(acc, 0.0f);
    for (int i = 0; i < k; i += 16) {
        wmma::load_matrix_sync(af, a + static_cast<int64_t>(warp_m * 16) * k + i, k);
        wmma::load_matrix_sync(bf, b + static_cast<int64_t>(i) * n + warp_n * 16, n);
        wmma::mma_sync(acc, af, bf, acc);
    }
    wmma::store_matrix_sync(out + static_cast<int64_t>(warp_m * 16) * n + warp_n * 16,
                           acc, n, wmma::mem_row_major);
}

torch::Tensor gemm_template(const torch::Tensor& a, const torch::Tensor& b, int64_t warps) {
    check_gemm_contract(a, b);
    TORCH_CHECK(warps == 1 || warps == 2 || warps == 4 || warps == 8, "Expected warps 1/2/4/8");
    const int m = a.size(0), n = b.size(1), k = a.size(1);
    const auto ap = reinterpret_cast<const __half*>(a.data_ptr<at::Half>());
    const auto bp = reinterpret_cast<const __half*>(b.data_ptr<at::Half>());
    if (m % 16 || n % 16 || k % 16 || reinterpret_cast<uintptr_t>(ap) % 32 ||
        reinterpret_cast<uintptr_t>(bp) % 32) return gemm_cublas(a, b);
    const c10::cuda::CUDAGuard guard(a.device());
    auto out = torch::empty({m, n}, a.options().dtype(at::kFloat));
    const auto stream = c10::cuda::getCurrentCUDAStream(a.get_device());
    const dim3 grid((m + 16 * warps - 1) / (16 * warps), n / 16);
#define LAUNCH(W) gemm_wmma<W><<<grid, 32 * W, 0, stream.stream()>>>(ap, bp, out.data_ptr<float>(), m, n, k)
    switch (warps) {
        case 1: LAUNCH(1); break;
        case 2: LAUNCH(2); break;
        case 4: LAUNCH(4); break;
        case 8: LAUNCH(8); break;
    }
#undef LAUNCH
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
