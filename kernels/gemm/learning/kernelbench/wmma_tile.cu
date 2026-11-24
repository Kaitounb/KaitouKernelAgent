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

// Teaching adaptation of NVIDIA CUDA Samples v13.0 simple_wmma_gemm.
// Commit 3f1c50965017932fc81e6d94a3fc9e04c105b312.
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>
#include <mma.h>

#include <cstdint>

namespace wmma = nvcuda::wmma;
constexpr int TILE = 16;

__global__ void wmma_tile_kernel(const __half* a, const __half* b, float* out) {
    wmma::fragment<wmma::matrix_a, TILE, TILE, TILE, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, TILE, TILE, TILE, __half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, TILE, TILE, TILE, float> acc_frag;

    // 一个完整 warp 一起装载、计算和存储。
    wmma::fill_fragment(acc_frag, 0.0f);
    wmma::load_matrix_sync(a_frag, a, TILE);
    wmma::load_matrix_sync(b_frag, b, TILE);
    wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    wmma::store_matrix_sync(out, acc_frag, TILE, wmma::mem_row_major);
}

torch::Tensor wmma_tile_cuda(const torch::Tensor& a, const torch::Tensor& b) {
    TORCH_CHECK(a.is_cuda() && a.device() == b.device(), "Expected inputs on the same CUDA device");
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous(), "Expected contiguous inputs");
    TORCH_CHECK(a.scalar_type() == at::kHalf && b.scalar_type() == at::kHalf, "Expected FP16 inputs");
    TORCH_CHECK(a.dim() == 2 && a.size(0) == TILE && a.size(1) == TILE && a.sizes() == b.sizes(),
                "Expected two 16x16 inputs");

    auto a_ptr = reinterpret_cast<const __half*>(a.data_ptr<at::Half>());
    auto b_ptr = reinterpret_cast<const __half*>(b.data_ptr<at::Half>());
    TORCH_CHECK(reinterpret_cast<std::uintptr_t>(a_ptr) % 32 == 0 &&
                    reinterpret_cast<std::uintptr_t>(b_ptr) % 32 == 0,
                "WMMA input addresses must be 32-byte aligned");

    const c10::cuda::CUDAGuard guard(a.device());
    auto out = torch::empty({TILE, TILE}, a.options().dtype(at::kFloat));
    auto stream = c10::cuda::getCurrentCUDAStream(a.get_device());
    wmma_tile_kernel<<<1, 32, 0, stream.stream()>>>(a_ptr, b_ptr, out.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
