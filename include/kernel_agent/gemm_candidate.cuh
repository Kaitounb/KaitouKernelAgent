#ifndef KERNEL_AGENT_GEMM_CANDIDATE_CUH
#define KERNEL_AGENT_GEMM_CANDIDATE_CUH

#include "kernel_agent/gemm_problem.h"
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

/* P0-05 LOCAL PROTOTYPE ONLY: not the selected upstream integration ABI.
 * Engineering must first validate the pinned FlashInfer-Bench/KernelBench
 * bindings; this header is retained for the historical design exercise.
 * CUDA compile/load/execute validation is still pending.
 * Query allocates nothing. Launch only enqueues work on the supplied stream.
 * Caller owns all allocations, guards, references, Events and synchronization.
 * Inputs are immutable; all regions must be disjoint. No hidden streams,
 * host/device synchronization, allocation, graph capture or persistent answer cache.
 * Return KA_UNSUPPORTED for an unimplemented valid shape/layout.
 * KA_SUCCESS means enqueued, not numerical validation or device completion.
 */
extern "C" ka_status ka_gemm_workspace_v1(const ka_gemm_problem_v1* problem,
                                         uint64_t* required_bytes);

extern "C" ka_status ka_gemm_launch_v1(const ka_gemm_problem_v1* problem,
                                      const __half* a,
                                      const __half* b,
                                      float* c,
                                      void* workspace,
                                      uint64_t workspace_bytes,
                                      cudaStream_t stream);

#endif
