#ifndef KERNEL_AGENT_GEMM_PROBLEM_H
#define KERNEL_AGENT_GEMM_PROBLEM_H

#include <stdint.h>

#define KA_GEMM_ABI_VERSION 1u

/* P0-05 local metadata/learning prototype, not a frozen engineering ABI.
 * V1: C[M,N] = A[M,K] * B[K,N]; row-major; FP16 A/B, FP32 C.
 * lda/ldb/ldc count elements, not bytes. No transpose/bias/alpha/beta.
 * Header has no CUDA dependency so host contract checks also run on a Mac.
 */
typedef struct ka_gemm_problem_v1 {
    uint32_t abi_version;
    uint32_t struct_bytes;
    int64_t m;
    int64_t n;
    int64_t k;
    int64_t lda;
    int64_t ldb;
    int64_t ldc;
} ka_gemm_problem_v1;

typedef enum ka_status {
    KA_SUCCESS = 0,
    KA_INVALID_ARGUMENT = 1,
    KA_UNSUPPORTED = 2,
    KA_SIZE_OVERFLOW = 3,
    KA_WORKSPACE_TOO_SMALL = 4,
    KA_LAUNCH_ERROR = 5
} ka_status;

#endif
