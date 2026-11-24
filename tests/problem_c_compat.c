#include "kernel_agent/gemm_problem.h"
#include <stddef.h>

_Static_assert(sizeof(ka_gemm_problem_v1) == 56, "C ABI size");
_Static_assert(offsetof(ka_gemm_problem_v1, m) == 8, "C ABI offset");
_Static_assert(sizeof(ka_status) == 4, "C enum width");

int ka_problem_c_layout_check(void) {
    const ka_gemm_problem_v1 p = {KA_GEMM_ABI_VERSION, sizeof(p), 2, 3, 5, 5, 3, 3};
    return p.abi_version == 1 && p.struct_bytes == 56 && p.lda == 5;
}
