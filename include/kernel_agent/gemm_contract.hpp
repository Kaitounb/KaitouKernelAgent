#ifndef KERNEL_AGENT_GEMM_CONTRACT_HPP
#define KERNEL_AGENT_GEMM_CONTRACT_HPP

#include "kernel_agent/gemm_problem.h"

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace kernel_agent {

static_assert(sizeof(ka_gemm_problem_v1) == 56, "Unexpected GEMM ABI layout");
static_assert(offsetof(ka_gemm_problem_v1, m) == 8, "Unexpected GEMM ABI offset");
static_assert(sizeof(ka_status) == 4, "Do not compile with short enums");
static_assert(std::is_standard_layout<ka_gemm_problem_v1>::value, "GEMM POD required");

struct BufferSizes {
    std::uint64_t a_bytes;
    std::uint64_t b_bytes;
    std::uint64_t c_bytes;
};

// Minimum spans, excluding external guards. Leaves sizes untouched on failure.
// This validates metadata/overflow, not pointer allocations or GPU correctness.
ka_status validate_gemm_problem(const ka_gemm_problem_v1& problem,
                                BufferSizes& sizes) noexcept;

}  // namespace kernel_agent
#endif
