#include "kernel_agent/gemm_contract.hpp"

#include <limits>

namespace kernel_agent {
namespace {

bool span_bytes(std::uint64_t rows, std::uint64_t cols, std::uint64_t stride,
                std::uint64_t element_bytes, std::uint64_t& bytes) noexcept {
    constexpr auto max_u64 = std::numeric_limits<std::uint64_t>::max();
    // (rows-1)*stride+cols is the last row's end, not rows*stride.
    if (rows - 1 > (max_u64 - cols) / stride) {
        return false;
    }
    const auto elements = (rows - 1) * stride + cols;
    const auto max_bytes = static_cast<std::uint64_t>(
        std::numeric_limits<std::size_t>::max());
    if (elements > max_bytes / element_bytes) {
        return false;
    }
    bytes = elements * element_bytes;
    return true;
}

}  // namespace

ka_status validate_gemm_problem(const ka_gemm_problem_v1& p,
                                BufferSizes& sizes) noexcept {
    if (p.abi_version != KA_GEMM_ABI_VERSION || p.struct_bytes != sizeof(p)) {
        return KA_UNSUPPORTED;
    }
    if (p.m <= 0 || p.n <= 0 || p.k <= 0 ||
        p.lda < p.k || p.ldb < p.n || p.ldc < p.n) {
        return KA_INVALID_ARGUMENT;
    }
    BufferSizes checked{};
    if (!span_bytes(p.m, p.k, p.lda, 2, checked.a_bytes) ||
        !span_bytes(p.k, p.n, p.ldb, 2, checked.b_bytes) ||
        !span_bytes(p.m, p.n, p.ldc, 4, checked.c_bytes)) {
        return KA_SIZE_OVERFLOW;
    }
    sizes = checked;
    return KA_SUCCESS;
}

}  // namespace kernel_agent
