#include "kernel_agent/gemm_contract.hpp"

#include <iostream>
#include <limits>
#include <stdexcept>

extern "C" int ka_problem_c_layout_check(void);

namespace {
int checks = 0;
void require(bool passed, const char* message) {
    ++checks;
    if (!passed) {
        throw std::runtime_error(message);
    }
}
ka_gemm_problem_v1 example() {
    return {KA_GEMM_ABI_VERSION, sizeof(ka_gemm_problem_v1), 2, 3, 5, 5, 3, 3};
}
void rejected(ka_gemm_problem_v1 p, ka_status status) {
    kernel_agent::BufferSizes sizes{7, 11, 13};
    require(kernel_agent::validate_gemm_problem(p, sizes) == status, "wrong rejection status");
    require(sizes.a_bytes == 7 && sizes.b_bytes == 11 && sizes.c_bytes == 13,
            "failure changed buffer sizes");
}
}  // namespace

int main() {
    try {
        require(ka_problem_c_layout_check() == 1, "C/C++ problem layout compatibility");
        auto p = example();
        kernel_agent::BufferSizes sizes{};
        require(kernel_agent::validate_gemm_problem(p, sizes) == KA_SUCCESS, "contiguous rejected");
        require(sizes.a_bytes == 20 && sizes.b_bytes == 30 && sizes.c_bytes == 24,
                "wrong FP16/FP32 byte spans");

        p.lda = 8; p.ldb = 4; p.ldc = 7;
        require(kernel_agent::validate_gemm_problem(p, sizes) == KA_SUCCESS, "padded rejected");
        require(sizes.a_bytes == 26 && sizes.b_bytes == 38 && sizes.c_bytes == 40,
                "span must stop after last logical element, not final row padding");

        p = {1, sizeof(p), 1, 1, 1, 1, 1, 1};
        require(kernel_agent::validate_gemm_problem(p, sizes) == KA_SUCCESS, "1x1 rejected");
        require(sizes.a_bytes == 2 && sizes.b_bytes == 2 && sizes.c_bytes == 4, "1x1 spans");

        p = example(); p.abi_version = 2; rejected(p, KA_UNSUPPORTED);
        p = example(); --p.struct_bytes; rejected(p, KA_UNSUPPORTED);
        p = example(); p.m = 0; rejected(p, KA_INVALID_ARGUMENT);
        p = example(); p.n = -1; rejected(p, KA_INVALID_ARGUMENT);
        p = example(); p.k = -1; rejected(p, KA_INVALID_ARGUMENT);
        p = example(); p.lda = 4; rejected(p, KA_INVALID_ARGUMENT);
        p = example(); p.ldb = 2; rejected(p, KA_INVALID_ARGUMENT);
        p = example(); p.ldc = 0; rejected(p, KA_INVALID_ARGUMENT);

        const auto huge = std::numeric_limits<std::int64_t>::max();
        p = example(); p.m = huge; p.lda = huge; rejected(p, KA_SIZE_OVERFLOW);
        p = example(); p.m = 1; p.k = huge; p.lda = huge; rejected(p, KA_SIZE_OVERFLOW);
        // A/B each fit, but FP32 C's byte count overflows.
        p = {1, sizeof(p), 1, huge, 1, 1, huge, huge};
        rejected(p, KA_SIZE_OVERFLOW);

        // Row stride is irrelevant for a single row, even if its numeric value is huge.
        p = {1, sizeof(p), 1, 1, 1, huge, huge, huge};
        require(kernel_agent::validate_gemm_problem(p, sizes) == KA_SUCCESS, "single-row stride");
        require(sizes.a_bytes == 2 && sizes.b_bytes == 2 && sizes.c_bytes == 4,
                "single-row stride incorrectly increases allocation");
        std::cout << "{\"status\":\"PASS\",\"evidence_type\":\"HOST_CPP_TEST\",\"checks\":"
                  << checks << ",\"gpu_test\":false}\n";
    } catch (const std::exception& e) {
        std::cerr << "Contract test failed: " << e.what() << '\n';
        return 1;
    }
}
