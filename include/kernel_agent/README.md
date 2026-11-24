# P0-05 保留原型

本目录不是选定的工程 ABI。`gemm_problem.h`／`gemm_contract.hpp` 配合 `src/contracts.cpp` 用于主机元数据、地址跨度和溢出学习；`gemm_candidate.cuh` 是未编译运行的接口探索。

已有主机检查结果只对应历史原型。工程构建／绑定先采用成熟上游，不要求实现这些自定义导出符号或让上游适应此 struct。具体采用见 [UPSTREAM_REUSE.md](../../provenance/UPSTREAM_REUSE.md)。
