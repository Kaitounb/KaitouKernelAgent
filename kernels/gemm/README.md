# CUDA GEMM 学习与适配落点

2026-11-20，[KernelBench CUDA 教学源码](learning/kernelbench/README.md) 已按原 ModelNew/load_inline 入口编译并通过 14 组小 FP32 输入 × 2 次的实卡正确性／不可变检查。[证据](../../evidence/20261120-p1-01/REPORT.md)。memcheck/synccheck 因 WDDM 调试接口问题未完成，无性能结论。P1-01 先理解线程／地址，再逐步验证 shared tile／WMMA；教学版本与工程基线分开记录，当前没有工程 kernel 验收。

P1-02 已通过固定 KernelBench 入口在 SM86 上构建并验证 `cublas_sm86.cu`，作为可信 GEMM 初始实现。此批为 FP16 连续行优先输入、FP32 累加和输出；后续优化候选与该合同及可信回退对齐。[实测](../../evidence/20261121-p1-02-kernelbench-cublas/REPORT.md)

P0-05 的 `gemm_candidate.cuh` 仅为保留原型，不作为工程 ABI 或必要依赖。输出 dtype 和 binding 在上游最小接入后确定。[采用表](../../provenance/UPSTREAM_REUSE.md) · [接入顺序](../../docs/CUDA_ENGINEERING.md)

旧 `references/learning/gemm/` 保留原样，只作阅读；实际复制或改编前核对来源／许可并保存改动。

## P1-02 可信优化起点（2026-11-21）

`wmma_sm86.cu` 从已核验 NVIDIA CUDA Samples v13.0 的 `simple_wmma_gemm` 适配，保留原 BSD notice。每 warp 处理一个 16×16 输出 tile，FP16×FP16、FP32 累加；`warps=1/2/4/8` 改变 block 沿 M 的覆盖。B 改为合同的行优先布局，去除不需要的 alpha/beta epilogue。原始示例是 WMMA 起点，当前不宣称达到专家库性能。

整齐形状和 32-byte 对齐输入走 WMMA；尾块、较小尺寸、非对齐 storage offset 走 `cublas_sm86.cu`。两条路径共享 `gemm_contract.cuh` 的 shape/layout/dtype 校验。这里的 fallback 是固定模板内真实输入分派，不代表版本发布／运行时回滚已验收。

四种配置在 RTX 3070 上通过开发集、边界、非法输入及故障回归；默认 4 warps 补 memcheck/synccheck。详见 [工程合同与模板实测](../../evidence/20261121-p1-02-contract-template/REPORT.md)。冻结合同与 8+4 workload 见 [workloads](../../workloads/README.md)。后续模型只能选择允许的参数，未经 P3-02 验收不自动执行生成源码。
