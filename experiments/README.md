# 历史实验原型

`p0_03/` 为固定 Triton 任务的 A/A／正确性学习，`p0_04/` 为故障拒绝与 best 保护原型。它们的原始源码与证据保留，不改写为已经集成成熟项目，也不默认扩建成正式 agent 基础设施。

后续主要采用 KernelAgent、KDA、ForgeKernel、KernelBench、FlashInfer-Bench，kernel 尽量 CUDA；必要规则／用例迁移到实际接入路径后重新验证。[路线更正](../docs/ROUTE_CORRECTION_20261118.md)

`p1_01/` 为 KernelBench 原始 CUDA 教学源码及其 load_inline 入口的小规模正确性 glue／实验传输，不是新框架或 benchmark。2026-11-20 首次编译与 14 组输入检查通过，Sanitizer 因调试接口错误未完成。[实测范围](../evidence/20261120-p1-01/REPORT.md)
