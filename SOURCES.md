# 调研来源、适用范围与版本记录

初始核查：2026-11-17；五个核心项目固定版本补充核查：2026-11-18，Asia/Shanghai。对应 [计划书](PLAN.md) 和 [执行清单](TASKS.md)。

本目录引用一手项目、官方文档和原论文。公开项目的成绩是作者报告，不是本项目已复现结果；计划没有将其成绩作为目标保证。五个主要上游已固定 commit 和选定文件 hash，见第 7 节；其余 main／在线文档仍会变化，实际采用前另外固定版本和文件。

## 1. Agent 工作流

| 来源 | 本次核查的内容 | 计划中的用途 |
|---|---|---|
| [KernelAgent 官方博客，2026-03-06](https://pytorch.org/blog/kernelagent-hardware-guided-gpu-kernel-optimization-via-multi-agent-orchestration/) | NCU 数据进入诊断、优化与测量闭环 | 硬件反馈设计来源 |
| [meta-pytorch/KernelAgent](https://github.com/meta-pytorch/KernelAgent) | 优化流程、profiling／benchmark 组件、候选产物 | 核心 profile／诊断组件的优先复用来源，CUDA 入口必要适配另记 |
| [NVIDIA KDA](https://github.com/NVlabs/kda) | agent 工作流；社区 wishlist 与通用工作流的硬件范围不同 | 借鉴任务合同和证据，不把 wishlist 卡型限制套给本项目 |
| [KDA Agent Flow](https://github.com/NVlabs/kda/blob/main/docs/agent-flow.md) | 合同、验证、测量、保留／拒绝规则 | 状态机与验收流程 |
| [OpenBMB/ForgeKernel](https://github.com/OpenBMB/ForgeKernel) | 从参考实现演进、硬件实验支持的经验库 | 结构决策与带范围的经验记录 |
| [CUDA Agent 原论文](https://arxiv.org/abs/2602.24286) | 大规模 agentic reinforcement learning 路线 | 研究背景；本项目先做推理时搜索 |
| [KernelOPT 原论文](https://arxiv.org/abs/2609.30059) | 面向编译产物、保留库调用、验证与回退 | 作为后续模型子图集成阅读，不并入核心排期 |

## 2. 基准与推理工作负载

| 来源 | 内容与用途 |
|---|---|
| [KernelBench](https://github.com/ScalingIntelligence/KernelBench) | 核心 CUDA 评测／对抗检查的优先采用来源；额外外部任务范围另列，不将子集冒充全套成绩 |
| [KernelBench EVAL.md](https://github.com/ScalingIntelligence/KernelBench/blob/main/EVAL.md) | 评测陷阱、输入污染、缓存结果、stream 等对抗行为；用于评测器检查 |
| [FlashInfer-Bench](https://github.com/flashinfer-ai/flashinfer-bench) | 核心数据模型、CUDA Builder／binding／runner 的优先采用来源，以及部署相关 trace |
| [FlashInfer-Bench Quick Start](https://bench.flashinfer.ai/docs/start/quickstart) | 最小成熟评测接入时核对 warmup、trials、容差和任务入口，不预设另写完整 benchmark |
| [FlashInfer Trace Apply](https://docs.flashinfer.ai/trace_apply.html) | 可替换匹配 API 调用的运行时机制；作为以后集成方向，当前没有接入 |

## 3. GEMM、Attention 与 MLA

| 来源 | 内容与用途 |
|---|---|
| [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html) | 2026-11-17 路线调整后主阅读：CUDA C++、线程／访存、同步与编译；具体 API 仍按实机 CUDA 13.0 核对 |
| [cuBLAS API](https://docs.nvidia.com/cuda/cublas/index.html) | CUDA C/C++ GEMM 工程对照，核对布局、dtype／计算模式与计时边界；本项目尚未实测该对照 |
| [Triton Matrix Multiplication](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html) | 可选 GEMM tile／映射／autotune 算法对照，已不作为主工程语言 |
| [Triton Config](https://triton-lang.org/main/python-api/generated/triton.Config.html) | 可选 Triton 参数语义参考；num_warps／num_stages 不直接当作 CUDA 构建接口 |
| [Triton Persistent Matmul](https://triton-lang.org/main/getting-started/tutorials/09-persistent-matmul.html) | persistent/TMA 等实现分支；当前文档提示 RTX 4090 shared-memory 约束 |
| [FlashAttention 原论文](https://arxiv.org/abs/2205.14135) | IO-aware tiling、避免完整中间矩阵；本计划用其思想，不复述论文成绩 |
| [Triton Fused Attention](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html) | 融合前向与 online 状态的实现阅读；教程完整脚本按硬件适配 |
| [DeepSeek-V2，v5](https://arxiv.org/html/2405.04434v5#S2.SS1) | 2.1 节 MLA、低秩 KV 压缩、矩阵吸收与解耦 RoPE 的数学来源 |
| [DeepSeek-V3 参考配置](https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/configs/config_671B.json) | n_heads=128、kv_lora_rank=512、qk_nope=128、qk_rope=64、v_head=128；本项目使用同类单算子维度，不加载完整模型 |
| [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) | 当前 README 列出 SM90/SM100 及 CUDA 等要求；后期学习和可选对照 |

### FlashMLA 版本单独锁定

- [当前主分支 README](https://github.com/deepseek-ai/FlashMLA/blob/main/README.md)：包含 2026-09-30 breaking-change notice，说明 Hopper／旧模型支持移除和 cache 格式变化。
- [官方指定历史提交](https://github.com/deepseek-ai/FlashMLA/tree/ba89a3466e9470ad08ab39738d4e7bb66989e1e7)：`ba89a3466e9470ad08ab39738d4e7bb66989e1e7`。
- [历史 README 与支持矩阵](https://github.com/deepseek-ai/FlashMLA/blob/ba89a3466e9470ad08ab39738d4e7bb66989e1e7/README.md)：dense decoding 列为 SM90、V3/V3.1；该事实不等于本项目已安装／运行。
- [2025-04-22 kernel 深入分析](https://github.com/deepseek-ai/FlashMLA/blob/ba89a3466e9470ad08ab39738d4e7bb66989e1e7/docs/20250422-new-kernel-deep-dive.md)：关注 MLA 算术强度、输出累积资源、调度与 split/combine；本计划的设计不以复现其性能数字为前提。

本轮学习固定经典 dense MLA；当前仓库的新模型、DSA、FP8/FP4 路径不自动成为本项目目标。租卡前以要运行的具体 kernel 支持矩阵为准。

## 4. NVIDIA 工具与硬件

| 来源 | 用途 |
|---|---|
| [CUDA GPU Compute Capability](https://developer.nvidia.com/cuda/gpus) | RTX 3070=8.6、RTX 4090=8.9、H100/H200=9.0；核对目标架构 |
| [Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html) | TMA、资源和 thread block cluster 等架构机制；Hopper 扩展阅读 |
| [NCU Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/) | replay、开销、缓存／时钟控制和指标解释；正式测速与 profiler 分离 |
| [NCU Counter Permission](https://developer.nvidia.com/nvidia-development-tools-solutions-err-nvgpuctrperm-nsightcompute) | 实机／云环境计数器权限检查 |
| [CUDA on WSL](https://docs.nvidia.com/cuda/wsl-user-guide/) | Windows 驱动提供 WSL CUDA 支持；WSL 内的 Toolkit 与驱动区分 |
| [NCU Release Notes / System Requirements](https://docs.nvidia.com/nsight-compute/ReleaseNotes/#system-requirements) | WSL2 的 Windows／驱动门槛和具体 NCU 的 CUDA 驱动要求需同时满足 |
| [Compute Sanitizer](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html) | 越界、未对齐访问及按需要使用的同步检查 |

## 5. 实施阶段的来源锁定模板

每个实际使用或复制的第三方实现记录：

```json
{
  "project": "项目名",
  "repository": "上游 URL",
  "revision": "实际 checkout 的完整 commit",
  "files": [{"path": "文件路径", "sha256": "内容哈希"}],
  "license": "核查后的许可证及随附 NOTICE 路径",
  "role": "reference / benchmark / adapted_source / reading",
  "local_changes": "实际改动摘要",
  "verified_on": "GPU 与工具链；未运行时写 not_run"
}
```

模板为待填写规范，未填字段不代表已完成源码下载、许可审查或运行验证。

## 6. 2026-11-18 实施审查结果

[P0-05 复用审查](provenance/REUSE_AUDIT.md) 与 [机器来源锁定](provenance/reference-lock.json) 已记录 61 个原 reference 文件及采用边界。当前没有复制旧 Ascend 模块或第三方 kernel 到运行后端；P0-05 新 C++ 元数据／ABI 是主机原型，不作为工程协议。该批历史证据保留，当前采用决策见五个主要上游表。

GDN 两文件与本地固定 Git commit `622725cd44aa47186dc1f1c9f08485e9013fff7a` 的原路径内容一致，LICENSE 核验通过，COPYING/NOTICE 另存 [provenance/licenses](provenance/licenses)。此 pin 来自已有本地 Git 对象，不代表本次公开抓取成功。其余本地 optimizer／教学材料中缺失的 revision 或通用许可证如实标为未知，保持 reference_only。后续实际采用须补新的来源／许可／改动记录；P0-05 完成不自动批准所有参考材料的改编。

## 7. 五个核心上游的固定版本与采用表

2026-11-18 已取得 KernelAgent、KDA、ForgeKernel、KernelBench、FlashInfer-Bench 的固定 commit/tree，并下载核对 35 个选定原文件及其许可。版本／文件／Git blob SHA-1／SHA256 见 [upstream-lock.json](provenance/upstream-lock.json)，模块职责、适配要求、P0-05 ABI 草案修正见 [UPSTREAM_REUSE.md](provenance/UPSTREAM_REUSE.md)。当前只完成源码阅读与归档，未安装、集成或 GPU 验证。

采用原实现前仍检查对应文件头和第三方来源；KDA 文档／skills／assets 与源码的许可不同，KernelBench 的固定 LICENSE 为 MIT，不能仅按 GitHub API 的通用 license 标记推断。原文件与许可文本已保留在 provenance/upstream-reviews/20261118/；不以本轮 pin 证明任何上游在 3070 上已运行。
