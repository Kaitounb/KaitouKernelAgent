# KaitouKernelAgent

面向 NVIDIA GPU 的 CUDA 算子优化 Agent，将模型提案、受控工具执行、硬件分析和发布回退连成可追溯工作流。

**技术栈：** CUDA C++ · WMMA / Tensor Core · cuBLAS · PyTorch C++/CUDA Extension · DeepSeek · Nsight Compute · CUDA Graph / Event · Docker

## 项目解决的问题

模型产生的 CUDA 修改需要经过真实编译和运行才能判断效果；GPU 测量还会受到主机发射开销、缓存状态和设备波动影响。部署阶段则需要保证失败候选能够被隔离，运行版本可以校验和恢复。

项目围绕这些问题实现了候选提案、执行反馈、可信评测和版本管理，让优化过程具备明确的输入、工具边界、裁决依据和回退路径。

## 核心实现

### Agent 编排与反馈

- 将 DeepSeek 的结构化提案解析为受约束的 CUDA 源码补丁或算子配置，绑定父版本与候选身份。
- 串联编译、正确性检查、性能测量和发布裁决，按阶段记录输入、输出、错误与状态。
- 将编译错误、执行结果和硬件指标组织为反馈，支持执行反馈与硬件反馈的对照实验。
- 使用候选去重、调用预算和停止条件控制搜索成本；评测器独立决定候选是否满足晋升条件。

代码入口：[候选流水线](kernel_agent/optimization/pipeline.py) · [反馈通道](kernel_agent/optimization/feedback_channels.py) · [身份与去重](kernel_agent/optimization/identity.py) · [预算控制](kernel_agent/optimization/call_budget.py)

### CUDA 算子与 PyTorch 接入

- 通过 PyTorch C++/CUDA Extension 接入 GEMM，提供 WMMA Tensor Core 模板与 cuBLAS 路径。
- 对齐输入精度、张量形状、内存布局和地址对齐条件；不满足模板条件时切换到可信实现。
- 使用设备 guard 与当前 CUDA stream 发射 kernel，保证与 PyTorch 调用侧的设备、流语义一致。
- 接通 Attention 与 MLA Decode 计算后端，用任务合同约束各自的数据布局与运行配置。

代码入口：[WMMA 模板](kernels/gemm/wmma_sm86.cu) · [cuBLAS 接入](kernels/gemm/cublas_sm86.cu) · [算子后端](kernel_agent/operators/)

### 硬件分析与可信测量

- 用 Nsight Compute 采集 warp 占用、Tensor Core 活动、显存吞吐、缓存命中和等待原因，解析为可供提案使用的硬件事实。
- 通过 CUDA Graph 降低主机发射开销的影响，使用 CUDA Event 交替配对测量，区分重复输入和输入环场景。
- 先检查候选自身的 A/A 稳定性，再与当前版本做 A/B 对照；正确性、稳定性和收益分别判定。
- 结合独立数值参考、输入保护和 Sanitizer 检查约束候选，保留原始报告、日志和裁决记录。

代码入口：[配对评测流水线](kernel_agent/optimization/pair_pipeline.py) · [配对裁决](kernel_agent/optimization/pair_decision.py) · [正确性评测](kernel_agent/optimization/evaluate.py) · [NCU 运行说明](docs/RTX3060_NCU.md)

### 隔离执行与可回退部署

- 将构建与 GPU 执行放入受控 worker，使用低权限身份、只读挂载、网络隔离和资源约束限定执行边界。
- 对运行时间和日志大小设限；异常或超时后清理 worker，清理状态未确认时停止继续执行。
- 通过源码和产物指纹校验版本身份，维护候选、当前版本及可信后备版本。
- 使用版本注册与原子切换实现回滚，交付 wheel/CLI、运行记录及恢复备份。

代码入口：[版本注册](kernel_agent/registry.py) · [执行与回退](kernel_agent/deployment.py) · [有界进程控制](kernel_agent/bounded_io.py) · [GEMM CLI](kernel_agent/cdi_cli.py)

## 已交付能力

| 模块 | 已完成的工程能力 |
|---|---|
| GEMM | CUDA 模板与 cuBLAS 接入，正确性和稳定性评测，可信实现的部署、回退与恢复 |
| Attention / MLA Decode | 固定 SDK 后端集成、任务合同、受控配置实验与已验收版本运行入口 |
| 硬件反馈 | 真实 GPU 上的 NCU 采集、指标解析、原报告读取与输出复核 |
| 优化 Agent | 真实模型提案、受约束源码/配置修改、失败反馈、候选去重和外置裁决 |
| 工程交付 | 受控 worker、可安装命令行、版本校验、实际包切换和可追溯运行记录 |

性能候选只有在正确性、稳定性和收益门槛均满足后才晋升。当前部署默认保持已验收的可信实现，生产范围绑定对应的 GPU、驱动、镜像和输入合同。

## 运行与代码阅读

Python 环境中安装源码：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
kernel-agent-3060 --help
```

模型 provider 的可选依赖：

```bash
python -m pip install -e '.[model]'
```

NCU 离线解析另外需要 pandas。GPU 执行需要匹配的 CUDA、PyTorch、Docker 和 NVIDIA 环境；另一块卡或不同运行环境需要建立自己的资格，不能直接继承原目标的生产结论。

建议按以下顺序阅读：

1. `kernel_agent/optimization/pipeline.py`：候选在不同工具阶段之间如何流转。
2. `kernel_agent/optimization/identity.py` 与 `feedback_channels.py`：如何约束模型输出和反馈信息。
3. `kernels/gemm/`：CUDA kernel、张量合同和 PyTorch 接入。
4. `kernel_agent/optimization/pair_pipeline.py`：如何测量并裁决性能结果。
5. `kernel_agent/registry.py` 与 `deployment.py`：如何校验、执行和回退已验收版本。

[GPU 运行与回退](docs/RTX3060_GEMM_RUNTIME.md) · [源码发布范围](docs/GITHUB_PUBLICATION.md)

## 数据与许可

仓库提供实现、测试及选定工程记录。模型/SSH 凭据、私有独立验收清单和大型运行环境归档保存在本机，不进入版本控制。随代码分发的版权、许可证和署名见 [第三方声明](THIRD_PARTY_NOTICES.md) 及各文件附带许可。
