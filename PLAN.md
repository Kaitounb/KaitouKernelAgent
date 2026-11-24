# NVIDIA Kernel 优化 Agent 项目计划书

2026-11-24：**当前已确认范围的工程交付已结项**。3060 GEMM0.1.3、3070 Attention/MLA0.1.1保持各自生产资格；本次3060 NCU已真实采集3次cuBLAS kernel，每次19项指标有效，原报告重新导入/固定KernelAgent解析/三输出FP64复核通过。生产包、镜像、资格及原计时不变；最终qualified、GPU空闲/170W、无项目容器，0新增模型调用。原4090跨卡研究、3070 GEMM最终资格、通用源码生产及学习复盘保持各自未验收/延期状态。

[工程结项与延期范围](docs/PROJECT_CLOSEOUT.md) · [NCU实测与查看报告](docs/RTX3060_NCU.md) · [结项清单](releases/project-closeout-20261124/manifest.json)。原一次性入口不重跑，旧sealed不读不传不重跑。


2026-11-24：**RTX3060 GEMM 0.1.3正式交付已完成**，当前服务器入口为qualified，运行/故障回退/真实包切换和恢复备份均已验收。完整三AA三AB1728段：最大单对0.657401%、组0.251983%、最低保护0.999578594×；独立正确性16/16通过。

[运行、回退与恢复](docs/RTX3060_GEMM_RUNTIME.md) · [发行清单](releases/rtx3060-gemm-v0.1.3/manifest.json) · [验收汇总](docs/RTX3060_GEMM_RELEASE.md)。

范围为当前3060上的固定可信cuBLAS；3070 Attention/MLA0.1.1保持，3070 GEMM原拒绝不回填，4090目标C/任意模型源码生产仍未验收。无GPU/API任务在途，本轮0新模型（累计190/700）和0新CUDA编译。原失败、私有清单及旧原始证据保持；已消耗入口不重跑。

原计划与研究验收保留如下；生产资格只以实际发行范围声明。历史检查点见[归档](docs/history/PLAN_BEFORE_013_CLEANUP_20261124.md)。

版本：v1.5 · 2026-11-21 · Asia/Shanghai；CUDA C++ kernel 主线，成熟实现优先复用／适配，每天 6 小时。  
项目主线：GEMM → 稳定 Softmax → 融合 Attention → DeepSeek 风格 MLA Decode → Agent 自动优化与跨 GPU 验证。  
执行清单：[TASKS.md](TASKS.md) · 调研来源与版本：[SOURCES.md](SOURCES.md)

## 1. 项目定位与预期成果

**主要采用 KernelAgent、KDA、ForgeKernel、KernelBench、FlashInfer-Bench 的成熟流程与组件，适配一个能依据 NVIDIA 性能反馈优化 CUDA C++ kernel 的 agent。以 GEMM 建立基础，以经典 MLA decode 为主案例。**

先复用原接口／模块，再做必要适配；已证实的目标任务／硬件／精度缺口才新增设计。计算 kernel 尽量使用 CUDA，编译、验证、计时与 agent 控制层沿用成熟组件的语言与接口。完整纠正记录见 [ROUTE_CORRECTION_20261118.md](docs/ROUTE_CORRECTION_20261118.md)。

交付优先级是可部署、可安全运行、可回退的 KernelAgent。先从 GEMM 打通真实优化闭环，再扩展 Attention/MLA 和目标 GPU。教学材料、手算与机制复盘安排在工程交付后。

暂定项目名：**NVIDIA Kernel Optimization Agent：从 GEMM 到 MLA 的硬件反馈优化实践**。英文可用 **Profile-Guided Kernel Optimization Agent for GEMM and MLA**。

最终展示四项成果：

1. 有可核实来源的工程 kernel、可安装的执行入口、明确的支持矩阵与可信回退路径。
2. 复用／适配成熟工具组成的源码优化闭环，具备候选版本、失败回退和证据记录；不以重写整套框架为成果。
3. 两类核心任务 GEMM、MLA decode 的正确性、性能及 agent 消融实验。
4. RTX 3070 与一张目标服务器卡上的独立测量，以及一份完整优化案例报告。

**完成标准不预设必须超过 cuBLAS 或 FlashMLA，也不预设硬件反馈一定优于其他搜索方法。** 工程完成由功能与验证决定；研究结论由实验决定。若最终无稳定提速，保留有效的优化边界和失败分析，调整简历中的成果措辞。

### 首个生产发布门槛

首批支持范围从 RTX 3070/SM86 的 GEMM 和明确的 dtype/layout/shape 合同开始。WSL2 是当前开发与验证环境，正式部署只声明实际通过验收的系统、驱动和 GPU。保持现有 Attention/MLA 项目范围，按同一发布门槛扩展。

- **运行闭环**：通过文档命令安装并复现实卡运行。真实模型在固定预算内提出合法变更，确定性工具完成构建、正确性、计时、裁决和留档。
- **执行隔离**：模型生成的源码只在独立低权限 GPU worker 内构建与运行。worker 无模型凭据、默认无外网、评测与数据集只读；超时和 runtime failure 均经故障演练，失败可隔离并恢复。隔离验收前只实跑仓库已审查的变体或由受控参数生成的固定模板。
- **交付与回退**：部署入口只加载验收过的版本，候选质量、版本、编译产物和来源可追溯。晋升由独立正确性与配对复测裁定；未通过或不支持时回退可信实现，回滚经实测。
- **运维**：每次运行可追踪候选、资源、耗时、错误和裁决；定义资源上限、退出状态、GPU/软件支持范围以及可复现的验证命令。

## 2. 已知条件、假设与当前状态

| 项目 | 当前信息／采用的安排 |
|---|---|
| 本地 GPU | 已实测 RTX 3070 8 GiB，Windows 10 build 19045.6466 + Ubuntu 24.04.3 WSL2；CUDA/Triton 最小测试与 NCU 单个硬件指标采集通过 |
| 租卡 | 可以租 4090 或更高规格；优先按目标公司的实际卡型决定 |
| 时间 | 用户确认每天可投入 6 小时，希望尽快；按 18–24 个工作日推进核心交付，第 5–7 天争取完成第一版；另留 4 天工具链／调试缓冲 |
| 编程路线 | 学习版、工程 kernel 与候选尽量 CUDA C++；构建／正确性／计时／控制层优先复用成熟工具，可保留 Python；Triton 非必学、非默认工程 kernel |
| LLM | 使用用户已有可用模型 API；具体供应商、模型 ID、采样参数写入每轮记录 |
| 模型权重 | 核心阶段使用合成张量与单算子测试，无需下载完整 DeepSeek 权重 |
| MLA 范围 | 经典 DeepSeek-V2/V3 风格、dense、前向、先单 token decode，再扩展短 query |
| 当前交付状态 | P0、P1-02 首批 SM86 基线和 P1-03 首轮真实参数裁决已验收。DeepSeek 8-warps 候选数值正确但 A/A 不稳被拒绝，参照保持 4；固定无收益负对照与回放通过。P1-04 的 NCU 硬件反馈也已验收，2-warps 提案因收益不足／保护项退化被拒绝。P1-05报告／复算完成但长段A/A最坏3.324%仍未验收；15项受限容器可信GEMM／故障恢复通过，独立无GPU容器编译及新binary15项恢复／10项CPU解析测试通过，后续有界runtime29项及开发wheel安装／回退／回滚15次独立正确性通过；P3-02有界builder／审查源码闭环也已按固定image/SM86进程文件网络资源范围验收；P3-01首轮真实源码／160项独立正确性／Sanitizer及新A/A拒绝已按范围验收（最坏3.233%），GEMM-REL-01可安装离线入口已按开发范围验收（dev4），GEMM-REL-02双binary组件已按dev5安装／正确性／故障／诊断范围验收（35 CPU、72诊断Event段、不可晋升），继续GEMM-REL-03发布候选和独立验收协议冻结，无收益结论 |

### 2026-11-17 路线调整

用户更易理解 CUDA，因此 kernel 尽量使用 `.cu/.cuh`，不要求先学会 Triton。2026-11-18 进一步明确以五个成熟项目为主要来源：不再预设自有 C++ harness、FFI、加载器或完整 agent 模块；成熟 Python 编译／评测／控制组件可以保留。

P0-03/P0-04 是固定 Triton 任务的历史评测原型；P0-05 是本地主机元数据／接口探索与来源审查，原型 ABI 不作为工程规范。原始源码和实卡证据保留，不倒填为已经接入上游。新 CUDA 接入需独立做正确性、A/A 与必要故障回归，避免重复环境验收。

原排期暂作目标，选定成熟接口并完成 CUDA 基线验收后重估。当前编译能力依据既有 nvcc/sm_86 冒烟记录；cuBLAS 和工程后端仍待实测。

### 可复用的本地积累

| 现有材料 | 复用方法 | 尚需完成 |
|---|---|---|
| [Ascend agent README](references/ascend-agent/README.md) 与实现 | 仅作历史经验和失败场景参考；不替代五个主要上游 | 对照成熟项目选择可用组件，必要适配单独记录；新 CUDA 路径重新验证 |
| [Ascend 验证记录](references/ascend-agent/VALIDATION.md) | 学习如何保存真实失败、保留原版本和限定收益范围 | 旧 NPU 结果不计入新 NVIDIA 项目成绩 |
| [Attention 教程](references/learning/attention_tutorial/README.md) | 延续其十步路线，现有文件记录到 QK 和缩放 | 后续步骤及真实 GPU 验证 |
| [Tensor Core 教学例子](references/learning/gemm/README-tensor-core.md) | 用于理解 warp 协作、WMMA、对齐 | 当前文档只记录静态审查，需实卡编译运行；16×16 示例不作性能基线 |
| [FlashAttention 学习材料](references/learning/flash_attention/README.md) | 已有课程与源码副本，用作辅助讲解 | 本地 README 记录尚未运行；工程基线仍需验证和锁版本 |

本地材料的状态以上述文件记载为准；本次没有重新执行它们的测试。

## 3. 五个主要上游的实施分工

固定 commit／具体文件／许可见 [采用表](provenance/UPSTREAM_REUSE.md) 和 [上游 lock](provenance/upstream-lock.json)。已核查 35 个选定文件，当前只完成阅读／归档，未安装、集成或 GPU 验证。每个组件进入实施前写明 source → reuse/adapt → validation，不把“参考理念”当成实际复用。

| 主要来源 | 优先复用／借鉴 | 本项目必要适配与验收 |
|---|---|---|
| PyTorch KernelAgent | NCU profiler、指标清洗、诊断反馈结构 | CUDA 入口与 SM86/WSL 指标可用性；验证后再供 agent 使用 |
| NVIDIA KDA | 合同→计划→候选→验证→测量→证据／晋升的工作流 | GEMM/MLA 合同、候选 parent／当前参照与本项目验收条件 |
| OpenBMB ForgeKernel | 当前参照逐代优化、harness 保护、CUDA 模式和硬件知识条目 | 核对主要面向 Hopper/Blackwell 的组件；只适配可用于当前硬件的部分 |
| KernelBench | CUDA 评测入口、正确性／计时工具和对抗用例 | 任务／精度与 3070 支持；外部成绩和本项目配对筛选分别记录 |
| FlashInfer-Bench | Definition/Workload/Solution、CUDA Builder、现成 binding／runner 与 C++ 使用示例 | 先做最小成熟 CUDA 接入，检查已有依赖缺项，再冻结工程接口 |

先复用可用原组件；提取时保留来源、许可和改动说明；只有验证出原接口的具体不足才补最小适配。不因切换成 CUDA kernel 就把成熟 Python 控制层重写为 C++。五个项目按职责组合，不要求安装五套完整框架。

FlashAttention、FlashMLA、DeepGEMM 是后续算子实现的优先来源，进入对应阶段时核对版本、精度与目标卡。P0-05 自定义 ABI、手写 dlopen／C++ harness 是保留原型，不作为待交付基础设施。

首版仍为单个决策 agent 加确定性工具；是否采用更多 agent 由实际需要决定。贡献集中在上游适配、目标算子与 workload、可信证据及 B/C 对照，保留原作者归属。CUDA Agent 等论文是扩展阅读，不替代上述五个核心来源。

## 4. 主线与扩展范围

### 必做主线

- GEMM：CUDA 朴素 → shared-memory 分块 → 寄存器复用／Tensor Core → CUDA 参数与源码优化。
- Attention：稳定 softmax → online softmax → QK/softmax/PV 融合 → prefill/decode 差异。
- MLA：显式恢复参考 → 矩阵吸收参考 → latent cache 上的 decode → split-KV 与合并 → 分页／不等长。
- Agent：模板参数搜索 → 受约束源码改写 → 硬件反馈 → 失败经验复用。
- 评测：正确性、稳定性能、搜索预算对照、跨 GPU 独立测量。

### 有余力再做

- RMSNorm 作为最初的设备与评测冒烟任务，不单独扩成主研究线。
- GDN commit 作为跨算子迁移扩展，复用你已有经验；在 MLA 主线完成后进入。
- 扩大 FlashInfer-Bench／KernelBench 的额外外部任务集合；基础构建／评测组件的采用本身属于主线。
- Hopper 的 TMA、WGMMA、CuTe；FP8 KV cache；短 query／投机解码 workload。
- vLLM 的单层集成或小型回放实验。

本轮核心交付只覆盖前向和单卡。反向传播、大规模 RL 训练、MoE 通信、最新 DeepSeek 稀疏注意力与完整模型服务另立里程碑。它们不会阻塞本项目达到可展示状态。

## 5. 硬件、工具链与版本安排

### 5.1 两级硬件足够开始

| 硬件 | 用途 | 需要注意 |
|---|---|---|
| RTX 3070，SM86 | 学习、开发、正确性、初期 profiling 与 agent 闭环 | 以 FP32/FP16 起步，按实际显存限制 workload；不能运行 SM90 专属实现 |
| 目标服务器卡 | 最终性能与迁移实验 | 优先公司实际型号；经典 FlashMLA dense decode 对照优先检查 SM90/Hopper |
| RTX 4090，SM89，可选 | 若公司主要用 Ada，或需要中间规模实验 | 其测试不能替代 H100/H800 对照；先查所选实现的 shared-memory 要求 |
| B200 等 SM100，可选 | 公司明确要求 Blackwell 时再进入 | 核对具体模型、kernel、版本的支持矩阵；SM100 不等同于所有 Blackwell 产品 |

GPU compute capability 依据 [NVIDIA 官方列表](https://developer.nvidia.com/cuda/gpus)。这些卡上所得最佳配置分别保存，不能直接宣称跨架构通用最优。

**预算只作规划，不代替租赁授权。** 初期尽量使用已有 3070。远端先做不超过 1 小时的兼容性验证，再按结果安排约 6–12 GPU 小时的探索；最终对照与复测预留约 12–24 GPU 小时。总量是待实测修订的估计，API 与租卡费用按实际报价和用量记录。下单、续费和付费 API 额度须使用用户已经明确授权的资源或预算。

### 5.2 本次调研发现的重要版本限制

截至 2026-11-17，FlashMLA 当前主分支公告说明：2026-09-30 版本移除了 Hopper 和旧模型支持，并改变 KV cache 格式。官方给出的旧版本入口为：

`ba89a3466e9470ad08ab39738d4e7bb66989e1e7`

本计划以该历史版本的 **DeepSeek V3/V3.1 dense decoding、SM90** 路径作为拟采用的专家对照。其 README 支持矩阵已核实；安装、编译和运行仍待实机验证。[当前公告](https://github.com/deepseek-ai/FlashMLA/blob/main/README.md)、[历史支持矩阵](https://github.com/deepseek-ai/FlashMLA/tree/ba89a3466e9470ad08ab39738d4e7bb66989e1e7)

因此，经典 dense MLA 对照不是“租任意高端卡然后安装最新 FlashMLA”。若只有 3070/4090，就先完成自己的 CUDA MLA 与独立参考实验，将 FlashMLA 对照标为未运行，之后再租兼容服务器卡。

DeepGEMM 当前 README 列出 SM90/SM100 和工具链要求。它作为后期阅读与可选对照；3070 阶段优先适配可用于 SM86 的成熟 CUDA 实现并对照 cuBLAS；仅为未覆盖部分补小步实现。[DeepGEMM Requirements](https://github.com/deepseek-ai/DeepGEMM#quick-start)

Triton 当前 Persistent Matmul 教程也明确提示部分默认配置会在 RTX 4090 等 shared-memory 较小的设备上失败。先选择适配设备的实现分支和 tile；不能假定官方教程任意配置均可直接运行。[官方教程](https://triton-lang.org/main/getting-started/tutorials/09-persistent-matmul.html)

### 5.3 第一轮实机检查（历史已完成）

下列是原环境记录清单。当前不重复已通过检查，Triton 不作为新工程 kernel 的必备依赖；接入成熟组件时仅检查实际缺项。

先记录，不自动升级现有驱动或工具链：

```bash
nvidia-smi
nvcc --version
ncu --version
ncu --query-metrics
python -c 'import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))'
python -c 'import triton; print(triton.__version__)'
```

然后依次运行最小 CUDA kernel、最小 Triton kernel、NCU 指标采集、CUDA Event 测速、短时间重复测量。检查操作系统；远端与最终正式实验优先统一 Linux 环境。本地若为 Windows/WSL，先确认所用工具版本的支持情况，不在未经检查时承诺 NCU 全功能可用。

环境清单保存 GPU 型号／UUID、compute capability、显存、SM 数、驱动、CUDA Toolkit 与运行时、PyTorch、Triton、NCU、编译选项、软件 revision、功耗与时钟状态。编译能力和 `torch.cuda.is_available()` 分别核实。

若出现 `ERR_NVGPUCTRPERM`，先依据 [NVIDIA 文档](https://developer.nvidia.com/nvidia-development-tools-solutions-err-nvgpuctrperm-nsightcompute) 确认权限与平台支持。无法读取硬件计数器时，可继续代码正确性和普通测速；该环境下不把工具降级结果计作“硬件反馈”实验。

## 6. 交付后的学习与复盘

本节作为工程交付后的复盘资料。开发期间保留实现、地址和原始实测证据。

统一节奏：**查选定来源 → 手算一例 → 阅读／适配最小 CUDA 版本 → 独立验证 → 阅读 profiler → 小步修改 → 接入成熟 agent 流程对比。**

围绕地址、线程、同步和计算流程阅读并逐步适配 CUDA 实现。教学小例子优先从可核实来源简化，必要时写最小演示；工程基线优先采用已有 CUDA 实现，避免将教学代码逐层扩建成整套框架。复杂模板和 warp 指令按机制需要逐步引入。

### L1：GEMM，建立算力与数据复用意识

顺序：朴素 FP32 GEMM → 合并访存 → shared-memory tile → 每线程多个输出／寄存器复用 → WMMA/Tensor Core → CUDA tile、线程块、warp 分工与流水线优化。

学习题：

- 给定一个输出元素，指出 A/B 的读地址；给定一个 tile，计算复用次数。
- 分别估算 FLOPs、必要数据量和额外搬运；解释为什么加大 tile 也可能变慢。
- 解释 shared memory、register、occupancy 的制约关系，避免把 occupancy 越高当成统一目标。
- 说明 FP16 输入、FP32 累加与 FP32 输入不是同一数值契约。

学习记录覆盖 naive、tiled、Tensor Core 三种 CUDA 机制，允许保留上游命名／路径或明确标记的简化副本，不强制重写成自定文件布局。正式基线采用有来源、合理且正确的 CUDA tiled／Tensor Core 实现，记录实际改动、资源和适用 shape；教学 demo 不作为夸大收益的比较起点。

Agent 初期搜索 CUDA 的 block/warp tile、线程块配置及已实现的合法流水线选项；随后允许改变 tile 遍历顺序、支持范围内的融合 epilogue 和实现结构。参数必须映射到实际 CUDA 模板／launch 逻辑，不沿用 Triton 的配置字段冒充 CUDA 控制。任何融合必须另设等价的任务合同，例如 GEMM+bias 对照也必须包含 bias。

主阅读：[CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html) 与项目既有 CUDA GEMM 材料；[Triton GEMM](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html) 仅作可选算法对照，不作为必学语法。

### L2：Softmax，掌握数值稳定性与并行归约

先实现 `exp(x-max(x))/sum(exp(x-max(x)))`，再实现分块 online softmax。设上一块状态为最大值 m、归一化和 l、未归一化输出累积 a，新块分数为 s、对应 value 为 V：

```text
m_new = max(m, max(s))
alpha = exp(m - m_new)
p     = exp(s - m_new)
l_new = alpha * l + sum(p)
a_new = alpha * a + p @ V
最终输出 = a / l
```

实现时显式处理首个有效块、空分片与全 mask 行，避免 `-inf - (-inf)` 传播 NaN。普通 softmax 与携带 value 累积的 online attention 分开验证。

验收题：手算两块合并；解释为何最大值变化后 l 和 a 都要缩放；验证 tile 改变时结果仍在固定误差范围内。

### L3：Attention，把 GEMM 和 Softmax 连起来

先完成独立 QK、稳定 softmax、PV，再做融合前向。范围逐步增加：单头无 mask → batch/多头 → causal → ragged 长度。

学习 prefill 的多 query 行与 decode 的少 query 行有何区别；避免把二者放进同一性能结论。单 token decode 的 key cache 通常全部属于可见前缀；多 query decode 需要明确 query 在整个序列中的位置与 causal 对齐。

优化重点：消除完整 score/probability 中间张量写回、block 大小、head 分组、寄存器压力、不同 query 长度下的并行度。

验收：用 PyTorch 高精度小规模参考和已确认 backend 的 SDPA 分别检查；证明核心融合路径不物化完整 score 矩阵；报告分解版、合理融合版和成熟库的独立结果。[FlashAttention 原论文](https://arxiv.org/abs/2205.14135)、[Triton Fused Attention](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)

### L4：MLA 数学与缓存语义

学习对象固定为经典 V2/V3 风格 dense MLA。先用随机小矩阵实现两条参考路径：

1. 从 latent cache 显式恢复每个 head 的 K/V，再执行注意力。
2. 使用矩阵吸收后的 query，直接在 latent cache 上计算，再作对应的 value 上投影。

采用行向量记号，令 latent cache C 为 `[B,L,R]`，位置 key K_R 为 `[B,L,D_R]`。对 head h：

```text
K_h        = C @ U_K,h             # [B,L,D_N]
V_h        = C @ U_V,h             # [B,L,D_V]
Q_abs,h    = Q_nope,h @ U_K,h.T    # [B,Sq,R]
score_h    = (Q_abs,h @ C.T + Q_rope,h @ K_R.T) * sm_scale
P_h        = softmax(score_h + mask)
Z_h        = P_h @ C               # [B,Sq,R]
O_h        = Z_h @ U_V,h            # [B,Sq,D_V]
```

这里的 `.T` 代表最后两个维度转置；省略 batch 广播细节。U_K/U_V 是对应 head 的上投影，参考实现应包括模型需要的归一化与 RoPE 预处理，或者明确这些已体现在 kernel 输入中。

`sm_scale` 是冻结的输入语义，不能按吸收后的 R 或 R+D_R 重新计算。采用 V3 常见维度时，基础 QK 维度为 128+64，而 latent 为 512；额外位置缩放按任务规格显式传入。实际维度依据 [DeepSeek-V3 配置](https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/configs/config_671B.json)，数学依据 [DeepSeek-V2 2.1 节](https://arxiv.org/html/2405.04434v5#S2.SS1)。

核心 kernel 合同先固定为：输入 absorbed query 与 cache，输出 latent attention result Z 和指定底数的 LSE。query 投影、RoPE、cache append、value 上投影和输出投影分别标出计时边界。另提供小型完整 attention-layer 回放验证，不把核心 Z 的延迟冒充完整 MLA 层延迟。

LSE 在内部可采用自然底或 2 为底，但接口必须固定；跨实现比较时显式换底。

### L5：MLA decode 的性能路线

按顺序实现：

1. 连续 cache、单 token、FP16 输入／FP32 累积的直接实现。
2. 分块读取 latent 与位置 key，online softmax，避免完整分数张量。
3. 多 query head 共享 cache 读取的组织方式，以及 head tile／KV tile 的取舍。
4. split-KV：不同 CTA 处理不同 KV 段，保存 partial softmax 状态，第二个 kernel 正确合并。
5. 分页 cache 与 ragged 序列；覆盖页面重排和最后一页部分有效。
6. 第二张卡上重新优化，再按需要研究 Hopper TMA/WGMMA。

合并分片 j 的 `(m_j,l_j,a_j)`：

```text
m = max_j(m_j)
l = sum_j(exp(m_j-m) * l_j)
a = sum_j(exp(m_j-m) * a_j)
Z = a / l
```

空分片先剔除；不能平均 partial output。增加分片提升并行度，也增加临时存储和 combine 开销，是否采用由完整 core 路径的实测决定。

MLA decode 的瓶颈会随 query heads、query tokens、cache 长度、共享读取方式和设备变化。先估算算术强度，再以计数器和实验验证，不能预设所有 decode 都是带宽受限。官方 [FlashMLA 深入分析](https://github.com/deepseek-ai/FlashMLA/blob/ba89a3466e9470ad08ab39738d4e7bb66989e1e7/docs/20250422-new-kernel-deep-dive.md) 可作为后期阅读，关注其资源约束与调度推理；不把其中具体调度照抄成所有设备的默认策略。

## 7. Agent 工程设计

### 7.1 复用边界与目录职责

工程主干由五个成熟项目的可用组件组合。先选定上游构建／评测入口和工作流，再只建立所需适配模块；不预设 contracts/orchestrator/worker/LLM/memory 等全部由本项目重新实现。具体采用见 [UPSTREAM_REUSE.md](provenance/UPSTREAM_REUSE.md)，接入顺序见 [CUDA_ENGINEERING.md](docs/CUDA_ENGINEERING.md)。

```text
NvidiaKernelAgent/
  PLAN.md / TASKS.md / HANDOFF.md
  provenance/              # 上游 pin、许可、采用与必要改动
  kernels/                 # 有来源的 CUDA 学习／工程实现及适配
  evaluators/              # 选定上游评测器的最小任务适配，不预设新 harness
  workloads/               # 对齐上游定义后冻结的开发／独立验收集合
  src/                     # 仅放确需的连接代码；现有 contracts.cpp 是 P0-05 原型
  include/kernel_agent/    # 保留 P0-05 元数据／ABI 原型，不是工程 ABI
  experiments/ / evidence/ # 历史原型与原始证据，不改写
  references/              # 旧学习／NPU 材料，不作为默认运行依赖
  tools/                   # 必要的本地证据辅助，不扩建第二套基础设施
```

构建／runner／模型调用等优先使用上游现有 API。是否需要新 Python/C++ 模块、FFI 或共享库适配，由最小接入结果决定；本目录清单不等于这些功能已经完成。

### 7.2 合同字段

每个任务至少记录：op、输入／输出、shape/dtype/stride、mask、scale、输入可变性、数值容差、实现语言、硬件能力约束、baseline hash、benchmark 边界、计时模式、搜索预算、支持范围、开发与独立验收规则。

CUDA 路径另记录 nvcc 版本、目标架构（当前 sm_86）、完整编译／链接选项、头文件依赖、候选与评测器 hash、产物 hash。由受信任评测组件管理输入、参考与 Event 边界，优先复用已 pin 上游的 CUDA binding／Builder／runner；C++ 入口或 Python 控制层按实际接口适配，不为语言统一重写框架。P0-05 自定义 launch 只是设计草案，具体构建接口在 P1 最小接入验证后冻结。

决策 agent 只能修改候选源码和提案，不能修改 reference、benchmark、容差或原始测量结果。接口支持的 stride 和 mask 要明确；范围外返回“不支持”或走已声明回退，不默默产出错误。

### 7.3 工具与状态

以下是流程职责，不要求逐项自写同名 SDK。先映射到 KDA 工作流和选定上游现有函数／记录，仅补目标任务缺口；保留来源与原始结果。

| 工具 | 输入／输出要点 |
|---|---|
| `inspect_task` / `inspect_device` | 冻结合同、真实设备与工具链、能力与预算 |
| `read_candidate` / `read_history` | 源码、父版本、历史错误、已验证事实 |
| `submit_candidate` | 完整源码或合法配置、parent hash、变换说明、预期机制 |
| `compile` | 编译错误、静态资源用量、目标架构 |
| `validate` | 独立误差、边界／不可变检查、失败输入类型 |
| `benchmark` | 新鲜正确性检查、配对计时、原始样本、预算消耗 |
| `profile` | NCU 原始产物、单位、缺失项、环境和候选 hash |
| `decide` | 程序根据证据接受／拒绝；agent 解释不能代替裁决 |
| `record_finding` | 有适用范围与实验引用的假设或结论 |

候选状态：`proposed → compiled → validated → benchmarked → accepted/rejected`。可选 profile 不绕过必要验证；编译失败、正确性失败、超时、OOM 单独记录。源码变更后旧证据失效；恢复会话后重新检查计时与设备信息是否匹配。

### 7.4 两级变换能力

**第一版：参数与有限结构选择。** 搜索 tile、warps、pipeline stages、head group、split 数等。用于先验证工具链，简历明确称为受约束搜索。

**第二版：源码级变换。** 允许改变循环／tile 映射、局部融合、归约组织、split-KV 实现等。通过统一入口与外置 evaluator 检查行为。保留至少一次代码结构发生变化的真实运行，才能宣称源码级优化已完成。

源码执行放入无模型密钥的独立 worker，设置超时并终止进程组，评测器位于 agent 不可写的位置。若直接在宿主机运行候选，文档明确其不具备完整恶意代码隔离能力；正式自由源码运行优先使用受限容器／进程与文件权限，记录实际隔离范围。

### 7.5 硬件反馈与经验库

先收集少量信号：kernel 资源用量、SM／Tensor 活跃度、DRAM 与 L2 访问、occupancy、关键 stall、spill／local memory、launch grid。指标名称通过实际设备和 NCU 查询，缺失写为 unavailable，不能填成零。

诊断输出固定为：观察事实、瓶颈假设、支持指标、竞争解释、拟修改、预期变化、验证结果。低 occupancy 或高内存利用率本身不直接等于根因。NCU 的百分比和 roofline 用于提出假设，不设统一“达到 95% 就成功”的规则。

经验条目优先对齐 ForgeKernel 已有 CUDA 知识／来源格式，记录 GPU 架构、工具链、op/shape 范围、变换、资源与延迟、正确性、证据和结论状态。只补本项目需要且上游缺少的字段；跨卡当待验证假设，不预设自建第二套经验库。经验检索增益仍为可选消融。

## 8. 工作负载与参考实现

### 8.1 集合如何冻结

先定义 seed schedule、生成规则和固定 workload 清单，再运行搜索。推荐核心集合 24 个 workload：GEMM 12 个、MLA 12 个；各取 8 个开发、4 个独立验收。每类开发集中再选 3 个代表 workload 作为搜索锚点，减少成本。

先按 shape 家族分组再切分，避免只把同一 shape 换 seed 当成新任务。独立验收由 evaluator 持有，agent 只知道支持范围，不能读取精确清单和输入。选择完成并冻结候选后只做一次正式验收；若据此修改实现，该批次转为开发数据并重新建立独立验收集。

候选可以是支持多个 shape 的实现或带合法 runtime dispatch 的实现包。dispatch 只依赖实际输入规格，不依赖评测器标记或测试编号。统计时 24 个是 workload 数，核心 kernel 家族为 2 类。

### 8.2 GEMM 候选范围

| 家族 | 示例 M,N,K | 目的 |
|---|---|---|
| 中大矩阵 | 512/1024/2048 范围 | tile、复用和 Tensor Core |
| 小 M | M=1/8/32，N/K 在 512–4096 范围 | 低并行度、launch 成本 |
| 矩形 | 如 128×2048×512 | 不对称 tile 与调度 |
| 非整齐尺寸 | 如 127×1009×513 | mask、越界、padding 成本 |

正式列表在 P1 冻结；以上是设计范围，不是已运行结果。先做连续布局 FP16→FP32 累加，输出 dtype 按合同指定。FP32 scalar 版本与 Tensor Core 版本分组比较，TF32／精度开关全部记录。

参考分三档：CUDA 教学朴素版、合理 CUDA 初始版、cuBLAS 工程对照（PyTorch matmul 可作辅助核对）。Agent 主收益相对合理 CUDA 初始版计算，专家库差距另列。FP16/FP32 累积、布局、计算模式和计时边界须一致，不能把朴素 FP32 与 Tensor Core 实现混为同一合同。

### 8.3 MLA 候选范围

先用 R=64/128 的小规模例子验证吸收和 cache，再用 V3 风格 `R=512、D_R=64、D_N=128、D_V=128` 建立代表 workload。H 包含 8/32/128 等分组；H 小于模型全 heads 时明确是子配置。[官方维度依据](https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/configs/config_671B.json)

| 维度 | 初始开发范围 | 后续扩大 |
|---|---|---|
| B | 1/4/16 | 目标服务器卡上按显存增加 |
| L | 128/512/2048/8192，含非整齐长度 | 16384/32768，仅测量可容纳配置 |
| S_q | 1 | 2/4，进入扩展时明确 causal 对齐 |
| cache | 连续 → 分页 | 多请求 ragged 与页重排 |
| dtype | FP16 输入、FP32 累积 | BF16 单独验证；量化 cache 后续再做 |

不做所有维度的笛卡尔积；从中冻结 12 个代表 workload。另设正确性边界集合覆盖极小值、奇数长度、空有效段、mask、page 尾部及重复运行；它们不自动成为性能汇总中的独立 workload。

单次运行前估算显存，包含 reference、临时 workspace、计时 input ring 与 allocator 余量。在 3070 上控制规模，不能因 reference OOM 就跳过正确性。大规模以分块参考、小规模高精度检查及已验证专家库对照组合验收，并报告实际覆盖范围。

## 9. 正确性与可信测量协议

### 9.1 正确性

- 小规模用 CPU/PyTorch FP64 或高精度参考；较大规模使用已验证的 FP32 分块参考／成熟实现。
- 输入先量化到合同 dtype，再由这些相同值生成参考；固定 FP32/TF32 与累加规则。
- FP32 初始容差建议 `atol=1e-4, rtol=1e-4`；FP16/BF16 从 `1e-2/2e-2` 量级做独立参考校准，正式值必须在搜索前逐任务冻结。数值不适合时先审查输入分布、参考和计算规则，不由 agent 放宽标准。
- 同时报 max abs、带稳定分母的相对误差与 normalized RMS error；通过条件使用冻结的逐元素标准及必要的任务不变量，近零值不能只看相对误差。
- 覆盖多 seed、全零、混合正负、较大有限值、非整齐尺寸、全 mask 行约定、ragged、输入不可变和越界保护。
- 自写 CUDA 或有同步风险的候选在关键阶段运行 Compute Sanitizer memcheck；shared-memory 同步实现按需要补 racecheck/synccheck。[官方工具说明](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html)
- 页式 MLA 检查随机页面映射、尾页、请求重排；完整层回放检查 cache append 与 query position 语义。
- 用故意错误的候选验证评测器：输出常数、污染输入、读取缓存结果、漏算尾部等；确保会被拒绝。[KernelBench 评测讨论](https://github.com/ScalingIntelligence/KernelBench/blob/main/EVAL.md)

### 9.2 性能测量

1. 固定设备、输入、输出分配策略、软件版本和调用边界；编译、数据生成和热身排除。
2. 使用 CUDA Events，执行交替 A/B 配对；正式计时关闭 profiler。短 kernel 可选 CUDA Graph，但两边必须采用相同模式并独立报告。
3. 初始每次测量 9 个配对轮次；每段调用数在校准后冻结，使每段达到可稳定测量的毫秒量级。记录全部原始段时长，不只记录最小值。
4. 多 kernel MLA 路径覆盖 split、workspace 必要初始化与 combine。预处理、布局转换、metadata 构造分别列出，不能只报最短子 kernel。
5. 至少标明“重复输入／温缓存”与“多输入轮转”的缓存策略。NVIDIA 消费卡显存也称设备内存／DRAM，不把逻辑字节除以延迟直接写成实测 HBM 带宽。
6. 正式实验串行占用同一 GPU；记录温度、频率、功率和其他进程。无权锁频则记录状态和波动，不假定已锁频。
7. 搜索阶段的晋升暂设：锚点配对加速比几何均值至少 1.03、探索性区间下界大于 1.00、任一保护 workload 不低于 0.97，且所有正确性通过。先做 A/A 复测校准噪声，再在搜索开始前冻结阈值；噪声较大时增加重复或提高门槛。 2026-11-21 的 GEMM v1 已按双 Graph 实测偏差冻结为锚点几何均值 ≥1.05、探索性区间下界 >1.02、保护项 ≥0.97，三进程和两种缓存策略均需通过；详见 [冻结规则](workloads/gemm_measurement_v1.json) 与 [原始校准](evidence/20261121-p1-02-gemm-aa-anchors/REPORT.md)。此规则仅覆盖声明的 SM86/GEMM 模式，不推广到后续算子。
8. 这些小样本区间用于筛选，不作为经过多轮选择后的严格置信保证。最终冻结源码，与最初基线在独立会话中重新配对，至少 3 个独立 session；不将多次晋升比例相乘。

NCU 会引入 replay、缓存、时钟等测量影响，不能把 profiler 运行中的 CUDA Event 时间当正式成绩。[NCU Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)

加速比定义为 `baseline_latency / candidate_latency`。最终汇总给出全 workload 几何均值、胜／平／负数量、逐项结果和失败数。候选失败但原基线有效时记回退、该项加速比 1.0；连基线都无法运行的任务单列并保留原分母与缺失原因。OOM 或不支持不得悄悄从成功率分母删掉。

## 10. 如何证明 Agent 的作用

### 10.1 三种必做方法

| 方法 | 可以获得的反馈 | 用途 |
|---|---|---|
| A：参数搜索 | 同一初始实现的合法配置与计时 | 说明相对常规调参的表现 |
| B：LLM + 执行反馈 | 源码、任务、编译／正确性错误、延迟与候选历史 | 控制不使用硬件 profile 的 agent 基线 |
| C：LLM + 执行反馈 + NCU | B 的全部信息，再加实际硬件指标 | 检查硬件反馈的增益 |
| D：C + 经验检索，可选 | 有证据和适用范围的经验 | 检查经验复用，不作为第一版前提 |

A 和 B/C 的变换空间不同，因此 A 只作为实用参照；**B 与 C 才是隔离 NCU 作用的主要对照**。二者固定模型、提示主体、允许修改范围和初始源码。屏蔽 B 的硬件指标输入，不能仅在报告里不展示它。

### 10.2 固定预算与重复

完整实验建议 6 个搜索锚点（GEMM/MLA 各 3）× A/B/C × 3 次独立搜索 × 每次最多 12 个提案，共最多 648 个提案。编译失败、错误候选和重复提案都消耗提案预算；原样重复可复用结果但仍记一次尝试。第一版先做 2 锚点 × 3 方法 × 1 次 × 6 提案，确认成本后扩展。

记录 GPU 时间、profile 时间、编译时间、总墙钟时间、API token 用量和费用。以相同提案数比较最终质量，同时给出相同墙钟／费用预算下的质量曲线；NCU 的额外开销计入 C。独立搜索采用新会话，不泄露其他方法已经找到的最佳源码。

至少报告：

- 正确候选率和有稳定收益的 workload 比例。
- 最终相对同一初始实现的加速比。
- 达到预设 1.03／1.05 目标所需提案数与耗时；未达到者报告未达到。
- 搜索进度曲线；相同模型预算下 NCU 是否减少无效尝试。
- 独立验收集的结果与退化；3 次搜索之间的差异。

不预设结论。若 C 无优势，分析 profile 开销、指标解释、任务本身接近极限或模型决策质量，并据实报告。

## 11. 跨 GPU 实验

选相同支持范围的任务，冻结源 GPU 的最佳实现后进行：

1. 在目标 GPU 直接运行源 GPU 的方案。
2. 在目标 GPU 从同一个源方案起点运行普通参数调优。
3. 在目标 GPU 从同一个源方案起点运行硬件反馈 agent。

另列目标 GPU 上从通用初始实现重新搜索的结果，以区分“迁移后再优化”和“完全重新搜索”。不把跨卡直接运行失败改写成性能退化，失败原因单列。

比较目标 GPU 自身上的延迟、正确性与再优化成本。两张不同卡上的绝对延迟不直接用来证明 agent 增益。经验条目包含架构指纹；SM86 的成功结论迁到 SM90 后需重新验证。

## 12. 阶段、依赖与验收

时间为有效投入估计。任务进度以 [TASKS.md](TASKS.md) 为准，依赖满足后再推进。

| 阶段 | 工时估计 | 主要工作 | 进入下一阶段的验收 |
|---|---:|---|---|
| P0 环境与评测 | 6–8h | 检查 3070、工具链、NCU，建立合同、Event 计时和故障候选检查 | 最小 CUDA/Triton 正确；A/A 噪声已知；硬件指标可采或阻塞原因明确 |
| P1 GEMM + Agent v0 | 原估 16–22h，成熟 CUDA 接入后重估 | 上游构建／评测接入、合理基线与成熟 agent 流程适配 | 多 shape 正确；真实模型调用完成至少一次候选裁决；不要求必须晋升 |
| P2 Softmax/Attention | 12–16h | 稳定归约、online 状态、融合前向 | 独立参考正确；causal／边界覆盖；计时边界清楚 |
| P3 Agent v1 | 12–18h | 源码级修改、NCU 诊断、只看延迟的控制组 | 能处理代码变换及失败回退；B/C 可重复运行 |
| P4 MLA 参考与基础 kernel | 16–24h | 显式／吸收参考、RoPE、连续 cache、FP16 融合 decode | 两条参考等价；核心 Z 与完整层语义分别验证 |
| P5 MLA 优化与服务器卡 | 16–24h | split-KV、分页／ragged、目标卡、可用专家对照 | 边界正确；至少一条完整优化轨迹；专家库实跑或明确未运行 |
| P6 正式实验与交付 | 16–24h | 三方法消融、跨卡、独立验收、报告与演示 | 原始数据可复算；全任务成绩与限制明确；复现入口可运行 |

原 Triton 工程路线合计估计约 94–136 小时、每天 6 小时约 18–24 个工作日。改为 CUDA C++ 主线后，以下日期暂保留为冲刺目标，待工程基线验收后重估；若安装或复杂 kernel 调试超出预期，使用第 25–28 天缓冲。该估算不是已验证进度承诺。时间不足时先减少扩展范围，再减少 workload 与重复规模并标明探索性结论；保留正确性与合理基线。

### 工程优先的冲刺安排

按交付门槛推进。当前 P1-02 基线、P1-03 真实参数裁决与 P1-04 NCU 反馈已验收；P1-05按冻结4-warps完成一次独立最终复测和长段追加，仍有3.324% A/A离群，报告／复算已完成、计时保持未验收，不继续挑选重采；推进不依赖稳定性能的P3-02构建与隔离部署；有界runtime29项和开发wheel安装／回退／回滚15次正确性已实测，后续builder产物／日志及审查源码闭环也已按限定范围验收；P3-01首轮真实DeepSeek源码结构修改经新build、160项独立正确性和Sanitizer通过，但新identity三会话864段A/A最坏3.233%超界，按原门槛拒绝、不跑A/B；该源码／拒绝范围已验收，无收益／生产发布结论。GEMM-REL-01已完成dev4可安装离线优化入口；14项CPU故障回归、安装空cwd及指针回滚通过，dev3安装路径160项GPU输出及两项Sanitizer0errors由逐字不变的GPU组件继承；0新API／0计时。GEMM-REL-02已完成dev5双binary开发组件；35项安装CPU方法、真实双模块QA／故障拒绝／Sanitizer和72段非晋升诊断A/B通过。完整新source三A/A→三A/B尚无本轮实测，不能用诊断替代候选证据；0新API、未改best／默认cuBLAS。当前GEMM-REL-03先冻结发布候选／支持范围及独立验收协议，当前不读取／执行封存清单、不自动新增付费调用。[双binary报告](docs/DUAL_BINARY_EVALUATION.md)。[安装说明](docs/OFFLINE_OPTIMIZER.md)。[源码闭环报告](docs/P3_01_SOURCE.md)。已有教学、地址和手算题集中保留到交付后复盘。

推进顺序：核对五个上游的固定组件与接口；完成最小成熟 CUDA 接入；适配有来源的合理 kernel 与 KDA/KernelAgent 等工作流；先连续 cache，后 split-KV 和分页；先小规模对照校准成本，再做完整实验。控制／评测层保留可用上游实现。GDN、FP8、CuTe 深化和完整模型服务留在 V2 后。

### 版本里程碑

- **V0（目标第 5–7 天）**：3070 上 GEMM 任务、真实 API 调用、参数搜索和基础硬件反馈、正确性／计时／回退报告。对外明确这是第一版，MLA 尚未完成。
- **V1（目标第 12–15 天）**：源码改写、Attention、MLA 两条参考、基础 MLA decode；初步 B/C 对照。
- **V2（目标第 18–24 天）**：MLA 主案例、第二张 GPU、正式对照、独立验收、完整证据与简历材料；复杂安装或调试时使用第 25–28 天缓冲。

如果面试提前到来，展示最近一个实际完成的版本。版本名称不等于已经达到对应结果。

## 13. 工程执行流程

从当前工程清单选择依赖满足的任务，记录上游 commit／文件／接口、复用和必要新增部分及验证命令。实现后先正确性，再正式计时和单独 profile。保存运行命令、版本、输入、源码和产物 hash、原始结果与失败原因。满足验收才勾选。

遇到失败按环境、编译／资源、语义／数值、测量和性能策略逐层定位。连续尝试无新证据时，先做最小验证或采集诊断，再继续推进。

## 14. 证据与最终交付

优先沿用所选上游的目录和序列化记录，确保以下信息可追溯。下方仅为信息对应示例，不要求另造一套 schema：

```text
run_id/
  task.json                  # 冻结输入合同、容差、预算和计时范围
  environment.json           # GPU 与完整工具链、实际运行日期
  sources.json               # 上游 revision、许可、引用与改动
  baseline/                  # 最初实现
  candidates/<id>/           # 源码、父版本、提案和 hash
  checks/                    # 正确性与对抗检查
  measurements/              # 原始 A/B 配对时间与统计
  profiles/                  # NCU 原始报告和解析字段
  trajectory.jsonl           # 工具、错误、裁决
  usage.json                 # 模型 ID、token、GPU／wall time、预算
  final/                     # 冻结最佳实现、独立复测
  REPORT.md
```

交付内容：

- 一份环境恢复与一键运行说明；另有单 kernel、单 run 和全实验复现入口。
- 基于成熟评测组件、补齐任务检查的独立验收入口，能拒绝典型错误候选。
- 一条 GEMM、至少一条 MLA 的完整轨迹；包括被拒绝的候选。
- 一张逐 workload 表和一张 B/C 搜索曲线；只在读者需要时展示更多图。
- 10 分钟演示：任务 → 瓶颈 → 修改 → 验证 → 结果 → 限制。
- 3 条有证据的简历描述，以及性能无增益时也成立的工程表述。

本次计划不涉及替用户发布代码、发送消息或购买算力。后续这些动作按用户给定的具体授权执行。

## 15. 可写入简历的表述边界

以下是待结果填入的模板，不是现阶段成果：

> 基于 KDA、KernelAgent 等成熟流程与评测组件适配 CUDA kernel 优化 agent，面向 GEMM 与经典 MLA decode，完成 [实际适配与验证范围]。
>
> 在 RTX 3070 与 [目标卡] 上完成 [数量] 个 workload 的评测，相对 [明确的初始基线] 获得 [全任务几何均值]；同时报告与 cuBLAS／固定版本 FlashMLA 的性能差距。
>
> 固定模型和搜索预算，通过执行反馈／硬件反馈对照，观察到 [提案次数、耗时或性能的实测差异]，保留候选源码、profile 与可复算原始数据。

只有 V0 完成时，就写受约束 GEMM 搜索，不写完整 MLA 优化；没有第二张卡结果就不写跨卡验证；只有核心 latent attention 结果时，不宣称完整 DeepSeek 模型吞吐提升。

## 16. 常见阻塞与调整顺序

| 情况 | 调整方式 |
|---|---|
| GEMM 工程接入超预期 | 缩小首轮 shape／变换范围，优先完成上游调用和可信回退；正式支持范围按实测扩大 |
| NCU 权限不可用 | 继续普通验证／测速，换可采指标环境后再做硬件反馈结论 |
| 高端库安装失败 | 锁版本并做最小复现，优先使用可运行的成熟参考；仅为缺失语义补小规模高精度演示，专家库成绩留空 |
| 3070 显存不足 | 缩 batch／L、分块参考、控制缓存轮转；扩大配置移到服务器卡 |
| Agent 只会重复调参数 | 增加源码结构候选与失败诊断；以代码 diff 验证能力边界 |
| MLA 数值错误 | 先回到吸收等价、scale、RoPE、mask、LSE 底数和 split 合并逐层定位 |
| 最佳候选独立复测无收益 | 撤销收益结论，保留原实现与选择偏差分析，增加合理复测 |
| C 不优于 B | 据实分析硬件指标价值和开销；不追加挑选结果来改变全套结论 |
| 面试时间提前 | 交付最近已验收版本，优先可解释案例与原始证据 |

## 17. 下一次直接从哪里开始

截至 2026-11-18 已完成 **P0-01 至 P0-05**。环境见 [冒烟报告](evidence/20261117-wsl/REPORT.md)，噪声见 [P0-03](evidence/20261117-p0-03/REPORT.md)，故障拒绝见 [P0-04](evidence/20261117-p0-04/REPORT.md)，新工程职责与来源审查见 [P0-05](evidence/20261118-p0-05/REPORT.md)。本段为2026-11-18历史入口；2026-11-22最新进度是P1-02/P1-03/P1-04已验收，P1-05报告完成但最终计时未验收，P3-02隔离与P3-01首轮源码拒绝闭环均已按范围验收，GEMM-REL-01开发安装已验收，GEMM-REL-02双binary组件已验收，当前继续GEMM-REL-03发布候选／独立验收协议冻结；P1-01学习复盘延后。后续 CUDA GEMM 的 shape 与模式需重新校准，不能沿用向量噪声结论；P0-04 原型不等于完整多轮搜索或恶意代码隔离，P0-05 主机检查不等于 GPU 验收。

平台检查方法仍保留在 [WSL_SETUP.md](WSL_SETUP.md)；后续若换环境，可重新运行 [环境检查脚本](check_wsl_env.py)。当前实际运行应使用已找到的 `/home/dministrator/cuda-lab/.venv/bin/python` 和显式工具路径。

开始执行时读取本计划和 TASKS 的当前检查点。按工程依赖连续推进可验收交付，复习和手算安排在交付后。
