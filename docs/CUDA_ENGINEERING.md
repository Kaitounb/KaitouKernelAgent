# CUDA 工程接入：优先复用成熟组件

更新：2026-11-21。先交付工程闭环，学习复盘延后。路线以五个成熟项目为主要来源，kernel 尽量 CUDA C++。原 ABI 设计快照见 [P0-05 归档](../evidence/20261118-p0-05/REPORT.md)。

## 首个工程接入

- KernelBench 固定 commit `423217d9fda91e0c2d67e4a43bf62f96f6d104f1`。直接调用 `src/kernelbench/eval.py::load_custom_model`、`run_and_check_correctness` 及 `timing.py::time_execution_with_cuda_event`。整套 `eval_kernel_against_ref` 会额外导入依赖模型 SDK 的 `utils.py`。首轮只采用所需原始模块，保留上游 correctness 和原始 Event 结果。本项目补充输入域、CPU FP64 参考、输入不可变及非默认 stream 检查。
- 对比 FlashInfer-Bench commit `15618a67e8d7b7f49ca61b5ba3b9f26a876d0bee` 的 `flashinfer_bench/compile/builders/torch_builder.py::TorchBuilder.build`。该版本走 PyTorch `load`，但禁用了依赖库链接管理；3070 当前未安装 FlashInfer-Bench 或 TVM-FFI。首轮采用 KernelBench 入口，直接利用现成 CUDA/PyTorch 和显式 cuBLAS 链接。
- 初始可信实现为 CUDA 13.0 的 `cublasGemmEx`，保留库内算法选择。薄 CUDA C++ 包装负责张量验证、行优先映射和 PyTorch CUDA stream，采用 FP16 输入、FP32 累加及输出。三个开发锚点与两种缓存策略现已完成新路径 A/A，门槛已按实测偏差冻结，见 [测量规则](../workloads/gemm_measurement_v1.json)。[cuBLAS 13.0](https://docs.nvidia.com/cuda/archive/13.0.0/cublas/index.html#cublasgemmex)
- 使用与 pin 匹配的 KernelBench 源码运行 `python -m evaluators.gemm_kernelbench --output <新路径>`，先验证仓库内固定实现；生成源码执行进入独立 GPU worker 验收后再启用。

## 状态与边界

P1-02 首批 SM86 范围已验收：8+4 workload 冻结、可信 WMMA 模板与 cuBLAS 开发集正确、故障回归及双 Graph A/A 通过声明的校准规则。P1-03 真实首轮参数裁决及 P1-04 NCU 硬件反馈现也已验收；两次模型候选均拒绝，当前4-warps、无收益。独立验收尚未执行，完整隔离／发布回滚仍待后续验收。[最新实测](../evidence/20261121-p1-02-contract-template/REPORT.md)

五个项目已固定 commit 并核对选定文件。KernelBench 三个未修改的选定模块已经用于 cuBLAS GEMM 构建、正确性和 Event 实卡检查，其他工程接入按各自验收记录判断。[采用表](../provenance/UPSTREAM_REUSE.md) · [工程 GEMM 实测](../evidence/20261121-p1-02-kernelbench-cublas/REPORT.md)

`gemm_problem.h`、`gemm_contract.hpp`、`src/contracts.cpp` 与 `gemm_candidate.cuh` 是 P0-05 本地原型：主机元数据／跨度检查通过，但 CUDA ABI 未编译执行。它们不是工程接口，也不要求上游适应其符号、struct、输出 dtype 或 56 字节布局。`make host-check` 仅保留原型检查用途。

## 接入组件与候选路径

| 职责 | 优先来源／入口 | 本项目需要确认 |
|---|---|---|
| 工作流与证据 | KDA `docs/agent-flow.md` | task、candidate、parent、正确性／性能／拒绝证据的对应关系 |
| CUDA 构建与运行 | FlashInfer-Bench `data/solution.py`、现成 Builder、`examples/ffi/` | PyTorch extension／TVM-FFI 哪条能满足当前环境、dtype 与 SM86；先检查依赖，不预设自写加载器 |
| 正确性与计时／对抗检查 | KernelBench `src/kernelbench/eval.py`、`timing.py`、对抗测试；FlashInfer-Bench runner | 任务输入／参考／精度对齐、实际计时模式、超时及失败边界 |
| 硬件反馈 | KernelAgent profiler／metric schema | CUDA 入口薄适配，逐项核对本机指标；保留上游解析与诊断组件 |
| 评测器保护与经验 | ForgeKernel harness_guard、CUDA 模式与 KB 模板 | 采用可迁移机制；Hopper/Blackwell 专属路径不宣称适用 SM86 |

优先验证已有 API，再决定是否提取小模块。工程只新增目标算子／workload／精度、硬件兼容与必要连接代码；不预设全部用 C++ 重建。源码、构建和评测尽量保留上游目录／接口／数据格式，避免第二套重复协议。

## P1-02 最小接入顺序

1. 从已 pin 的路径明确选定构建／评测方案，记录 commit、具体文件／符号和选择理由。
2. 仅检查其实际依赖缺项、nvcc／架构与运行入口，优先复用已有 CUDA/PyTorch 环境。当前尚未确认 tvm-ffi/CMake 等额外依赖，不因示例提及就全部安装。
3. 通过选定上游入口构建并执行一个小 CUDA 候选，保留原始 argv、退出码、编译／运行日志和产物 hash；不要先实现自有 FFI／双库加载器。
4. 用同一批量化后输入对齐参考，检查输出、输入不可变和必要保护；输出 dtype、layout、shape 支持与容差依据实际任务和上游接口冻结。拟以 FP16 输入／FP32 累加起步，但不把旧原型的 FP32 输出写死为所有任务要求。
5. 沿用可用计时组件，新路径独立 A/A 校准；只为尚未覆盖的配对顺序／缓存／统计补最小适配，保留全部原始时间。
6. 将 P0-04 和 KernelBench 的可适用故障用例用于接入路径回归；校验输入污染、缓存旧答案、尾部、错误／超时等不能晋升。正常 CUDA 工作后的主机超时不等于无限 GPU kernel 恢复。
7. 最小接入通过后，再冻结正式工程接口和 12 个 GEMM workload；未运行或不支持的项目保持未验收。

## GEMM 学习与工程语义

row-major 的地址示例：`A[r,k] = A_base + sizeof(A_element)*(r*lda+k)`，`B[k,c] = B_base + sizeof(B_element)*(k*ldb+c)`，`C[r,c] = C_base + sizeof(C_element)*(r*ldc+c)`。leading dimension 按元素计；R×C、行 stride L 的有效跨度为 `(R-1)*L+C` 个元素。

这些公式可用于理解和检查，不能替代完整内存／精度合同。教学 FP32、工程 FP16→FP32 累加及不同输出类型分别验证；const 不能替代执行后的输入快照比较。工程 kernel 优先采用有来源、适配目标卡的 CUDA 实现，必要修改单独记录。

## 证据和责任

优先复用上游的任务、solution、构建缓存与 trace 记录；确保可以追溯 source／依赖、编译器／flags、候选和当前 incumbent、评测器及产物 hash、正确性、原始时间与裁决。上游未记录且本项目必须的信息才补充，不为统一格式另造完整系统。

agent 不得修改评测标准、参考结果或自己的晋升结论。先复用已核查的保护机制，再验证实际权限／进程边界；不把目录名称或 hash 当完整隔离。新候选与当前 incumbent 配对，最终冻结版本另与最初 baseline 复测。

本地主机原型可用 `make host-check` 检查；旧来源可用 `python3 tools/check_provenance.py` 检查。这两个命令都不构建／验收上游 CUDA 后端。
