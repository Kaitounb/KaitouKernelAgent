# KaitouKernelAgent

基于成熟组件的 NVIDIA CUDA 算子优化 Agent：真实模型提案、受控编译、正确性与硬件反馈、配对测量、独立裁决和可回退部署。

本仓库是已完成工程交付的源码快照，包含 CUDA/Python 实现、测试、固定上游来源、工作负载合同及选定验收摘要。生产资格只覆盖实际验收过的固定实现与 GPU/驱动/镜像组合。

## 已验证的交付

| 内容 | 实际验收 |
|---|---|
| RTX3060 GEMM / v0.1.3 | FP16输入、FP32输出的固定可信cuBLAS；320公开正确性、32 Sanitizer检查、独立16/16输出；三轮A/A与A/B共1728段 |
| GEMM测量稳定性 | A/A最大单对偏差0.657401%，最大组偏差0.251983%，最低A/B保护项0.999578594×；通过预先冻结的3% / 1.5% / 0.97部署门槛 |
| RTX3070 Attention / MLA / v0.1.1 | 已验收的固定FlashInfer Attention与page16/split1 MLA core，保留原限定生产资格 |
| RTX3060 Nsight Compute | 3次真实cuBLAS kernel，每次19项指标有效；原报告重导出与固定KernelAgent parser解析一致；三输出FP64复核通过 |
| 交付与恢复 | wheel/CLI、版本与产物校验、低权限worker、有界日志/超时、可信回退、实际包回滚和镜像恢复验证 |

部署成绩不代表新的算法提速。正式A/B/C消融完成54组搜索、221个提案、158次模型调用：没有候选通过完整晋升门槛，默认保留可信实现。包含先前工程调用的新预算累计190/700（历史3次另计）。未建立硬件反馈优于其他方法的收益结论。

[工程结项与延期范围](docs/PROJECT_CLOSEOUT.md) · [正式实验结果](docs/FORMAL_STUDY_RESULTS.md) · [3060验收汇总](docs/RTX3060_GEMM_RELEASE.md) · [NCU指标与原始CSV](docs/RTX3060_NCU.md)

## Agent 与可信执行流程

```text
冻结合同与工作负载
  → DeepSeek提出受控CUDA源码/配置
  → 变更范围与身份检查
  → 独立构建与CUDA正确性/Sanitizer
  → 候选自身A/A稳定性
  → 与当前参照配对A/B测量
  → 外置门槛裁决与证据留档
  → 只允许已验收固定产物进入部署/回退目录
```

硬件反馈由Nsight Compute原报告与CSV提供，复用固定KernelAgent指标解析。模型提案与发布裁决分离；未通过、重复、超时或退化的候选均有记录。GEMM源码和MLA SDK配置采用不同受控变更范围。

生产GEMM worker使用UID/GID65534、零cap、NoNewPrivs、只读根目录、私有PID/network边界、两个只读产物/输入挂载、2GiB内存和有界输出。管理员NCU诊断通道单独受控，不执行任意模型生成源码。可信控制器与worker职责见[执行边界](docs/BOUNDED_BUILDER.md)及[运行说明](docs/RTX3060_GEMM_RUNTIME.md)。

## 代码结构

| 目录 | 职责 |
|---|---|
| `kernel_agent/` | 可安装CLI、registry、执行/回退、优化与算子后端 |
| `kernels/` | CUDA GEMM基线、可信WMMA模板及教学kernel |
| `evaluators/`, `workloads/` | 合同、正确性/测量适配、固定工作负载 |
| `experiments/` | 参数/源码提案、NCU、正式消融和一次性验收控制器 |
| `tests/` | CPU gate、协议、发布、回退与适配测试 |
| `provenance/` | 上游commit、采用接口、修改范围、原始许可和来源锁 |
| `releases/` | 各目标的限定发行/资格清单与工程结项声明 |
| `evidence/` | 选定审计摘要和NCU CSV；完整原始档案保存在原本地工作区 |

## 本地安装与CPU回归

Python3.12或更新版本；GPU运行另外需要匹配的CUDA/PyTorch/Docker/NVIDIA环境。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m pip install pandas==2.3.3  # 仅NCU离线解析需要
kernel-agent-3060 --help

PYTHONPATH=tests python -m unittest   test_registry test_deployment_policy test_qualified_channel   test_sm89_runtime test_sm89_qualified test_operator_qualification   test_gemm_deployment_policy test_gemm_target_binding test_cdi_release
```

以上54项CPU回归不代替GPU验收。模型provider的可选依赖通过`pip install -e '.[model]'`安装；密钥仅从本机环境变量注入，仓库不含模型或SSH凭据。

GPU部署的镜像和资格绑定原验收环境。在另一块卡上安装源码不会自动获得生产资格；请先构建/核对自己的依赖环境，再以新目录建立对应目标的正确性、稳定性与隔离证据。不要重跑历史一次性stage/freeze/qualification/independent/install脚本来覆盖既有结果。

## 主要复用来源

复用/适配PyTorch KernelAgent的profiler/provider，KernelBench的CUDA评测路径，以及KDA/ForgeKernel/FlashInfer-Bench的既定流程、保护和数据接口。算子后端按硬件支持核对现成CUDA SDK。各组件的实际集成范围有单独记录，并非五套框架全部直接集成。

[上游采用表](provenance/UPSTREAM_REUSE.md) · [版本锁](provenance/upstream-lock.json) · [第三方许可](THIRD_PARTY_NOTICES.md)

## 数据与发布范围

公开的是工程源码、测试、上游归属及选定验收摘要。SSH/API凭据、私有独立验收清单、大型Docker/SDK/工具恢复归档、完整原始运行档案及未明确再分发许可的参考材料不进入Git。链接到这些未分发本地材料的历史路径用于记录审计来源，在线仓库中可能不可直接下载。

[源码发布范围](docs/GITHUB_PUBLICATION.md)说明目录选择；[完整结项声明](docs/PROJECT_CLOSEOUT.md)列出延期项。3070 GEMM最终资格、原4090目标C完整跨卡研究、通用源码生产执行和学习复盘保持各自未验收/延期状态。
