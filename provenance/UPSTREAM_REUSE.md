# 成熟项目优先的实施约束与采用表

更新：2026-11-18。用户要求后续优先借鉴既定五个成熟项目，尽量减少自行设计；该约束覆盖此前 P0-05 的工程草案。

当前计划、任务、交接、工程／目录说明及原型状态已按本表统一更正，详见 [ROUTE_CORRECTION_20261118.md](../docs/ROUTE_CORRECTION_20261118.md)。CUDA 优先指 kernel，成熟 Python 构建／评测／计时与控制层可直接保留；本表不要求整套框架重写成 C++。

实施顺序固定为：**查已有接口／模块 → 尝试直接复用 → 必要时提取适配 → 只为已证实的缺口新增代码**。每项工程工作先写出上游 commit、文件／符号、拟保留部分、必要改动和验证命令，再动手实现。没有适用实现时记录具体硬件、精度或接口差异，不能仅因“自己写更清楚”就另造一套状态机、benchmark 或 FFI。

CUDA C++ 仍是 kernel 主线；Python 控制层可以复用成熟工具。教学小例子服务线程／地址理解，工程基线与 agent 工具链优先采用有来源的成熟实现。组件组合成同一条工作流，各自承担明确职责。

## 已核查的固定版本与采用位置

本次从五个指定仓库取得 commit/tree 元数据，下载 **35 个选定原文件**，逐个验证 Git blob SHA-1 和本地 SHA256。文件总计约 424 KB。均未安装、导入或执行；当前状态为 **源码／文档核查，尚未集成或 GPU 验证**。完整文件与 hash 见 [upstream-lock.json](upstream-lock.json)，原始数据在 `upstream-reviews/20261118/`。

| 项目与固定 commit | 已查看的关键入口 | 后续优先采用 | 需要适配／验证 |
|---|---|---|---|
| [KernelAgent](https://github.com/meta-pytorch/KernelAgent/tree/e0647170da36ef9b059ac0bd3d60103aa4ed378b) `e0647170da36ef9b059ac0bd3d60103aa4ed378b` | `kernel_perf_agent/kernel_opt/profiler/ncu_profiler.py`、`diagnose_prompt/metric_schema.py`；原始计时工具 | P1-04 的 NCU 采集、CSV 清洗、指标到诊断反馈；先检查现成函数再实现 adapter | 已查看 profiler 面向 Triton Python 入口并依赖 pandas/numpy；CUDA 可执行入口需薄适配。各指标在 SM86/WSL 上逐项验证，不能把上游列表当作本机已可用 |
| [KDA](https://github.com/NVlabs/kda/tree/acdd79343c5e9d0c7cb96289f26fd3f1051ce785) `acdd79343c5e9d0c7cb96289f26fd3f1051ce785` | `docs/agent-flow.md`、`prompts/basic-flow.md` | 主工作流直接按 contract→draft/plan→candidate→correctness→measurement→evidence/promotion；任务工作区持有候选、parent 和拒绝记录 | 它主要提供可复用工作流／文档，不是可直接替换所有 CUDA backend 的执行库；本项目补 shape、精度、3070 调用与验收边界 |
| [ForgeKernel](https://github.com/OpenBMB/ForgeKernel/tree/b2c8ae8510e02c28576808910a6b8827721ab7dc) `b2c8ae8510e02c28576808910a6b8827721ab7dc` | `src/forgekernel/loop/lang_mode.py`、`harness_guard.py`、`window_contract.py`、`hw_kb/cuda/TEMPLATE.md` | CUDA 语言与 prompt/KB 分离、固定 evaluator 的保护方式、当前参照逐代优化、硬件事实／探针／经验条目 | 已查看 CUDA 模式为 `cuda-sol`，含自己的 binding 约定；README 的主要 GPU 要求为 Hopper/Blackwell。3070 先采用可迁移流程／格式，不声称整套工具在 SM86 即插即用 |
| [KernelBench](https://github.com/ScalingIntelligence/KernelBench/tree/423217d9fda91e0c2d67e4a43bf62f96f6d104f1) `423217d9fda91e0c2d67e4a43bf62f96f6d104f1` | `src/kernelbench/eval.py`、`timing.py`、`unit_tests/test_eval_adversarial.py`、`EVAL.md` | P1-02/P1-03 的 CUDA 候选评测入口、错误／输入污染／缓存答案／stream 绕过等负例；外部验收通道 | 使用其 model/input 入口时遵守原任务定义；外部结果和本项目配对筛选分列。保留官方代码检查能力的实际范围，不把 regex 检查当完整隔离 |
| [FlashInfer-Bench](https://github.com/flashinfer-ai/flashinfer-bench/tree/15618a67e8d7b7f49ca61b5ba3b9f26a876d0bee) `15618a67e8d7b7f49ca61b5ba3b9f26a876d0bee` | `data/definition.py`、`data/solution.py`、`compile/builders/tvm_ffi_builder.py`、`examples/ffi/README.md`、`cpp_example.cc` | 工程数据模型／源码与产物绑定，CUDA/C++ Solution、现成 Builder、destination-passing 输出、C++ 加载共享库示例 | 优先做与已装 PyTorch extension 或 TVM-FFI 的最小接入试验；先检查实际依赖缺项。CMake／tvm-ffi 在本项目尚未确认具备，示例的固定 GEMM／dtype／GPU 不能直接当作本项目合同 |

五个项目的 README、选定实现和许可文本已归档，**不等于五套系统均已集成**。采用状态必须继续区分 reading / adaptation-planned / adapted / executed / GPU-validated。

## 对 P0-05 草案的明确修正

- `gemm_problem.h` 和主机跨度检查保留为教学／内部元数据工具，已有主机测试事实保持有效。
- **`gemm_candidate.cuh` 的自定义 C ABI、手写 dlopen 双加载方案不作为已定型工程要求。** 先对照 FlashInfer-Bench 的 CUDA Builder、PyTorch extension 与 TVM-FFI 路径，优先复用可用绑定。只有接入试验证明需要时，才保留最小自定义适配并写明原因。
- 不继续按“旧 NPU 模块不适用，所以整个 NVIDIA workflow 独立重写”的方式推进。KDA 的工作流、KernelAgent 的 profile 组件、KernelBench／FlashInfer-Bench 的评测与构建工具要逐项尝试复用。
- 自有代码重点收敛为 GEMM／经典 MLA 的目标 shape、精度／语义适配，SM86 入口，以及本项目的配对统计、保护 workload、同预算 B/C 对照和独立验收。即使这些部分需要新增，也先核对上游是否已有实现。
- P0-03/P0-04/P0-05 原始报告与源码快照不改写；本页是后续实施约束的补充，不能把本次阅读倒填为原批次实际使用了上游代码。

## 许可核查状态

选定版本的 KernelAgent、ForgeKernel、FlashInfer-Bench 根许可为 Apache-2.0；KernelBench LICENSE 为 MIT。KDA LICENSE 明确区分文档／skills／assets 的 CC-BY-4.0 和源码的 Apache-2.0，不能只依赖 GitHub API 的 NOASSERTION 标识。原许可文本与 FlashInfer NOTICE／KDA third-party notice 已随选定文件保存；实际复制时还要检查目标文件的 header／第三方来源、保留 attribution 并记录改动，不推断根许可覆盖所有子目录。

本次归档位于 provenance 下，未改动已锁定的 references 快照，也未把上游文件加入运行时 import 路径。各项目实际依赖／平台支持以最小接入试验为准，当前没有安装新依赖或调用模型。

## 后续算子来源

FlashAttention、FlashMLA、DeepGEMM 是后续算子实现的优先来源；进入相应阶段时固定具体版本／文件、核对硬件与精度。FlashMLA 已计划的历史 commit 保留为待实机验证的经典 dense 路径；不把它或新 main 的支持范围自动推广到 3070。此三项本次没有新增源码运行或兼容性结论。

## 紧接着怎样推进

2026-11-21 工程优先，P1-01 学习复盘延后。P1-02 已用固定 KernelBench 的 `load_custom_model`、`run_and_check_correctness` 和 CUDA Event 入口跑通 cuBLAS GEMM。增加的核对文件、pin 和取舍已归档。P1-03/P1-04 继续对接成熟裁决与 profile 组件。[工程实测](../evidence/20261121-p1-02-kernelbench-cublas/REPORT.md) · [新增查阅文件](upstream-reviews/20261121-p1-02/source-record.json)


## 2026-11-20 教学源码首次执行（范围补充）

另外选择同一固定 KernelBench commit 的 `src/kernelbench/prompts/few_shot/model_new_ex_tiled_matmul.py`，原文件／CUDA kernel／load_inline 入口未改，已在 3070 编译并通过 14 组小 FP32 输入各 2 次的正确性与输入不可变检查。[原始证据](../evidence/20261120-p1-01/REPORT.md)。这两个新增教学原文件独立归档在 upstream-reviews/20261119-p1-01/，不改变原 35 文件的 lock。该结果是教学片段采用，不是整个 eval.py、工程基线或五套系统接入；没有性能结论。memcheck/synccheck 因 WDDM 调试接口初始化错误未完成，不能标通过。

## 2026-11-21 P1-02 合同与可信模板续接

实施前来源、保留／改动与验证命令见 [适配记录](adaptations/20261121-p1-02-contract-template.md)。继续执行原始 KernelBench loader/correctness；CUDA 模板从已核验 NVIDIA Samples `simple_wmma_gemm` 适配，保留原版权。合法参数限 1/2/4/8 warps，尾块／未对齐时复用 cuBLAS。8+4 workload 已冻结，独立清单尚未执行。KernelBench 选定 loader/correctness/timing 与可信模板已在 SM86 实测，见 [合同／模板](../evidence/20261121-p1-02-contract-template/REPORT.md)、[双 Graph 校准](../evidence/20261121-p1-02-gemm-aa-anchors/REPORT.md)。upstream-lock 仅更新这些选定组件的采用状态，不代表完整上游或其余四个框架集成。

## 2026-11-21 P1-03 参数裁决与模型 provider

实施前 [draft](../docs/p1_03/draft.md) 与 [plan](../docs/p1_03/plan.md) 记录具体符号／保留与改动。新增核对 KernelAgent 同一固定 commit 的 6 个 provider 原文件，逐项 Git blob／SHA256 见 `upstream-reviews/20261121-p1-03/KernelAgent/source-record.json`。实际只导入原 `base.py`、`env_config.py`、`openai_base.py`，子类补 DeepSeek JSON/非 thinking、60s timeout、零重试；不改上游文件。此为模型 provider 接入，不代表 profiler／NCU 已适配。

KDA 采用文档工作流与 candidates.jsonl/parent；ForgeKernel 采用可信 evaluator 和当前参照／对称计时窗口原则。当前项目无 Git lineage，未执行其 Git revert 或云 job overlay，不声称完整 ForgeKernel 集成。继续复用 KernelBench 已验收 CUDA .so 和 `check_case`；新代码只补本合同的合法参数、配对、门槛与回放。实测结果以本批证据为准。

## 2026-11-22 P1-04 profiler／原 parser 与 schema 运行

KernelAgent 同一固定 commit 的原 `load_ncu_metrics/metrics_to_prompt`、`NCU_METRIC_SECTIONS` 已配合适配 CUDA collector 在 SM86 完成真实 NCU 接入，原文件未改。4/2-warps × 512³/1024³ 共四份 `.ncu-rep`，每份三 launch，19 字段含 launch 软件统计；旧 memory_dependency 标 unavailable。原 parser 按 launch 分片直接复用，全部行／单位保留，实际零与缺失分开。具体取舍见 [实施前记录](adaptations/20261122-p1-04-ncu.md)。

一次真实 DeepSeek 根据 NCU 事实提出 2 warps，指标预测未支持，独立 Event 保护项退化而拒绝。不是 NCU 收益或 B/C 对照结论；KernelAgent 整套 agent、完整 ForgeKernel/KDA backend 等仍不宣称已集成。[完整实测](../evidence/20261122-p1-04-feedback/REPORT.md)

## 2026-11-22 P1-05 与 P3-02 阶段性进展

P1-05沿用原KernelBench检查／计时与冻结可信binary。唯一长段适配是原paired函数写死replay数参数化，CPU逐函数核对；一次长段追加仍3.324% A/A离群，计时未验收，无晋升。[来源记录](adaptations/20261122-p1-05-final.md) · [V0报告](../docs/V0_REPORT.md)

P3-02先核对原FlashInfer-Bench `isolated_runner.py`：同UID且发送ref_outputs/ref_latency，不能直接作为本项目安全边界，只参考生命周期，不导入其完整runner。实际新增Docker适配补非root、只读、网络／PID隔离，独立宿主持有参考和裁决。固定现有镜像15项可信GEMM／五故障／五恢复已实测；builder继续复用原KernelBench loader及未改CUDA。不是五套框架完整集成、通用恶意代码隔离或生产发布。[来源／具体缺口](adaptations/20261122-p3-02-worker.md) · [阶段入口](../experiments/p3_02/README.md)

后续独立builder已从空cache成功执行原KernelBench load_custom_model／Torch load_inline，CUDA不改；UID65534、无GPU／无网络。新binary通过15项运行恢复与10项CPU输出解析。[实测报告](../evidence/20261122-p3-02-build-fixed-tool-path/REPORT.md)。仍依赖既有镜像／公开依赖，不等于全新安装、通用源码隔离或生产发布；原freeze保持。

## 2026-11-22 P3-02 有界runtime与开发部署

固定上游版本不变，实施前逐符号和缺口记录在[适配记录](adaptations/20261122-p3-02-bounded-runtime.md)。复用既有Torch CUDA二进制／绑定、原NPY检查函数、NumPy与stdlib/Docker能力，补原FlashInfer同UID/Pipe发送ref及本项目宿主RW输出／无限日志缺口。无宿主RW mount、有界字节流／metadata／NPY、tmpfs／PID／CPU内存实测29项通过，12故障／12恢复。不是新CUDA ABI或新的计时后端。

用标准setuptools/pip离线wheel，在新prefix／空cwd验证默认cuBLAS、来源／合同／image／binary校验、独立fallback与原子版本rollback。15次独立正确性、3个CLI负例和4个CPU mock策略测试通过。只注册两个包内可信cuBLAS开发版本，不将worker自报成功当晋升，不改变4-warps搜索freeze。[实测／wheel](../evidence/20261122-p3-02-install-rollback/REPORT.md)。源码入口／builder输出边界、正式独立验收和生产发布仍未整体验收，完整五框架集成仍不宣称。

最终开发包为[0.1.0.dev2](../evidence/20261122-p3-02-install-rollback-dev2/REPORT.md)，收尾增加启动前日志准备、host I/O失败仍清理、cleanup异常保留未知状态。16项本机I/O／协议、10项registry及目标4项CPU mock测试分列，新prefix重新15次独立正确性／回退／回滚通过。dev0/dev1及所有原始证据保留。CUDA／合同／搜索best与上游pin均不变。

## 2026-11-22 P3-02有界builder／审查源码闭环验收

[实施前记录](adaptations/20261122-p3-02-bounded-builder.md)列出固定KernelBench load_custom_model／原load_trusted／Torch load_inline与FlashInfer TorchBuilder具体缺口；实际继续执行原KernelBench构建绑定，未改其原文件，不把引用概念说成完整FlashInfer Builder集成。builder无GPU、无宿主RW挂载、tmpfs cache及有界日志／固定四产物；host不import新.so，独立GPU输出由host事先算的FP64参考裁定。11个builder场景、27次GPU调用验收，26正确、1错误源码拒绝，实际freeze不变，4个loader pin相符。[报告](../evidence/20261122-p3-02-bounded-builder/REPORT.md)。

P3-01后续真实DeepSeek一次2,232tokens，Git --recount显式派生后审查，复用KernelBench loader／P3-02无GPU builder新编译；新源码160项独立正确性、memcheck／synccheck各8项0errors。原配对Event路径只移除child参考、改回hash witness；可信host ledger不发child，计时AST核验未变。新A/A三会话864段最坏3.233%超界，外置REJECT，不跑A/B；原失败与全部证据保留，本机CPU审计PASS。源码／拒绝闭环按限定范围验收，非性能收益或生产发布。[调用／格式来源](adaptations/20261122-p3-01-source.md) · [新评测具体缺口与适配](adaptations/20261122-p3-01-source-evaluation.md) · [实测](../docs/P3_01_SOURCE.md)。


## 2026-11-22 GEMM-REL-01 可安装离线入口

GEMM-REL-01可安装离线入口已按开发范围验收，最终dev4 wheel可在新prefix／空cwd运行。dev3安装路径fresh build＋160项独立正确性、Sanitizer各8项0errors及一次默认cuBLAS通过；dev4仅command／版本／bundle改动，13条空cwd命令含14项CPU回归、历史失败幂等拒绝和实际版本回滚通过，GPU组件逐字不变，明确继承刚取得的实卡证据而未重复GPU。3项无损传输CPU测试／两阶段独立审计PASS。0新API／0计时，默认cuBLAS／allowlist未变，生产发布仍未通过。[说明／wheel](../docs/OFFLINE_OPTIMIZER.md) · [范围验收](../evidence/20261122-gemm-rel01-dev4-delta/acceptance.json)。 原KernelAgent provider文件、KernelBench loader与许可／SHA随包；P3-01输入／数值／Event函数体AST不变，只做package import／资源定位及必要单次预算持久化。实际借鉴、安装资源、传输分片缺口与验证见[实施前记录](adaptations/20261122-gemm-rel-01-installable.md)。最终dev4不重跑GPU；只继承与dev3逐字相同组件的刚完成实卡证据，0收益／生产结论。


## 2026-11-22 GEMM-REL-02 双binary开发组件

GEMM-REL-02双binary评测／外置筛选已按开发组件范围验收，最终dev5 wheel新prefix／空cwd35项CPU方法通过。真实双扩展16项QA含4个零输出候选拒绝、baseline正确；两项Sanitizer各4项0errors。全轮92项输出校验含4个预期失败；空闲时一批WMMA/cuBLAS诊断72原始Event段、16合同witness／256图输出hash通过。诊断不可晋升，完整新source三A/A→三A/B未实跑；0新API、不重采旧A/A，默认cuBLAS／freeze／allowlist未变，无收益／生产发布。[报告／wheel](../docs/DUAL_BINARY_EVALUATION.md) · [范围验收](../evidence/20261122-gemm-rel02-dual-installed-fixed-test-import/acceptance.json)。 CPython原importlib只改变完整模块命名空间，原paired_witness计时函数体不改；原gate数值聚合在包内旧文件直接对照一致。双artifact／两侧ledger／protocol／worker／bundle绑定补原单binary入口缺口；外置开发建议不写runtime／freeze。具体复用和预注册见[来源](adaptations/20261122-gemm-rel-02-dual-binary.md)。

## 2026-11-22 P2/P4成熟CUDA接入与独立GEMM验收

GEMM RC1固定dev5/默认cuBLAS后，一次性独立正确性16/16与6公开回退/回滚场景通过；精确清单不展示给模型，195份私有文件完整hash复核。既有KernelBench/Docker路径逐字复用，仅独立worker观察增加input/guard/stream检查。[实测](../docs/GEMM_RC1_ACCEPTANCE.md)

补充算子来源不是五个主框架的替代：PyTorch v2.7.1 `e2d141dbde55c2a4370fac5165b0561b6af4798b` 的原生PersistentSoftmax header直接调用；FlashInfer v0.2.8 `3f8317c67ac129facb8525b2fb581be42f634459` 的fa2 prefill/MLA、RoPE、page append、state merge实际在3070执行。编译/运行薄适配保留Torch/Python接口；2处C++20编译标志适配原Torch2.14，SDK CUDA算法未改，原C++17失败保留。40项公开检查、guard/stream/memcheck/synccheck0errors、冻结8开发workload的50项正确性通过。4独立MLA只生成私有清单/公开commitment，未执行。原生binary、源码和SDK源hash均归档；性能/Agent/发布仍另验。[采用和缺口](adaptations/20261122-operators-cuda.md) · [实卡](../evidence/20261122-attention-mla-sm86-cxx20/verification.json)

B/C共用所pin KernelAgent provider、现有patch guard与已有真实profile事实，仅新增信息白名单和总预算reservation。2次真实接口pilot6,135tokens完成，B不读取profile，C错误源码被静态拒绝；不是正式消融。总预算已授权新增700次，失败/重复/未知占额度，当前用2次。历史3次独立保留。[接口审计](../evidence/20261122-feedback-live-pilot/verification.json)

2026-11-23服务器兼容复核：经典FlashMLA dense固定commit ba89a3466e9470ad08ab39738d4e7bb66989e1e7的README/LICENSE已逐hash归档于provenance/expert-references/FlashMLA-ba89a3466e94；支持矩阵要求SM90，现有SM86/SM89均不支持。该专家路径不构建/不运行、成绩null，不能把实际FlashInfer fa2结果标成FlashMLA。主五pin及运行CUDA选择不变。

## 2026-11-23 SM86 Attention/MLA生产交付

固定FlashInfer3f8317c...的fa2 CUDA/既有AOT与PyTorch、KernelBench/KDA等pin保留；复用原OperatorRunner、NKA/NPY、bounded_io、原子发布/包指针切换和source_timing共享ring，只补真实资格/完整SDK准入/边界见证。三AA三AB及独立正确性、40CPU/最终9输出11尝试、实际包回滚/5160文件审计通过；scope只限当前SM86固定Attention/MLA，不将其写成全上游整合、自由源码或全项目性能通过。[完整来源与失败](adaptations/20261123-sm86-operator-release.md) · [运行](../docs/SM86_ATTENTION_MLA_RUNTIME.md)。

## 2026-11-24 SM86 GEMM 发布续接与热条件诊断

沿用 KernelBench `423217d9fda91e0c2d67e4a43bf62f96f6d104f1` 的现有 CUDA 接入、两份已冻结 cuBLAS、共享 ring/Graph/Event/外置 gate、DockerRunner 和离线 pip 新 prefix 发布路径。仅新增证据绑定入口与 NVIDIA 官方 power-limit 控制适配；未重写数学或构建框架。普通时钟三 AA / 864 段 7.699345% 拒绝且独立审计通过；100W 热诊断 90 秒/54°C/热计数无增加通过，随后单次冻结100W三AA/864段15.011395%仍拒绝、0AB/0证书；最高55°C/热计数不增、240W恢复。49CPU及预览11输出13尝试/592文件独立审计通过，另native诊断54段/73040 Graph仅作根因线索。用户新提供3060，独立预检后正在准备同版本环境，后续再回3070。[来源/命令/原始失败](adaptations/20261124-sm86-gemm-thermal.md) · [状态](../docs/SM86_GEMM_STATUS.md)。

## 2026-11-24 3060公开对照与3070机制确认

上述KernelBench423217d...、两可信SM86 cuBLAS、低权限DockerRunner/SourceRunner、shared ring/Graph/Event/外置gate继续复用；数学、原封存及原生产组件未改。新增目标3060采用NVIDIA原生CDI，补实际运行镜像绑定与可信父容器Docker控制，不放宽worker边界。320QA/32Sanitizer/两故障恢复、三AA三AB/1728段与960文件CPU审计PASS，最大0.479820%。随后3070固定8block ON/OFF+RAW/CPU耗时144段/141文件审计PASS但波动仍存在（ON17.024809%、OFF5.272610%），无新证书。原始失败与启动编码0Event失败均保留，未重复旧身份。[来源/命令](adaptations/20261124-3060-gemm-counterpart.md) · [3060结果](../docs/SM86_GEMM_3060_COUNTERPART.md) · [3070结果](../docs/SM86_NVML_DIAGNOSTIC.md)。

## 2026-11-24 RTX3060正式交付准备

用户已明确推进3060正式收尾。沿用上述KernelBench pin/原CUDA及Registry(data,catalog)、deployment(registry_factory,runner_factory)、已验收trusted Docker父/子控制；独立CDI runner只替换image/GPU注入参数，专用catalog/CLI参考现有SM89发布。54CPU、三launcher CPU检查与不可变0.1.3rc1构建通过，原3070运行文件不改。当前SSH KEX前关闭，实卡安装/新独立验收/证书/GA均未完成；拒绝模拟签证已CPU验证。[来源/精确差异/命令](adaptations/20261124-3060-production-release.md) · [当前准备状态](../docs/RTX3060_GEMM_RELEASE.md)。

## 2026-11-24 RTX3060正式交付验收完成

沿用上述KernelBench pin/原cuBLAS、Registry/deployment/worker/Graph与gate，仅适配nativeCDI/image和证据绑定CLI。新GPU三AA三AB、独立16项、54CPU/实卡恢复回退及入口均通过；PID域诊断修正和原FAIL完整留存。Docker官方save/load与OCI规范复用，镜像整体/各层及实际load通过；应用707文件恢复归档CPU校验通过。无新CUDA/模型调用，无任意源码生产。[来源/符号/差异/验证命令](adaptations/20261124-3060-production-release.md) · [正式交付](../docs/RTX3060_GEMM_RELEASE.md)。

## 2026-11-24 RTX3060 NCU实测及工程结项

固定KernelAgent e0647170...原parser/schema与现有ncu_adapter原字节复用；仅适配生产cuBLAS公共目标和独立NCU/CDI可信管理员诊断，生产runtime不改。3次真实kernel/19指标、原报告重新导入和三输出FP64审计通过；原0kernel启动失败/查询差异/截断工具备份记录保留。官方工具SHA、全部源码/原始报告与最终健康检查就绪。当前限定工程范围结项，原研究未验收项保持。[来源/符号/改动/验证命令](adaptations/20261124-3060-ncu.md) · [工程结项](../docs/PROJECT_CLOSEOUT.md)。
