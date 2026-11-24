# 执行清单与检查点

2026-11-24：**当前已确认范围的工程交付已结项**。3060 GEMM0.1.3、3070 Attention/MLA0.1.1保持各自生产资格；本次3060 NCU已真实采集3次cuBLAS kernel，每次19项指标有效，原报告重新导入/固定KernelAgent解析/三输出FP64复核通过。生产包、镜像、资格及原计时不变；最终qualified、GPU空闲/170W、无项目容器，0新增模型调用。原4090跨卡研究、3070 GEMM最终资格、通用源码生产及学习复盘保持各自未验收/延期状态。

[工程结项与延期范围](docs/PROJECT_CLOSEOUT.md) · [NCU实测与查看报告](docs/RTX3060_NCU.md) · [结项清单](releases/project-closeout-20261124/manifest.json)。原一次性入口不重跑，旧sealed不读不传不重跑。


2026-11-24：**RTX3060 GEMM 0.1.3正式交付已完成**，当前服务器入口为qualified，运行/故障回退/真实包切换和恢复备份均已验收。完整三AA三AB1728段：最大单对0.657401%、组0.251983%、最低保护0.999578594×；独立正确性16/16通过。

[运行、回退与恢复](docs/RTX3060_GEMM_RUNTIME.md) · [发行清单](releases/rtx3060-gemm-v0.1.3/manifest.json) · [验收汇总](docs/RTX3060_GEMM_RELEASE.md)。

范围为当前3060上的固定可信cuBLAS；3070 Attention/MLA0.1.1保持，3070 GEMM原拒绝不回填，4090目标C/任意模型源码生产仍未验收。无GPU/API任务在途，本轮0新模型（累计190/700）和0新CUDA编译。原失败、私有清单及旧原始证据保持；已消耗入口不重跑。

对应 [完整计划](PLAN.md) · [来源与版本](SOURCES.md)  
创建：2026-11-17 · 路线更新：v1.5，五个成熟项目为主要来源、CUDA kernel 主线，每天 6 小时冲刺。状态：2026-11-22，P0-01 至 P0-05 已验收，P1-02 首批 SM86 工程基线已验收，P1-03 首轮真实参数裁决已验收，P1-04 最小硬件反馈已验收；P1-05 报告／复算完成、计时未验收，P3-02固定环境下隔离／构建／恢复已验收，P3-01首轮真实源码／拒绝闭环已按范围验收，GEMM-REL-01开发安装范围已验收，GEMM-REL-02开发组件已验收，当前推进GEMM-REL-03发布候选与独立验收协议冻结，学习复盘延后。所有未勾选项均未整体验收。

## 当前检查点

3060-REL-00至04全部按实际验收完成，正式wheel、资格证书、入口、回退与恢复备份已交付。PID域诊断错误的原FAIL保持，RC2/GA真实宿主PID补充已独立审计；不重跑私有16项或任何计时。详[汇总](docs/RTX3060_GEMM_RELEASE.md)。

原3070 GEMM最终性能仍不合格；其Attention/MLA0.1.1继续保持原生产资格。P6原研究范围、4090目标C及任意源码生产仍按各项原状态，不因限定发行完成而全部勾选。已完成的54组A/B消融及零晋升结论保持；新模型预算190/700未增加。

## 使用规则

按生产交付依赖选择工程任务。P1-01 的学习复盘排在交付后。通过验收后在该项下追加日期、命令／报告路径、实测范围和限制，再勾选。仅写好代码、模型宣称完成、工具替身测试通过，都不能勾选实卡验收项。

所有 P1–P6 工程任务继承成熟实现优先原则：先列上游来源／接口和实际缺口，记录采用状态；控制层语言不要求 C++。

发现 deadline、GPU 或版本变化时先更新当前检查点与对应计划，不删除历史失败记录。遇到某条任务阻塞，可以继续其他依赖已满足的任务。

## P0：环境与可信评测

- [x] **P0-01 环境盘点**。2026-11-17 已确认连接、GPU、WSL2、原虚拟环境及工具路径，未安装／升级软件。[报告](evidence/20261117-wsl/REPORT.md)、[环境 JSON](evidence/20261117-wsl/environment.json)。
- [x] **P0-02 实卡冒烟**。2026-11-17 最小 CUDA 编译／执行通过，Triton 三种长度正确，CUDA Events 可用，NCU 单个 SM 周期计数器实际采集成功。[原始结果](evidence/20261117-wsl/smoke-results.json)。范围仅限最小测试，未建立性能收益。
- [x] **P0-03 评测合同与噪声**。2026-11-17，RTX 3070 / WSL2 实卡通过。冻结 N=1,048,589 FP32 add-one、精确参考与源码 hash；校准后冻结每段 512 次 CUDA Graph 节点调用。两种缓存策略 × 3 独立进程 × 9 对，无 profiler，108 个原始段全部保留。输入 hash 不变、输出精确、两侧保护区完整；CPU 复算与快照 hash 核对通过。单对最大偏差 2.4202%，六组几何均值偏离 1 最多 0.5040%，区间均含 1。[报告与复现命令](evidence/20261117-p0-03/REPORT.md)、[统计](evidence/20261117-p0-03/summary.json)。未锁频；仅该向量任务／Graph 模式的噪声校准，不构成 GEMM／MLA 或优化收益结论。该批未做故障注入／Sanitizer；故障与来源审查后来分别完成于 P0-04/P0-05。
- [x] **P0-04 故障与证据**。2026-11-17，3070 / WSL2 实测常数输出、输入污染、漏算尾部、缓存旧答案、延迟污染、主机异常、主机等待超时共 7 个故障候选，全部拒绝且 best 未变；正确 baseline 初始化与超时后 recovery 均通过首次及测速后检查。超时进程组含 worker/child，SIGTERM 后无残留；54 个原始 Event 段、源码／合同 hash、逐候选状态／日志／检查、best 轨迹留档。[报告与复现](evidence/20261117-p0-04/REPORT.md)、[独立审计](evidence/20261117-p0-04/verification.json)。12 项 CPU gate 模拟单独标识，不代替实卡验收；runtime_error 是主机异常。实测仅有限 GPU 工作后的主机超时，不证明 GPU 挂死恢复；同 UID/root worker 不构成完整安全隔离。P0-05 后来完成来源／主机原型审查；P3-02 完整隔离仍未验收，无新性能收益。
- [x] **P0-05 实施目录与来源锁定**。2026-11-18，已建立 include/src/kernels/evaluators/workloads/provenance 等职责及 CUDA v1 workspace/query/launch 设计；主机 C/C++ 合同检查 31 项、来源工具 6 项 CPU 夹具测试通过。61 个 reference、21 个历史源文件与旧证据清单锁定；固定本地 Git commit 的 GDN 两文件及 LICENSE 内容核验一致。旧 optimizer 无全包统一许可证／Git revision，教学许可不完整，明确 reference_only，本次未复制进主实现。[报告与复现](evidence/20261118-p0-05/REPORT.md)、[来源 lock](provenance/reference-lock.json)、[审查](provenance/REUSE_AUDIT.md)。未运行 GPU、未实现工程 GEMM／评测接入；原型 ABI 不作为 P1 的默认工程规范或必做编译任务。五个主要上游的成熟 binding／Builder 最小接入仍待 P1，主机检查不算 GPU 验收。

## P1：GEMM 与 Agent v0

- [ ] **P1-01 GEMM 学习版（交付后复盘）**。先选择有可核实来源的 CUDA 朴素、shared tile、WMMA 教学实现，按机制阅读／简化并逐步验证；解释线程地址、复用和参与同步条件。验收：小规模参考通过，能手算一个 tile。依赖 P0-02。
  - 2026-11-20 shared tile 跨 K 分块实测通过。复现命令、来源与独立复算见[报告](evidence/20261120-p1-01-shared-tail/REPORT.md)。手算题待用户作答。
  - 2026-11-20 朴素 CUDA GEMM 实卡通过 4 组输入共 8 次调用。复现命令、源码快照和独立复算见[报告](evidence/20261120-p1-01-naive/REPORT.md)。
  - 2026-11-20 单 warp WMMA 实卡通过 3 组输入共 6 次调用，独立复算 1536 个输出并核对静态 SASS。[报告与复现](evidence/20261120-p1-01-wmma/REPORT.md)
  - 2026-11-20 补齐原 shared tile 二进制的 memcheck/synccheck，启用 NVIDIA GPU 调试接口后重跑，各 28 次调用、0 errors。[修复与复验](evidence/20261120-p1-01-sanitizer-repair/REPORT.md)
- [x] **P1-02 合理工程基线**。先按采用表选择／验证 FlashInfer-Bench 或 KernelBench 的现成 CUDA 构建／评测入口，再适配有明确来源的 CUDA GEMM。拟以 FP16 输入／FP32 累加起步，输出 dtype 与接口在最小接入后冻结；对照 cuBLAS；冻结 12 个 workload 的划分与参考容差。验收：上游 commit／文件／改动与必要新增部分清楚，最小 CUDA 接入、开发集及边界正确；nvcc 构建、可信 Event 计时与 CUDA 路径 A/A 校准通过，保存编译选项与产物 hash。优先使用成熟评测组件，不以自写 C++ harness 为前提；若需新增，记录上游接口不能满足的具体原因。依赖 P0-03、P0-05。
  - 2026-11-21 已实卡接入固定 KernelBench 与 cuBLAS，建立可信参考路径并保存原始未配对 Event 数据。[工程接入](evidence/20261121-p1-02-kernelbench-cublas/REPORT.md)
  - 2026-11-21 同一二进制的 Graph A/A 试跑保存 27 对、54 段。最大单对偏离约 12.1%，项目进程退出后 GPU 利用率连续为 96%，需要独占 GPU 才能正式校准。[配对试跑](evidence/20261121-p1-02-gemm-aa/REPORT.md)
  - 2026-11-21 首批 SM86 范围验收：冻结 12 workload（8 开发 + 4 私有独立验收，后者仅 commitment、未执行），CPU FP64／原容差、输出合同、输入 guard、尾部 fallback 与 producer/candidate/consumer stream 检查通过。cuBLAS + NVIDIA 来源的 WMMA 模板，四种 warp 配置各 56 组检查、14 项非法输入、9 种故障拒绝通过；默认模板 memcheck/synccheck 各 0 errors。编译选项、源码／产物 hash 与日志见[实测](evidence/20261121-p1-02-contract-template/REPORT.md)。
  - 2026-11-21 GPU 空闲窗口完成原二进制同图 54 段复测，随后完成合同版 cuBLAS／默认模板双 Graph 的 3 锚点 × 2 缓存策略 × 3 进程，共 324 对／648 段。独立图 cuBLAS 存在约 1% 偏差、最大单对偏离 1.3832%，未将其算为收益；搜索前冻结 GM≥1.05、探索区间下界>1.02、保护 workload≥0.97，36 组 A/A 均不会误晋升。[校准与 CPU 复算](evidence/20261121-p1-02-gemm-aa-anchors/REPORT.md) · [冻结规则](workloads/gemm_measurement_v1.json)。模式或候选延迟范围变化需重新校准。
  - 2026-11-21 有限 CUDA 工作后主机等待超时被拒绝，进程组 worker/child 无残留，新进程 cuBLAS／WMMA 恢复通过、二进制 hash 未变。这不是 GPU 挂死恢复或完整容器隔离。[超时原始记录](evidence/20261121-p1-02-gemm-aa-anchors/run/timeout.json)。无真实模型调用、无跨实现正式 A/B 收益、无版本发布／回滚验收。
- [x] **P1-03 参数搜索闭环**。按 KDA 的合同／证据／晋升工作流，优先复用所选上游构建、执行和评测组件，再接入真实模型工具调用与 CUDA 合法参数提案。验收：新增适配路径的错误输出／输入污染／超时拒绝回归通过，KernelBench 的可适用对抗场景得到覆盖；至少一次真实候选裁决，错误和无增益均可正确结束。依赖 P0-04、P1-02。
  - 2026-11-21 首轮真实参数裁决验收：复用固定 KernelAgent 的 OpenAI-compatible provider，用户指定的 DeepSeek key 仅本机环境使用，一次 `deepseek-flash` 返回 8 warps，339 input + 92 output = 431 tokens。实际请求／response ID／usage、模型假设、local parent 绑定均留档。首轮仅公开 WMMA 描述；未发送精确开发输入或独立清单。[报告](evidence/20261121-p1-03-deepseek/REPORT.md)
  - 3070 真实候选三进程测前／后共 168 组 CPU FP64／输入／输出／stream 检查通过，A/B 864 段 + 候选自身 A/A 864 段保留；session 2/dev-04/repeated 的 25.3012% 离群超冻结稳定界限，外置裁决 REJECT，当前参照仍 4 warps，无晋升／收益。首次 SSH 认证失败及新目录重试保留，未重复 API 调用。
  - 新入口常数、输入污染、主机异常、有限 CUDA 后 host timeout 均拒绝；进程组无残留、新进程恢复通过。另固定 4-vs-4 负对照三进程实测：两缓存锚点 GM=1.000061/1.000376，收益门槛均 false，REJECT/duplicate；不替换原模型候选失败批次。[负对照](evidence/20261121-p1-03-deepseek/negative-control/decision.json)
  - CPU 轨迹重放与真实裁决／最终参照一致；10 个纯 CPU gate 测试方法通过，包含模拟晋升正例及证据篡改／缺失负例。模拟正例不算实卡收益。当前是一轮受约束参数裁决，持续多轮状态服务、生成源码、NCU、独立验收、容器隔离与发布回滚仍按后续任务验收。[回放](evidence/20261121-p1-03-deepseek/replay.json)
- [x] **P1-04 最小硬件反馈**。优先采用已 pin 的 KernelAgent profiler／指标解析与诊断结构，对 CUDA 入口做必要适配；采集 GEMM 的资源和少量本机可用 NCU 指标。验收：适配文件与验证明确，至少一轮“事实／假设／修改”能追溯到真实数据，缺失指标不伪造；不将上游指标列表当作 WSL/SM86 已全部支持。依赖 P1-03 与 P0-02 的 NCU 能力。
  - 2026-11-22 验收：固定 KernelAgent `load_ncu_metrics/metrics_to_prompt` 与 `NCU_METRIC_SECTIONS` 原文件直接导入，collector 对 CUDA 作指标筛选、kernel filter、API 区间和 .ncu-rep 薄适配。4/2-warps × 512³/1024³，各 3 launch records，实际 11 profile 指标 + 8 launch 统计；原 CSV/.ncu-rep、query gzip、hash、argv、stderr 与原始来源保留。旧 memory_dependency 未在 query 列出，标 unavailable，未填零。[采集与取舍](provenance/adaptations/20261122-p1-04-ncu.md)
  - 实际 1 次 DeepSeek、5,629 input + 498 output = 6,127 tokens，给出 fact ID／假设／竞争解释／2-warps 修改／方向预测。1024³ 活跃 warp 88.22→62.67、long-scoreboard 62.34→92.62，未支持改善预测；收集的 block/thread 限额支持 2-warps 线程容量上界 66.7% 的竞争解释，非唯一因果证明。[事实与验证](evidence/20261122-p1-04-feedback/REPORT.md)
  - NCU 完成后独立普通 Event 通道验证：168 组测前／后正确性通过，864 A/B 段 + 864 候选 A/A 段；A/A 最大偏离约 1.254%，两缓存锚点 GM 1.000899/1.000533、1024³ 保护项 0.962918/0.962517，REJECT/insufficient_gain_or_regression，当前参照保持 4。四种故障拒绝、超时清理／新进程恢复和 CPU 回放通过。[裁决](evidence/20261122-p1-04-feedback/evaluation/decision.json)
  - 4 个 parser CPU 测试方法、3 个反馈 schema 测试方法通过；四份原生报告只读导入、原 CSV 逐 launch 的上游 parser 重解析一致。缺失、真实零与单位分开保存。该轮不代表 B/C 对照、不把 NCU duration 当正式 Event 成绩，无生成源码／独立验收／完整容器隔离／生产发布／回滚验收。
- [ ] **P1-05 V0 报告**。报告与完整轨迹／CPU复算完成，计时仍未验收。冻结 best 对最初同一4-warps，首批A/A最坏3.042%；一次250ms长段追加仍最坏3.324%，不放宽原界限，不再挑选复测。无收益结论；4项封存验收未执行。[报告](docs/V0_REPORT.md) · [原始证据](evidence/20261122-p1-05-final/REPORT.md)。依赖P1-03/P1-04。

## P2：Softmax 与融合 Attention

- [x] **P2-01 稳定 Softmax**。验证 max/sum 归约；覆盖零值、较大有限值、非整齐长度。验收：数值标准冻结且通过。依赖 P0-03。
- [x] **P2-02 Online 状态**。实现分块 `(m,l,a)` 更新与两块合并。验收：能解释两个缩放因子来源；tile 改变不改变语义；全 mask／空块规则明确。依赖 P2-01。
- [x] **P2-03 Attention 全流程**。结合已有教程理解 QK→softmax→PV，优先参考／适配 FlashAttention 的适用 CUDA 路径再做融合；先锁定具体来源与硬件支持。验收：小规模独立参考及 causal/batch/head 测试通过。依赖 P2-02、P1-02。
- [x] **P2-04 性能与解释（后端诊断范围）**。分别测分解版、合理融合版和可用 SDPA backend，标记 prefill/decode。验收：核心融合路径不物化完整 score；报告精度、backend 和计时边界。依赖 P2-03。

P2-03工程验收（2026-11-22）：固定FlashInfer fa2在3070/Torch2.14/CUDA13的新prefix中，仅2处JIT C++17→C++20适配后公开CUDA共40/40项（含Softmax23项）通过。Attention含batch2/head3、D64/128、causal和custom mask，输入不变；后续offset8/非默认stream/全mask与causal交集/输出合同/未对齐拒绝delta在native、memcheck、synccheck下通过，0errors。原C++17失败完整保留。GPU核心不物化完整score；尚无性能/部署结论。[原始40项](evidence/20261122-attention-mla-sm86-cxx20/verification.json) · [guard/Sanitizer](evidence/20261122-attention-mla-sm86-guards/verification.json)

P2-01工程验收（2026-11-22）：固定PyTorch CUDA PersistentSoftmax薄绑定在3070编译，22项FP16/FP32公开QA通过；宽度1..4097含非整齐及大有限值，>2048明确ATen CUDA回退，FP32输出。新增16项offset17/输入guard/非默认producer-consumer stream检查及4个非法合同拒绝，native/memcheck/synccheck三模式均通过，两工具0errors。不是性能或生产部署验收。[实卡审计](evidence/20261122-softmax-sm86/verification.json)

P2-02工程数学验收：CPU FP64中tile=1/7/32/64/500与整块attention一致，空块/全mask输出零且LSE=-inf，块合并的两项分别乘exp(m_old-m_new)；5项参考测试含本项通过。手算/口述复盘仍延后。[CPU证据](evidence/20261122-operator-reference-math/result.json)

P2-04已按后端诊断范围验收：3070/4090各3独立进程，prefill/decode×分解版/FlashInfer fa2/强制SDPA MATH/FLASH，24行、240原始Event样本/卡；先后正确性、具体backend和完整score边界均记录。3070首进程prefill部分高值全部保留，不形成稳定提速或Agent收益结论；原SMI路径0样本失败也保留。[全表与边界](docs/ATTENTION_BACKEND_DIAGNOSTIC.md)

## P3：源码改写与 Agent v1

- [x] **P3-01 源码提案**。优先适配 KDA／ForgeKernel 等已有候选与固定评测流程，补齐 parent hash、源码提交和受约束入口，评测器独立保存。验收：实际完成一次结构变换；源码变化使旧证据失效。依赖 P1-03、P0-05。
  - 2026-11-22 P3-01首轮真实CUDA源码提案已按限定范围验收：DeepSeek一次2,232 tokens，Git --recount仅恢复diff计数后审查；fresh build、160项独立正确性、memcheck／synccheck各8项且0 errors。新源码三会话864段A/A最坏3.233%超2%，外置REJECT，原best／cuBLAS默认未变，无A/B／收益／生产发布。原格式失败、Sanitizer协议拒绝、测后汇总NameError和SSH失败均保留，本机独立审计PASS。累计三次真实调用8,790 tokens。[报告](docs/P3_01_SOURCE.md) · [裁决／范围验收](evidence/20261122-p3-01-source-sanitizer-channel/acceptance.json)。21项源码相关CPU测试通过，与GPU实测分列。源码／parent／binary独立绑定，旧成绩失效。补丁格式恢复与源码审查由可信控制侧完成，不宣称任意源码完全自动执行或生产发布。
- [x] **P3-02 执行隔离与恢复**。建设独立低权限 GPU worker、凭据和网络隔离、只读评测、超时与恢复。验收：未受信任候选的进程、文件和网络边界经实测；失败／恢复不污染 best。依赖 P1-02；在自动执行 P3-01 的模型生成源码之前完成。
  - 2026-11-22 已实测固定现有镜像下可信GEMM／边界／五故障／五恢复共15项，控制侧保存参考和裁决，worker UID65534、network none、只读根目录、cap-drop、no-new-privileges、PID/CPU内存上限均观察一致。超时是有限CUDA后主机等待；不宣称GPU挂死恢复或显存配额。[首批证据](evidence/20261122-p3-02-runtime/verification.json)
  - 独立非root、无GPU／无网络builder从空cache编译通过，新binary `66fcc80f…9dfc3` 再通过15项运行／恢复及10项CPU输出解析检查。三次依赖／Ninja失败留档，只修改新副本。[构建／回归报告](evidence/20261122-p3-02-build-fixed-tool-path/REPORT.md)。后续runtime输出／资源及开发wheel安装回退回滚已通过；后续builder输出／日志及审查源码闭环已验收；真实模型源码提案、正式独立验收／生产发布仍未验收。
  - 2026-11-21 非 root、无网络、只读挂载的容器实卡预检通过。候选镜像及完整隔离、超时恢复尚未验收。[容器预检](evidence/20261121-gpu-worker-preflight/REPORT.md)
  - [x] 2026-11-22 runtime宿主输出RW挂载已移除，stdout/stderr及metadata/NPY限长；固定29项实测通过，12类故障／12恢复。[CPU证据审计](evidence/20261122-p3-02-bounded-runtime-confirmation-fix/verification.json)
  - [x] 2026-11-22 开发wheel新prefix安装、cuBLAS默认、五场景实际回退与显式版本回滚通过；15次独立正确性，另3个CLI负例、4个CPU mock策略测试。仅development-only，不是生产发布。[安装审计](evidence/20261122-p3-02-install-rollback-dev2/verification.json)
  - [x] 2026-11-22 builder无GPU、无宿主RW挂载、tmpfs产物／有界日志；11场景通过，3次fresh build。新产物在独立GPU worker共27次调用，26次正确、1个真实零输出源码拒绝，编译错／native init abort／带child超时恢复及真实freeze不变均验证。[报告](evidence/20261122-p3-02-bounded-builder/REPORT.md) · [组合验收](evidence/20261122-p3-02-bounded-builder/acceptance.json)
  - 本项仅验收声明镜像／SM86下普通容器的进程、文件、网络、资源与恢复边界；不包含GPU驱动恶意代码隔离、显存配额、无限GPU kernel取消或生产发布。正式封存验收／最终计时另验，保留P1-05未通过状态。
- [x] **P3-03 B/C 对照开关（接口与成本记账范围）**。同模型、同初始代码与修改空间，分别提供执行反馈或执行+NCU 反馈。验收：B 不读取硬件指标；API usage 和 profile 成本可追溯。依赖 P3-01、P1-04。
- [x] **P3-04 小规模试验（修正版36提案，单次搜索范围）**。2 个锚点×A/B/C×1 次×6 提案。验收：全过程可完成，统计与预算记账一致；确认完整实验成本后再扩大。依赖 P3-02、P3-03。
- [ ] **P3-05 经验条目，可选**。优先采用 ForgeKernel 的 CUDA 知识／来源格式，记录适用硬件、shape、变换、证据和反例，必要时补本项目缺失字段。验收：检索结果是有范围的事实／假设，跨卡需重验。依赖 P3-04。

## P4：经典 MLA 参考与基础 Decode

- [x] **P4-01 数学等价**。随机小权重实现显式恢复 K/V 和矩阵吸收两条参考。验收：同输入输出一致；解释 latent cache、head up-projection、scale。依赖 P2-03。
- [x] **P4-02 解耦 RoPE 与层边界**。增加位置 key、相同 RoPE 预处理、cache append 语义；定义 core Z/LSE 与完整层输出边界。验收：位置、mask、LSE 底数、原始 scale 清楚。依赖 P4-01。
- [x] **P4-03 基础融合 MLA**。先核对固定 FlashMLA／FlashAttention 路径及本机硬件差异，再适配连续 cache、Sq=1、FP16、FP32 累积的 CUDA 实现；缺口用最小变更补齐。验收：与两条参考一致；没有完整 score 张量落地。依赖 P4-02、P2-02。
- [x] **P4-04 Workload 冻结**。冻结 12 个 MLA workload 与独立验收划分；扩大到 R512/Drope64 等代表维度。验收：显存预算包含参考和 workspace，不能用小 R 测试冒充真实维度。依赖 P4-03。
- [x] **P4-05 MLA 接入 Agent（固定CUDA SDK配置／显式split图范围）**。沿用 GEMM 阶段已验证的上游构建／评测与 agent 组件，适配 head/KV tile、warp、split 选择及 CUDA 源码入口。验收：一轮真实 API→编译→验证→测速→裁决完成。依赖 P4-04、P3-03。

P4-02/P4-03工程验收（2026-11-22）：3070实际运行固定FlashInfer MLA fa2，8个R512/Drope64公开case，heads16/32/64/128，L1/15/17/257/1025/4097、ragged/空请求、随机页表，CPU FP64逐元素校验及自然LSE通过。显式位置RoPE至8191、两阶段cache append跨尾页且旧值/tail保留、split/combine空分片通过；后续offset/stream及core Z→U_V projection在三模式含两Sanitizer0errors。只改编译标准，不改native算法；不是完整DeepSeek模型/生产发布。P5-01/P5-02特性已有公开GPU证据，仍待P4-05 agent依赖与冻结workload的完整回归。[合同](docs/OPERATOR_CONTRACTS.md) · [CUDA证据](evidence/20261122-attention-mla-sm86-cxx20/verification.json) · [guard证据](evidence/20261122-attention-mla-sm86-guards/verification.json)

P4-01工程数学验收（2026-11-22）：同一随机FP64输入、head up-projection下，显式恢复K/V与吸收Q/恢复V两路径逐元素在2e-13内一致，含空/非整齐有效段。latent cache存C；q_abs=Q_nope·U_K^T；core输出Z，层输出Z·U_V；scale仍按吸收前D_nope+D_rope计算。仅CPU数学，不代表GPU完整层。[测试](tests/test_operator_references.py) · [证据](evidence/20261122-operator-reference-math/result.json)

## P5：MLA 优化与服务器卡

- [x] **P5-01 Split-KV**。正确保存／合并 partial `(m,l,a)`，计时包含 combine。验收：极短、不等长、空分片、非整齐分片通过；与不 split 版本等价。依赖 P4-05。
- [x] **P5-02 分页与 ragged**。处理页表、尾页、请求重排与边界。验收：连续参考及随机页映射测试通过。依赖 P5-01。
- [x] **P5-03 已授权服务器兼容检查（现有4090）**。确定公司主力卡／实验卡、预算授权、FlashMLA 固定版本和 NCU 权限。验收：明确要运行的 kernel 支持该卡，预算已获授权；短时环境验证成功后才扩大时长。依赖 P0-01；可以提前做只读准备。
- [x] **P5-04 专家对照与跨卡初测（不支持／已运行边界明确）**。在兼容设备运行锁版本 FlashMLA dense decode；与自身核心输出、scale、dtype、mask、workspace/metadata 边界对齐。验收：真实结果或明确的未运行原因，禁止编造成熟库成绩。依赖 P5-03、P4-04。
- [x] **P5-05 完整优化案例（成熟MLA SDK配置／拒绝轨迹范围）**。选择一个公开开发 workload，记录基线→瓶颈→多个候选→最终复测。验收：原始代码／计数器／配对测量齐全，含失败假设。依赖 P5-01、P3-03；paged 案例另需 P5-02。

## P6：正式实验与交付

- [x] **P6-01 冻结实验协议**。54组独立搜索/最多648提案，B/C最多432新API且受总700次硬上限；目标、父源码、模型、18份初始profile、cache、停止/噪声规则均在首次正式调用前冻结。B不调用profile loader，C只见绑定的初始诊断，独立值未读。协议SHA56b6bd…；[协议与范围](docs/FORMAL_STUDY_PROTOCOL.md)。5项CPU状态回归及正式请求/自身历史复算通过，P6-02已按本冻结范围完成，见正式结果报告。
- [x] **P6-02 三方法消融（冻结固定父版本/初始硬件反馈范围）**。54/54组、221提案、158真实API/858008tokens；112重复、109拒绝，0筛选通过。13新源码身份实际评测（1编译失败、12GPU正确性/Sanitizer通过），14688段原始Event；响应→patch/config→descriptor、信息隔离、停止与成本独立CPU复算PASS。逐提案/搜索/1632延迟行/成本/曲线已导出；C收益未建立，局限明确，未称完整自适应或跨卡C。[结果与复算](docs/FORMAL_STUDY_RESULTS.md)。
- [ ] **P6-03 跨卡迁移**。MLA直接复用和普通配置空间已完成，3候选共5184段、全部拒绝；源方案与通用初始相同，按相同身份复用。目标4090 C因ERR_NVGPUCTRPERM未运行，完整项保持未验收。2026-11-23用户转3070单卡交付，此研究延后并移出当前交付依赖，非通过。[结果/限制](docs/CROSSCARD_RESULTS.md)。
- [ ] **P6-04 独立验收与复测**。GEMM16/16与MLA8/8调用、16/16输出的冻结独立正确性均已通过且禁止重跑sealed。旧cuBLAS4.274%/锁频250ms2.849%/1秒2.040408%拒绝保留。用户重启后实际HAGS On，新protocol ec6bbbc...保持原阈值：320QA通过、仅AA0/288段，单对最坏5.142371%；下一轮前连续空闲未满足停止，3AA未完成、0AB，锁频复原，590文件CPU审计PASS但资格不通过。不是完整三会话结论，也不再重采该失败身份。[新实测](evidence/20261123-cublas-hags-qualification/verification.json) · [当前门槛](docs/PRODUCTION_BLOCKERS.md)。 新增用户3%部署v2的历史复核：旧完整1秒2.040408%及250ms2.849277%的A/A单项通过；明确观察后修订，原2%拒绝不改、无AB/证书。新4090按v2单独冻结实测待连接，不能替3070回填完整资格。[v2复核](docs/DEPLOYMENT_STABILITY_V2.md)。 新SM89可信cuBLAS已按预先冻结v2完成三AA+三AB/1728段，0.664015%/AB最低0.999909889×，917文件CPU审计通过；只解决该目标路径性能单项，不覆盖SM86或完整生产隔离。[SM89实测](docs/SM89_CUBLAS_STABILITY.md)。 新SM89独立16/16及正式GEMM scoped qualified通道完成，见SM89-REL系列；不将其写回旧SM86源候选性能通过。
- [ ] **P6-05 发布与运行文档**。0.1.0rc1未发布预览已新prefix/空cwd安装验证19CPU、5GPU输出7尝试、crash/timeout恢复及rc1→dev6实际回滚，独立审计PASS；无资格证书时qualified通道GPU前拒绝。原dev6 wheel、旧bundle45/RC冻结9文件保持；正式报告/逐workload数据/成本/PNG/SVG/运行文档已准备。生产准入未过，不勾正式发布。[预览包](docs/RC1_PREVIEW.md) · [交付状态](docs/DELIVERY_STATUS.md)。 2026-11-23新增SM89固定可信GEMM 0.1.0生产通道已本地发行并部署；本大项的完整跨卡/全部算子范围保留未整体验收，明细见SM89-REL-04。

## 首个 GEMM 交付补齐

- [x] **GEMM-REL-01 可安装离线优化入口**。把已验收provider／source guard、独立builder、参考不进入候选的GPU验证、外置裁决和证据目录接到可安装的一次优化命令；预算／重复运行／失败／恢复状态明确。依赖P3-01、P3-02。先离线及固定夹具验证，不自动扩大API预算。运行时仍只激活已验收artifact，未通过候选不能进入allowlist。完成后再冻结正式发布版本及封存验收协议；P1-05旧失败不变。
  - 2026-11-22 GEMM-REL-01可安装离线入口已按开发范围验收，最终dev4 wheel可在新prefix／空cwd运行。dev3安装路径fresh build＋160项独立正确性、Sanitizer各8项0errors及一次默认cuBLAS通过；dev4仅command／版本／bundle改动，13条空cwd命令含14项CPU回归、历史失败幂等拒绝和实际版本回滚通过，GPU组件逐字不变，明确继承刚取得的实卡证据而未重复GPU。3项无损传输CPU测试／两阶段独立审计PASS。0新API／0计时，默认cuBLAS／allowlist未变，生产发布仍未通过。[说明／wheel](docs/OFFLINE_OPTIMIZER.md) · [范围验收](evidence/20261122-gemm-rel01-dev4-delta/acceptance.json)。 原empty choices／调用预留计数CPU失败及.DS_Store审计错误留档，最终14项安装CPU回归通过。真实provider复用但本轮未新增网络API；源码审查为控制侧SHA绑定，未实现双binary A/B，A/A通过也只能NO_PROMOTION_PENDING_AB。
- [x] **GEMM-REL-02 双binary A/B与发布裁决（开发组件范围）**。复用原paired Event与外置门槛，为独立baseline／candidate二进制绑定编译、合同、输出witness及原始段；未知／缺失／未稳定证据不得晋升，落盘决定与运行时激活分离。先用CPU／固定夹具验证，再按未重复的合适实卡协议验收。依赖GEMM-REL-01；不重采旧P1-05或已拒绝P3-01 A/A，不自动增加API预算，不读取封存验收。正式冻结和封存发布另验。
  - 2026-11-22 GEMM-REL-02双binary评测／外置筛选已按开发组件范围验收，最终dev5 wheel新prefix／空cwd35项CPU方法通过。真实双扩展16项QA含4个零输出候选拒绝、baseline正确；两项Sanitizer各4项0errors。全轮92项输出校验含4个预期失败；空闲时一批WMMA/cuBLAS诊断72原始Event段、16合同witness／256图输出hash通过。诊断不可晋升，完整新source三A/A→三A/B未实跑；0新API、不重采旧A/A，默认cuBLAS／freeze／allowlist未变，无收益／生产发布。[报告／wheel](docs/DUAL_BINARY_EVALUATION.md) · [范围验收](evidence/20261122-gemm-rel02-dual-installed-fixed-test-import/acceptance.json)。 CPU模拟KEEP只作规则验证，落盘SCREENING_PASS不激活运行时；诊断template/cublas会被source gate排除。首次空cwd对照测试仓库import失败留档，GPU未启动；包内快照导入修复后35项安装CPU方法通过，本机独立审计PASS。
- [x] **GEMM-REL-03 首批发布候选与独立验收协议冻结**。冻结包／artifact／默认backend／支持矩阵和可回退版本，明确开发KEEP与正式发布的区别、原始证据／成本／停止条件。先完成可审查RC及独立验收／回滚协议，不读取／执行封存清单、不新调用模型；协议固定后才安排独立验收。新source完整三A/A→三A/B若用于收益必须另有新合格候选，不能重采旧拒绝或借诊断晋升。正式发布状态仍false，按实际后续验收推进。

- [x] **GEMM-REL-04 一次性独立正确性与公开恢复验收（限定范围）**。冻结 dev5/两份cuBLAS产物后执行，4个封存workload × 2 seeds × 2 artifacts，16次独立GPU输出全部正确（最大绝对误差1.14441e-5），输入hash/offset guards/producer-consumer stream/输出合同/UID65534及cleanup均通过。195份私有文件逐hash复核；模型不见精确shape/seed。公开crash、timeout、损坏产物回退及显式版本回滚6场景/8次尝试通过。0新API、0性能采样，默认/allowlist/best不变；production仍false，不能替代P1-05或P6的最终三session性能验收。首次自动审批拒绝后，用户明确授权传输和执行，原拒绝记录保留。[CPU审计](evidence/20261122-gemm-independent-acceptance/verification.json) · [公开恢复](evidence/20261122-gemm-rc1-public-recovery/result.json)

### 完整项目连续实施记录

- 4090/SM89/CUDA12.6已只读确认。项目独立目录安装Python3.12.12及固定PyTorch/FlashInfer中；无Docker且user/network namespace被宿主拒绝，服务器目前仅执行已审查源码/成熟库。
- DeepSeek新增总预算已获用户明确授权最多700次；历史3次8,790 tokens另列。先小规模、后正式实验；失败/重复/未知结果占额度，不自动重试或超额。[授权记录](evidence/20261122-full-project-readiness/model-budget.json)
- 新增真实B/C接口pilot各1次，共6,135 tokens；同初始源码/任务/模型，B不读profile，C提供历史真实NCU事实。C提案覆盖当前tile，静态语义拒绝不执行；B审查后待fresh build/QA。历史profile墙钟未记录，明确unknown，不能当正式ABC。累计5次14,925 tokens，本次700次上限已用2次。[接口审计](evidence/20261122-feedback-live-pilot/verification.json)
- P3-03 B/C提示构造边界5项CPU检查通过，B不调用profile loader，C附带来源hash/采集成本；全局调用账本3项CPU测试通过。尚无本轮真实B/C搜索。[离线验证](evidence/20261122-feedback-channels/verification.json)
- MLA 8开发+4独立workload已生成并commit，R512/Drope64真实维度；独立值只存.private，模型只见hash，尚未读取/执行。8项公开开发全量回归已通过50/50（2 seeds×3 page sizes=48，另zero/signed4边界2项）。实测峰值154,310,656字节；16输入ring/4 split workspace及1GiB runtime+256MiB flush保守预算最高2,216,755,200字节；CPU参考预算单列，符合3070。[开发集审计](evidence/20261122-mla-development-sm86/verification.json)。P4-04按冻结/开发范围勾选，4项独立清单仍未读取或执行。[冻结记录](evidence/20261122-mla-workload-freeze/freeze.json)
- P2/P4可信CPU FP64参考5项数学测试通过，涵盖稳定softmax、online状态分块/空块/全mask、bottom-right causal、MLA显式与吸收等价、RoPE位置与内积；不是CUDA验收。[证据](evidence/20261122-operator-reference-math/result.json)
- 已锁PyTorch v2.7.1/FlashInfer v0.2.8源，薄CUDA绑定和fa2适配正在验证。P2/P4未勾选。FlashMLA固定历史版本不支持SM86/SM89，对照保持未运行，不编造成绩。

## SM89首个生产通道（2026-11-23，独立范围）

- [x] **SM89-REL-01 可安装目标版本**：dev7基线与rc2预览两个新不可变wheel、空cwd/新prefix安装，27CPU/9GPU输出11尝试、crash/timeout/坏产物fallback、版本与真实包回滚通过；1203文件CPU审计PASS。
- [x] **SM89-REL-02 固定SDK执行边界**：实卡UID/GID65534、CapEff/CapBnd零、NoNewPrivs、非Unix socket拒绝、私有目录不可读、产物/输入不可写、控制信号拒绝、canary/凭据环境清除、byte/time/rlimit与日志洪泛拒绝/新worker恢复通过。不声明通用源码沙箱或namespace/cgroup隔离。
- [x] **SM89-REL-03 新目标独立验收**：固定rc2后服务端生成并封存全新合成清单，16/16 GPU输出通过、246私有文件及本机FP64重算PASS，最大误差1.907349e-6；只公开commit/汇总，旧sealed无读取/传输/重跑。
- [x] **SM89-REL-04 0.1.0 scoped qualified发行/部署**：真实证据证书、9运行文件/两CUDA绑定与rc2逐字相同；30CPU及6GPU输出8尝试、qualified版本回滚和实际0.1.0→rc2→0.1.0切换通过，581文件CPU审计PASS。本地wheel/发行manifest/入口/新状态目录均就绪；production=true只限固定可信SM89 GEMM。[运行](docs/SM89_RUNTIME.md)。

## SM86 Attention / MLA 剩余发布工作（2026-11-23）

- [x] **SM86-OPS-00 切换准备与4090归档**：3070 SSH/固定镜像通过；4090最终部署593文件、本地原备份+差异覆盖24956软件文件/1052链接。GPU97%时未执行新GPU任务，服务器未关机/退租。[范围](docs/SM86_PRIMARY_DELIVERY.md)。
- [x] **SM86-OPS-01 运行时资格与合同边界（门禁实现/开发实测范围）**：固定GPU/驱动/image及运行文件证据门禁、真实容器边界witness、37CPU与5项GPU前拒绝通过；无证书qualified在GPU前拒绝，真实生产资格归OPS-03，不以CPU模拟代替。
- [x] **SM86-OPS-02 安装与恢复（0.1.1rc1开发范围）**：新prefix/空cwd37CPU、13GPU输出17尝试、两算子crash/timeout/fresh恢复、rc1→已归档dev6→rc1真实包切换通过；含冒烟15份成功输出独立NumPy FP64复算，2465文件SHA审计PASS，原两worker metadata拒绝完整保留。[证据](evidence/20261123-sm86-operators-rc1/verification.json)。
- [x] **SM86-OPS-01b 完整SDK目录准入**：GPU前完整文件集合/无symlink和special/有界manifest及非对象拒绝，rc2新包40CPU/2真实输出/四目录拒绝通过，4773文件独立审计PASS；正式包真实证书下又验证启动/bytecode/篡改拒绝。[rc2证据](evidence/20261123-sm86-operators-rc2/verification.json)。
- [x] **SM86-OPS-03 目标性能与发布（固定SDK范围）**：原三AA540段拒绝保留；恢复成熟共享物理ring、预先冻结唯一方法更正批，三AA+三AB1080段原2%/1.5%门槛通过，最坏1.935923%/0.447360%、AB最低0.999478866×。新Attention独立8/8及旧MLA同数值/AOT证据继承；正式0.1.1新prefix/空cwd40CPU、9输出11尝试、五真实资格拒绝、GA→rc2→GA实际包切换及5160文件NumPy FP64审计PASS。12运行文件与rc2逐字一致，本地发行并部署。[运行](docs/SM86_ATTENTION_MLA_RUNTIME.md) · [正式审计](evidence/20261123-sm86-operators-qualified-installed/verification.json)。
- [x] **SM86-OPS-04 运行文档与示例**：持久launcher/current/SDK/轮子就绪；Attention/MLA两例工具已上传及SHA核验、CPU合同预检通过，恢复/预览回退/正式恢复命令和明确scope已写入运行说明。

## SM86 GEMM最终生产资格续接（2026-11-23）

- [x] **SM86-GEMM-00 新环境协议冻结**：普通动态时钟/HAGS On、既有cuBLAS两binary/固定image，用户3%部署政策在首次Event前绑定；旧锁频失败不重采。49CPU含独立政策/目标/功耗温度绑定负例通过。
- [ ] **SM86-GEMM-01 目标性能实测与独立审计**：普通时钟320QA/三AA864段、615文件CPU审计完成，最大单对7.699345%拒绝；一次100W新条件320QA/三AA864段、616文件审计完成，最坏15.011395%/组1.387542%、最高55°C/热计数无增仍拒绝，0AB。温度受控不等于抖动消除，0新生产证书，不再重采两身份。[100W审计](evidence/20261124-sm86-gemm-power-qualification/verification.json)。
- [ ] **SM86-GEMM-02 真实资格与正式安装**：实测/审计通过后签发绑定当前GPU/driver/image/源码/policy的真实证据；新不可变包实卡执行、故障恢复/产物及实际包回滚，同时保持Attention/MLA已验收范围。
- [x] **SM86-GEMM-02a 可安装预览与恢复（0.1.2rc1开发范围）**：新prefix/空cwd49CPU、GEMM8输出10尝试+两既有算子qualified冒烟+直接边界1输出，共11输出13尝试；crash/timeout/产物损坏回退、版本回滚和rc1→0.1.1→rc1实际包切换通过。592文件及全部NPY/帧/运行边界独立CPU审计PASS；缺GEMM证书qualified在GPU前拒绝，无正式GEMM资格。[审计](evidence/20261124-sm86-gemm-0.1.2rc1-installed/verification.json) · [清单](releases/sm86-gemm-v0.1.2rc1-preview/manifest.json)。

## 用户新增3060对照（2026-11-24，随后回3070）

- [x] **3060-CMP-00 独立预检与连接隔离**：RTX3060/12GiB/driver580.95.05/原生Ubuntu22.04、GPU0%/1MiB；Docker/NVIDIA runtime/免交互sudo可用，宿主两Python无Torch/CUDA。新连接私有0600，不覆盖旧服务器。只读预检非GPU资格。[预检](evidence/20261124-sm86-3060-preflight/preflight.json)。
- [x] **3060-CMP-01 同版本运行时与公开正确性**：新image7ed9d551...、nativeCDI真实UID65534/零cap/只读/无网络/无socket边界通过，320QA/32Sanitizer零错误及crash/finiteCUDA超时两fresh恢复通过；五核心数值库与3070旧image SHA完全相同。失败准备和legacy GPU注入失败原始记录保留。尚不代表正式性能或生产资格。[precheck](evidence/20261124-sm86-3060-counterpart/precheck-launch-result.json)。
- [x] **3060-CMP-02 冻结目标性能对照**：完整三AA三AB/1728段，最坏0.479820%/组0.185669%、最低保护0.999335845×，960文件CPU逐SHA/帧/见证/pilot/gate复算PASS；收尾0%/1MiB/40°C，无项目容器，170W未改。0模型/封存/编译，不是新生产证书，不回填3070。[报告](docs/SM86_GEMM_3060_COUNTERPART.md)。
- [x] **3060-CMP-03 回3070机制确认及后续准备**：固定8block/144段，ON17.024809%、OFF5.272610%，关闭监测不足；141文件CPU逐SHA/帧/见证/双时钟/全部统计PASS，内外时钟恢复、240W保持。准备失败0Event/10文件保留；原HAGS恢复计划只读核验完毕，未改设置/重启，下一批正式资格仍属未完成的SM86-GEMM-01。[诊断](docs/SM86_NVML_DIAGNOSTIC.md)。

## RTX3060 GEMM正式交付（2026-11-24，用户已选择）

- [x] **3060-REL-00 专用入口/不可变预览构建（静态与CPU范围）**：原Registry/deployment扩展点及固定runner两处绑定差异复用，原3070共享运行文件不变；54CPU、3launcher CPU检查、12运行文件冻结通过。0.1.3rc1 wheel c81d6a10d78b034d60f5d2163bdf55720eeb06a5ee0e9140076dc26dcad102e9，尚无GPU验收。[准备说明](docs/RTX3060_GEMM_RELEASE.md)。
- [x] **3060-REL-01a 新目标边界/正确性及实际预览安装**：GPU0140b895.../image6ce20122...，320QA/32Sanitizer零错误、两fault fresh恢复及真实低权限边界通过；848公开precheck文件归档。0.1.3rc2 wheel5b45358a...实际安装54CPU、7输出9尝试、回退/版本回滚/边界及419文件CPU审计PASS。[预览审计](evidence/20261124-3060-0.1.3rc2-installed/verification.json)。
- [x] **3060-REL-01 新实例独立资格及预览安装恢复**：GPU0140b895.../image6ce20122...，320QA/32Sanitizer/两fault恢复通过；新协议19912fbc...完整三AA三AB1728段，最坏0.657401%/组0.251983%、最低保护0.999578594×。939文件逐SHA/帧/NPY/见证/pilot/gate CPU审计PASS；收尾0%/1MiB/32°C/170W/无项目容器。RC2实际安装419文件审计保持。长连接回执丢失后仅只读恢复服务器已保存exit0结果，未重跑样本。[资格审计](evidence/20261124-3060-next-qualification/verification.json)。
- [x] **3060-REL-02 新目标独立验收**：冻结RC2后服务端全新私有清单一次16/16输出通过，220私有文件逐SHA/帧/NPY/FP64复算PASS，最大绝对误差9.536743e-6。精确值未展示模型，旧sealed不读不传不重跑。控制容器PID探针域错误已披露，不作宿主隔离结论；同包/镜像真实宿主PID公开补充通过，448文件CPU审计PASS并绑定补充SHA。[独立审计](evidence/20261124-3060-next-independent/verification.json)。
- [x] **3060-REL-03 真实证书/GA安装与回滚**：证书627007af...、唯一0.1.3 wheel b3a98c60...；54CPU/9输出11尝试、两fault/坏产物回退、版本及GA→RC2→GA实际包切换、driver/policy负例完成。原validator误用容器PID导致FAIL保留，GA真实宿主PID另补1公开输出/497文件审计通过；468文件CPU安装审计对完整输出及所有边界和补充逐项复核后接受，不重跑原批。[安装审计](evidence/20261124-3060-0.1.3-installed/verification.json)。
- [x] **3060-REL-04 生产入口/发行清单/运行文档及恢复备份**：正式launcher默认qualified，串行GPU锁、独立workspace、目标/image/profile门禁；三输出、实际GA→RC2→GA包切换及160W拒绝/170W恢复和589文件审计PASS。镜像服务器实际load+Mac逐层DiffID/整体SHA通过；应用和回退prefix707文件归档SHA/权限通过。最终qualified/空闲/170W/无项目容器。发行manifest及运行/恢复文档就绪，production=true仅本卡固定cuBLAS。[运行](docs/RTX3060_GEMM_RUNTIME.md) · [发行](releases/rtx3060-gemm-v0.1.3/manifest.json)。

## RTX3060 NCU与当前工程结项（2026-11-24）

- [x] **3060-NCU-01 官方工具及权限**：NCU2025.3.1/官方包2025.3.1.4-1按NVIDIA索引SHA校验，独立解包，不升级生产image/驱动/CUDA。普通用户拒绝、sudo指标枚举通过；驱动admin-only保持。314475378字节官方工具本机离线备份SHA通过；首次截断包保留并逐前缀SHA只读补齐。
- [x] **3060-NCU-02 真实公开kernel采集及审计**：固定生产cuBLAS/512³，3次kernel×19指标全部有效；原ncu-rep可重新导入，CSV逐指标一致，三输出CPU FP64最大误差4.291534e-6。成功38文件、首次启动前0kernel权限失败29文件逐SHA审计通过。可信管理员诊断边界独立，生产worker零cap不改；0新CUDA/模型/正式Event/私有数据。[审计](evidence/20261124-3060-ncu-smoke-v2/verification.json)。
- [x] **ENGINEERING-CLOSEOUT 当前交付范围结项**：3060 GEMM与原3070 Attention/MLA、NCU及原始证据/运行/回退/镜像和工具备份就绪。NCU后再次实查qualified、原包指针/证书不变、GPU空闲/170W/无项目容器。结项manifest明确完整原计划并非逐项全通过，原P6范围/3070 GEMM/通用源码/学习延期不冒充通过。[结项](docs/PROJECT_CLOSEOUT.md) · [清单](releases/project-closeout-20261124/manifest.json)。

## GitHub与简历交付（2026-11-24）

- [x] **PUBLICATION-01 公开源码仓库**：按用户指定名称建立[Kaitounb/KaitouKernelAgent](https://github.com/Kaitounb/KaitouKernelAgent)，发布核心源码、CUDA产物、测试、上游归属与公开验收摘要；独立Git快照54项CPU回归通过，已知凭据与常见令牌扫描无匹配。私有连接/API key/独立清单、镜像和大型归档未上传，原本地证据不改。
- [x] **RESUME-01 第二项目更新**：TexPage第二项目由Ascend Agent替换为当前NVIDIA KernelAgent，写入真实闭环/NCU/实卡验收/回退结果与新GitHub链接；第一项目及专业技能/奖项原文逐段保持，编译一页且日志通过。原/新LaTeX及截图仅在本机.private备份，未进入GitHub。

## 当前交付顺序

工程阶段已按上述范围结项；NCU亦已完成。后续按运行文档使用/维护，新研究或学习任务按用户新指令另行开始。

3060 GEMM 0.1.3正式交付已完成，REL-00至04按实测勾选。日常使用正式launcher和production-store；应用故障可切RC2并显式development，随后switch release恢复。不要重跑任何历史stage/freeze/qualification/install/independent入口，不重建同版本wheel。学习复盘延后；原3070 GEMM与跨卡研究保持原结论。

## 每次结束填写的记录

```text
日期与任务 ID：
本次问题／预测：
运行硬件与环境 revision：
代码／候选 hash：
命令与工作目录：
正确性证据：
性能／profile 证据：
结论与限制：
是否满足验收：
下一项与阻塞：
```

## 下一次会话可直接使用的说明

先读README/HANDOFF/TASKS/PLAN及当前运行文档。3060 GEMM 0.1.3已正式交付，无在途GPU/API；按运行说明使用或恢复。禁止重放已消耗stage/freeze/qualification/independent/install入口或重建同版本wheel，不读取/传输/重跑旧sealed；任何新目标需独立资格。未启动新的研究或学习任务。此前接续说明和历史检查点已完整保存在[归档](docs/history/TASKS_BEFORE_013_CLEANUP_20261124.md)。
