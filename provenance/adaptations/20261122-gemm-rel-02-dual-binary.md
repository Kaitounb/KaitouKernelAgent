# GEMM-REL-02 双binary A/B外置裁决（实施前）

2026-11-22。不读／执行封存清单，不新增API，不重采P1-05或已拒绝P3-01 A/A。当前dev4保持原交付artifact，新增开发版本独立安装验收；原始失败与全部新证据留档。

| 来源／固定版本／符号 | 保留 | 必要适配与验证 |
|---|---|---|
| KernelBench 423217d9fda91e0c2d67e4a43bf62f96f6d104f1；本项目gemm_candidate.paired / source_timing.paired_witness | 原32节点、12replay、9对、capture／Event交替及warmup函数体，不改数值／计时阈值 | 现有函数已接受两个callable。用CPython原importlib loader的不同完整模块名、相同末级nka_gemm_v1加载两个.so，不新增FFI／ABI。实卡同名正确／已审查零输出artifact验证两侧独立，错误候选不能污染baseline |
| 原固定P1-03 gate／KDA acdd79343c5e9d0c7cb96289f26fd3f1051ce785 证据→晋升流程 | group_log_ratio及3-session锚点log聚合、4.303区间、1.05／1.02／0.97门槛 | 原gate只认一个固定binary及28项合同，新增双artifact descriptor和两侧私有ledger，保留32项现合同；实际report／cleanup／source／protocol／worker／binary须绑定，CPU正例不作为实卡收益 |
| ForgeKernel b2c8ae8510e02c28576808910a6b8827721ab7dc harness_guard | 固定公开输入、不可改harness／合同、失败保留incumbent | 包内bundle hash绑定两侧评测，known-rejection默认短路不采样。KEEP仅开发筛选建议，不更新冻结best／运行allowlist／发布指针；未完成封存与正式发布不自动激活 |
| KernelAgent e0647170da36ef9b059ac0bd3d60103aa4ed378b provider | 继续沿用已验收单次provider，当前工作0新调用 | 不为测试A/B而追加模型预算或NCU；API故障／中断策略不改变 |
| FlashInfer-Bench 15618a67e8d7b7f49ca61b5ba3b9f26a876d0bee生命周期＋已有P3-02 runner | 非root、无网络、两RO挂载、bounded pipes、tmpfs、timeout／cleanup | 把两份artifact放在同一只读artifact树，参考／ledger／裁决仍在host。真实故障回归与静态／mock分别标明 |

预注册：生产候选路径先两侧独立FP64 QA／输入保护，再candidate双加载上下文Sanitizer，三session同上下文candidate A/A通过才运行三session双binary A/B。mode、range变化不借用旧A/A；busy停止不采样。所有原始段与invalid witness保留，门槛不放宽。

本轮实卡适配验收只用已验证、不同SHA的正确93fb／66fcc及已审查零输出a4a25产物，覆盖尾部／offset／明确stream，验证双import身份与错误输出拒绝；必要时Compute Sanitizer。它不是优化候选，不重跑它们的原失败A/A；不通过identity换名绕开重采约束。CPU模拟三session A/A／A/B完整裁决与篡改负例、缺失／忙／不稳时不启动A/B；安装空cwd验证新入口／资源。正式性能门槛保持，实际是否做timing仅取决于后续新合格source，而非本轮适配冒烟。

实施前补充诊断协议：仅在GPU连续3次util0时，固定93fb-WMMA4对66fcc-cuBLAS（明确不同算法，不是旧WMMA A/A）运行一个session、两公开诊断shape(128,128,128)/(32,48,48)、两缓存策略、9对、32节点／12replay，72原始Event段。先两角色×两shape×16seeds=64项独立输出QA，再测前／后四项合同witness。该mode固定report.backends标成template/cublas，production gate只接受template/template且必须新候选三session A/A，因此此诊断不能用于KEEP；忙时跳过，不追加挑选测量。它只验收新双DSO Event通道，不是候选收益／原失败复测。

实际完成：首次dev5安装只在CPU统计对照测试import evaluators失败，GPU尚未开始，原wheel／stderr保留。测试改从包内固定原gate快照导入，package组件不变；新prefix／空cwd16项入口＋19项gate／调度CPU测试通过。真实双DSO16项QA中12正确／4零输出故障拒绝，baseline保持正确；两项Sanitizer各4项0errors、两侧hash错前置拒绝通过。新增64项图输入＋4项cuBLAS合同输出由可信host FP64验收。GPU空闲后只一批WMMA/cuBLAS诊断72原始Event段，16测前后合同witness／256图输出hash匹配，诊断明确不具备筛选资格，0真实candidate A/A／0新API。完整本机独立审计PASS，dev5 wheel SHA348a6e6ed8fca9051b570fb96548d4491c59a14c8f261df2b3e0e2172e5d2c7a。

普通新source路径已接full双侧ledger、candidate双上下文A/A和A/B；本轮硬件验收仅上述固定产物／故障／诊断通道，没有新source完整三A/A／三A/B的收益或KEEP实测。gate KEEP是开发筛选建议，CLI落盘SCREENING_PASS，不激活runtime或改冻结best。新版本分发通知已明确包中KernelBench／KernelAgent源码及其原完整MIT／Apache LICENSE位置。原paired_witness函数体／9份原保护文件／正式阈值不改。报告docs/DUAL_BINARY_EVALUATION.md；证据evidence/20261122-gemm-rel02-dual-installed-fixed-test-import。
