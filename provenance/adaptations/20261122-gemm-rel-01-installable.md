# GEMM-REL-01 可安装离线优化入口（实施前）

2026-11-22。目标是脱离仓库／空cwd安装后组合已验证组件；本轮不新增API调用、不读取封存验收、不重采P1-05或P3-01失败A/A，不改cuBLAS allowlist。

| 来源／固定revision／符号 | 保留 | 必要适配与验证 |
|---|---|---|
| KernelAgent e0647170da36ef9b059ac0bd3d60103aa4ed378b utils/providers/base.py/env_config.py/openai_base.py；已验证P3-01 call_once | 原provider文件和DeepSeek JSON/non-thinking、60秒／768输出／零重试 | 原文件随开发wheel附许可／SHA；package内加载，不依赖仓库。只读DEEPSEEK_API_KEY环境；本轮只用历史响应／CPU mock，0新付费API |
| KernelBench 423217d9fda91e0c2d67e4a43bf62f96f6d104f1 eval/timing/dataset；gemm_regression/load_trusted | 原文件、loader、输入和固定Event路径；P3-02 DockerBuilder／P3-01 source runner | 固定只读资源树随包分发，bundle manifest校验。builder只挂公共源和依赖；host reference/ledger、控制状态和密钥不入worker。安装wheel空cwd实际编译／正确性验证 |
| KDA acdd79343c5e9d0c7cb96289f26fd3f1051ce785 docs/agent-flow.md | parent→proposal→review→build→verify→decision证据阶段 | argparse入口用现有registry的flock／fsync／原子写方式补一次调用预算与持久状态。中断不隐式重试模型或GPU；同目录幂等／锁互斥，CPU故障夹具 |
| ForgeKernel b2c8ae8510e02c28576808910a6b8827721ab7dc harness_guard | 原single CUDA patch、license／host launcher／parent guard | 原patch应用器迁入package只改import。显式recount及review SHA绑定；审查源码和源哈希绑定是控制侧流程，不新增用户批准请求。已知失败identity默认不再测量 |
| FlashInfer-Bench 15618a67e8d7b7f49ca61b5ba3b9f26a876d0bee 生命周期与已验收本地Docker适配 | 有界管道／无RW宿主挂载／资源／cleanup | 只移动已验证适配到包命名空间；检查模式不计时／不晋升，新编译／GPU QA验证安装后组合路径。未实现双binary A/B则明确NO_PROMOTION，不宣称正式发布 |

新增设计仅为已确认缺口：现有experiments依赖ROOT/仓库路径且一次预算未覆盖中断重入，不能安装空cwd运行。包中保留固定公开CUDA／合同／开发集／上游许可证；不打包`.private`、模型凭据或封存清单。运行包新开发版本，历史dev2 wheel与证据不覆盖。

预定验证：源码来源／bundle SHA；CPU预算、错误响应、锁／中断／重复运行／secret redaction；新prefix安装空cwdCLI；历史DeepSeek响应保留FAILED格式信息及显式recount，已知A/A失败拒绝复用；固定审查源码checks-only走无GPU构建＋独立GPU正确性／Sanitizer，不重新计时；默认cuBLAS回退不变。模型和GPU mock明确与实卡分列，按实际范围更新TASKS。

传输适配：上次完整raw包255MiB文本传输约20分钟，重复NPY同时存在于frame ZIP与独立输出。新collector只对固定NKA1内ZIP_STORED成员按原字节边界分片，复用已有只读SHA blob；重建每个原文件后必须匹配whole-file SHA／长度，仍保留所有raw frame／NPY／日志及完整manifest，不重新压缩或重写候选数据，不把复用旧字节视为沿用旧GPU结论。CPU测试验证分片可无损重建；新增GPU判定仍来自本次安装后的独立宿主FP64。

首个dev3 wheel实际安装／空cwd11条命令、160项输出与Sanitizer各8项0errors、一次cuBLAS输出通过；独立本机审计PASS。CPU审计曾因collector记录.DS_Store而误要求wheel包含它，失败留档；wheel正确未包含Finder元数据，修正audit／后续collector过滤，不重跑GPU。

收尾CPU回归发现empty choices触发IndexError、live预留被标成已完成调用，原失败测试保存。dev4只修command的拒绝分支和调用状态（预留后未知用null/UNCONFIRMED，收到response才1），版本／bundle更新；14项CPU回归通过。最终wheel新prefix安装，从空cwd再跑这14项、已知拒绝／坏响应／幂等／预算与CLI版本指针检查；与dev3 wheel逐文件比较，只有command、包版本和bundle manifest可变。builder／worker／数值／Event／运行时及公开资源必须逐字相同，因此继承dev3刚取得的GPU组合证据，不重复编译／GPU／计时。新artifact hash与dev3原始实测分别保留。

最终实测完成：dev3空cwd11条命令、160项输出／16项Sanitizer／一次cuBLAS全部通过；dev4空cwd13条命令、14项已安装CPU回归和实际版本回滚通过。两个wheel只三份package文件不同，独立GPU及最终delta审计均PASS；默认release allowlist／部署策略及9份原保护文件hash不变。最终wheel SHA256 5a30f76189e56b1094591a1c802329f6e43c70a572674163bb447d190c0cdf49；完整证据与acceptance见evidence/20261122-gemm-rel01-dev4-delta，操作说明docs/OFFLINE_OPTIMIZER.md。17项本机CPU方法（14优化政策＋3传输）和已安装14项分别标记；0新API、0 Event、无封存集／生产发布。历史P3-01审计的cuBLAS不变检查从整个CLI字节hash改为未改的registry/deployment/releases三文件hash，以允许新增独立CLI子命令；旧源码完整审计重放再次PASS，原计时拒绝不变。
