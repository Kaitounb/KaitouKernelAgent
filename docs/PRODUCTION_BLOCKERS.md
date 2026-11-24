# 当前生产准入与完整研究的剩余项

2026-11-24：3060 NCU真实3 kernel/19指标、报告重导入与FP64复核完成，当前限定工程交付已结项。[工程结项与延期范围](PROJECT_CLOSEOUT.md) · [NCU](RTX3060_NCU.md)。旧目标研究状态保持，以下历史阻塞不覆盖本次已交付范围。

2026-11-24 最新范围：RTX3060 GEMM 0.1.3正式交付已完成，完整三AA三AB/独立16项/真实回退/入口和恢复备份通过。[3060正式运行](RTX3060_GEMM_RUNTIME.md)。本次无新增阻塞；3070 GEMM原拒绝及4090目标C延期仍保留，3070 Attention/MLA0.1.1保持。下方为历史或其他目标，不能据其旧production=false覆盖本发行。


2026-11-23 当前交付已完成：3070固定可信Attention/MLA的0.1.1 qualified通道已实际部署。三AA三AB1080段通过，最坏单对1.935923%（原2%门槛）、组偏差0.447360%、AB最低0.999478866×；1668文件和1632输出数组独立CPU复核PASS。Attention新合成独立8/8、239私有文件/NumPy FP64重算PASS；MLA继承同AOT/未变数值合同的原独立8调用16输出，旧sealed不读不传不重跑。rc1安装/恢复37CPU、13输出17尝试保持；rc2严格SDK准入40CPU/2输出和四拒绝PASS；正式0.1.1安装40CPU、9输出11尝试（7qualified+2显式预览回退）、五项真实资格负例、正式→rc2→正式包切换及5160文件/NPY/边界CPU审计PASS。12运行文件与rc2逐字相同，wheel f2ef0ec1f054a9331a4134e6545e6ce2a84b5aae22923f81063cb5c7418a2706，证书261b6fcb4c44ba3de7c88998b6d7e7f375350c3eb3f37b1d208be6eab0f9d3f4。原独立分配输入540段/5.598675%拒绝保留；共享物理只读input ring方法更正在采样前冻结且只执行一次，原标准不变。生产范围为当前SM86固定FlashInfer Attention和page16/split1 MLA core，不覆盖Standalone Softmax、3070 GEMM旧失败或自由源码；原4090目标C按用户决定延后未运行。无项目GPU/API在途，无新增DeepSeek（仍190/700），无新CUDA编译或系统设置变化。入口及两个可运行示例见docs/SM86_ATTENTION_MLA_RUNTIME.md；发行清单releases/sm86-operators-v0.1.1/manifest.json。以下旧检查点按当时范围阅读，不重新启动已消耗批次/独立验收或重建旧版本wheel。

当前用户交付范围（3070固定Attention/MLA）已无未完成准入项。其余未通过项目维持各自状态；下方SM89/3070旧GEMM和NCU阻塞描述为历史或其他范围。

当前分项：SM89固定可信GEMM的稳定性、独立正确性、安装/回退/版本及包回滚、声明执行边界均已解决，0.1.0通道已部署。其余范围分别推进：SM86当前环境不继承资格；SM89任意源码worker仍缺通用隔离；Softmax/Attention/MLA目标生产化尚待；完整目标C仍无NCU。下面旧cuBLAS拒绝和双阻塞描述为历史，不表示当前SM89固定通道未通过。

2026-11-23 首个生产通道已交付：0.1.0的SM89固定可信cuBLAS qualified已在当前4090部署，入口/状态见docs/SM89_RUNTIME.md。证据：320QA+32Sanitizer；三AA三AB1728段(最大0.664015%、最低保护0.999909889×)；新服务端合成独立16/16、246私有文件本机FP64复核PASS(1.907349e-6)，未读/传/重跑旧3070/MLA封存数据；rc2预览安装27CPU/9输出11尝试及实际dev7包回滚PASS；正式0.1.0安装30CPU/6输出8尝试(5qualified+1显式预览退回)、581文件CPU审计PASS。正式wheel76224f3435764e5484e3434235e9f95a4b70cbe92476ac9f72c9df86bd3fb18f，资格证书9451476fc9ac796ff1e5c4617c9414b0b7006366a0790f184618a8747a386c77，发行清单releases/sm89-gemm-v0.1.0/manifest.json。可信控制root、worker UID/GID65534/零cap/NoNewPrivs、IPv4/IPv6拒绝/Unix IPC保留、控制目录不可读/产物输入只读/有界日志超时、故障和版本/实际包回滚均实测；无PID/mount/network namespace或RAM cgroup独立限额，不执行任意模型源码。production=true仅限SM89_REVIEWED_FIXED_CUBLAS；SM86和Attention/MLA及完整目标C保持各自状态，不冒称全项目生产通过。所有GPU/API任务已结束，0新DeepSeek(仍190/700)，无新增Event重采。新控制Python和入口在/data，旧包/证据/权限保留；当前SSH显式用.private/server-gpu-next-connection.json。以下历史检查点按当时范围阅读。

2026-11-23 4090 v2完整实测已通过并归档：SM89/Torch2.7.1+cu126/CUDA12.6.85/driver570.86.10；两可信cuBLAS绑定fresh build，320QA、32Sanitizer(0errors)、crash/timeout拒绝及两fresh恢复通过。三AA+三AB共1728原始Event段，AA最坏单对0.664015%、最坏组偏差0.570454%（也满足旧2%），AB最低保护0.999909889×>=.97。917文件逐SHA/NKA1/NPY/见证/Graph/pilot/统计CPU审计AUDIT_PASS_STABILITY_PASS；protocol8a023ee57b57948c72866304e74dabd4f50fe27a2249278473c31e4572961065，归档68eb546027baf486b5db2ff46bca89876f7fa6bcf42696d8e0bb99d89ea5a86b。报告docs/SM89_CUBLAS_STABILITY.md，evidence/20261123-sm89-cublas-v2。所有会话已结束，GPU0%/1MiB/无compute进程；0新API(仍190/700)/0NCU/0sealed/0宿主时钟变更，无best晋升或优化收益。实例无Docker，本轮是可信native固定SDK，不构成完整源码worker隔离或生产发布；production仍false。下一项SM89目标安装/回滚及相应执行边界，P6-03完整目标C、P6-04/05仍未整体验收；SM89结论不回填3070。旧连接/备份保持，新连接须显式使用.private/server-gpu-next-connection.json。不要重跑本批GPU入口，只读audit_sm89_cublas。

2026-11-23 用户修订部署标准：新增workloads/gemm_deployment_stability_v2.json，A/A单对<3%、组均值偏差<1.5%，完整三session等原采样要求不变；仅部署准入，原正式研究/搜索门槛及旧2%拒绝不改。四批历史CPU逐SHA/原始段复核完成：完整锁频1秒2.040408%（最大组偏差0.286466%）与250ms2.849277%在新v2下A/A单项通过；原4.274482%及不完整HAGS On/5.142371%不通过。明确为观察后标准修订/回顾复核，0新GPU/API、0AB/证书；不把HAGS Off成绩继承给当前HAGS On。下一步等4090连接，在新SM89环境先正确性再冻结v2/三AA→三AB；NCU可选，隔离条件另验收，目标卡通过不回填3070。报告docs/DEPLOYMENT_STABILITY_V2.md，结果evidence/20261123-deployment-stability-v2/verification.json。P6-03/04/05未整体验收，预算190/700。以下旧2%检查点按当时政策阅读。

2026-11-23 重启后最终检查点：用户手动重启3070已完成，DxDiag实际HAGS Supported/Enabled均True，bootUTC=2026-11-23T06:52:14.5000000Z；同GPU/驱动591.74/固定Docker镜像。原dev6/AOT从归档按15控制文件/1456文件SHA恢复到持久目录。HAGS新协议ec6bbbc...保留原1秒host与全部阈值：320QA通过，仅完成AA0/288段，最坏单对5.142371%（原2%门槛）；下一轮前GPU空闲条件失败停止，3AA未完成、0AB、锁频恢复成功。590文件归档+物理环境/帧/参考/Graph/clock CPU审计AUDIT_PASS_INCOMPLETE_QUALIFICATION_REJECT；首次后台启动的0QA/0Event WSL失败21文件另存，不重采任何已采身份。证据evidence/20261123-cublas-hags-qualification/verification.json；本轮无GPU/API在途，无资格证书/生产发布。4090可不用NCU执行可信SDK正确性和Event；NCU只阻塞完整P6-03目标C，不是首个SM86 GEMM生产运行门槛。现4090无已验收源码隔离，仍不执行自由生成CUDA；不为NCU单独要求换服务器，旧实例和备份未动。P6-03/04/05保持未勾，预算仍190/700，独立sealed不再执行。详docs/4090_WITHOUT_NCU.md、docs/PRODUCTION_BLOCKERS.md。下方旧检查点按历史阅读。

2026-11-23更新：4090不需要NCU即可运行可信CUDA SDK、正确性检查和普通CUDA Event计时。按照PLAN.md“首个生产发布门槛”，NCU不是SM86 GEMM生产运行依赖；此前文档把首版产品准入与完整跨卡研究并列为双阻塞，分类过严，在此纠正。没有修改已冻结的研究协议或验收阈值。

| 范围 | 仍需完成的实际验收 | 当前状态 |
|---|---|---|
| 首个SM86 GEMM生产发布 | 默认cuBLAS最终稳定配对、真实资格证书、包含资格/环境边界的版本安装与回退 | 旧4.274%/2.849%/2.040408%均拒绝。用户重启后DxDiag实际HAGS On，新boot与原字节产物已冻结；原1秒evaluator、2%/1.5%/.97不变；320QA通过、AA0/288段最坏5.142371%，随后GPU忙停止，三AA未完成，0AB，恢复成功，无资格 |
| 完整跨卡C研究/P6-03 | 4090实际硬件反馈与对应重新优化 | 目标NCU实际ERR_NVGPUCTRPERM；保持未运行，不用3070指标代替，不将普通Event结果冒充C |
| 4090自由源码worker扩展 | 已验证的独立低权限构建/运行隔离 | 现实例无Docker/user-net namespace，仍只运行可信SDK/受控配置；NCU权限与此隔离要求分别处理 |

GEMM和MLA冻结独立正确性已完成，禁止重跑sealed。生产资格必须来自真实配对与审计，不可由CPU模拟证书替代。已有[RC1预览](RC1_PREVIEW.md)可运行；缺少真实证书时qualified通道在GPU前拒绝。完整P6-03/04/05均尚未整体验收，不把首版范围说明写成完整计划完成。

HAGS新环境：Windows bootUTC `2026-11-23T06:52:14.5000000Z`、驱动591.74、同3070 UUID，DxDiag实际Supported/Enabled均True。恢复原dev6 wheel和许可AOT到持久目录，15控制文件和1456 AOT文件逐SHA一致；8秒有限活动检查14个高负载样本1200/6800MHz，结束后锁频复原。新资格protocol `ec6bbbc176a4c67df83bdc362feb3f95ce1cad1e73001db4a88f1692e947959d` 首次前置WSL失败0样本保留，同协议继续后因GPU忙停止；原始证据 `evidence/20261123-cublas-hags-qualification`。未读取私有清单、未调用模型、未改变旧wheel/二进制/失败记录。

以下段落保留旧拒绝与修复历史，不覆盖上述重启后事实。

cuBLAS失败的公开原始目录已用SHA分段归档，610个文件逐字核验；独立CPU审计返回 `AUDIT_PASS_QUALIFICATION_REJECT`。三个超限组均为16输入ring：session0/dev-03为4.274%、session1/dev-06为2.267%、session2/dev-03为3.713%；配对几何均值仍接近1。没有足够连续时钟/调度证据确定根因，不能把失败归因写成已证明的WDDM、DVFS或外部占用，也不选择单输入策略另称资格通过。

MLA此前的授权阻塞已经解决：用户具体回复“批准”后，按原冻结协议一次执行8/8 GPU调用、16/16输出通过，204私有文件CPU复核PASS。精确值未显示给模型，参考与完整清单不挂载给候选；0新模型调用、0性能采样。[独立验收报告](MLA_RC1_ACCEPTANCE.md)。

原自动审批拒绝原文继续保留：

> 该命令会把未获具体授权的私有MLA验收清单（精确shape/seed及派生输入）传到3070并执行；用户此前仅明确批准过GEMM清单传输，泛化的“完成项目”不足以授权这项敏感外发。

历史拒绝位于 `evidence/20261123-mla-private-approval-block`，后续具体授权位于 `evidence/20261123-mla-independent-approval`，新GPU/CPU汇总位于 `evidence/20261123-mla-independent-acceptance`。没有修改旧冻结预览manifest或拒绝原文；新状态清单另存 `releases/full-project-rc1-mla-acceptance-update`。

原GEMM独立16/16、公开恢复与版本回滚继续有效；已经完成的两类封存验收不再运行。FlashMLA专家固定版本在SM86/SM89不支持，专家成绩为null；不以FlashInfer冒称替代成绩。P6-04仍因最终性能资格未完成而保持未勾，P6-03目标C及P6-05生产发布也未勾。

本次修复不是已完成准入：HAGS配置已写入并读回，Windows未重启，实际运行模式仍待确认；维护中断应用/WSL/SSH，需要单独安排。[实际诊断、两次新拒绝与复核](STABILITY_REPAIR.md) · [3070维护与恢复](3070_HAGS_MAINTENANCE.md)。原wheel及冻结文件未变。
